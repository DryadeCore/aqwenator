//! Qwen3.5 (`qwen3_5`) dense hybrid model: weights on device + a batched forward step.
//!
//! One `step` runs a mixed batch: any number of sequences, each contributing a chunk of
//! tokens (a prefill chunk or a single decode token). Full-attention layers read a paged KV
//! cache; gated-delta-net layers keep a fixed-size recurrent state per sequence slot.

use crate::config::{LayerKind, TextConfig};
use crate::cublaslt::CUDA_R_16BF;
use crate::gpu::{Gpu, Prof, cfg, cfg_smem, de};
use cuda_core::LaunchConfig;
use crate::weights::Checkpoint;
use anyhow::Result;
use half::bf16;
use std::path::Path;
use std::time::Instant;

const P: &str = "model.language_model.";

pub struct Layer {
    pub kind: LayerKind,
    in_norm: u64,
    post_norm: u64,
    // gated delta net
    gdn_proj: u64,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    gnorm: u64,
    out_proj: u64,
    // full attention
    qkv: u64,
    q_norm: u64,
    k_norm: u64,
    o_proj: u64,
    // mlp
    gate_up: u64,
    down: u64,
    // per-slot state pools
    delta_state: u64,
    conv_state: u64,
    k_cache: u64,
    v_cache: u64,
}

pub struct Scratch {
    amax_v: u64,
    amax_i: u64,
    // split-K decode attention partials
    attn_part: u64,
    attn_ml: u64,
    // chunked GDN + tiled attention (prefill)
    chunk_start: u64,
    chunk_len: u64,
    seq_chunk0: u64,
    seq_nchunk: u64,
    tile_start: u64,
    tile_len: u64,
    gu: u64,
    gw: u64,
    gqp: u64,
    gkp: u64,
    gintra: u64,
    gdecay: u64,
    go: u64,
    // tensor-core chunked GDN (per chunk-head: Qn, Kn, U, W, G, h snapshot, Vn^T)
    tqn: u64,
    tkn: u64,
    tu: u64,
    tw: u64,
    tg: u64,
    tht: u64,
    tvnt: u64,
    ids: u64,
    tok_pos: u64,
    tok_seq: u64,
    seq_start: u64,
    seq_len: u64,
    seq_slot: u64,
    seq_slot_out: u64,
    // state taps (prompt checkpoints inside a prefill chunk)
    seq_tap_n: u64,
    seq_tap_len: u64,
    seq_tap_slot: u64,
    rows: u64,
    tables: u64,
    hfinal: u64,
    x: u64,
    h: u64,
    proj: u64,
    conv: u64,
    mix: u64,
    delta: u64,
    q: u64,
    logits: u64,
    tok: u64,
    // MTP head (stubs unless the head is loaded)
    next: u64,
    mtp_hfinal: u64,
    mtp_logits: u64,
    mtp_tok: u64,
}

/// The multi-token-prediction head: one full-attention layer over
/// `fc([norm_e(embed(next token)), norm_h(final-normed hidden)])`, then its own norm and the
/// shared output head. At position p it predicts token p + 2.
pub struct Mtp {
    layer: Layer,
    fc: u64,
    norm: u64,
    pre_e: u64,
    pre_h: u64,
}

pub struct LoadStats {
    pub open_s: f64,
    pub upload_s: f64,
    pub weight_bytes: usize,
}

/// Chunk length of the CUDA-core chunked gated delta rule (must match the kernels).
pub const GDN_CHUNK: usize = 32;
/// Chunk length of the tensor-core chunked gated delta rule (must match the kernels).
pub const GDN_TC_CHUNK: usize = 64;
/// Dynamic shared memory of the tensor-core GDN kernels (bytes; must match the kernels).
const GDN_TC_PREP_SMEM: u32 = 70672;
const GDN_TC_SCAN_SMEM: u32 = 93696;
const GDN_TC_OUT_SMEM: u32 = 88832;
/// Value columns per `gdn_tc_scan` block (must match the kernel).
const GDN_TC_SCAN_VB: u32 = 32;
/// Query rows per tile of the prefill attention kernel (must match the kernels).
pub const ATTN_QTILE: usize = 16;
/// Blocks per row in the two-phase argmax.
pub const ARGMAX_PARTS: usize = 32;
/// Capacity of the split-K partials in (row x slice) units: one row per sequence with up to
/// `MAX_ATTN_SPLITS` slices, or up to `MAX_DECODE_ROWS_PER_SEQ` rows per sequence with
/// row-local slicing (at most `ROW_LOCAL_SPLITS` slices).
fn attn_part_rows(max_seqs: usize) -> usize {
    max_seqs * MAX_ATTN_SPLITS.max(MAX_DECODE_ROWS_PER_SEQ * ROW_LOCAL_SPLITS)
}

/// Max tokens one decode-path sequence may feed in a step (speculative verification).
pub const MAX_DECODE_ROWS_PER_SEQ: usize = 9;
/// Max slices per row with row-local (fixed) decode-attention slicing (must match the kernel).
pub const ROW_LOCAL_SPLITS: usize = 32;
/// Upper bound on split-K slices for decode attention.
pub const MAX_ATTN_SPLITS: usize = 128;

/// Engine sizing, fixed at load time (every buffer is preallocated).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Max tokens in one step (prefill chunks + decode tokens).
    pub max_tokens: usize,
    /// Max concurrent sequences (recurrent-state slots).
    pub max_seqs: usize,
    /// Max context length per sequence.
    pub max_ctx: usize,
    /// KV cache page size in tokens.
    pub block_size: usize,
    /// KV page pool size in bytes; 0 = enough for every slot at `max_ctx`. With a budget,
    /// slots can outnumber what fits at full context (idle sessions share the pool).
    pub kv_budget_bytes: usize,
    /// Pages in the pool, resolved at load time from the budget.
    pub kv_blocks: usize,
}

impl Limits {
    /// Token rows a step may hold: the token budget plus headroom for speculative draft rows
    /// (drafts do not count against the budget).
    pub fn max_rows(&self) -> usize {
        self.max_tokens + self.max_seqs * (MAX_DECODE_ROWS_PER_SEQ - 1)
    }
    pub fn max_chunks(&self) -> usize {
        self.max_tokens / GDN_CHUNK + self.max_seqs
    }
    /// Upper bound on 64-token chunks in one step: full chunks plus one partial per sequence.
    pub fn max_tc_chunks(&self) -> usize {
        self.max_tokens / GDN_TC_CHUNK + self.max_seqs.min(self.max_tokens)
    }
    pub fn max_tiles(&self) -> usize {
        self.max_tokens / ATTN_QTILE + self.max_seqs
    }
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_ctx.div_ceil(self.block_size)
    }
    pub fn num_blocks(&self) -> usize {
        if self.kv_blocks > 0 { self.kv_blocks } else { self.max_seqs * self.max_blocks_per_seq() }
    }
}

/// One step's work. Token-level arrays have one entry per token; sequence-level arrays one
/// entry per sequence in the batch. `sample_rows` lists the token rows whose logits are
/// sampled (the last row of each sequence that finishes its chunk of input this step).
#[derive(Debug, Default, Clone)]
pub struct Batch {
    pub tokens: Vec<u32>,
    pub tok_pos: Vec<u32>,
    pub tok_seq: Vec<u32>,
    pub seq_start: Vec<u32>,
    pub seq_len: Vec<u32>,
    pub seq_slot: Vec<u32>,
    pub block_tables: Vec<u32>,
    pub sample_rows: Vec<u32>,
    /// Sequences `0..n_decode` take the decode path: DeltaNet layers run them token by token
    /// with the state in registers, attention runs one split-K row per token. Normally one
    /// token each; a speculative step feeds pending tokens plus drafts.
    pub n_decode: usize,
    /// Slot each sequence's updated state is written to (empty = same as `seq_slot`). A
    /// speculative step writes to a spare slot so the old state survives a rejection.
    pub seq_slot_out: Vec<u32>,
    /// State taps, one `(chunks, slot)` per sequence (empty = none): after `chunks` full
    /// 64-token chunks of a prefill sequence, its recurrent and conv state are also written to
    /// `slot`. Only with the tensor-core chunked DeltaNet path (`Model::supports_tap`).
    pub seq_tap: Vec<(u32, u32)>,
    /// Per sequence, the token that follows its last row if it is already known (a prefill
    /// chunk that does not end the prompt), else `u32::MAX`. Only the MTP head reads it; empty
    /// = unknown for every sequence.
    pub next_tok_last: Vec<u32>,
}

pub struct Model {
    pub cfg: TextConfig,
    pub gpu: Gpu,
    embed: u64,
    final_norm: u64,
    lm_head: u64,
    pub layers: Vec<Layer>,
    /// MTP draft head (AQ_MTP=1 and the checkpoint has one).
    mtp: Option<Mtp>,
    /// Run the MTP head this step (set by the engine; needs `has_mtp`).
    pub mtp_on: bool,
    /// MTP predictions of the last step, one per sampled row: the token expected after the
    /// one that row sampled. Empty when the head did not run.
    pub last_mtp: Vec<u32>,
    s: Scratch,
    pub limits: Limits,
    pub load: LoadStats,
    pub prof: Option<Prof>,
    /// Captured decode graphs, keyed by (padded batch size, attention splits), with the
    /// profile mark names recorded inside them (when profiling).
    graphs: std::collections::HashMap<(usize, usize), (u64, Vec<&'static str>)>,
    num_sms: usize,
    /// Target decode-attention blocks per SM when choosing split-K slices (AQ_ATTN_FILL).
    attn_fill: usize,
    /// Split-K slices for this step's decode attention (set by `upload`).
    cur_splits: usize,
    /// Decode-path token rows of this step (set by `upload`).
    cur_nd_rows: usize,
    /// This step has at least one state tap (set by `upload`).
    cur_tap: bool,
    /// Logits buffer of the last step that sampled, whichever lane ran it.
    last_logits: u64,
    /// Fixed decode-attention slice size in use this step (0 = slices sized per step).
    cur_split_keys: u32,
    /// Fixed decode-attention slice size (keys); 0 = size slices from the batch. Fixed slices
    /// make decode attention batch-invariant (AQ_ATTN_SPLIT_KEYS, or set by speculation).
    pub attn_split_keys: usize,
    pub use_graphs: bool,
    /// Prefill work lists of the step being run (set by `upload`).
    cur_chunks: usize,
    cur_tiles: usize,
    /// A/B switches back to the reference kernels (AQ_GDN_RECURRENT=1, AQ_ATTN_NAIVE=1,
    /// AQ_GDN_CHUNK_CC=1 for the CUDA-core chunked GDN).
    ref_gdn: bool,
    ref_attn: bool,
    /// AQ_ATTN_TILE=1: use the CUDA-core tiled prefill attention instead of tensor cores.
    tile_attn: bool,
    /// Dynamic shared memory of `attention_tc` for this model's GQA group size.
    attn_tc_smem: u32,
    /// 16-token sub-tiles per `attention_tc` block (AQ_ATTN_QT, default 2): each K/V tile a
    /// block loads serves `16 * attn_qt` query tokens.
    attn_qt: usize,
    cc_gdn: bool,
    /// Rows of the active lane's scratch: the GEMM row cap and the per-step token limit.
    s_cap: usize,
    /// The inactive lane's per-step state when a decode lane exists (co-execution); swapped
    /// with the active fields by `set_lane`.
    alt: Option<LaneState>,
    /// One completion event per lane (co-execution).
    lane_events: Vec<cuda_core::Event>,
}

/// Everything that is per step and per lane: scratch and metadata buffers, the work lists of
/// the step being run, and the CUDA graphs captured against this lane's buffers.
struct LaneState {
    s: Scratch,
    cap: usize,
    graphs: std::collections::HashMap<(usize, usize), (u64, Vec<&'static str>)>,
    cur_chunks: usize,
    cur_tiles: usize,
    cur_splits: usize,
    cur_nd_rows: usize,
    cur_split_keys: u32,
}

/// Allocate one lane's scratch for `lim.max_rows()` token rows (and `lim.max_seqs` sequences).
fn alloc_scratch(gpu: &mut Gpu, cfg: &TextConfig, lim: &Limits, cc_gdn: bool, mtp: bool) -> Result<Scratch> {
    let mrows = if mtp { lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ } else { 0 };
    let hd = cfg.hidden_size;
    let wide = cfg.gdn_proj_ld().max(cfg.attn_proj_dim()).max(2 * cfg.intermediate_size);
    let mixw = cfg.value_dim().max(cfg.num_attention_heads * cfg.head_dim).max(cfg.intermediate_size);
    let t = lim.max_rows();
    let sq = lim.max_seqs;
    let hvd = cfg.linear_num_value_heads * 128;
    let hq = cfg.num_attention_heads;
    // Only the selected chunked-GDN path gets real scratch; the other gets a stub.
    let (cct, tcc) = if cc_gdn { (t, 0) } else { (0, lim.max_tc_chunks() * cfg.linear_num_value_heads) };
    Ok(Scratch {
        amax_v: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * ARGMAX_PARTS * 4)?,
        amax_i: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * ARGMAX_PARTS * 4)?,
        attn_part: gpu.alloc(attn_part_rows(sq) * hq * cfg.head_dim * 4)?,
        attn_ml: gpu.alloc(attn_part_rows(sq) * hq * 2 * 4)?,
        chunk_start: gpu.alloc(lim.max_chunks() * 4)?,
        chunk_len: gpu.alloc(lim.max_chunks() * 4)?,
        seq_chunk0: gpu.alloc(sq * 4)?,
        seq_nchunk: gpu.alloc(sq * 4)?,
        tile_start: gpu.alloc(lim.max_tiles() * 4)?,
        tile_len: gpu.alloc(lim.max_tiles() * 4)?,
        gu: gpu.alloc(cct * hvd * 4)?,
        gw: gpu.alloc(cct * hvd * 4)?,
        gqp: gpu.alloc(cct * hvd * 4)?,
        gkp: gpu.alloc(cct * hvd * 4)?,
        gintra: gpu.alloc(if cc_gdn { lim.max_chunks() * cfg.linear_num_value_heads * GDN_CHUNK * GDN_CHUNK * 4 } else { 0 })?,
        gdecay: gpu.alloc(lim.max_chunks() * cfg.linear_num_value_heads * 4)?,
        go: gpu.alloc(cct * hvd * 2)?,
        tqn: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tkn: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tu: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tw: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tg: gpu.alloc(tcc * GDN_TC_CHUNK * 4)?,
        tht: gpu.alloc(tcc * 128 * 128 * 2)?,
        tvnt: gpu.alloc(tcc * 128 * GDN_TC_CHUNK * 2)?,
        ids: gpu.alloc(t * 4)?,
        tok_pos: gpu.alloc(t * 4)?,
        tok_seq: gpu.alloc(t * 4)?,
        seq_start: gpu.alloc((sq + 1) * 4)?,
        seq_len: gpu.alloc((sq + 1) * 4)?,
        seq_slot: gpu.alloc((sq + 1) * 4)?,
        seq_slot_out: gpu.alloc((sq + 1) * 4)?,
        seq_tap_n: gpu.alloc((sq + 1) * 4)?,
        seq_tap_len: gpu.alloc((sq + 1) * 4)?,
        seq_tap_slot: gpu.alloc((sq + 1) * 4)?,
        rows: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * 4)?,
        tables: gpu.alloc((sq + 1) * lim.max_blocks_per_seq() * 4)?,
        hfinal: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * hd * 2)?,
        x: gpu.alloc(t * hd * 2)?,
        h: gpu.alloc(t * hd * 2)?,
        proj: gpu.alloc(t * wide * 2)?,
        conv: gpu.alloc(t * cfg.conv_dim() * 2)?,
        mix: gpu.alloc(t * mixw * 2)?,
        delta: gpu.alloc(t * hd * 2)?,
        q: gpu.alloc(t * cfg.num_attention_heads * cfg.head_dim * 2)?,
        logits: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * cfg.vocab_size * 2)?,
        tok: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * 4)?,
        next: gpu.alloc(if mtp { t * 4 } else { 0 })?,
        mtp_hfinal: gpu.alloc(mrows * hd * 2)?,
        mtp_logits: gpu.alloc(mrows * cfg.vocab_size * 2)?,
        mtp_tok: gpu.alloc(mrows * 4)?,
    })
}

/// Shared-memory bytes of `attention_tc`: Q rows (16 per sub-tile and query head of the group)
/// plus one 32-key K tile and one V tile, all 256-wide bf16.
fn attention_tc_smem(group: usize, qt: usize) -> u32 {
    ((qt * group * 16 + 2 * 32) * 256 * 2) as u32
}

/// Padded batch sizes that get a captured decode graph.
const DECODE_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512];

/// Upload several bf16 matrices concatenated along rows into one buffer.
fn concat_bf16(gpu: &mut Gpu, ck: &Checkpoint, names: &[String]) -> Result<u64> {
    concat_bf16_padded(gpu, ck, names, 0)
}

/// Same, with `pad_bytes` of zeros after the last matrix (zero rows that round the GEMM's
/// N up to a tile-friendly size).
fn concat_bf16_padded(gpu: &mut Gpu, ck: &Checkpoint, names: &[String], pad_bytes: usize) -> Result<u64> {
    let views = names.iter().map(|n| ck.bf16(n)).collect::<Result<Vec<_>>>()?;
    let total: usize = views.iter().map(|v| v.bytes.len()).sum::<usize>() + pad_bytes;
    // Not zeroed: the uploads overwrite everything but the padding.
    let p = gpu.alloc_uninit(total)?;
    let mut off = 0u64;
    for v in views {
        gpu.upload_bytes(p + off, v.bytes)?;
        off += v.bytes.len() as u64;
    }
    if pad_bytes > 0 {
        gpu.zero(p + off, pad_bytes)?;
        gpu.sync()?;
    }
    Ok(p)
}

impl Model {
    pub fn load(dir: &Path, limits: Limits) -> Result<Self> {
        let t0 = Instant::now();
        let cfg = TextConfig::load(dir)?;
        let ck = Checkpoint::open(dir)?;
        let open_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut gpu = Gpu::new(0)?;
        let one = |n: &str| vec![format!("{P}{n}")];

        let embed = concat_bf16(&mut gpu, &ck, &one("embed_tokens.weight"))?;
        let final_norm = concat_bf16(&mut gpu, &ck, &one("norm.weight"))?;
        let lm_head = if cfg.tied() || !ck.has("lm_head.weight") {
            embed
        } else {
            concat_bf16(&mut gpu, &ck, &["lm_head.weight".to_string()])?
        };

        let hv = cfg.linear_num_value_heads;
        let mut lim = limits;
        // The MTP head is opt-in: it adds one attention layer's KV to every page.
        let want_mtp = std::env::var("AQ_MTP").as_deref() == Ok("1") && ck.has("mtp.fc.weight") && ck.has("mtp.layers.0.self_attn.q_proj.weight");
        if lim.kv_budget_bytes > 0 {
            let n_full = (0..cfg.num_hidden_layers).filter(|&l| cfg.kind(l) == LayerKind::Full).count() + want_mtp as usize;
            let page_bytes = n_full * 2 * lim.block_size * cfg.num_key_value_heads * cfg.head_dim * 2;
            lim.kv_blocks = lim.kv_budget_bytes / page_bytes;
        }
        let kv_bytes = (lim.num_blocks() + 1) * lim.block_size * cfg.num_key_value_heads * cfg.head_dim * 2;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        // Seconds spent allocating and zeroing recurrent state and KV pages (not weights).
        let mut state_s = 0.0f64;
        for l in 0..cfg.num_hidden_layers {
            let n = |s: &str| format!("{P}layers.{l}.{s}");
            let kind = cfg.kind(l);
            let mut layer = Layer {
                kind,
                in_norm: concat_bf16(&mut gpu, &ck, &[n("input_layernorm.weight")])?,
                post_norm: concat_bf16(&mut gpu, &ck, &[n("post_attention_layernorm.weight")])?,
                gdn_proj: 0,
                conv_w: 0,
                a_log: 0,
                dt_bias: 0,
                gnorm: 0,
                out_proj: 0,
                qkv: 0,
                q_norm: 0,
                k_norm: 0,
                o_proj: 0,
                gate_up: concat_bf16(&mut gpu, &ck, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?,
                down: concat_bf16(&mut gpu, &ck, &[n("mlp.down_proj.weight")])?,
                delta_state: 0,
                conv_state: 0,
                k_cache: 0,
                v_cache: 0,
            };
            match kind {
                LayerKind::Linear => {
                    let la = |s: &str| n(&format!("linear_attn.{s}"));
                    let pad = (cfg.gdn_proj_ld() - cfg.gdn_proj_dim()) * cfg.hidden_size * 2;
                    layer.gdn_proj = concat_bf16_padded(
                        &mut gpu,
                        &ck,
                        &[la("in_proj_qkv.weight"), la("in_proj_z.weight"), la("in_proj_b.weight"), la("in_proj_a.weight")],
                        pad,
                    )?;
                    layer.conv_w = concat_bf16(&mut gpu, &ck, &[la("conv1d.weight")])?;
                    layer.a_log = gpu.alloc_upload(&ck.f32_vec(&la("A_log"))?)?;
                    layer.dt_bias = gpu.alloc_upload(&ck.f32_vec(&la("dt_bias"))?)?;
                    layer.gnorm = gpu.alloc_upload(&ck.f32_vec(&la("norm.weight"))?)?;
                    layer.out_proj = concat_bf16(&mut gpu, &ck, &[la("out_proj.weight")])?;
                    let t = Instant::now();
                    layer.delta_state = gpu.alloc((lim.max_seqs + 1) * hv * 128 * 128 * 4)?;
                    layer.conv_state = gpu.alloc((lim.max_seqs + 1) * cfg.conv_dim() * 3 * 2)?;
                    gpu.sync()?;
                    state_s += t.elapsed().as_secs_f64();
                }
                LayerKind::Full => {
                    let sa = |s: &str| n(&format!("self_attn.{s}"));
                    layer.qkv = concat_bf16(
                        &mut gpu,
                        &ck,
                        &[sa("q_proj.weight"), sa("k_proj.weight"), sa("v_proj.weight")],
                    )?;
                    layer.q_norm = gpu.alloc_upload(&ck.bf16_vec(&sa("q_norm.weight"))?)?;
                    layer.k_norm = gpu.alloc_upload(&ck.bf16_vec(&sa("k_norm.weight"))?)?;
                    layer.o_proj = concat_bf16(&mut gpu, &ck, &[sa("o_proj.weight")])?;
                    let t = Instant::now();
                    layer.k_cache = gpu.alloc(kv_bytes)?;
                    layer.v_cache = gpu.alloc(kv_bytes)?;
                    gpu.sync()?;
                    state_s += t.elapsed().as_secs_f64();
                }
            }
            layers.push(layer);
        }
        let mtp = if want_mtp {
            let n = |s: &str| format!("mtp.layers.0.{s}");
            let sa = |s: &str| n(&format!("self_attn.{s}"));
            let layer = Layer {
                kind: LayerKind::Full,
                in_norm: concat_bf16(&mut gpu, &ck, &[n("input_layernorm.weight")])?,
                post_norm: concat_bf16(&mut gpu, &ck, &[n("post_attention_layernorm.weight")])?,
                gdn_proj: 0,
                conv_w: 0,
                a_log: 0,
                dt_bias: 0,
                gnorm: 0,
                out_proj: 0,
                qkv: concat_bf16(&mut gpu, &ck, &[sa("q_proj.weight"), sa("k_proj.weight"), sa("v_proj.weight")])?,
                q_norm: gpu.alloc_upload(&ck.bf16_vec(&sa("q_norm.weight"))?)?,
                k_norm: gpu.alloc_upload(&ck.bf16_vec(&sa("k_norm.weight"))?)?,
                o_proj: concat_bf16(&mut gpu, &ck, &[sa("o_proj.weight")])?,
                gate_up: concat_bf16(&mut gpu, &ck, &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?,
                down: concat_bf16(&mut gpu, &ck, &[n("mlp.down_proj.weight")])?,
                delta_state: 0,
                conv_state: 0,
                k_cache: gpu.alloc(kv_bytes)?,
                v_cache: gpu.alloc(kv_bytes)?,
            };
            Some(Mtp {
                layer,
                fc: concat_bf16(&mut gpu, &ck, &["mtp.fc.weight".to_string()])?,
                norm: concat_bf16(&mut gpu, &ck, &["mtp.norm.weight".to_string()])?,
                pre_e: concat_bf16(&mut gpu, &ck, &["mtp.pre_fc_norm_embedding.weight".to_string()])?,
                pre_h: concat_bf16(&mut gpu, &ck, &["mtp.pre_fc_norm_hidden.weight".to_string()])?,
            })
        } else {
            None
        };
        if mtp.is_some() {
            eprintln!("MTP draft head loaded");
        }
        let weight_bytes = gpu.allocated_bytes;
        gpu.sync()?;
        gpu.free_staging();
        let layers_s = t1.elapsed().as_secs_f64();
        eprintln!(
            "load: open {open_s:.2}s, weights {:.2}s, state+KV alloc {state_s:.2}s",
            layers_s - state_s
        );
        eprintln!(
            "memory: {:.2} GiB weights+state+KV ({} KV pages = {} tokens, {} state slots of {:.1} MiB)",
            weight_bytes as f64 / (1u64 << 30) as f64,
            lim.num_blocks(),
            lim.num_blocks() * lim.block_size,
            lim.max_seqs,
            cfg.state_bytes_per_seq() as f64 / (1 << 20) as f64
        );

        let cc_gdn = std::env::var("AQ_GDN_CHUNK_CC").as_deref() == Ok("1");
        let s = alloc_scratch(&mut gpu, &cfg, &lim, cc_gdn, mtp.is_some())?;
        gpu.sync()?;
        let lane_events = vec![gpu.ctx.new_event().map_err(de)?];
        let num_sms = gpu.num_sms()?;
        let grp = cfg.num_attention_heads / cfg.num_key_value_heads;
        // Sub-tiles per block: capped by 1024 threads and 100 KiB of shared memory (the most a
        // block may ask for on the GB10; 4 sub-tiles of a 4-head group need 160 KiB).
        let mut attn_qt: usize = std::env::var("AQ_ATTN_QT").ok().and_then(|v| v.parse().ok()).unwrap_or(2).clamp(1, 4);
        while attn_qt > 1 && (32 * grp * attn_qt > 1024 || attention_tc_smem(grp, attn_qt) > 100 << 10) {
            attn_qt -= 1;
        }
        let attn_tc_smem = attention_tc_smem(grp, attn_qt);
        gpu.opt_in_dynamic_smem(
            <aq_kernels::kernels::__attention_tc_CudaKernel as cuda_host::CudaKernel>::PTX_NAME,
            attn_tc_smem,
        )?;
        gpu.set_max_dynamic_smem("gdn_tc_prep", GDN_TC_PREP_SMEM)?;
        gpu.set_max_dynamic_smem("gdn_tc_scan", GDN_TC_SCAN_SMEM)?;
        gpu.set_max_dynamic_smem("gdn_tc_out", GDN_TC_OUT_SMEM)?;
        let upload_s = t1.elapsed().as_secs_f64();
        eprintln!("load: scratch + kernel setup {:.2}s", upload_s - layers_s);
        Ok(Self {
            cfg,
            gpu,
            embed,
            final_norm,
            lm_head,
            layers,
            mtp,
            mtp_on: false,
            last_mtp: Vec::new(),
            s,
            limits: lim,
            load: LoadStats { open_s, upload_s, weight_bytes },
            prof: Prof::from_env(),
            graphs: Default::default(),
            use_graphs: std::env::var("AQ_NO_GRAPHS").is_err(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
            cur_tap: false,
            last_logits: 0,
            cur_split_keys: 0,
            attn_split_keys: std::env::var("AQ_ATTN_SPLIT_KEYS").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            num_sms,
            attn_fill: std::env::var("AQ_ATTN_FILL").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            ref_gdn: std::env::var("AQ_GDN_RECURRENT").as_deref() == Ok("1"),
            ref_attn: std::env::var("AQ_ATTN_NAIVE").as_deref() == Ok("1"),
            tile_attn: std::env::var("AQ_ATTN_TILE").as_deref() == Ok("1"),
            attn_tc_smem,
            attn_qt,
            cc_gdn,
            s_cap: lim.max_rows(),
            alt: None,
            lane_events,
        })
    }

    /// Create the decode lane for decode/prefill co-execution: a high-priority stream with its
    /// own cuBLASLt workspace, plus scratch and metadata buffers sized for one decode token per
    /// sequence (plus the usual decode-row headroom). Lane 0 keeps the full-size scratch for
    /// prefill.
    pub fn enable_decode_lane(&mut self) -> Result<()> {
        if self.alt.is_some() {
            return Ok(());
        }
        self.set_lane_raw(0);
        // AQ_COEXEC_PRIO: stream priority of the decode lane (default: the device's highest;
        // 0 = same priority as prefill).
        let prio = match std::env::var("AQ_COEXEC_PRIO").ok().and_then(|v| v.parse::<i32>().ok()) {
            Some(p) => p,
            None => self.gpu.highest_priority()?,
        };
        let lane = self.gpu.add_lane(prio)?;
        anyhow::ensure!(lane == 1, "decode lane must be lane 1");
        let mut l2 = self.limits;
        l2.max_tokens = l2.max_seqs;
        let s = alloc_scratch(&mut self.gpu, &self.cfg, &l2, self.cc_gdn, self.mtp.is_some())?;
        self.gpu.sync()?;
        self.lane_events.push(self.gpu.ctx.new_event().map_err(de)?);
        self.alt = Some(LaneState {
            s,
            cap: l2.max_rows(),
            graphs: Default::default(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
            cur_split_keys: 0,
        });
        Ok(())
    }

    /// Create a background prefill lane: a second, low-priority stream with its own cuBLASLt
    /// workspace and full-size scratch, so a long prompt can be prefilled while steps for the
    /// other sequences keep running on lane 0.
    pub fn enable_prefill_lane(&mut self) -> Result<()> {
        if self.alt.is_some() {
            return Ok(());
        }
        self.set_lane_raw(0);
        let prio = self.gpu.lowest_priority()?;
        let lane = self.gpu.add_lane(prio)?;
        anyhow::ensure!(lane == 1, "prefill lane must be lane 1");
        let lim = self.limits;
        let s = alloc_scratch(&mut self.gpu, &self.cfg, &lim, self.cc_gdn, self.mtp.is_some())?;
        self.gpu.sync()?;
        self.lane_events.push(self.gpu.ctx.new_event().map_err(de)?);
        self.alt = Some(LaneState {
            s,
            cap: lim.max_rows(),
            graphs: Default::default(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
            cur_split_keys: 0,
        });
        Ok(())
    }

    /// Which lane is active (0 without co-execution).
    pub fn lane(&self) -> usize {
        self.gpu.lane
    }

    fn set_lane_raw(&mut self, lane: usize) {
        self.gpu.set_lane(lane);
    }

    /// Make `lane` active: swap in its scratch, work lists and graphs, and point the GPU at its
    /// stream and workspace. Lane 1 exists only after `enable_decode_lane`.
    pub fn set_lane(&mut self, lane: usize) {
        if lane == self.gpu.lane {
            return;
        }
        let alt = self.alt.as_mut().expect("set_lane(1) without enable_decode_lane");
        std::mem::swap(&mut self.s, &mut alt.s);
        std::mem::swap(&mut self.s_cap, &mut alt.cap);
        std::mem::swap(&mut self.graphs, &mut alt.graphs);
        std::mem::swap(&mut self.cur_chunks, &mut alt.cur_chunks);
        std::mem::swap(&mut self.cur_tiles, &mut alt.cur_tiles);
        std::mem::swap(&mut self.cur_splits, &mut alt.cur_splits);
        std::mem::swap(&mut self.cur_nd_rows, &mut alt.cur_nd_rows);
        std::mem::swap(&mut self.cur_split_keys, &mut alt.cur_split_keys);
        self.gpu.set_lane(lane);
    }

    /// Enqueue a whole step on the active lane without waiting for it: upload, forward, argmax.
    /// Returns the number of sampled rows; collect them with `collect` once `lane_done`.
    pub fn launch(&mut self, b: &Batch) -> Result<usize> {
        let nseq = b.seq_start.len();
        let ns = b.sample_rows.len();
        anyhow::ensure!(!b.tokens.is_empty() && b.tokens.len() <= self.s_cap, "{} tokens exceeds lane capacity", b.tokens.len());
        anyhow::ensure!(nseq <= self.limits.max_seqs, "{nseq} sequences exceeds max_seqs");
        self.upload(b)?;
        self.forward(b.tokens.len(), nseq, ns, b.n_decode, None)?;
        let lane = self.gpu.lane;
        self.lane_events[lane].record(&self.gpu.stream).map_err(de)?;
        Ok(ns)
    }

    /// Whether everything enqueued on `lane` up to its last `launch` has finished.
    pub fn lane_done(&self, lane: usize) -> Result<bool> {
        self.lane_events[lane].query().map_err(de)
    }

    /// Sampled tokens of the active lane's last `launch` (waits for it).
    pub fn collect(&mut self, ns: usize) -> Result<Vec<u32>> {
        if ns == 0 {
            self.gpu.sync()?;
            return Ok(Vec::new());
        }
        self.gpu.download(self.s.tok, ns)
    }

    /// Clear a slot's recurrent state before a new sequence uses it (KV pages are written
    /// before they are read and need no reset).
    /// Tune every cuBLASLt plan the engine can hit (each M bucket up to `max_tokens` for every
    /// weight shape, and the LM head up to `max_seqs`) on scratch buffers, so no step ever
    /// stalls on a first-use autotune mid-run. Results in the scratch buffers are garbage and
    /// are overwritten by the next real step. Returns the number of M values tried.
    pub fn prewarm_gemms(&mut self) -> Result<usize> {
        let c = self.cfg.clone();
        let lim = self.limits;
        let mut ms: Vec<usize> = (1..=16).map(|i| i * 16).collect();
        ms.extend((3..=16).map(|i| i * 128));
        ms.extend((5..=16).map(|i| i * 512));
        ms.retain(|&m| m <= lim.max_tokens);
        let (s, g) = (&self.s, &mut self.gpu);
        let mut shapes: Vec<(usize, usize, u64, u64)> = Vec::new(); // (n, k, x, y)
        let mut seen = std::collections::HashSet::new();
        for l in &self.layers {
            let w = match l.kind {
                LayerKind::Linear => vec![(c.gdn_proj_ld(), c.hidden_size, s.h, s.proj, l.gdn_proj), (c.hidden_size, c.value_dim(), s.mix, s.delta, l.out_proj)],
                LayerKind::Full => vec![
                    (c.attn_proj_dim(), c.hidden_size, s.h, s.proj, l.qkv),
                    (c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, s.delta, l.o_proj),
                ],
            };
            for (n, k, x, y, wt) in w.into_iter().chain([
                (2 * c.intermediate_size, c.hidden_size, s.h, s.proj, l.gate_up),
                (c.hidden_size, c.intermediate_size, s.mix, s.delta, l.down),
            ]) {
                if seen.insert((n, k)) {
                    shapes.push((n, k, x, y));
                    for &m in &ms {
                        g.gemm(m, lim.max_tokens, n, k, x, wt, y, CUDA_R_16BF)?;
                    }
                }
            }
        }
        for m in 1..=lim.max_seqs {
            g.gemm(m, lim.max_seqs, c.vocab_size, c.hidden_size, s.hfinal, self.lm_head, s.logits, CUDA_R_16BF)?;
        }
        g.sync()?;
        Ok(ms.len())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        let dsz = self.cfg.linear_num_value_heads * 128 * 128 * 4;
        let csz = self.cfg.conv_dim() * 3 * 2;
        for l in &self.layers {
            if l.kind == LayerKind::Linear {
                self.gpu.zero(l.delta_state + (slot * dsz) as u64, dsz)?;
                self.gpu.zero(l.conv_state + (slot * csz) as u64, csz)?;
            }
        }
        Ok(())
    }

    /// Whether prefill sequences can carry a state tap (`Batch::seq_tap`).
    pub fn supports_tap(&self) -> bool {
        !self.cc_gdn && !self.ref_gdn
    }

    /// Copy one KV page (every full-attention layer's K and V rows) into another, on the stream.
    pub fn copy_page(&mut self, dst: usize, src: usize) -> Result<()> {
        let psz = self.limits.block_size * self.cfg.num_key_value_heads * self.cfg.head_dim * 2;
        let st = self.gpu.stream.cu_stream();
        for l in &self.layers {
            if l.kind == LayerKind::Full {
                for base in [l.k_cache, l.v_cache] {
                    unsafe { cuda_core::memory::memcpy_dtod_async(base + (dst * psz) as u64, base + (src * psz) as u64, psz, st) }.map_err(de)?;
                }
            }
        }
        Ok(())
    }

    /// Copy one slot's recurrent state (DeltaNet + conv) into another, on the stream.
    pub fn copy_slot(&mut self, dst: usize, src: usize) -> Result<()> {
        let dsz = self.cfg.linear_num_value_heads * 128 * 128 * 4;
        let csz = self.cfg.conv_dim() * 3 * 2;
        let st = self.gpu.stream.cu_stream();
        for l in &self.layers {
            if l.kind == LayerKind::Linear {
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(l.delta_state + (dst * dsz) as u64, l.delta_state + (src * dsz) as u64, dsz, st)
                        .map_err(de)?;
                    cuda_core::memory::memcpy_dtod_async(l.conv_state + (dst * csz) as u64, l.conv_state + (src * csz) as u64, csz, st)
                        .map_err(de)?;
                }
            }
        }
        Ok(())
    }

    /// Convenience for a single sequence in slot 0 with an identity block table.
    pub fn single(&self, ids: &[u32], pos: usize) -> Batch {
        let n = ids.len() as u32;
        Batch {
            tokens: ids.to_vec(),
            tok_pos: (0..n).map(|i| pos as u32 + i).collect(),
            tok_seq: vec![0; ids.len()],
            seq_start: vec![0],
            seq_len: vec![n],
            seq_slot: vec![0],
            block_tables: (0..self.limits.max_blocks_per_seq() as u32).collect(),
            sample_rows: vec![n - 1],
            n_decode: (n == 1) as usize,
            seq_slot_out: Vec::new(),
            seq_tap: Vec::new(),
            next_tok_last: Vec::new(),
        }
    }

    /// Run one batch and return the greedy token for each entry of `b.sample_rows`.
    /// With `dump`, also return the residual stream after every layer (as f32).
    pub fn step(&mut self, b: &Batch, dump: Option<&mut Vec<Vec<f32>>>) -> Result<Vec<u32>> {
        let nseq = b.seq_start.len();
        let ns = b.sample_rows.len();
        let lim = self.limits;
        anyhow::ensure!(!b.tokens.is_empty() && b.tokens.len() <= self.s_cap, "{} tokens exceeds the lane's capacity", b.tokens.len());
        anyhow::ensure!(nseq <= lim.max_seqs, "{nseq} sequences exceeds max_seqs");
        anyhow::ensure!(ns <= lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ, "too many sampled rows");
        self.last_mtp.clear();
        let pure_decode = nseq > 0 && b.n_decode == nseq && ns == nseq && b.tokens.len() == nseq;
        if self.use_graphs && pure_decode && dump.is_none() {
            return self.step_graph(b);
        }
        let decode_only = nseq > 0 && b.n_decode == nseq && b.tokens.len() > nseq;
        if self.use_graphs && decode_only && dump.is_none() && self.attn_split_keys > 0 {
            return self.step_graph_spec(b);
        }
        self.upload(b)?;
        self.forward(b.tokens.len(), nseq, ns, b.n_decode, dump)?;
        if let Some(p) = self.prof.as_mut() {
            p.finish()?;
        }
        if ns == 0 {
            return Ok(Vec::new());
        }
        if self.mtp_active() {
            self.last_mtp = self.gpu.download(self.s.mtp_tok, ns)?;
        }
        self.gpu.download(self.s.tok, ns)
    }

    /// Pure-decode step through a captured CUDA graph. The batch is padded to a bucket size
    /// with rows that use the reserved dummy state slot (`max_seqs`) and dummy KV page
    /// (`num_blocks`), so padding never touches a live sequence.
    fn step_graph(&mut self, b: &Batch) -> Result<Vec<u32>> {
        let lim = self.limits;
        let n = b.seq_start.len();
        let bucket = DECODE_BUCKETS.iter().copied().find(|&x| x >= n).unwrap_or(n).min(lim.max_seqs).max(n);
        let mut p = b.clone();
        let mb = lim.max_blocks_per_seq();
        for k in n..bucket {
            p.tokens.push(0);
            p.tok_pos.push(0);
            p.tok_seq.push(k as u32);
            p.seq_start.push(k as u32);
            p.seq_len.push(1);
            p.seq_slot.push(lim.max_seqs as u32);
            if !p.seq_slot_out.is_empty() {
                p.seq_slot_out.push(lim.max_seqs as u32);
            }
            p.block_tables.extend(std::iter::repeat_n(lim.num_blocks() as u32, mb));
            p.sample_rows.push(k as u32);
        }
        p.n_decode = bucket;
        self.upload(&p)?;
        let key = (bucket + self.mtp_active() as usize * 50_000_000, self.cur_splits * 2 + (self.cur_split_keys > 0) as usize);
        if let Some((exec, names)) = self.graphs.get(&key) {
            if let Some(p) = self.prof.as_mut() {
                // The graph replays the event records captured with it.
                p.active = p.filter != 2;
                p.names = names.clone();
            }
            self.gpu.launch_graph(*exec)?;
            if let Some(p) = self.prof.as_mut() {
                p.finish()?;
            }
        } else {
            // Eager run first: produces this step's tokens and tunes every GEMM plan the
            // graph will use (tuning synchronizes, which capture forbids).
            self.forward(bucket, bucket, bucket, bucket, None)?;
            if let Some(p) = self.prof.as_mut() {
                p.finish()?;
            }
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.forward(bucket, bucket, bucket, bucket, None);
            let exec = self.gpu.end_capture()?;
            r?;
            let names = self.prof.as_mut().map(|p| std::mem::take(&mut p.names)).unwrap_or_default();
            self.graphs.insert(key, (exec, names));
        }
        if self.mtp_active() {
            self.last_mtp = self.gpu.download(self.s.mtp_tok, bucket)?;
            self.last_mtp.truncate(n);
        }
        let mut toks = self.gpu.download(self.s.tok, bucket)?;
        toks.truncate(n);
        Ok(toks)
    }

    /// Decode-only step where some sequences feed several tokens (speculative verification),
    /// through a captured graph of fixed shape: the sequence count is padded to a bucket, every
    /// sequence gets W rows of room (W = max tokens of any sequence), and the leftover rows go
    /// to a padding sequence of length 0 on the dummy slot and dummy KV page. All rows are
    /// sampled; the caller's sample rows are picked out afterwards. Needs fixed-size attention
    /// slices, so padding the slice count is exact (empty slices add nothing).
    fn step_graph_spec(&mut self, b: &Batch) -> Result<Vec<u32>> {
        let lim = self.limits;
        let n = b.seq_start.len();
        let w = *b.seq_len.iter().max().unwrap() as usize;
        let bucket = DECODE_BUCKETS.iter().copied().find(|&x| x >= n).unwrap_or(n).min(lim.max_seqs).max(n);
        // Rows padded to a multiple of 16 (not bucket * W): sequences keep their own lengths,
        // so the padding is only the round-up, bounded by `max_rows`.
        let rows = b
            .tokens
            .len()
            .next_multiple_of(16)
            .max(bucket)
            .min(self.s_cap)
            .min(lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ)
            .max(b.tokens.len());
        let _ = w;
        let used = b.tokens.len();
        let mb = lim.max_blocks_per_seq();
        let dummy_slot = lim.max_seqs as u32;
        let dummy_page = lim.num_blocks() as u32;
        let mut p = b.clone();
        if p.seq_slot_out.is_empty() {
            p.seq_slot_out = p.seq_slot.clone();
        }
        // Empty dummy sequences up to the bucket, then the padding sequence (index `bucket`).
        for _ in n..=bucket {
            p.seq_start.push(used as u32);
            p.seq_len.push(0);
            p.seq_slot.push(dummy_slot);
            p.seq_slot_out.push(dummy_slot);
            p.block_tables.extend(std::iter::repeat_n(dummy_page, mb));
        }
        for r in used..rows {
            p.tokens.push(0);
            p.tok_pos.push(((r - used) % lim.block_size) as u32);
            p.tok_seq.push(bucket as u32);
        }
        p.sample_rows = (0..rows as u32).collect();
        p.n_decode = bucket + 1;
        self.upload(&p)?;
        // Every row runs through attention (padding rows read only the dummy page), and the
        // slice count is rounded up so a few graphs cover all context lengths.
        self.cur_nd_rows = rows;
        self.cur_splits = self.cur_splits.next_power_of_two().min(if self.cur_split_keys > 0 { ROW_LOCAL_SPLITS } else { MAX_ATTN_SPLITS });
        anyhow::ensure!(rows * self.cur_splits <= attn_part_rows(lim.max_seqs), "decode attention partials overflow");
        let key = (bucket * 100_000 + rows * 10 + 1 + self.mtp_active() as usize * 50_000_000, self.cur_splits * 2 + (self.cur_split_keys > 0) as usize);
        if let Some((exec, _)) = self.graphs.get(&key) {
            self.gpu.launch_graph(*exec)?;
        } else {
            self.forward(rows, bucket + 1, rows, bucket + 1, None)?;
            if let Some(pr) = self.prof.as_mut() {
                pr.finish()?;
            }
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.forward(rows, bucket + 1, rows, bucket + 1, None);
            let exec = self.gpu.end_capture()?;
            r?;
            let names = self.prof.as_mut().map(|pr| std::mem::take(&mut pr.names)).unwrap_or_default();
            self.graphs.insert(key, (exec, names));
        }
        if self.mtp_active() {
            let all: Vec<u32> = self.gpu.download(self.s.mtp_tok, rows)?;
            self.last_mtp = b.sample_rows.iter().map(|&r| all[r as usize]).collect();
        }
        let all = self.gpu.download(self.s.tok, rows)?;
        Ok(b.sample_rows.iter().map(|&r| all[r as usize]).collect())
    }

    /// Whether the checkpoint's MTP head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn mtp_active(&self) -> bool {
        self.mtp_on && self.mtp.is_some()
    }

    fn upload(&mut self, b: &Batch) -> Result<()> {
        // Prefill work lists: GDN chunks and attention query tiles for sequences n_decode..
        let (mut cs, mut cl, mut c0, mut cn, mut ts, mut tl) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        let chunk = if self.cc_gdn { GDN_CHUNK } else { GDN_TC_CHUNK };
        for k in b.n_decode..b.seq_start.len() {
            let (start, len) = (b.seq_start[k] as usize, b.seq_len[k] as usize);
            c0.push(cs.len() as u32);
            let mut o = 0;
            while o < len {
                cs.push((start + o) as u32);
                cl.push((len - o).min(chunk) as u32);
                o += chunk;
            }
            cn.push(cs.len() as u32 - *c0.last().unwrap());
            let qtile = if self.tile_attn || self.ref_attn { ATTN_QTILE } else { ATTN_QTILE * self.attn_qt };
            let mut o = 0;
            while o < len {
                ts.push((start + o) as u32);
                tl.push((len - o).min(qtile) as u32);
                o += qtile;
            }
        }
        self.cur_chunks = cs.len();
        self.cur_tiles = ts.len();
        // Split-K slices for decode attention: enough blocks to fill the GPU twice, at least
        // 64 keys per slice, power of two (bounds the number of captured graphs).
        let nd_seqs = b.n_decode;
        let nd = if nd_seqs == 0 { 0 } else { (b.seq_start[nd_seqs - 1] + b.seq_len[nd_seqs - 1]) as usize };
        self.cur_nd_rows = nd;
        self.cur_split_keys = 0;
        let max_ctx_rows = if nd == 0 { 0 } else { b.tok_pos[..nd].iter().copied().max().unwrap_or(0) as usize + 1 };
        self.cur_splits = if nd == 0 {
            1
        } else if self.attn_split_keys > 0 {
            // Row-local slice size (see attn_decode_split): a row's result is independent of
            // the other rows. Slices needed = max over rows of ceil(len / slice(len)) <= 32.
            self.cur_split_keys = self.attn_split_keys as u32;
            b.tok_pos[..nd]
                .iter()
                .map(|&p| {
                    let n = p as usize + 1;
                    n.div_ceil(self.attn_split_keys.max(n.div_ceil(ROW_LOCAL_SPLITS)))
                })
                .max()
                .unwrap_or(1)
        } else {
            let max_ctx = max_ctx_rows;
            let fill = (self.attn_fill * self.num_sms).div_ceil(nd * self.cfg.num_key_value_heads);
            let cap = (attn_part_rows(self.limits.max_seqs) / nd).max(1);
            fill.min(max_ctx.div_ceil(64)).clamp(1, MAX_ATTN_SPLITS).next_power_of_two().min(MAX_ATTN_SPLITS).min(cap)
        };
        anyhow::ensure!(
            nd * self.cur_splits <= attn_part_rows(self.limits.max_seqs),
            "decode attention partials overflow ({nd} rows x {} slices)",
            self.cur_splits
        );
        // State taps (dummy slot and zero chunks where a sequence has none).
        let ns_all = b.seq_start.len();
        let dummy = self.limits.max_seqs as u32;
        let (mut tn, mut tlen, mut tslot) = (vec![0u32; ns_all], vec![0u32; ns_all], vec![dummy; ns_all]);
        for (k, &(n, slot)) in b.seq_tap.iter().enumerate() {
            if n > 0 {
                anyhow::ensure!(k >= b.n_decode && (n as usize * GDN_TC_CHUNK) <= b.seq_len[k] as usize && self.supports_tap(), "invalid state tap");
                (tn[k], tlen[k], tslot[k]) = (n, n * GDN_TC_CHUNK as u32, slot);
            }
        }
        self.cur_tap = tn.iter().any(|&n| n > 0);
        let (s, g) = (&self.s, &self.gpu);
        if !cs.is_empty() {
            g.upload(s.seq_tap_n, &tn)?;
            g.upload(s.seq_tap_len, &tlen)?;
            g.upload(s.seq_tap_slot, &tslot)?;
            g.upload(s.chunk_start, &cs)?;
            g.upload(s.chunk_len, &cl)?;
            g.upload(s.seq_chunk0, &c0)?;
            g.upload(s.seq_nchunk, &cn)?;
            g.upload(s.tile_start, &ts)?;
            g.upload(s.tile_len, &tl)?;
        }
        g.upload(s.ids, &b.tokens)?;
        g.upload(s.tok_pos, &b.tok_pos)?;
        g.upload(s.tok_seq, &b.tok_seq)?;
        g.upload(s.seq_start, &b.seq_start)?;
        g.upload(s.seq_len, &b.seq_len)?;
        g.upload(s.seq_slot, &b.seq_slot)?;
        g.upload(s.seq_slot_out, if b.seq_slot_out.is_empty() { &b.seq_slot } else { &b.seq_slot_out })?;
        g.upload(s.tables, &b.block_tables)?;
        g.upload(s.rows, &b.sample_rows)?;
        if self.mtp_on && self.mtp.is_some() {
            // The token after each row: the next row of its sequence, the token it samples
            // (top bit + sample index, resolved on the GPU), or what the caller says follows.
            let mut next = vec![0u32; b.tokens.len()];
            for (k, (&st, &len)) in b.seq_start.iter().zip(&b.seq_len).enumerate() {
                let (st, len) = (st as usize, len as usize);
                for r in st..st + len {
                    next[r] = if r + 1 < st + len { b.tokens[r + 1] } else { b.next_tok_last.get(k).copied().filter(|&t| t != u32::MAX).unwrap_or(0) };
                }
            }
            for (j, &r) in b.sample_rows.iter().enumerate() {
                next[r as usize] = 0x8000_0000 | j as u32;
            }
            g.upload(s.next, &next)?;
        }
        Ok(())
    }

    /// Enqueue the whole forward pass for `t` tokens in `nseq` sequences (`nd` of them pure
    /// decode, listed first), ending with argmax for `ns` sampled rows. No host sync unless
    /// profiling or dumping, so it can be captured into a graph.
    fn forward(&mut self, t: usize, nseq: usize, ns: usize, nd_total: usize, mut dump: Option<&mut Vec<Vec<f32>>>) -> Result<()> {
        let lim = self.limits;
        // Row capacity of the active lane's scratch (GEMM M-bucket cap).
        let cap = self.s_cap;
        let c = self.cfg.clone();
        let hd = c.hidden_size as u32;
        let eps = c.rms_norm_eps as f32;
        let (s, g) = (&self.s, &mut self.gpu);
        let prof = &mut self.prof;
        macro_rules! mark {
            ($name:literal) => {
                if let Some(p) = prof.as_mut() {
                    p.mark(g, $name)?;
                }
            };
        }
        if let Some(p) = prof.as_mut() {
            let pure_decode = nd_total == nseq;
            p.active = match p.filter {
                1 => pure_decode,
                2 => !pure_decode,
                _ => true,
            };
            p.start(g)?;
        }
        let st = g.stream.clone();
        let mb = lim.max_blocks_per_seq() as u32;
        let bs = lim.block_size as u32;
        unsafe {
            g.k.embed(&st, cfg((t as u32, 1, 1), 256), s.ids as *const u32, self.embed as *const u16, s.x as *mut u16, hd)
        }
        .map_err(de)?;
        mark!("embed");

        // Final norm, output head and argmax for the sampled rows, from the residual stream
        // (x + delta) as the last layer left it.
        macro_rules! head {
            ($norm:expr, $hfinal:expr, $logits:expr, $tok:expr) => {{
                unsafe {
                    g.k.final_norm_rows(
                        &st,
                        cfg((ns as u32, 1, 1), 256),
                        s.x as *const u16,
                        s.delta as *const u16,
                        $norm as *const u16,
                        $hfinal as *mut u16,
                        s.rows as *const u32,
                        hd,
                        eps,
                    )
                }
                .map_err(de)?;
                mark!("final_norm");
                g.gemm(ns, lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ, c.vocab_size, c.hidden_size, $hfinal, self.lm_head, $logits, CUDA_R_16BF)?;
                mark!("gemm_lm_head");
                unsafe {
                    g.k.argmax_part(
                        &st,
                        cfg((ns as u32, ARGMAX_PARTS as u32, 1), 256),
                        $logits as *const u16,
                        c.vocab_size as u32,
                        s.amax_v as *mut f32,
                        s.amax_i as *mut u32,
                    )
                }
                .map_err(de)?;
                unsafe {
                    g.k.argmax_final(&st, cfg((ns as u32, 1, 1), 32), s.amax_v as *const f32, s.amax_i as *const u32, ARGMAX_PARTS as u32, $tok as *mut u32)
                }
                .map_err(de)?;
                mark!("argmax");
            }};
        }
        let nl = self.layers.len();
        let mtp = if self.mtp_on { self.mtp.as_ref() } else { None };
        for (li, l) in self.layers.iter().chain(mtp.map(|m| &m.layer)).enumerate() {
            if li == nl {
                // The backbone is done: sample, then start the MTP head's residual stream.
                let m = mtp.unwrap();
                if ns > 0 {
                    self.last_logits = s.logits;
                    head!(self.final_norm, s.hfinal, s.logits, s.tok);
                }
                unsafe {
                    g.k.mtp_prep(
                        &st,
                        cfg((t as u32, 1, 1), 256),
                        s.x as *const u16,
                        s.delta as *const u16,
                        self.final_norm as *const u16,
                        m.pre_h as *const u16,
                        m.pre_e as *const u16,
                        self.embed as *const u16,
                        s.next as *const u32,
                        s.tok as *const u32,
                        s.proj as *mut u16,
                        hd,
                        eps,
                    )
                }
                .map_err(de)?;
                g.gemm(t, cap, c.hidden_size, 2 * c.hidden_size, s.proj, m.fc, s.x, CUDA_R_16BF)?;
                mark!("mtp_in");
            }
            unsafe {
                g.k.add_rmsnorm(
                    &st,
                    cfg((t as u32, 1, 1), 256),
                    s.x as *mut u16,
                    s.delta as *const u16,
                    l.in_norm as *const u16,
                    s.h as *mut u16,
                    hd,
                    eps,
                    (li > 0 && li != nl) as u32,
                )
            }
            .map_err(de)?;

            match l.kind {
                LayerKind::Linear => {
                    let n = c.gdn_proj_ld();
                    mark!("norm");
                    g.gemm(t, cap, n, c.hidden_size, s.h, l.gdn_proj, s.proj, CUDA_R_16BF)?;
                    mark!("gemm_gdn_in");
                    let cd = c.conv_dim() as u32;
                    unsafe {
                        g.k.conv1d_tok(
                            &st,
                            cfg((cd.div_ceil(256), t as u32, 1), 256),
                            s.proj as *const u16,
                            n as u32,
                            l.conv_state as *const u16,
                            l.conv_w as *const u16,
                            s.conv as *mut u16,
                            s.tok_seq as *const u32,
                            s.seq_start as *const u32,
                            s.seq_slot as *const u32,
                            cd,
                        )
                    }
                    .map_err(de)?;
                    if self.cur_tap {
                        // Conv state at each tap: the last 3 inputs before it, into the tap slot.
                        unsafe {
                            g.k.conv_state_update(
                                &st,
                                cfg((cd.div_ceil(256), nseq as u32, 1), 256),
                                s.proj as *const u16,
                                n as u32,
                                l.conv_state as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_tap_len as *const u32,
                                s.seq_slot as *const u32,
                                s.seq_tap_slot as *const u32,
                                cd,
                            )
                        }
                        .map_err(de)?;
                    }
                    unsafe {
                        g.k.conv_state_update(
                            &st,
                            cfg((cd.div_ceil(256), nseq as u32, 1), 256),
                            s.proj as *const u16,
                            n as u32,
                            l.conv_state as *mut u16,
                            s.seq_start as *const u32,
                            s.seq_len as *const u32,
                            s.seq_slot as *const u32,
                            s.seq_slot_out as *const u32,
                            cd,
                        )
                    }
                    .map_err(de)?;
                    mark!("conv1d");
                    let z_off = c.conv_dim() as u32;
                    let b_off = z_off + c.value_dim() as u32;
                    let a_off = b_off + c.linear_num_value_heads as u32;
                    let (hv_u, hk_u) = (c.linear_num_value_heads as u32, c.linear_num_key_heads as u32);
                    let nd = nd_total as u32;
                    if nd > 0 {
                        unsafe {
                            g.k.gdn_decode(
                                &st,
                                cfg((hv_u, nd, 1), 128),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                l.gnorm as *const f32,
                                l.delta_state as *mut f32,
                                s.mix as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_len as *const u32,
                                s.seq_slot as *const u32,
                                s.seq_slot_out as *const u32,
                                hk_u,
                                hv_u,
                                eps,
                            )
                        }
                        .map_err(de)?;
                    }
                    if (nseq as u32) > nd && self.ref_gdn {
                        unsafe {
                            g.k.gdn_recurrent(
                                &st,
                                cfg((hv_u, nseq as u32 - nd, 1), 512),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                l.gnorm as *const f32,
                                l.delta_state as *mut f32,
                                s.mix as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_len as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                hk_u,
                                hv_u,
                                eps,
                            )
                        }
                        .map_err(de)?;
                    } else if (nseq as u32) > nd && !self.cc_gdn {
                        let nc = self.cur_chunks as u32;
                        unsafe {
                            g.k.gdn_tc_prep(
                                &st,
                                cfg_smem((nc, hv_u, 1), 128, GDN_TC_PREP_SMEM),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.tqn as *mut u16,
                                s.tkn as *mut u16,
                                s.tu as *mut u16,
                                s.tw as *mut u16,
                                s.tg as *mut f32,
                                hk_u,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_prep");
                        unsafe {
                            g.k.gdn_tc_scan(
                                &st,
                                cfg_smem((128 / GDN_TC_SCAN_VB, hv_u, nseq as u32 - nd), 256, GDN_TC_SCAN_SMEM),
                                s.tkn as *const u16,
                                s.tu as *const u16,
                                s.tw as *const u16,
                                s.tg as *const f32,
                                s.seq_chunk0 as *const u32,
                                s.seq_nchunk as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                l.delta_state as *mut f32,
                                s.tht as *mut u16,
                                s.tvnt as *mut u16,
                                hv_u,
                                s.seq_tap_n as *const u32,
                                s.seq_tap_slot as *const u32,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_scan");
                        unsafe {
                            g.k.gdn_tc_out(
                                &st,
                                cfg_smem((nc, hv_u, 1), 128, GDN_TC_OUT_SMEM),
                                s.tqn as *const u16,
                                s.tkn as *const u16,
                                s.tg as *const f32,
                                s.tht as *const u16,
                                s.tvnt as *const u16,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                l.gnorm as *const f32,
                                s.mix as *mut u16,
                                hv_u,
                                eps,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_out");
                    } else if (nseq as u32) > nd {
                        let nc = self.cur_chunks as u32;
                        unsafe {
                            g.k.gdn_chunk_prep(
                                &st,
                                cfg((nc, hv_u, 1), 128),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.gu as *mut f32,
                                s.gw as *mut f32,
                                s.gqp as *mut f32,
                                s.gkp as *mut f32,
                                s.gintra as *mut f32,
                                s.gdecay as *mut f32,
                                hk_u,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_prep");
                        unsafe {
                            g.k.gdn_chunk_scan(
                                &st,
                                cfg((hv_u, nseq as u32 - nd, 1), 128),
                                s.gu as *const f32,
                                s.gw as *const f32,
                                s.gqp as *const f32,
                                s.gkp as *const f32,
                                s.gintra as *const f32,
                                s.gdecay as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.seq_chunk0 as *const u32,
                                s.seq_nchunk as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                l.delta_state as *mut f32,
                                s.go as *mut u16,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_scan");
                        unsafe {
                            g.k.gated_norm(
                                &st,
                                cfg((t as u32, hv_u, 1), 128),
                                s.go as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                l.gnorm as *const f32,
                                s.mix as *mut u16,
                                s.tok_seq as *const u32,
                                nd,
                                hv_u,
                                eps,
                            )
                        }
                        .map_err(de)?;
                    }
                    mark!("gdn");
                    g.gemm(t, cap, c.hidden_size, c.value_dim(), s.mix, l.out_proj, s.delta, CUDA_R_16BF)?;
                    mark!("gemm_gdn_out");
                }
                LayerKind::Full => {
                    let n = c.attn_proj_dim();
                    let (h, hkv) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);
                    mark!("norm");
                    g.gemm(t, cap, n, c.hidden_size, s.h, l.qkv, s.proj, CUDA_R_16BF)?;
                    mark!("gemm_qkv");
                    unsafe {
                        g.k.attn_prep(
                            &st,
                            cfg((t as u32, h + hkv, 1), c.head_dim as u32),
                            s.proj as *const u16,
                            n as u32,
                            l.q_norm as *const u16,
                            l.k_norm as *const u16,
                            s.q as *mut u16,
                            l.k_cache as *mut u16,
                            l.v_cache as *mut u16,
                            s.tok_pos as *const u32,
                            s.tok_seq as *const u32,
                            s.tables as *const u32,
                            mb,
                            bs,
                            h,
                            hkv,
                            c.rot_dim() as u32,
                            c.rope_parameters.rope_theta as f32,
                            eps,
                        )
                    }
                    .map_err(de)?;
                    mark!("attn_prep");
                    let scale = 1.0 / (c.head_dim as f32).sqrt();
                    if self.ref_attn {
                        unsafe {
                            g.k.attention(
                                &st,
                                cfg((t as u32, h, 1), 256),
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                            )
                        }
                        .map_err(de)?;
                    } else if nd_total > 0 {
                        let ns = self.cur_splits as u32;
                        unsafe {
                            g.k.attn_decode_split(
                                &st,
                                LaunchConfig { grid_dim: (self.cur_nd_rows as u32, hkv, ns), block_dim: (32 * (h / hkv), 1, 1), shared_mem_bytes: 0 },
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                                s.attn_part as *mut f32,
                                s.attn_ml as *mut f32,
                                self.cur_split_keys,
                            )
                        }
                        .map_err(de)?;
                        mark!("attn_dec_split");
                        unsafe {
                            g.k.attn_decode_combine(
                                &st,
                                cfg((self.cur_nd_rows as u32, h, 1), 256),
                                s.attn_part as *const f32,
                                s.attn_ml as *const f32,
                                ns,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                h,
                            )
                        }
                        .map_err(de)?;
                        mark!("attn_dec_combine");
                    }
                    if self.cur_tiles > 0 && !self.ref_attn && !self.tile_attn {
                        let grp = (h / hkv) as u32;
                        let launch = cuda_core::LaunchConfig {
                            grid_dim: (self.cur_tiles as u32, hkv, 1),
                            block_dim: (32 * grp * self.attn_qt as u32, 1, 1),
                            shared_mem_bytes: self.attn_tc_smem,
                        };
                        unsafe {
                            g.k.attention_tc(
                                &st,
                                launch,
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tile_start as *const u32,
                                s.tile_len as *const u32,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                            )
                        }
                        .map_err(de)?;
                    }
                    if self.cur_tiles > 0 && !self.ref_attn && self.tile_attn {
                        unsafe {
                            g.k.attention_tile(
                                &st,
                                cfg((self.cur_tiles as u32, h, 1), 256),
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tile_start as *const u32,
                                s.tile_len as *const u32,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                            )
                        }
                        .map_err(de)?;
                    }
                    mark!("attn_prefill");
                    g.gemm(t, cap, c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, l.o_proj, s.delta, CUDA_R_16BF)?;
                    mark!("gemm_o");
                }
            }

            unsafe {
                g.k.add_rmsnorm(
                    &st,
                    cfg((t as u32, 1, 1), 256),
                    s.x as *mut u16,
                    s.delta as *const u16,
                    l.post_norm as *const u16,
                    s.h as *mut u16,
                    hd,
                    eps,
                    1,
                )
            }
            .map_err(de)?;
            mark!("norm");
            let inter = c.intermediate_size;
            g.gemm(t, cap, 2 * inter, c.hidden_size, s.h, l.gate_up, s.proj, CUDA_R_16BF)?;
            mark!("gemm_gate_up");
            let nvec = (t * inter / 8) as u32;
            unsafe {
                g.k.silu_mul(&st, cfg((nvec.div_ceil(256), 1, 1), 256), s.proj as *const u16, s.mix as *mut u16, inter as u32, t as u32)
            }
            .map_err(de)?;
            mark!("silu_mul");
            g.gemm(t, cap, c.hidden_size, inter, s.mix, l.down, s.delta, CUDA_R_16BF)?;
            mark!("gemm_down");

            if let Some(d) = dump.as_deref_mut().filter(|_| li < nl) {
                let x: Vec<u16> = g.download(s.x, t * c.hidden_size)?;
                let dl: Vec<u16> = g.download(s.delta, t * c.hidden_size)?;
                d.push(
                    x.iter()
                        .zip(&dl)
                        .map(|(&a, &b)| {
                            bf16::from_f32(bf16::from_bits(a).to_f32() + bf16::from_bits(b).to_f32()).to_f32()
                        })
                        .collect(),
                );
            }
        }

        if ns == 0 {
            return Ok(());
        }
        match mtp {
            Some(m) => head!(m.norm, s.mtp_hfinal, s.mtp_logits, s.mtp_tok),
            None => {
                self.last_logits = s.logits;
                head!(self.final_norm, s.hfinal, s.logits, s.tok);
            }
        }
        Ok(())
    }

    /// Logits of the first sampled row of the last step.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let src = if self.last_logits != 0 { self.last_logits } else { self.s.logits };
        let v: Vec<u16> = self.gpu.download(src, self.cfg.vocab_size)?;
        Ok(v.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }

    /// Logits of the first `n` sampled rows of the last step, row after row.
    pub fn logits_rows(&self, n: usize) -> Result<Vec<f32>> {
        let src = if self.last_logits != 0 { self.last_logits } else { self.s.logits };
        let v: Vec<u16> = self.gpu.download(src, n * self.cfg.vocab_size)?;
        Ok(v.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }

    /// Post-final-norm hidden state of the first sampled row of the last step, as f32.
    pub fn final_hidden(&self) -> Result<Vec<f32>> {
        let h: Vec<u16> = self.gpu.download(self.s.hfinal, self.cfg.hidden_size)?;
        Ok(h.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }
}
