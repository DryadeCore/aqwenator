//! Qwen3.5 (`qwen3_5`) dense hybrid model: weights on device + a batched forward step.
//!
//! One `step` runs a mixed batch: any number of sequences, each contributing a chunk of
//! tokens (a prefill chunk or a single decode token). Full-attention layers read a paged KV
//! cache; gated-delta-net layers keep a fixed-size recurrent state per sequence slot.

use crate::config::{LayerKind, TextConfig};
use crate::cublaslt::CUDA_R_16BF;
use crate::gpu::{Gpu, Prof, cfg, de};
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
    ids: u64,
    tok_pos: u64,
    tok_seq: u64,
    seq_start: u64,
    seq_len: u64,
    seq_slot: u64,
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
}

pub struct LoadStats {
    pub open_s: f64,
    pub upload_s: f64,
    pub weight_bytes: usize,
}

/// Chunk length of the chunked gated delta rule (must match the kernels).
pub const GDN_CHUNK: usize = 32;
/// Query rows per tile of the prefill attention kernel (must match the kernels).
pub const ATTN_QTILE: usize = 16;
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
}

impl Limits {
    pub fn max_chunks(&self) -> usize {
        self.max_tokens / GDN_CHUNK + self.max_seqs
    }
    pub fn max_tiles(&self) -> usize {
        self.max_tokens / ATTN_QTILE + self.max_seqs
    }
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_ctx.div_ceil(self.block_size)
    }
    pub fn num_blocks(&self) -> usize {
        self.max_seqs * self.max_blocks_per_seq()
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
    /// Sequences `0..n_decode` feed exactly one token (decode fast path).
    pub n_decode: usize,
}

pub struct Model {
    pub cfg: TextConfig,
    pub gpu: Gpu,
    embed: u64,
    final_norm: u64,
    lm_head: u64,
    pub layers: Vec<Layer>,
    s: Scratch,
    pub limits: Limits,
    pub load: LoadStats,
    pub prof: Option<Prof>,
    /// Captured decode graphs, keyed by (padded batch size, attention splits).
    graphs: std::collections::HashMap<(usize, usize), u64>,
    num_sms: usize,
    /// Target decode-attention blocks per SM when choosing split-K slices (AQ_ATTN_FILL).
    attn_fill: usize,
    /// Split-K slices for this step's decode attention (set by `upload`).
    cur_splits: usize,
    pub use_graphs: bool,
    /// Prefill work lists of the step being run (set by `upload`).
    cur_chunks: usize,
    cur_tiles: usize,
    /// A/B switches back to the reference kernels (AQ_GDN_RECURRENT=1, AQ_ATTN_NAIVE=1).
    ref_gdn: bool,
    ref_attn: bool,
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
    let p = gpu.alloc(total)?;
    gpu.sync()?;
    let mut off = 0u64;
    for v in views {
        gpu.upload(p + off, v.bytes)?;
        off += v.bytes.len() as u64;
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
        let lim = limits;
        let kv_bytes = (lim.num_blocks() + 1) * lim.block_size * cfg.num_key_value_heads * cfg.head_dim * 2;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
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
                    layer.delta_state = gpu.alloc((lim.max_seqs + 1) * hv * 128 * 128 * 4)?;
                    layer.conv_state = gpu.alloc((lim.max_seqs + 1) * cfg.conv_dim() * 3 * 2)?;
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
                    layer.k_cache = gpu.alloc(kv_bytes)?;
                    layer.v_cache = gpu.alloc(kv_bytes)?;
                }
            }
            layers.push(layer);
        }
        let weight_bytes = gpu.allocated_bytes;

        let hd = cfg.hidden_size;
        let wide = cfg.gdn_proj_ld().max(cfg.attn_proj_dim()).max(2 * cfg.intermediate_size);
        let mixw = cfg.value_dim().max(cfg.num_attention_heads * cfg.head_dim).max(cfg.intermediate_size);
        let t = lim.max_tokens;
        let sq = lim.max_seqs;
        let hvd = cfg.linear_num_value_heads * 128;
        let hq = cfg.num_attention_heads;
        let s = Scratch {
            attn_part: gpu.alloc(sq * hq * MAX_ATTN_SPLITS * cfg.head_dim * 4)?,
            attn_ml: gpu.alloc(sq * hq * MAX_ATTN_SPLITS * 2 * 4)?,
            chunk_start: gpu.alloc(lim.max_chunks() * 4)?,
            chunk_len: gpu.alloc(lim.max_chunks() * 4)?,
            seq_chunk0: gpu.alloc(sq * 4)?,
            seq_nchunk: gpu.alloc(sq * 4)?,
            tile_start: gpu.alloc(lim.max_tiles() * 4)?,
            tile_len: gpu.alloc(lim.max_tiles() * 4)?,
            gu: gpu.alloc(t * hvd * 4)?,
            gw: gpu.alloc(t * hvd * 4)?,
            gqp: gpu.alloc(t * hvd * 4)?,
            gkp: gpu.alloc(t * hvd * 4)?,
            gintra: gpu.alloc(lim.max_chunks() * cfg.linear_num_value_heads * GDN_CHUNK * GDN_CHUNK * 4)?,
            gdecay: gpu.alloc(lim.max_chunks() * cfg.linear_num_value_heads * 4)?,
            go: gpu.alloc(t * hvd * 2)?,
            ids: gpu.alloc(t * 4)?,
            tok_pos: gpu.alloc(t * 4)?,
            tok_seq: gpu.alloc(t * 4)?,
            seq_start: gpu.alloc(sq * 4)?,
            seq_len: gpu.alloc(sq * 4)?,
            seq_slot: gpu.alloc(sq * 4)?,
            rows: gpu.alloc(sq * 4)?,
            tables: gpu.alloc(sq * lim.max_blocks_per_seq() * 4)?,
            hfinal: gpu.alloc(sq * hd * 2)?,
            x: gpu.alloc(t * hd * 2)?,
            h: gpu.alloc(t * hd * 2)?,
            proj: gpu.alloc(t * wide * 2)?,
            conv: gpu.alloc(t * cfg.conv_dim() * 2)?,
            mix: gpu.alloc(t * mixw * 2)?,
            delta: gpu.alloc(t * hd * 2)?,
            q: gpu.alloc(t * cfg.num_attention_heads * cfg.head_dim * 2)?,
            logits: gpu.alloc(sq * cfg.vocab_size * 2)?,
            tok: gpu.alloc(sq * 4)?,
        };
        gpu.sync()?;
        let num_sms = gpu.num_sms()?;
        let upload_s = t1.elapsed().as_secs_f64();
        Ok(Self {
            cfg,
            gpu,
            embed,
            final_norm,
            lm_head,
            layers,
            s,
            limits: lim,
            load: LoadStats { open_s, upload_s, weight_bytes },
            prof: Prof::from_env(),
            graphs: Default::default(),
            use_graphs: std::env::var("AQ_NO_GRAPHS").is_err(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            num_sms,
            attn_fill: std::env::var("AQ_ATTN_FILL").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            ref_gdn: std::env::var("AQ_GDN_RECURRENT").as_deref() == Ok("1"),
            ref_attn: std::env::var("AQ_ATTN_NAIVE").as_deref() == Ok("1"),
        })
    }

    /// Clear a slot's recurrent state before a new sequence uses it (KV pages are written
    /// before they are read and need no reset).
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
        }
    }

    /// Run one batch and return the greedy token for each entry of `b.sample_rows`.
    /// With `dump`, also return the residual stream after every layer (as f32).
    pub fn step(&mut self, b: &Batch, dump: Option<&mut Vec<Vec<f32>>>) -> Result<Vec<u32>> {
        let nseq = b.seq_start.len();
        let ns = b.sample_rows.len();
        let lim = self.limits;
        anyhow::ensure!(!b.tokens.is_empty() && b.tokens.len() <= lim.max_tokens, "{} tokens exceeds max_tokens", b.tokens.len());
        anyhow::ensure!(nseq <= lim.max_seqs, "{nseq} sequences exceeds max_seqs");
        let pure_decode = nseq > 0 && b.n_decode == nseq && ns == nseq;
        if self.use_graphs && pure_decode && dump.is_none() && self.prof.is_none() {
            return self.step_graph(b);
        }
        self.upload(b)?;
        self.forward(b.tokens.len(), nseq, ns, b.n_decode, dump)?;
        if let Some(p) = self.prof.as_mut() {
            p.finish()?;
        }
        if ns == 0 {
            return Ok(Vec::new());
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
            p.block_tables.extend(std::iter::repeat_n(lim.num_blocks() as u32, mb));
            p.sample_rows.push(k as u32);
        }
        p.n_decode = bucket;
        self.upload(&p)?;
        let key = (bucket, self.cur_splits);
        if let Some(&exec) = self.graphs.get(&key) {
            self.gpu.launch_graph(exec)?;
        } else {
            // Eager run first: produces this step's tokens and tunes every GEMM plan the
            // graph will use (tuning synchronizes, which capture forbids).
            self.forward(bucket, bucket, bucket, bucket, None)?;
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.forward(bucket, bucket, bucket, bucket, None);
            let exec = self.gpu.end_capture()?;
            r?;
            self.graphs.insert(key, exec);
        }
        let mut toks = self.gpu.download(self.s.tok, bucket)?;
        toks.truncate(n);
        Ok(toks)
    }

    fn upload(&mut self, b: &Batch) -> Result<()> {
        // Prefill work lists: GDN chunks and attention query tiles for sequences n_decode..
        let (mut cs, mut cl, mut c0, mut cn, mut ts, mut tl) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for k in b.n_decode..b.seq_start.len() {
            let (start, len) = (b.seq_start[k] as usize, b.seq_len[k] as usize);
            c0.push(cs.len() as u32);
            let mut o = 0;
            while o < len {
                cs.push((start + o) as u32);
                cl.push((len - o).min(GDN_CHUNK) as u32);
                o += GDN_CHUNK;
            }
            cn.push(cs.len() as u32 - *c0.last().unwrap());
            let mut o = 0;
            while o < len {
                ts.push((start + o) as u32);
                tl.push((len - o).min(ATTN_QTILE) as u32);
                o += ATTN_QTILE;
            }
        }
        self.cur_chunks = cs.len();
        self.cur_tiles = ts.len();
        // Split-K slices for decode attention: enough blocks to fill the GPU twice, at least
        // 64 keys per slice, power of two (bounds the number of captured graphs).
        let nd = b.n_decode;
        self.cur_splits = if nd == 0 {
            1
        } else {
            let max_ctx = b.tok_pos[..nd].iter().copied().max().unwrap_or(0) as usize + 1;
            let fill = (self.attn_fill * self.num_sms).div_ceil(nd * self.cfg.num_key_value_heads);
            fill.min(max_ctx.div_ceil(64)).clamp(1, MAX_ATTN_SPLITS).next_power_of_two().min(MAX_ATTN_SPLITS)
        };
        let (s, g) = (&self.s, &self.gpu);
        if !cs.is_empty() {
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
        g.upload(s.tables, &b.block_tables)?;
        g.upload(s.rows, &b.sample_rows)?;
        Ok(())
    }

    /// Enqueue the whole forward pass for `t` tokens in `nseq` sequences (`nd` of them pure
    /// decode, listed first), ending with argmax for `ns` sampled rows. No host sync unless
    /// profiling or dumping, so it can be captured into a graph.
    fn forward(&mut self, t: usize, nseq: usize, ns: usize, nd_total: usize, mut dump: Option<&mut Vec<Vec<f32>>>) -> Result<()> {
        let lim = self.limits;
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

        for (li, l) in self.layers.iter().enumerate() {
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
                    (li > 0) as u32,
                )
            }
            .map_err(de)?;

            match l.kind {
                LayerKind::Linear => {
                    let n = c.gdn_proj_ld();
                    mark!("norm");
                    g.gemm(t, lim.max_tokens, n, c.hidden_size, s.h, l.gdn_proj, s.proj, CUDA_R_16BF)?;
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
                                s.seq_slot as *const u32,
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
                    g.gemm(t, lim.max_tokens, c.hidden_size, c.value_dim(), s.mix, l.out_proj, s.delta, CUDA_R_16BF)?;
                    mark!("gemm_gdn_out");
                }
                LayerKind::Full => {
                    let n = c.attn_proj_dim();
                    let (h, hkv) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);
                    mark!("norm");
                    g.gemm(t, lim.max_tokens, n, c.hidden_size, s.h, l.qkv, s.proj, CUDA_R_16BF)?;
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
                                LaunchConfig { grid_dim: (nd_total as u32, hkv, ns), block_dim: (32 * (h / hkv), 1, 1), shared_mem_bytes: 0 },
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
                            )
                        }
                        .map_err(de)?;
                        unsafe {
                            g.k.attn_decode_combine(
                                &st,
                                cfg((nd_total as u32, h, 1), 256),
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
                        mark!("attn_decode");
                    }
                    if self.cur_tiles > 0 && !self.ref_attn {
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
                    g.gemm(t, lim.max_tokens, c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, l.o_proj, s.delta, CUDA_R_16BF)?;
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
            g.gemm(t, lim.max_tokens, 2 * inter, c.hidden_size, s.h, l.gate_up, s.proj, CUDA_R_16BF)?;
            mark!("gemm_gate_up");
            unsafe {
                g.k.silu_mul(&st, cfg((t as u32, 1, 1), 256), s.proj as *const u16, s.mix as *mut u16, inter as u32)
            }
            .map_err(de)?;
            mark!("silu_mul");
            g.gemm(t, lim.max_tokens, c.hidden_size, inter, s.mix, l.down, s.delta, CUDA_R_16BF)?;
            mark!("gemm_down");

            if let Some(d) = dump.as_deref_mut() {
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
        unsafe {
            g.k.final_norm_rows(
                &st,
                cfg((ns as u32, 1, 1), 256),
                s.x as *const u16,
                s.delta as *const u16,
                self.final_norm as *const u16,
                s.hfinal as *mut u16,
                s.rows as *const u32,
                hd,
                eps,
            )
        }
        .map_err(de)?;
        mark!("final_norm");
        g.gemm(ns, lim.max_seqs, c.vocab_size, c.hidden_size, s.hfinal, self.lm_head, s.logits, CUDA_R_16BF)?;
        mark!("gemm_lm_head");
        unsafe { g.k.argmax(&st, cfg((ns as u32, 1, 1), 1024), s.logits as *const u16, c.vocab_size as u32, s.tok as *mut u32) }
            .map_err(de)?;
        mark!("argmax");
        Ok(())
    }

    /// Logits of the first sampled row of the last step.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let v: Vec<u16> = self.gpu.download(self.s.logits, self.cfg.vocab_size)?;
        Ok(v.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }

    /// Post-final-norm hidden state of the first sampled row of the last step, as f32.
    pub fn final_hidden(&self) -> Result<Vec<f32>> {
        let h: Vec<u16> = self.gpu.download(self.s.hfinal, self.cfg.hidden_size)?;
        Ok(h.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }
}
