//! Qwen3.5 (`qwen3_5`) dense hybrid model: weights on device + a single-sequence forward pass.
//!
//! Milestone 1 (correctness): one sequence, contiguous KV cache, token-recurrent gated delta
//! rule, cuBLASLt bf16 GEMMs. Batching, paging and the fast kernels build on this.

use crate::config::{LayerKind, TextConfig};
use crate::cublaslt::{CUDA_R_16BF, CUDA_R_32F};
use crate::gpu::{Gpu, cfg, de};
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
    // per-sequence state
    delta_state: u64,
    conv_state: u64,
    k_cache: u64,
    v_cache: u64,
}

pub struct Scratch {
    ids: u64,
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

pub struct Model {
    pub cfg: TextConfig,
    pub gpu: Gpu,
    embed: u64,
    final_norm: u64,
    lm_head: u64,
    pub layers: Vec<Layer>,
    s: Scratch,
    pub max_tokens: usize,
    pub max_ctx: usize,
    pub load: LoadStats,
}

/// Upload several bf16 matrices concatenated along rows into one buffer.
fn concat_bf16(gpu: &mut Gpu, ck: &Checkpoint, names: &[String]) -> Result<u64> {
    let views = names.iter().map(|n| ck.bf16(n)).collect::<Result<Vec<_>>>()?;
    let total: usize = views.iter().map(|v| v.bytes.len()).sum();
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
    pub fn load(dir: &Path, max_tokens: usize, max_ctx: usize) -> Result<Self> {
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
        let kv_bytes = max_ctx * cfg.num_key_value_heads * cfg.head_dim * 2;
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
                    layer.gdn_proj = concat_bf16(
                        &mut gpu,
                        &ck,
                        &[la("in_proj_qkv.weight"), la("in_proj_z.weight"), la("in_proj_b.weight"), la("in_proj_a.weight")],
                    )?;
                    layer.conv_w = concat_bf16(&mut gpu, &ck, &[la("conv1d.weight")])?;
                    layer.a_log = gpu.alloc_upload(&ck.f32_vec(&la("A_log"))?)?;
                    layer.dt_bias = gpu.alloc_upload(&ck.f32_vec(&la("dt_bias"))?)?;
                    layer.gnorm = gpu.alloc_upload(&ck.f32_vec(&la("norm.weight"))?)?;
                    layer.out_proj = concat_bf16(&mut gpu, &ck, &[la("out_proj.weight")])?;
                    layer.delta_state = gpu.alloc(hv * 128 * 128 * 4)?;
                    layer.conv_state = gpu.alloc(cfg.conv_dim() * 3 * 2)?;
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
        let wide = cfg.gdn_proj_dim().max(cfg.attn_proj_dim()).max(2 * cfg.intermediate_size);
        let mixw = cfg.value_dim().max(cfg.num_attention_heads * cfg.head_dim).max(cfg.intermediate_size);
        let t = max_tokens;
        let s = Scratch {
            ids: gpu.alloc(t * 4)?,
            x: gpu.alloc(t * hd * 2)?,
            h: gpu.alloc(t * hd * 2)?,
            proj: gpu.alloc(t * wide * 2)?,
            conv: gpu.alloc(t * cfg.conv_dim() * 2)?,
            mix: gpu.alloc(t * mixw * 2)?,
            delta: gpu.alloc(t * hd * 2)?,
            q: gpu.alloc(t * cfg.num_attention_heads * cfg.head_dim * 2)?,
            logits: gpu.alloc(cfg.vocab_size * 4)?,
            tok: gpu.alloc(4)?,
        };
        gpu.sync()?;
        let upload_s = t1.elapsed().as_secs_f64();
        Ok(Self {
            cfg,
            gpu,
            embed,
            final_norm,
            lm_head,
            layers,
            s,
            max_tokens,
            max_ctx,
            load: LoadStats { open_s, upload_s, weight_bytes },
        })
    }

    /// Clear recurrent state (KV caches are position-indexed and need no reset).
    pub fn reset(&mut self) -> Result<()> {
        let hv = self.cfg.linear_num_value_heads;
        for l in &self.layers {
            if l.kind == LayerKind::Linear {
                self.gpu.zero(l.delta_state, hv * 128 * 128 * 4)?;
                self.gpu.zero(l.conv_state, self.cfg.conv_dim() * 3 * 2)?;
            }
        }
        Ok(())
    }

    /// Run `ids` (one sequence, starting at position `pos`) and return the greedy next token.
    /// With `dump`, also return the residual stream after every layer (as f32).
    pub fn forward(&mut self, ids: &[u32], pos: usize, mut dump: Option<&mut Vec<Vec<f32>>>) -> Result<u32> {
        let t = ids.len();
        anyhow::ensure!(t > 0 && t <= self.max_tokens, "chunk of {t} tokens exceeds max_tokens");
        anyhow::ensure!(pos + t <= self.max_ctx, "context overflow");
        let c = self.cfg.clone();
        let hd = c.hidden_size as u32;
        let eps = c.rms_norm_eps as f32;
        let (s, g) = (&self.s, &mut self.gpu);
        let st = g.stream.clone();

        g.upload(s.ids, ids)?;
        unsafe {
            g.k.embed(&st, cfg((t as u32, 1, 1), 256), s.ids as *const u32, self.embed as *const u16, s.x as *mut u16, hd)
        }
        .map_err(de)?;

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
                    let n = c.gdn_proj_dim();
                    g.gemm(t, n, c.hidden_size, s.h, l.gdn_proj, s.proj, CUDA_R_16BF)?;
                    let cd = c.conv_dim() as u32;
                    unsafe {
                        g.k.conv1d_silu(
                            &st,
                            cfg((cd.div_ceil(256), 1, 1), 256),
                            s.proj as *const u16,
                            n as u32,
                            l.conv_state as *mut u16,
                            l.conv_w as *const u16,
                            s.conv as *mut u16,
                            t as u32,
                            cd,
                        )
                    }
                    .map_err(de)?;
                    let z_off = c.conv_dim() as u32;
                    let b_off = z_off + c.value_dim() as u32;
                    let a_off = b_off + c.linear_num_value_heads as u32;
                    unsafe {
                        g.k.gdn_recurrent(
                            &st,
                            cfg((c.linear_num_value_heads as u32, 1, 1), 512),
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
                            t as u32,
                            c.linear_num_key_heads as u32,
                            c.linear_num_value_heads as u32,
                            eps,
                        )
                    }
                    .map_err(de)?;
                    g.gemm(t, c.hidden_size, c.value_dim(), s.mix, l.out_proj, s.delta, CUDA_R_16BF)?;
                }
                LayerKind::Full => {
                    let n = c.attn_proj_dim();
                    let (h, hkv) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);
                    g.gemm(t, n, c.hidden_size, s.h, l.qkv, s.proj, CUDA_R_16BF)?;
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
                            pos as u32,
                            h,
                            hkv,
                            c.rot_dim() as u32,
                            c.rope_parameters.rope_theta as f32,
                            eps,
                        )
                    }
                    .map_err(de)?;
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
                            pos as u32,
                            h,
                            hkv,
                            1.0 / (c.head_dim as f32).sqrt(),
                        )
                    }
                    .map_err(de)?;
                    g.gemm(t, c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, l.o_proj, s.delta, CUDA_R_16BF)?;
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
            let inter = c.intermediate_size;
            g.gemm(t, 2 * inter, c.hidden_size, s.h, l.gate_up, s.proj, CUDA_R_16BF)?;
            unsafe {
                g.k.silu_mul(&st, cfg((t as u32, 1, 1), 256), s.proj as *const u16, s.mix as *mut u16, inter as u32)
            }
            .map_err(de)?;
            g.gemm(t, c.hidden_size, inter, s.mix, l.down, s.delta, CUDA_R_16BF)?;

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

        unsafe {
            g.k.add_rmsnorm(
                &st,
                cfg((t as u32, 1, 1), 256),
                s.x as *mut u16,
                s.delta as *const u16,
                self.final_norm as *const u16,
                s.h as *mut u16,
                hd,
                eps,
                1,
            )
        }
        .map_err(de)?;
        let last = s.h + ((t - 1) * c.hidden_size * 2) as u64;
        g.gemm(1, c.vocab_size, c.hidden_size, last, self.lm_head, s.logits, CUDA_R_32F)?;
        unsafe { g.k.argmax(&st, cfg((1, 1, 1), 1024), s.logits as *const f32, c.vocab_size as u32, s.tok as *mut u32) }
            .map_err(de)?;
        let tok: Vec<u32> = g.download(s.tok, 1)?;
        Ok(tok[0])
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.gpu.download(self.s.logits, self.cfg.vocab_size)
    }

    /// Post-final-norm hidden state of the last forward (all tokens), as f32.
    pub fn final_hidden(&self, t: usize) -> Result<Vec<f32>> {
        let h: Vec<u16> = self.gpu.download(self.s.h, t * self.cfg.hidden_size)?;
        Ok(h.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }
}
