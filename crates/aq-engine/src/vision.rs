//! The vision tower of a Qwen3.5 checkpoint (`model.visual.*`): the patches of one image in,
//! one embedding row per merged patch out, as `Batch::rows` takes them.
//!
//! Patch embedding, the learned position table resampled to the image's grid, then blocks of
//! LayerNorm, attention over every patch with a rotation by row and column, and an MLP; the
//! merger then turns each `merge x merge` block of patches into one row of the language model.
//! Scratch is allocated per image and freed: an image is rare next to a decode step.

use crate::cublaslt::CUDA_R_32F;
use crate::gpu::{Gpu, cfg, de};
use crate::weights::Checkpoint;
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Deserialize;
use std::path::Path;

const P: &str = "model.visual.";
/// `rope_theta` of every Qwen vision config.
const THETA: f32 = 10000.0;

#[derive(Deserialize)]
struct Config {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    out_hidden_size: usize,
    num_position_embeddings: usize,
}

#[derive(Deserialize)]
struct Root {
    vision_config: Config,
}

/// A linear layer or a norm: weight and bias on the device, bf16.
type Wb = (u64, u64);

pub struct Vision {
    c: Config,
    patch: Wb,
    /// The position table, `[side * side][hidden]`.
    pos: Vec<f32>,
    /// norm1, qkv, proj, norm2, fc1, fc2 of each block.
    blocks: Vec<[Wb; 6]>,
    merger: [Wb; 3],
}

impl Vision {
    pub fn load(g: &mut Gpu, dir: &Path, ck: &Checkpoint) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let c = serde_json::from_str::<Root>(&raw).context("config.json has no vision_config")?.vision_config;
        let mut wb = |n: &str| -> Result<Wb> {
            Ok((g.alloc_upload(ck.bf16(&format!("{P}{n}.weight"))?.bytes)?, g.alloc_upload(ck.bf16(&format!("{P}{n}.bias"))?.bytes)?))
        };
        let blocks = (0..c.depth)
            .map(|i| {
                let mut l = [(0, 0); 6];
                for (slot, n) in l.iter_mut().zip(["norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.linear_fc2"]) {
                    *slot = wb(&format!("blocks.{i}.{n}"))?;
                }
                Ok(l)
            })
            .collect::<Result<_>>()?;
        let (patch, merger) = (wb("patch_embed.proj")?, [wb("merger.norm")?, wb("merger.linear_fc1")?, wb("merger.linear_fc2")?]);
        Ok(Self { patch, pos: ck.f32_vec(&format!("{P}pos_embed.weight"))?, blocks, merger, c })
    }

    /// The rows of one image of `h x w` patches. `patches` is `[h * w][channels * frames * patch
    /// * patch]` bf16 in the processor's order (blocks of `merge x merge`, row-major inside and
    /// between); the result is `[h * w / merge^2][out_hidden]` bf16. With `dump`, the rows into
    /// the first block and out of each block are pushed to it.
    pub fn encode(&self, g: &mut Gpu, patches: &[u16], h: usize, w: usize, mut dump: Option<&mut Vec<Vec<u16>>>) -> Result<Vec<u16>> {
        let c = &self.c;
        let (n, d, m, mlp) = (h * w, c.hidden_size, c.spatial_merge_size, c.intermediate_size);
        let pd = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        ensure!(n > 0 && patches.len() == n * pd && h % m == 0 && w % m == 0, "{} patch values for a {h}x{w} grid", patches.len());
        let (heads, hd) = (c.num_heads, d / c.num_heads);
        let side = c.num_position_embeddings.isqrt();

        // Row and column of each patch, and the position table resampled at it (bilinear,
        // corners aligned).
        let mut rc = Vec::with_capacity(n * 2);
        let mut pos = Vec::with_capacity(n * d);
        let tap = |x: usize, size: usize| {
            let f = if size > 1 { x as f32 * (side - 1) as f32 / (size - 1) as f32 } else { 0.0 };
            (f as usize, (f as usize + 1).min(side - 1), f - f.floor())
        };
        for i in 0..n {
            let (blk, inner) = (i / (m * m), i % (m * m));
            let (row, col) = (blk / (w / m) * m + inner / m, blk % (w / m) * m + inner % m);
            rc.extend([row as u32, col as u32]);
            let ((r0, r1, fr), (c0, c1, fc)) = (tap(row, h), tap(col, w));
            let taps = [(r0, c0, (1.0 - fr) * (1.0 - fc)), (r0, c1, (1.0 - fr) * fc), (r1, c0, fr * (1.0 - fc)), (r1, c1, fr * fc)];
            pos.extend((0..d).map(|j| bf16::from_f32(taps.iter().map(|&(r, c, wt)| self.pos[(r * side + c) * d + j] * wt).sum()).to_bits()));
        }

        let sizes = [n * pd * 2, n * 8, n * d * 2, n * d * 2, n * (3 * d).max(mlp) * 4, n * mlp * 2, n * d * 2, n * d * 2, n * d * 2, n * n * 4, n * n * 2];
        let mut bufs = [0u64; 11];
        for (b, bytes) in bufs.iter_mut().zip(sizes) {
            *b = g.alloc_uninit(bytes)?;
        }
        let [px, rcd, x, y, f, a, q, k, vt, sc, pr] = bufs;
        g.upload(px, patches)?;
        g.upload(rcd, &rc)?;
        g.upload(y, &pos)?;
        let rows = |r: usize| cfg((r as u32, 1, 1), 256);
        // y[r, :] = x[r, :] . w^T + b, through `f`, with an activation and a residual.
        let linear = |g: &mut Gpu, r: usize, cols: usize, k: usize, x: u64, (w, b): Wb, res: u64, out: u64, act: u32| -> Result<()> {
            let s = g.raw_stream();
            g.lt.gemm(s, r, cols, k, x, w, f, CUDA_R_32F)?;
            unsafe { g.k.vis_affine(&g.stream, rows(r), f as *const f32, b as *const u16, res as *const u16, out as *mut u16, cols as u32, act, (res != 0) as u32) }.map_err(de)
        };
        let norm = |g: &Gpu, (w, b): Wb, x: u64, out: u64| unsafe { g.k.layer_norm(&g.stream, rows(n), x as *const u16, w as *const u16, b as *const u16, out as *mut u16, d as u32, 1e-6) }.map_err(de);

        linear(g, n, d, pd, px, self.patch, y, x, 0)?;
        for &[norm1, qkv, proj, norm2, fc1, fc2] in &self.blocks {
            if let Some(dump) = dump.as_deref_mut() {
                dump.push(g.download(x, n * d)?);
            }
            norm(g, norm1, x, y)?;
            let s = g.raw_stream();
            g.lt.gemm(s, n, 3 * d, d, y, qkv.0, f, CUDA_R_32F)?;
            unsafe { g.k.vis_qkv(&g.stream, cfg((n as u32, heads as u32, 1), 64), f as *const f32, qkv.1 as *const u16, rcd as *const u32, q as *mut u16, k as *mut u16, vt as *mut u16, n as u32, heads as u32, hd as u32, THETA) }.map_err(de)?;
            for head in 0..heads {
                let (qk, v) = ((head * n * hd * 2) as u64, (head * hd * n * 2) as u64);
                g.lt.gemm(s, n, n, hd, q + qk, k + qk, sc, CUDA_R_32F)?;
                unsafe { g.k.vis_softmax(&g.stream, rows(n), sc as *const f32, pr as *mut u16, n as u32, (hd as f32).powf(-0.5)) }.map_err(de)?;
                g.lt.gemm_rows(s, n, hd, n, pr, vt + v, y + (head * hd * 2) as u64, d)?;
            }
            linear(g, n, d, d, y, proj, x, x, 0)?;
            norm(g, norm2, x, y)?;
            linear(g, n, mlp, d, y, fc1, 0, a, 1)?;
            linear(g, n, d, mlp, a, fc2, x, x, 0)?;
        }
        if let Some(dump) = dump {
            dump.push(g.download(x, n * d)?);
        }
        let ([mnorm, fc1, fc2], out, md) = (self.merger, c.out_hidden_size, m * m * d);
        ensure!(d <= mlp && out <= md, "merger wider than the tower's scratch");
        norm(g, mnorm, x, y)?;
        linear(g, n / (m * m), md, md, y, fc1, 0, a, 2)?;
        linear(g, n / (m * m), out, md, a, fc2, 0, y, 0)?;
        let rows = g.download(y, n / (m * m) * out)?;
        for b in bufs {
            g.free(b)?;
        }
        Ok(rows)
    }
}
