//! Device kernels for the Qwen3.5 (`qwen3_5`) hybrid architecture, written in Rust and
//! compiled to PTX by cuda-oxide.
//!
//! Conventions:
//! - bf16 values travel as raw `u16` bits; math happens in f32.
//! - Every kernel takes raw device pointers: the engine owns one preallocated arena and
//!   hands out views into it, which keeps buffer addresses stable for CUDA graph capture.
//! - `rb(x)` rounds an f32 to bf16 and back. It is used where the reference (transformers,
//!   bf16 eager) materializes a bf16 tensor, so intermediate rounding matches.
//!
//! These are correctness-first versions (milestone 1). The fast paths (paged attention,
//! chunked gated delta rule, fused epilogues) come later and get A/B tested against these.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments, non_snake_case)]

use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::{thread, warp};

const FULL: u32 = 0xffff_ffff;

#[inline(always)]
unsafe fn ld(p: *const u16, i: usize) -> f32 {
    bf16_to_f32(unsafe { *p.add(i) })
}

#[inline(always)]
unsafe fn st(p: *mut u16, i: usize, v: f32) {
    unsafe { *p.add(i) = f32_to_bf16_rne(v) }
}

#[inline(always)]
fn rb(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(v))
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + libm::expf(-x))
}

/// torch.nn.functional.softplus with beta=1, threshold=20.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { libm::log1pf(libm::expf(x)) }
}

#[inline(always)]
fn warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(FULL, v, 16);
    v += warp::shuffle_xor_f32_sync(FULL, v, 8);
    v += warp::shuffle_xor_f32_sync(FULL, v, 4);
    v += warp::shuffle_xor_f32_sync(FULL, v, 2);
    v += warp::shuffle_xor_f32_sync(FULL, v, 1);
    v
}

#[allow(dead_code)]
#[inline(always)]
fn warp_max(mut v: f32) -> f32 {
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 16));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 8));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 4));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 2));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 1));
    v
}

/// Sum over the whole block; every thread gets the result. `smem` holds >= 32 f32.
#[inline(always)]
unsafe fn block_sum(v: f32, smem: *mut f32) -> f32 {
    let v = warp_sum(v);
    let tid = thread::threadIdx_x();
    let nw = thread::blockDim_x().div_ceil(32);
    if tid % 32 == 0 {
        unsafe { *smem.add((tid / 32) as usize) = v };
    }
    thread::sync_threads();
    let mut t = 0.0f32;
    let mut i = 0u32;
    while i < nw {
        t += unsafe { *smem.add(i as usize) };
        i += 1;
    }
    thread::sync_threads();
    t
}

#[cuda_device::cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::{SharedArray, kernel};

    /// out[t, :] = table[ids[t], :]. grid = (T), block = (256).
    #[kernel]
    pub unsafe fn embed(ids: *const u32, table: *const u16, out: *mut u16, hidden: u32) {
        let t = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let id = unsafe { *ids.add(t) } as usize;
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            unsafe { *out.add(t * h + i) = *table.add(id * h + i) };
            i += thread::blockDim_x() as usize;
        }
    }

    /// Zero-centered RMSNorm (`x * rsqrt(mean(x^2) + eps) * (1 + w)`), optionally fused with the
    /// residual add: if `add` != 0, first `x = bf16(x + delta)` in place.
    /// grid = (T), block = (256).
    #[kernel]
    pub unsafe fn add_rmsnorm(
        x: *mut u16,
        delta: *const u16,
        w: *const u16,
        out: *mut u16,
        hidden: u32,
        eps: f32,
        add: u32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let row = thread::blockIdx_x() as usize * hidden as usize;
        let h = hidden as usize;
        let bd = thread::blockDim_x() as usize;
        let mut ss = 0.0f32;
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            let mut v = unsafe { ld(x, row + i) };
            if add != 0 {
                v = rb(v + unsafe { ld(delta, row + i) });
                unsafe { st(x, row + i, v) };
            }
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            let v = unsafe { ld(x, row + i) };
            unsafe { st(out, row + i, v * inv * (1.0 + ld(w, i))) };
            i += bd;
        }
    }

    /// out[t, i] = silu(gu[t, i]) * gu[t, inter + i]. grid = (T), block = (256).
    #[kernel]
    pub unsafe fn silu_mul(gu: *const u16, out: *mut u16, inter: u32) {
        let t = thread::blockIdx_x() as usize;
        let n = inter as usize;
        let mut i = thread::threadIdx_x() as usize;
        while i < n {
            let g = unsafe { ld(gu, t * 2 * n + i) };
            let u = unsafe { ld(gu, t * 2 * n + n + i) };
            unsafe { st(out, t * n + i, rb(silu(g)) * u) };
            i += thread::blockDim_x() as usize;
        }
    }

    /// Full-attention prologue for one token and one head.
    ///
    /// `qkv` rows are `[q|gate interleaved per head: H * 2D][k: Hkv * D][v: Hkv * D]`.
    /// Query heads get q_norm + partial RoPE into `q_out[T, H, D]`; key heads get
    /// k_norm + partial RoPE written into the paged cache, and v copied next to it.
    /// Paged layout: `cache[block][slot in block][Hkv][D]`, block ids from the token's
    /// sequence row of `block_tables` (`max_blocks` entries per sequence).
    /// grid = (T, H + Hkv), block = (D).
    #[kernel]
    pub unsafe fn attn_prep(
        qkv: *const u16,
        ld_qkv: u32,
        q_norm: *const u16,
        k_norm: *const u16,
        q_out: *mut u16,
        k_cache: *mut u16,
        v_cache: *mut u16,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        rot_dim: u32,
        rope_theta: f32,
        eps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut ROW: SharedArray<f32, 512> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let rowbuf = (&raw mut ROW).cast::<f32>();

        let t = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let dim = thread::blockDim_x() as usize;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let row = t * ld_qkv as usize;
        let pos = unsafe { *tok_pos.add(t) } as usize;

        let is_q = head < h;
        let (src, w) = if is_q {
            (row + head * 2 * dim + d, q_norm)
        } else {
            (row + h * 2 * dim + (head - h) * dim + d, k_norm)
        };
        let x = unsafe { ld(qkv, src) };
        let ss = unsafe { block_sum(x * x, red) };
        let inv = 1.0 / libm::sqrtf(ss / dim as f32 + eps);
        // RMSNorm output is materialized in bf16 by the reference.
        let xn = rb(x * inv * (1.0 + unsafe { ld(w, d) }));
        unsafe { *rowbuf.add(d) = xn };
        thread::sync_threads();

        let rot = rot_dim as usize;
        let half = rot / 2;
        let mut y = xn;
        if d < rot {
            let i = d % half;
            let inv_freq = 1.0 / libm::powf(rope_theta, (2 * i) as f32 / rot as f32);
            let f = inv_freq * pos as f32;
            let c = rb(libm::cosf(f));
            let s = rb(libm::sinf(f));
            // rotate_half: [-x2, x1]
            let other = if d < half {
                -unsafe { *rowbuf.add(d + half) }
            } else {
                unsafe { *rowbuf.add(d - half) }
            };
            y = rb(rb(xn * c) + rb(other * s));
        }

        if is_q {
            unsafe { st(q_out, (t * h + head) * dim + d, y) };
        } else {
            let kv = head - h;
            let seq = unsafe { *tok_seq.add(t) } as usize;
            let bs = block_size as usize;
            let blk = unsafe { *block_tables.add(seq * max_blocks as usize + pos / bs) } as usize;
            let off = ((blk * bs + pos % bs) * hkv + kv) * dim + d;
            unsafe { st(k_cache, off, y) };
            let v = unsafe { *qkv.add(row + h * 2 * dim + hkv * dim + kv * dim + d) };
            unsafe { *v_cache.add(off) = v };
        }
    }

    /// Causal GQA attention over the paged KV cache, with the output gate fused:
    /// `out[t, h*D + d] = attn[t, h, d] * sigmoid(gate[t, h, d])`.
    ///
    /// grid = (T, H), block = (256) = 8 warps. Each warp walks a strided subset of keys with
    /// an online softmax over its own lane slice of D (D = 256, 8 dims per lane); the 8
    /// partial states are merged through shared memory at the end.
    #[kernel]
    pub unsafe fn attention(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
    ) {
        const D: usize = 256;
        const PER: usize = 8;
        static mut M: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut L: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut ACC: SharedArray<f32, 2048> = SharedArray::UNINIT;
        let sm = (&raw mut M).cast::<f32>();
        let sl = (&raw mut L).cast::<f32>();
        let sacc = (&raw mut ACC).cast::<f32>();

        let t = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let kv = head / (h / hkv);
        let n_keys = unsafe { *tok_pos.add(t) } as usize + 1;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t) as usize * max_blocks as usize) };

        let mut qv = [0.0f32; PER];
        let qbase = (t * h + head) * D + lane * PER;
        let mut i = 0;
        while i < PER {
            qv[i] = unsafe { ld(q, qbase + i) } * scale;
            i += 1;
        }
        let mut acc = [0.0f32; PER];
        let mut m = -1.0e30f32;
        let mut l = 0.0f32;

        let mut j = w;
        while j < n_keys {
            let blk = unsafe { *table.add(j / bs) } as usize;
            let kb = ((blk * bs + j % bs) * hkv + kv) * D + lane * PER;
            let mut dot = 0.0f32;
            let mut i = 0;
            while i < PER {
                dot += qv[i] * unsafe { ld(k_cache, kb + i) };
                i += 1;
            }
            let s = warp_sum(dot);
            let m_new = m.max(s);
            let corr = libm::expf(m - m_new);
            let p = libm::expf(s - m_new);
            l = l * corr + p;
            let mut i = 0;
            while i < PER {
                acc[i] = acc[i] * corr + p * unsafe { ld(v_cache, kb + i) };
                i += 1;
            }
            m = m_new;
            j += 8;
        }

        if lane == 0 {
            unsafe {
                *sm.add(w) = m;
                *sl.add(w) = l;
            }
        }
        let mut i = 0;
        while i < PER {
            unsafe { *sacc.add(w * D + lane * PER + i) = acc[i] };
            i += 1;
        }
        thread::sync_threads();

        // Thread tid now owns output dim d = tid.
        let d = tid;
        let mut mm = -1.0e30f32;
        let mut k = 0;
        while k < 8 {
            mm = mm.max(unsafe { *sm.add(k) });
            k += 1;
        }
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        let mut k = 0;
        while k < 8 {
            let f = libm::expf(unsafe { *sm.add(k) } - mm);
            den += unsafe { *sl.add(k) } * f;
            num += unsafe { *sacc.add(k * D + d) } * f;
            k += 1;
        }
        let o = rb(num / den);
        let g = unsafe { ld(qkv, t * ld_qkv as usize + head * 2 * D + D + d) };
        unsafe { st(out, t * h * D + head * D + d, o * rb(sigmoid(g))) };
    }

    /// Depthwise causal conv1d (kernel 4) + SiLU over the first `conv_dim` columns of `x`
    /// (row stride `ld_x`), carrying the last 3 inputs per channel in the sequence's state
    /// slot: `state[slot][c * 3 + j]`. Sequence `s` owns rows `seq_start[s]..+seq_len[s]`.
    /// grid = (ceil(conv_dim / 256), S), block = (256).
    #[kernel]
    pub unsafe fn conv1d_silu(
        x: *const u16,
        ld_x: u32,
        state_base: *mut u16,
        w: *const u16,
        out: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if c >= conv_dim as usize {
            return;
        }
        let sq = thread::blockIdx_y() as usize;
        let cd = conv_dim as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n_tokens = unsafe { *seq_len.add(sq) };
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let w0 = unsafe { ld(w, c * 4) };
        let w1 = unsafe { ld(w, c * 4 + 1) };
        let w2 = unsafe { ld(w, c * 4 + 2) };
        let w3 = unsafe { ld(w, c * 4 + 3) };
        let mut s0 = unsafe { ld(state, c * 3) };
        let mut s1 = unsafe { ld(state, c * 3 + 1) };
        let mut s2 = unsafe { ld(state, c * 3 + 2) };
        let mut t = start;
        while t < start + n_tokens as usize {
            let xt = unsafe { ld(x, t * ld_x as usize + c) };
            let acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt;
            unsafe { st(out, t * cd + c, silu(rb(acc))) };
            s0 = s1;
            s1 = s2;
            s2 = xt;
            t += 1;
        }
        unsafe {
            st(state, c * 3, s0);
            st(state, c * 3 + 1, s1);
            st(state, c * 3 + 2, s2);
        }
    }

    /// Gated delta rule, token-recurrent form, fused with the gated RMSNorm epilogue.
    ///
    /// One block per (value head `h`, sequence) (grid = (Hv, S)), block = (512): thread `(r, c)` with
    /// `c = tid % 128` (value column) and `r = tid / 128` keeps rows `32r..32r+32` of column
    /// `c` of the f32 state `S[h] (Dk = 128 x Dv = 128)` in registers.
    ///
    /// Per token: `S *= exp(g)`, `delta = beta * (v - S^T k)`, `S += k delta^T`, `o = S^T q`,
    /// then `out = rmsnorm(o) * w * silu(z)`. q and k are L2-normalized; q is scaled by
    /// `1/sqrt(Dk)`. `conv` rows are `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`; `proj` rows hold z at
    /// `z_off`, b at `b_off`, a at `a_off`.
    #[kernel]
    pub unsafe fn gdn_recurrent(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const f32,
        state_base: *mut f32,
        out: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        seq_off: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        eps: f32,
    ) {
        const DK: usize = 128;
        const DV: usize = 128;
        const R: usize = 32;
        static mut QS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut RED1: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut RED2: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut SC: SharedArray<f32, 8> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<f32>();
        let ks = (&raw mut KS).cast::<f32>();
        let red1 = (&raw mut RED1).cast::<f32>();
        let red2 = (&raw mut RED2).cast::<f32>();
        let sc = (&raw mut SC).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let c = tid % DV;
        let r = tid / DV;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let cd = conv_dim as usize;
        let ldp = ld_proj as usize;
        let key_dim = hk * DK;

        let sq = (thread::blockIdx_y() + seq_off) as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n_tokens = unsafe { *seq_len.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * hv * DK * DV) };
        let sbase = h * DK * DV;
        let mut s = [0.0f32; R];
        let mut i = 0;
        while i < R {
            s[i] = unsafe { *state.add(sbase + (r * R + i) * DV + c) };
            i += 1;
        }
        let neg_a = -libm::expf(unsafe { *a_log.add(h) });
        let dtb = unsafe { *dt_bias.add(h) };
        let nw = unsafe { *norm_w.add(c) };
        let q_scale = 1.0 / libm::sqrtf(DK as f32);

        let mut t = start;
        while t < start + n_tokens {
            let crow = t * cd;
            if tid < DK {
                unsafe { *qs.add(tid) = ld(conv, crow + kh * DK + tid) };
            } else if tid < 2 * DK {
                let k = tid - DK;
                unsafe { *ks.add(k) = ld(conv, crow + key_dim + kh * DK + k) };
            }
            thread::sync_threads();
            if warp_id < 2 {
                let src = if warp_id == 0 { qs } else { ks };
                let mut ss = 0.0f32;
                let mut i = 0;
                while i < 4 {
                    let v = unsafe { *src.add(lane * 4 + i) };
                    ss += v * v;
                    i += 1;
                }
                let tot = warp_sum(ss);
                if lane == 0 {
                    unsafe { *sc.add(warp_id) = 1.0 / libm::sqrtf(tot + 1.0e-6) };
                }
            } else if tid == 64 {
                let a = unsafe { ld(proj, t * ldp + a_off as usize + h) };
                let b = unsafe { ld(proj, t * ldp + b_off as usize + h) };
                let g = neg_a * softplus(a + dtb);
                unsafe {
                    *sc.add(2) = libm::expf(g);
                    *sc.add(3) = rb(sigmoid(b));
                }
            }
            thread::sync_threads();
            let inv_q = unsafe { *sc.add(0) } * q_scale;
            let inv_k = unsafe { *sc.add(1) };
            let decay = unsafe { *sc.add(2) };
            let beta = unsafe { *sc.add(3) };

            let mut part = 0.0f32;
            let mut i = 0;
            while i < R {
                s[i] *= decay;
                part += s[i] * unsafe { *ks.add(r * R + i) } * inv_k;
                i += 1;
            }
            unsafe { *red1.add(r * DV + c) = part };
            thread::sync_threads();
            let kv_mem = unsafe {
                *red1.add(c) + *red1.add(DV + c) + *red1.add(2 * DV + c) + *red1.add(3 * DV + c)
            };
            let v = unsafe { ld(conv, crow + 2 * key_dim + h * DV + c) };
            let delta = (v - kv_mem) * beta;
            let mut po = 0.0f32;
            let mut i = 0;
            while i < R {
                s[i] += unsafe { *ks.add(r * R + i) } * inv_k * delta;
                po += s[i] * unsafe { *qs.add(r * R + i) } * inv_q;
                i += 1;
            }
            unsafe { *red2.add(r * DV + c) = po };
            thread::sync_threads();

            if r == 0 {
                let o = rb(unsafe {
                    *red2.add(c) + *red2.add(DV + c) + *red2.add(2 * DV + c) + *red2.add(3 * DV + c)
                });
                let ss = warp_sum(o * o);
                if lane == 0 {
                    unsafe { *red1.add(warp_id) = ss };
                }
                unsafe { *red2.add(c) = o };
            }
            thread::sync_threads();
            if r == 0 {
                let var = unsafe { *red1.add(0) + *red1.add(1) + *red1.add(2) + *red1.add(3) }
                    / DV as f32;
                let o = unsafe { *red2.add(c) };
                let y = nw * rb(o / libm::sqrtf(var + eps));
                let z = unsafe { ld(proj, t * ldp + z_off as usize + h * DV + c) };
                unsafe { st(out, t * hv * DV + h * DV + c, y * silu(z)) };
            }
            thread::sync_threads();
            t += 1;
        }

        let mut i = 0;
        while i < R {
            unsafe { *state.add(sbase + (r * R + i) * DV + c) = s[i] };
            i += 1;
        }
    }

    /// Gated delta rule for sequences that feed exactly one token this step (decode).
    ///
    /// grid = (Hv, n_decode), block = (128): thread `j` owns column `j` of the head's f32
    /// state, all 128 rows in registers, so `S^T k` and `S^T q` are thread-local sums and
    /// the state is read and written exactly once. Same math and epilogue as `gdn_recurrent`.
    #[kernel]
    pub unsafe fn gdn_decode(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const f32,
        state_base: *mut f32,
        out: *mut u16,
        seq_start: *const u32,
        seq_slot: *const u32,
        n_k_heads: u32,
        n_v_heads: u32,
        eps: f32,
    ) {
        const D: usize = 128;
        static mut QS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut R1: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut R2: SharedArray<f32, 4> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<f32>();
        let ks = (&raw mut KS).cast::<f32>();
        let r1 = (&raw mut R1).cast::<f32>();
        let r2 = (&raw mut R2).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let sq = thread::blockIdx_y() as usize;
        let j = thread::threadIdx_x() as usize;
        let lane = j % 32;
        let w = j / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let t = unsafe { *seq_start.add(sq) } as usize;
        let slot = unsafe { *seq_slot.add(sq) } as usize;
        let sp = unsafe { state_base.add((slot * hv + h) * D * D + j) };

        // Issue all state loads first: 128 independent coalesced loads per thread.
        let mut s = [0.0f32; 128];
        #[unroll]
        for i in 0..128 {
            s[i] = unsafe { *sp.add(i * D) };
        }

        let crow = t * conv_dim as usize;
        let key_dim = hk * D;
        let qj = unsafe { ld(conv, crow + kh * D + j) };
        let kj = unsafe { ld(conv, crow + key_dim + kh * D + j) };
        let vj = unsafe { ld(conv, crow + 2 * key_dim + h * D + j) };
        let prow = t * ld_proj as usize;
        let a = unsafe { ld(proj, prow + a_off as usize + h) };
        let b = unsafe { ld(proj, prow + b_off as usize + h) };
        let z = unsafe { ld(proj, prow + z_off as usize + h * D + j) };
        let qq = warp_sum(qj * qj);
        let kk = warp_sum(kj * kj);
        unsafe {
            *qs.add(j) = qj;
            *ks.add(j) = kj;
        }
        if lane == 0 {
            unsafe {
                *r1.add(w) = qq;
                *r1.add(4 + w) = kk;
            }
        }
        thread::sync_threads();
        let inv_q = 1.0
            / libm::sqrtf(unsafe { *r1.add(0) + *r1.add(1) + *r1.add(2) + *r1.add(3) } + 1.0e-6)
            / libm::sqrtf(D as f32);
        let inv_k = 1.0 / libm::sqrtf(unsafe { *r1.add(4) + *r1.add(5) + *r1.add(6) + *r1.add(7) } + 1.0e-6);
        let g = -libm::expf(unsafe { *a_log.add(h) }) * softplus(a + unsafe { *dt_bias.add(h) });
        let decay = libm::expf(g);
        let beta = rb(sigmoid(b));

        let mut kv = 0.0f32;
        #[unroll]
        for i in 0..128 {
            s[i] *= decay;
            kv += s[i] * unsafe { *ks.add(i) };
        }
        let delta = (vj - kv * inv_k) * beta;
        let dk = delta * inv_k;
        let mut o = 0.0f32;
        #[unroll]
        for i in 0..128 {
            s[i] += unsafe { *ks.add(i) } * dk;
            o += s[i] * unsafe { *qs.add(i) };
        }
        #[unroll]
        for i in 0..128 {
            unsafe { *sp.add(i * D) = s[i] };
        }

        let ob = rb(o * inv_q);
        let oo = warp_sum(ob * ob);
        if lane == 0 {
            unsafe { *r2.add(w) = oo };
        }
        thread::sync_threads();
        let var = unsafe { *r2.add(0) + *r2.add(1) + *r2.add(2) + *r2.add(3) } / D as f32;
        let y = unsafe { *norm_w.add(j) } * rb(ob / libm::sqrtf(var + eps));
        unsafe { st(out, t * hv * D + h * D + j, y * silu(z)) };
    }

    // ------------------------------------------------------------------------------------
    // Prefill path: token-parallel conv, chunked gated delta rule, tiled attention.
    // ------------------------------------------------------------------------------------

    /// Token-parallel causal conv1d + SiLU: thread (c, t) reads its 4-tap window from the
    /// projection rows of its own sequence, or from the slot's carried state before the
    /// chunk start. State is advanced afterwards by `conv_state_update`.
    /// grid = (ceil(conv_dim / 256), T), block = (256).
    #[kernel]
    pub unsafe fn conv1d_tok(
        x: *const u16,
        ld_x: u32,
        state_base: *const u16,
        w: *const u16,
        out: *mut u16,
        tok_seq: *const u32,
        seq_start: *const u32,
        seq_slot: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let cd = conv_dim as usize;
        if c >= cd {
            return;
        }
        let t = thread::blockIdx_y() as usize;
        let sq = unsafe { *tok_seq.add(t) } as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let i = t - start;
        let ldx = ld_x as usize;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < 4 {
            // window element j holds input at local index i - 3 + j
            let v = if i + j >= 3 {
                unsafe { ld(x, (start + i + j - 3) * ldx + c) }
            } else {
                unsafe { ld(state, c * 3 + i + j) }
            };
            acc += unsafe { ld(w, c * 4 + j) } * v;
            j += 1;
        }
        unsafe { st(out, t * cd + c, silu(rb(acc))) };
    }

    /// Advance each sequence's conv state to its last 3 inputs. grid = (ceil(cd/256), S).
    #[kernel]
    pub unsafe fn conv_state_update(
        x: *const u16,
        ld_x: u32,
        state_base: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let cd = conv_dim as usize;
        if c >= cd {
            return;
        }
        let sq = thread::blockIdx_y() as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n = unsafe { *seq_len.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let old = unsafe { [*state.add(c * 3), *state.add(c * 3 + 1), *state.add(c * 3 + 2)] };
        let mut j = 0usize;
        while j < 3 {
            // new state j = input at local index n - 3 + j
            let v = if n + j >= 3 {
                unsafe { *x.add((start + n + j - 3) * ld_x as usize + c) }
            } else {
                old[n + j]
            };
            unsafe { *state.add(c * 3 + j) = v };
            j += 1;
        }
    }

    /// Chunked gated delta rule, phase 1 (fully parallel): for one 32-token chunk and one
    /// value head, build everything the sequential scan needs.
    ///
    /// With `Kn`, `Qn` the L2-normalized keys/queries (q also scaled by 1/sqrt(Dk)), `g` the
    /// log decays and `G = cumsum(g)` within the chunk:
    ///   A[i][j]  = beta_i exp(G_i - G_j) Kn_i.Kn_j            (j < i)
    ///   X        = (I + A)^-1                                (forward substitution)
    ///   U        = X (beta V)              W  = X (beta exp(G) Kn)
    ///   Qp       = Qn exp(G)               Kp = Kn exp(G_last - G)
    ///   intra    = exp(G_i - G_j) Qn_i.Kn_j                 (j <= i)
    ///   decay    = exp(G_last)
    /// grid = (n_chunks, Hv), block = (128).
    #[kernel]
    pub unsafe fn gdn_chunk_prep(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        chunk_start: *const u32,
        chunk_len: *const u32,
        u_out: *mut f32,
        w_out: *mut f32,
        qp_out: *mut f32,
        kp_out: *mut f32,
        intra_out: *mut f32,
        decay_out: *mut f32,
        n_k_heads: u32,
        n_v_heads: u32,
    ) {
        const C: usize = 32;
        const D: usize = 128;
        static mut KN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut QN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut AM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut XM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut GB: SharedArray<f32, 96> = SharedArray::UNINIT;
        let kn = (&raw mut KN).cast::<f32>();
        let qn = (&raw mut QN).cast::<f32>();
        let am = (&raw mut AM).cast::<f32>();
        let xm = (&raw mut XM).cast::<f32>();
        let gb = (&raw mut GB).cast::<f32>(); // [0..32) gamma, [32..64) beta, [64..96) scratch

        let ci = thread::blockIdx_x() as usize;
        let h = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let lane = d % 32;
        let wid = d / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let cd = conv_dim as usize;
        let key_dim = hk * D;
        let c0 = unsafe { *chunk_start.add(ci) } as usize;
        let len = unsafe { *chunk_len.add(ci) } as usize;

        // Load raw q/k columns; rows past the chunk end are zero.
        let mut r = 0usize;
        while r < C {
            let (kv, qv) = if r < len {
                let row = (c0 + r) * cd;
                unsafe { (ld(conv, row + key_dim + kh * D + d), ld(conv, row + kh * D + d)) }
            } else {
                (0.0, 0.0)
            };
            unsafe {
                *kn.add(r * D + d) = kv;
                *qn.add(r * D + d) = qv;
            }
            r += 1;
        }
        if d < C {
            let (g, beta) = if d < len {
                let prow = (c0 + d) * ld_proj as usize;
                let a = unsafe { ld(proj, prow + a_off as usize + h) };
                let b = unsafe { ld(proj, prow + b_off as usize + h) };
                let g = -libm::expf(unsafe { *a_log.add(h) }) * softplus(a + unsafe { *dt_bias.add(h) });
                (g, rb(sigmoid(b)))
            } else {
                (0.0, 0.0)
            };
            unsafe {
                *gb.add(64 + d) = g;
                *gb.add(32 + d) = beta;
            }
        }
        thread::sync_threads();

        // L2-normalize rows (warp w handles rows 8w..8w+8); q also gets 1/sqrt(D).
        let mut rr = 0usize;
        while rr < 8 {
            let row = wid * 8 + rr;
            let mut sk = 0.0f32;
            let mut sq = 0.0f32;
            let mut e = 0usize;
            while e < 4 {
                let k = unsafe { *kn.add(row * D + lane * 4 + e) };
                let q = unsafe { *qn.add(row * D + lane * 4 + e) };
                sk += k * k;
                sq += q * q;
                e += 1;
            }
            let ik = 1.0 / libm::sqrtf(warp_sum(sk) + 1.0e-6);
            let iq = 1.0 / libm::sqrtf(warp_sum(sq) + 1.0e-6) / libm::sqrtf(D as f32);
            let mut e = 0usize;
            while e < 4 {
                unsafe {
                    *kn.add(row * D + lane * 4 + e) *= ik;
                    *qn.add(row * D + lane * 4 + e) *= iq;
                }
                e += 1;
            }
            rr += 1;
        }
        if d == 0 {
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < C {
                acc += unsafe { *gb.add(64 + i) };
                unsafe { *gb.add(i) = acc };
                i += 1;
            }
        }
        thread::sync_threads();

        // A (strict lower) and intra (lower incl. diagonal). Warp w owns rows 8w..8w+8, lane
        // owns column j = lane; the dim loop is skewed by j so lanes hit distinct banks.
        let ib = ci * hv + h;
        let j = lane;
        let mut e = 0usize;
        while e < 8 {
            let i = wid * 8 + e;
            let idx = i * C + j;
            let mut a = 0.0f32;
            let mut q = 0.0f32;
            if j <= i {
                let f = libm::expf(unsafe { *gb.add(i) - *gb.add(j) });
                let mut kk = 0.0f32;
                let mut qk = 0.0f32;
                let mut x = 0usize;
                while x < D {
                    let xx = (x + j) % D;
                    let kj = unsafe { *kn.add(j * D + xx) };
                    kk += unsafe { *kn.add(i * D + xx) } * kj;
                    qk += unsafe { *qn.add(i * D + xx) } * kj;
                    x += 1;
                }
                if j < i {
                    a = unsafe { *gb.add(32 + i) } * f * kk;
                }
                q = f * qk;
            }
            unsafe {
                *am.add(idx) = a;
                *xm.add(idx) = if i == j { 1.0 } else { 0.0 };
                *intra_out.add(ib * C * C + idx) = q;
            }
            e += 1;
        }
        thread::sync_threads();

        // X = (I + A)^-1, row by row: X[i][m] = -A[i][m] - sum_{m<j<i} A[i][j] X[j][m].
        let mut i = 1usize;
        while i < C {
            if d < i {
                let m = d;
                let mut v = -unsafe { *am.add(i * C + m) };
                let mut j = m + 1;
                while j < i {
                    v -= unsafe { *am.add(i * C + j) * *xm.add(j * C + m) };
                    j += 1;
                }
                unsafe { *xm.add(i * C + m) = v };
            }
            thread::sync_threads();
            i += 1;
        }

        // Per column d: U = X (beta V), W = X (beta e^G Kn), plus decayed q/k rows.
        let mut vb = [0.0f32; 32];
        let mut kb = [0.0f32; 32];
        let glast = unsafe { *gb.add(C - 1) };
        #[unroll]
        for m in 0..32 {
            let beta = unsafe { *gb.add(32 + m) };
            let v = if m < len { unsafe { ld(conv, (c0 + m) * cd + 2 * key_dim + h * D + d) } } else { 0.0 };
            vb[m] = beta * v;
            kb[m] = beta * libm::expf(unsafe { *gb.add(m) }) * unsafe { *kn.add(m * D + d) };
        }
        let mut i = 0usize;
        while i < len {
            let mut u = 0.0f32;
            let mut w = 0.0f32;
            #[unroll]
            for m in 0..32 {
                let x = unsafe { *xm.add(i * C + m) };
                u += x * vb[m];
                w += x * kb[m];
            }
            let row = ((c0 + i) * hv + h) * D + d;
            let gi = unsafe { *gb.add(i) };
            unsafe {
                *u_out.add(row) = u;
                *w_out.add(row) = w;
                *qp_out.add(row) = *qn.add(i * D + d) * libm::expf(gi);
                *kp_out.add(row) = *kn.add(i * D + d) * libm::expf(glast - gi);
            }
            i += 1;
        }
        if d == 0 {
            unsafe { *decay_out.add(ib) = libm::expf(glast) };
        }
    }

    /// Chunked gated delta rule, phase 2: sequential over one sequence's chunks, state in
    /// registers (thread `j` owns state column `j`). Per chunk:
    ///   Vn = U - W S;  O = Qp S + intra Vn;  S = decay S + Kp^T Vn.
    /// Writes bf16 `o` (pre-norm) rows. grid = (Hv, n_prefill_seqs), block = (128).
    #[kernel]
    pub unsafe fn gdn_chunk_scan(
        u_in: *const f32,
        w_in: *const f32,
        qp_in: *const f32,
        kp_in: *const f32,
        intra_in: *const f32,
        decay_in: *const f32,
        chunk_start: *const u32,
        chunk_len: *const u32,
        seq_chunk0: *const u32,
        seq_nchunk: *const u32,
        seq_slot: *const u32,
        seq_off: u32,
        state_base: *mut f32,
        o_out: *mut u16,
        n_v_heads: u32,
    ) {
        const C: usize = 32;
        const D: usize = 128;
        static mut BUF: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut VN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut IN: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let buf = (&raw mut BUF).cast::<f32>();
        let vn = (&raw mut VN).cast::<f32>();
        let intra = (&raw mut IN).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let p = thread::blockIdx_y() as usize; // index among prefill sequences
        let j = thread::threadIdx_x() as usize;
        let hv = n_v_heads as usize;
        let slot = unsafe { *seq_slot.add(p + seq_off as usize) } as usize;
        let sp = unsafe { state_base.add((slot * hv + h) * D * D + j) };
        let mut s = [0.0f32; 128];
        #[unroll]
        for k in 0..128 {
            s[k] = unsafe { *sp.add(k * D) };
        }
        let c_first = unsafe { *seq_chunk0.add(p) } as usize;
        let n_chunks = unsafe { *seq_nchunk.add(p) } as usize;

        let mut cc = 0usize;
        while cc < n_chunks {
            let ci = c_first + cc;
            let c0 = unsafe { *chunk_start.add(ci) } as usize;
            let len = unsafe { *chunk_len.add(ci) } as usize;
            let ib = ci * hv + h;
            // W rows -> buf, intra -> smem
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *w_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            let mut e = 0usize;
            while e < 8 {
                unsafe { *intra.add(j * 8 + e) = *intra_in.add(ib * C * C + j * 8 + e) };
                e += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                let mut a = [0.0f32; 4];
                #[unroll]
                for k in 0..128 {
                    a[k % 4] += unsafe { *buf.add(i * D + k) } * s[k];
                }
                let acc = unsafe { *u_in.add(((c0 + i) * hv + h) * D + j) } - ((a[0] + a[1]) + (a[2] + a[3]));
                unsafe { *vn.add(i * D + j) = acc };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *qp_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                let mut a = [0.0f32; 4];
                #[unroll]
                for k in 0..128 {
                    a[k % 4] += unsafe { *buf.add(i * D + k) } * s[k];
                }
                let mut o = (a[0] + a[1]) + (a[2] + a[3]);
                let mut m = 0usize;
                while m <= i {
                    o += unsafe { *intra.add(i * C + m) * *vn.add(m * D + j) };
                    m += 1;
                }
                unsafe { st(o_out, ((c0 + i) * hv + h) * D + j, o) };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *kp_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            thread::sync_threads();
            let dec = unsafe { *decay_in.add(ib) };
            #[unroll]
            for k in 0..128 {
                s[k] *= dec;
            }
            let mut i = 0usize;
            while i < len {
                let v = unsafe { *vn.add(i * D + j) };
                #[unroll]
                for k in 0..128 {
                    s[k] += unsafe { *buf.add(i * D + k) } * v;
                }
                i += 1;
            }
            thread::sync_threads();
            cc += 1;
        }
        #[unroll]
        for k in 0..128 {
            unsafe { *sp.add(k * D) = s[k] };
        }
    }

    /// Gated RMSNorm epilogue for prefill tokens: `out = rmsnorm(o) * w * silu(z)` per
    /// (token, value head). Tokens of decode sequences (`tok_seq < n_decode`) are skipped:
    /// `gdn_decode` already wrote them. grid = (T, Hv), block = (128).
    #[kernel]
    pub unsafe fn gated_norm(
        o_in: *const u16,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        norm_w: *const f32,
        out: *mut u16,
        tok_seq: *const u32,
        n_decode: u32,
        n_v_heads: u32,
        eps: f32,
    ) {
        const D: usize = 128;
        static mut R: SharedArray<f32, 4> = SharedArray::UNINIT;
        let red = (&raw mut R).cast::<f32>();
        let t = thread::blockIdx_x() as usize;
        if unsafe { *tok_seq.add(t) } < n_decode {
            return;
        }
        let h = thread::blockIdx_y() as usize;
        let j = thread::threadIdx_x() as usize;
        let hv = n_v_heads as usize;
        let o = unsafe { ld(o_in, (t * hv + h) * D + j) };
        let ss = warp_sum(o * o);
        if j % 32 == 0 {
            unsafe { *red.add(j / 32) = ss };
        }
        thread::sync_threads();
        let var = unsafe { *red.add(0) + *red.add(1) + *red.add(2) + *red.add(3) } / D as f32;
        let y = unsafe { *norm_w.add(j) } * rb(o / libm::sqrtf(var + eps));
        let z = unsafe { ld(proj, t * ld_proj as usize + z_off as usize + h * D + j) };
        unsafe { st(out, t * hv * D + h * D + j, y * silu(z)) };
    }

    /// Tiled causal GQA prefill attention with the output gate fused. One block per
    /// (tile of up to 16 query rows from one sequence, query head); K/V are streamed in
    /// 32-key tiles through shared memory and reused by all 16 queries.
    /// Thread `d` owns output dim `d` for all 16 queries. grid = (n_tiles, H), block = (256).
    #[kernel]
    pub unsafe fn attention_tile(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        tile_start: *const u32,
        tile_len: *const u32,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
    ) {
        const D: usize = 256;
        const QT: usize = 16;
        const KT: usize = 32;
        static mut QS: SharedArray<u16, 4096> = SharedArray::UNINIT;
        static mut KS: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut VS: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut SS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut CORR: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut LSUM: SharedArray<f32, 16> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<u16>();
        let ks = (&raw mut KS).cast::<u16>();
        let vs = (&raw mut VS).cast::<u16>();
        let ss = (&raw mut SS).cast::<f32>();
        let corr = (&raw mut CORR).cast::<f32>();
        let lsum = (&raw mut LSUM).cast::<f32>();

        let tile = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let kv = head / (h / hkv);
        let t0 = unsafe { *tile_start.add(tile) } as usize;
        let nq = unsafe { *tile_len.add(tile) } as usize;
        let pos0 = unsafe { *tok_pos.add(t0) } as usize;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t0) as usize * max_blocks as usize) };
        let n_keys = pos0 + nq; // keys 0..=last query position

        let mut r = 0usize;
        while r < QT {
            unsafe { *qs.add(r * D + d) = if r < nq { *q.add(((t0 + r) * h + head) * D + d) } else { 0 } };
            r += 1;
        }
        // Row state (m, l) lives in thread r < 16.
        let mut m_row = -1.0e30f32;
        let mut l_row = 0.0f32;
        let mut acc = [0.0f32; 16];

        let mut kb = 0usize;
        while kb < n_keys {
            let nk = (n_keys - kb).min(KT);
            // Load K/V tile: 32 keys x 256 dims; thread d loads dim d of each key.
            let mut k = 0usize;
            while k < KT {
                let (kvv, vvv) = if k < nk {
                    let j = kb + k;
                    let blk = unsafe { *table.add(j / bs) } as usize;
                    let off = ((blk * bs + j % bs) * hkv + kv) * D + d;
                    unsafe { (*k_cache.add(off), *v_cache.add(off)) }
                } else {
                    (0, 0)
                };
                unsafe {
                    *ks.add(k * D + d) = kvv;
                    *vs.add(k * D + d) = vvv;
                }
                k += 1;
            }
            thread::sync_threads();
            // Scores: warp w owns query rows 2w and 2w+1, lane owns key `lane`. The dim loop is
            // skewed by the lane so the 32 lanes hit 32 different banks of the K tile.
            let lane = d % 32;
            let q0 = (d / 32) * 2;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut x = 0usize;
            while x < D {
                let xx = (x + 2 * lane) % D;
                let kv = bf16_to_f32(unsafe { *ks.add(lane * D + xx) });
                s0 += bf16_to_f32(unsafe { *qs.add(q0 * D + xx) }) * kv;
                s1 += bf16_to_f32(unsafe { *qs.add((q0 + 1) * D + xx) }) * kv;
                x += 1;
            }
            let valid = |qi: usize| qi < nq && lane < nk && kb + lane <= pos0 + qi;
            unsafe {
                *ss.add(q0 * KT + lane) = if valid(q0) { s0 * scale } else { -1.0e30 };
                *ss.add((q0 + 1) * KT + lane) = if valid(q0 + 1) { s1 * scale } else { -1.0e30 };
            }
            thread::sync_threads();
            // Online softmax per query row.
            if d < QT {
                let mut mx = m_row;
                let mut k = 0usize;
                while k < KT {
                    mx = mx.max(unsafe { *ss.add(d * KT + k) });
                    k += 1;
                }
                let c = libm::expf(m_row - mx);
                let mut l = l_row * c;
                let mut k = 0usize;
                while k < KT {
                    let sv = unsafe { *ss.add(d * KT + k) };
                    let pv = if sv <= -1.0e29 { 0.0 } else { libm::expf(sv - mx) };
                    unsafe { *ss.add(d * KT + k) = pv };
                    l += pv;
                    k += 1;
                }
                m_row = mx;
                l_row = l;
                unsafe { *corr.add(d) = c };
            }
            thread::sync_threads();
            #[unroll]
            for qi in 0..16 {
                let mut a = acc[qi] * unsafe { *corr.add(qi) };
                let mut k = 0usize;
                while k < nk {
                    a += unsafe { *ss.add(qi * KT + k) } * bf16_to_f32(unsafe { *vs.add(k * D + d) });
                    k += 1;
                }
                acc[qi] = a;
            }
            thread::sync_threads();
            kb += KT;
        }
        if d < QT {
            unsafe { *lsum.add(d) = l_row };
        }
        thread::sync_threads();
        #[unroll]
        for qi in 0..16 {
            if qi < nq {
                let t = t0 + qi;
                let o = rb(acc[qi] / unsafe { *lsum.add(qi) });
                let g = unsafe { ld(qkv, t * ld_qkv as usize + head * 2 * D + D + d) };
                unsafe { st(out, t * h * D + head * D + d, o * rb(sigmoid(g))) };
            }
        }
    }

    /// Final residual add + RMSNorm for the rows that will be sampled only:
    /// `out[s] = rmsnorm(x[rows[s]] + delta[rows[s]])`. grid = (S), block = (256).
    #[kernel]
    pub unsafe fn final_norm_rows(
        x: *const u16,
        delta: *const u16,
        w: *const u16,
        out: *mut u16,
        rows: *const u32,
        hidden: u32,
        eps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut ROWV: SharedArray<f32, 8192> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let rv = (&raw mut ROWV).cast::<f32>();
        let s = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let src = unsafe { *rows.add(s) } as usize * h;
        let bd = thread::blockDim_x() as usize;
        let mut ss = 0.0f32;
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            let v = rb(unsafe { ld(x, src + i) + ld(delta, src + i) });
            unsafe { *rv.add(i) = v };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            unsafe { st(out, s * h + i, *rv.add(i) * inv * (1.0 + ld(w, i))) };
            i += bd;
        }
    }

    /// Row-wise argmax over `[S, n]` f32 logits; grid = (S), block = (1024).
    /// Ties resolve to the lowest index.
    #[kernel]
    pub unsafe fn argmax(logits: *const u16, n: u32, out_base: *mut u32) {
        let x = unsafe { logits.add(thread::blockIdx_x() as usize * n as usize) };
        let out = unsafe { out_base.add(thread::blockIdx_x() as usize) };
        static mut BV: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut BI: SharedArray<u32, 32> = SharedArray::UNINIT;
        let bv = (&raw mut BV).cast::<f32>();
        let bi = (&raw mut BI).cast::<u32>();
        let tid = thread::threadIdx_x();
        let mut best = -3.0e38f32;
        let mut idx = u32::MAX;
        let mut i = tid;
        while i < n {
            let v = unsafe { ld(x, i as usize) };
            if v > best {
                best = v;
                idx = i;
            }
            i += thread::blockDim_x();
        }
        let mut off = 16u32;
        while off > 0 {
            let ov = warp::shuffle_xor_f32_sync(FULL, best, off);
            let oi = warp::shuffle_xor_sync(FULL, idx, off);
            if ov > best || (ov == best && oi < idx) {
                best = ov;
                idx = oi;
            }
            off /= 2;
        }
        if tid % 32 == 0 {
            unsafe {
                *bv.add((tid / 32) as usize) = best;
                *bi.add((tid / 32) as usize) = idx;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let mut k = 0usize;
            while k < thread::blockDim_x().div_ceil(32) as usize {
                let v = unsafe { *bv.add(k) };
                let j = unsafe { *bi.add(k) };
                if v > best || (v == best && j < idx) {
                    best = v;
                    idx = j;
                }
                k += 1;
            }
            unsafe { *out = idx };
        }
    }
}
