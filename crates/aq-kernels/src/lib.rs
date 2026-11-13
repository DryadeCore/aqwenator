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
    /// k_norm + partial RoPE into `k_cache[pos, Hkv, D]`, and copy v into `v_cache`.
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
        start_pos: u32,
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
        let pos = start_pos as usize + t;

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
            let off = (pos * hkv + kv) * dim + d;
            unsafe { st(k_cache, off, y) };
            let v = unsafe { *qkv.add(row + h * 2 * dim + hkv * dim + kv * dim + d) };
            unsafe { *v_cache.add(off) = v };
        }
    }

    /// Causal GQA attention over a contiguous KV cache, with the output gate fused:
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
        start_pos: u32,
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
        let n_keys = start_pos as usize + t + 1;

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
            let kb = (j * hkv + kv) * D + lane * PER;
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
    /// (row stride `ld_x`), carrying the last 3 inputs per channel in `state[c * 3 + j]`.
    /// grid = (ceil(conv_dim / 256)), block = (256).
    #[kernel]
    pub unsafe fn conv1d_silu(
        x: *const u16,
        ld_x: u32,
        state: *mut u16,
        w: *const u16,
        out: *mut u16,
        n_tokens: u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if c >= conv_dim as usize {
            return;
        }
        let cd = conv_dim as usize;
        let w0 = unsafe { ld(w, c * 4) };
        let w1 = unsafe { ld(w, c * 4 + 1) };
        let w2 = unsafe { ld(w, c * 4 + 2) };
        let w3 = unsafe { ld(w, c * 4 + 3) };
        let mut s0 = unsafe { ld(state, c * 3) };
        let mut s1 = unsafe { ld(state, c * 3 + 1) };
        let mut s2 = unsafe { ld(state, c * 3 + 2) };
        let mut t = 0usize;
        while t < n_tokens as usize {
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
    /// One block per value head `h` (grid = (Hv)), block = (512): thread `(r, c)` with
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
        state: *mut f32,
        out: *mut u16,
        n_tokens: u32,
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

        let mut t = 0usize;
        while t < n_tokens as usize {
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

    /// Argmax over `n` f32 values; one block of 1024 threads. Ties resolve to the lowest index.
    #[kernel]
    pub unsafe fn argmax(x: *const f32, n: u32, out: *mut u32) {
        static mut BV: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut BI: SharedArray<u32, 32> = SharedArray::UNINIT;
        let bv = (&raw mut BV).cast::<f32>();
        let bi = (&raw mut BI).cast::<u32>();
        let tid = thread::threadIdx_x();
        let mut best = -3.0e38f32;
        let mut idx = u32::MAX;
        let mut i = tid;
        while i < n {
            let v = unsafe { *x.add(i as usize) };
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
