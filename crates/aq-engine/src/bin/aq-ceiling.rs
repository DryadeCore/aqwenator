//! Hardware ceilings: what the device can do at most, and what that means for one engine step.
//!
//!     aq-ceiling [<model>] [--gb 2] [--distinct] [--out ceiling.json]
//!
//! Measures, on the visible GPU:
//! - memory bandwidth: read (16-byte loads), copy, and memset write, over `--gb` GiB;
//! - peak bf16 GEMM rate (cuBLASLt, f32 accumulate), square problems;
//! - kernel launch overhead (enqueue cost and launch + sync round trip).
//!
//! With a model, it lists the GEMMs of one forward step from the config (no weights are
//! loaded) and, for each row count M, prints two lower bounds on the step's GEMM time --
//! every weight byte read once at the measured read bandwidth, and every multiply-add at the
//! measured peak rate -- next to what cuBLASLt actually takes for the same sequence of GEMMs.
//! By default every GEMM of a shape reuses one weight buffer; `--distinct` gives each GEMM of
//! the step its own (the model's full weight footprint), as the engine has.
//! `AQ_PROFILE=decode aq-bench ...` gives the engine's real step time to set against these.

use anyhow::{Context, Result};
use aq_engine::config::{LayerKind, TextConfig};
use aq_engine::cublaslt::{CUDA_R_8F_E4M3, CUDA_R_8I, CUDA_R_16BF, CUDA_R_32I};
use aq_engine::gpu::{Gpu, cfg, de};
use aq_engine::weights::resolve_model;
use serde_json::json;
use std::time::Instant;

/// Best wall time of `reps` runs of `f` followed by a stream sync, in seconds.
fn best(g: &mut Gpu, reps: usize, mut f: impl FnMut(&mut Gpu) -> Result<()>) -> Result<f64> {
    let mut b = f64::INFINITY;
    for _ in 0..reps {
        g.sync()?;
        let t = Instant::now();
        f(g)?;
        g.sync()?;
        b = b.min(t.elapsed().as_secs_f64());
    }
    Ok(b)
}

struct Gemm {
    name: &'static str,
    n: usize,
    k: usize,
    /// How many times per step.
    count: usize,
    /// Row cap (the output head only runs on rows that need logits).
    max_m: usize,
}

fn main() -> Result<()> {
    let mut it = std::env::args().skip(1);
    let mut model = None;
    let mut gb = 2usize;
    let mut out = None;
    let mut distinct = false;
    let mut i8chunk = false;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--distinct" => distinct = true,
            "--i8chunk" => i8chunk = true,
            "--gb" => gb = it.next().context("--gb N")?.parse()?,
            "--out" => out = it.next(),
            _ => model = Some(a),
        }
    }
    let mut g = Gpu::new(0)?;
    let sms = g.num_sms()?;
    println!("device: {sms} SMs");

    // ---- memory bandwidth
    let bytes = gb << 30;
    let src = g.alloc(bytes)?;
    let dst = g.alloc(bytes)?;
    g.sync()?;
    let gbs = |t: f64, moved: usize| moved as f64 / t / 1e9;
    let mut read_best = 0f64;
    let mut copy_best = 0f64;
    let mut bw_rows = Vec::new();
    for per in [1024usize, 16384, 262144] {
        let blocks = bytes / (per * 16);
        let outp = g.alloc(blocks * 256 * 4)?;
        let st = g.stream.clone();
        let tr = best(&mut g, 5, |g| unsafe {
            g.k.hw_read(&st, cfg((blocks as u32, 1, 1), 256), src as *const u32, per as u32, outp as *mut u32).map_err(de)
        })?;
        let tc = best(&mut g, 5, |g| unsafe {
            g.k.hw_copy(&st, cfg((blocks as u32, 1, 1), 256), src as *const u32, dst as *mut u32, per as u32).map_err(de)
        })?;
        let (r, c) = (gbs(tr, bytes), gbs(tc, 2 * bytes));
        println!("bandwidth, {:>4} KiB per block: read {r:7.1} GB/s   copy {c:7.1} GB/s (read + write bytes)", per * 16 / 1024);
        read_best = read_best.max(r);
        copy_best = copy_best.max(c);
        bw_rows.push(json!({"block_kib": per * 16 / 1024, "read_gb_s": r, "copy_gb_s": c}));
    }
    let tw = best(&mut g, 5, |g| g.zero(dst, bytes))?;
    let write = gbs(tw, bytes);
    println!("bandwidth, memset: write {write:7.1} GB/s");
    println!("=> read bandwidth ceiling {read_best:.1} GB/s");

    // ---- launch overhead
    let st = g.stream.clone();
    const N: usize = 2000;
    g.sync()?;
    let t = Instant::now();
    for _ in 0..N {
        unsafe { g.k.hw_noop(&st, cfg((1, 1, 1), 32), src as *const u32) }.map_err(de)?;
    }
    let enq = t.elapsed().as_secs_f64() / N as f64 * 1e6;
    g.sync()?;
    let thru = t.elapsed().as_secs_f64() / N as f64 * 1e6;
    let rt = best(&mut g, 200, |g| unsafe { g.k.hw_noop(&st, cfg((1, 1, 1), 32), src as *const u32) }.map_err(de))? * 1e6;
    println!("launch: {enq:.2} us to enqueue, {thru:.2} us per empty kernel back to back, {rt:.1} us launch + sync round trip");

    // ---- peak GEMM rate
    let mut peak = 0f64;
    let mut sq_rows = Vec::new();
    for n in [2048usize, 4096, 8192] {
        if n * n * 2 > bytes {
            continue;
        }
        let stream = g.raw_stream();
        g.lt.gemm(stream, n, n, n, src, dst, src + (bytes / 2) as u64, CUDA_R_16BF)?;
        let t = best(&mut g, 5, |g| g.lt.gemm(stream, n, n, n, src, dst, src + (bytes / 2) as u64, CUDA_R_16BF))?;
        let tf = 2.0 * (n as f64).powi(3) / t / 1e12;
        println!("gemm bf16 {n}^3: {:8.3} ms, {tf:7.1} TFLOP/s", t * 1e3);
        peak = peak.max(tf);
        sq_rows.push(json!({"n": n, "ms": t * 1e3, "tflops": tf}));
        if t > 0.05 {
            break;
        }
    }

    // ---- int8 weights expanded to bf16 in chunks of rows, each chunk multiplied right away
    // (`--i8chunk`): which chunk size is fastest?
    if i8chunk {
        use aq_engine::gpu::Fmt;
        let (n, k) = (24576usize, 4096usize);
        let wb = g.alloc(n * k * 2)?;
        g.sync()?;
        let w8 = g.quantize_weight(wb, n, k, Fmt::Int8, true)?;
        let tmp = g.alloc(n * k * 2)?;
        let x = g.alloc(512 * k * 2)?;
        let y = g.alloc(512 * n * 2)?;
        g.sync()?;
        let stream = g.raw_stream();
        let st = g.stream.clone();
        for m in [16usize, 64, 128] {
            g.lt.gemm(stream, m, n, k, x, wb, y, CUDA_R_16BF)?;
            let t16 = best(&mut g, 5, |g| g.lt.gemm(stream, m, n, k, x, wb, y, CUDA_R_16BF))?;
            println!("i8chunk M={m}: bf16 GEMM {:.3} ms ({:.0} GB/s of weights)", t16 * 1e3, (n * k * 2) as f64 / t16 / 1e9);
            for nc in [n, 8192, 4096, 2048, 1024, 512, 256, 128] {
                g.lt.gemm(stream, m, nc, k, x, tmp, y, CUDA_R_16BF)?;
                let t = best(&mut g, 5, |g| {
                    for c in (0..n).step_by(nc) {
                        unsafe {
                            g.k.dequant_i8b(&st, cfg((((nc * k / 32) as u32).div_ceil(256), 1, 1), 256), (w8.p + (c * k) as u64) as *const u8, (w8.scale + (c * k / 32 * 2) as u64) as *const u16, tmp as *mut u16, (nc * k / 32) as u32)
                        }
                        .map_err(de)?;
                        g.lt.gemm(stream, m, nc, k, x, tmp, y, CUDA_R_16BF)?;
                    }
                    Ok(())
                })?;
                let td = best(&mut g, 5, |g| {
                    for c in (0..n).step_by(nc) {
                        unsafe {
                            g.k.dequant_i8b(&st, cfg((((nc * k / 32) as u32).div_ceil(256), 1, 1), 256), (w8.p + (c * k) as u64) as *const u8, (w8.scale + (c * k / 32 * 2) as u64) as *const u16, tmp as *mut u16, (nc * k / 32) as u32)
                        }
                        .map_err(de)?;
                    }
                    Ok(())
                })?;
                println!(
                    "  chunk {nc:>5} rows ({:>6.1} MiB bf16): expand + GEMM {:.3} ms, expand alone {:.3} ms, {:.2}x the bf16 GEMM",
                    (nc * k * 2) as f64 / (1 << 20) as f64,
                    t * 1e3,
                    td * 1e3,
                    t / t16
                );
            }
        }
    }

    // ---- one model step
    let mut step_rows = Vec::new();
    let mut model_info = json!(null);
    if let Some(spec) = model {
        let dir = resolve_model(&spec)?;
        let c = TextConfig::load(&dir)?;
        let n_lin = (0..c.num_hidden_layers).filter(|&l| c.kind(l) == LayerKind::Linear).count();
        let n_full = c.num_hidden_layers - n_lin;
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let gemms = [
            Gemm { name: "gdn_proj", n: c.gdn_proj_ld(), k: h, count: n_lin, max_m: usize::MAX },
            Gemm { name: "gdn_out", n: h, k: c.value_dim(), count: n_lin, max_m: usize::MAX },
            Gemm { name: "attn_qkv", n: c.attn_proj_dim(), k: h, count: n_full, max_m: usize::MAX },
            Gemm { name: "attn_o", n: h, k: c.num_attention_heads * c.head_dim, count: n_full, max_m: usize::MAX },
            Gemm { name: "mlp_gate_up", n: 2 * inter, k: h, count: c.num_hidden_layers, max_m: usize::MAX },
            Gemm { name: "mlp_down", n: h, k: inter, count: c.num_hidden_layers, max_m: usize::MAX },
            Gemm { name: "lm_head", n: c.vocab_size, k: h, count: 1, max_m: 512 },
        ];
        let wbytes: usize = gemms.iter().map(|x| x.n * x.k * 2 * x.count).sum();
        let params: usize = gemms.iter().map(|x| x.n * x.k * x.count).sum();
        let kv_per_tok = n_full * 2 * c.num_key_value_heads * c.head_dim * 2;
        println!(
            "\nmodel {spec}: {} layers ({n_lin} linear, {n_full} full), hidden {h}; GEMM weights {:.2} GB ({:.2} B params); attention KV {} KiB per context token",
            c.num_hidden_layers,
            wbytes as f64 / 1e9,
            params as f64 / 1e9,
            kv_per_tok / 1024
        );
        let mem_ms = wbytes as f64 / (read_best * 1e9) * 1e3;
        println!("weight-read bound: {mem_ms:.2} ms per step at {read_best:.0} GB/s (any M)");
        println!(
            "attention KV read, 16 sequences x 20k context: {:.1} GB per step = {:.1} ms at that bandwidth",
            16.0 * 20000.0 * kv_per_tok as f64 / 1e9,
            16.0 * 20000.0 * kv_per_tok as f64 / (read_best * 1e9) * 1e3
        );
        let wmax = gemms.iter().map(|x| x.n * x.k * 2).max().unwrap();
        let w = g.alloc(wmax)?;
        // Weight buffer of each GEMM of the step, in the order the step runs them.
        let mut ws: Vec<Vec<u64>> = Vec::new();
        for q in &gemms {
            let mut v = Vec::new();
            for _ in 0..q.count {
                v.push(if distinct { g.alloc(q.n * q.k * 2)? } else { w });
            }
            ws.push(v);
        }
        const MS: [usize; 11] = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let kmax = gemms.iter().map(|x| x.k).max().unwrap();
        let x = g.alloc(4096 * kmax * 2)?;
        let ymax = gemms.iter().map(|x| x.n * x.max_m.min(4096) * 4).max().unwrap();
        let y = g.alloc(ymax)?;
        g.zero(x, 4096 * kmax * 2)?;
        g.zero(w, wmax)?;
        g.sync()?;
        println!("\n{:>5} {:>10} {:>10} {:>10} {:>9} {:>11} {:>11}   per-GEMM ms (x count)", "M", "flop ms", "mem ms", "cublas ms", "of bound", "rows/s", "ceil rows/s");
        for m in MS {
            let stream = g.raw_stream();
            let mut flops = 0f64;
            let mut parts = String::new();
            let mut per = Vec::new();
            for q in &gemms {
                let mm = m.min(q.max_m);
                flops += 2.0 * mm as f64 * q.n as f64 * q.k as f64 * q.count as f64;
                g.lt.gemm(stream, mm, q.n, q.k, x, w, y, CUDA_R_16BF)?;
                let t = best(&mut g, 5, |g| {
                    for _ in 0..4 {
                        g.lt.gemm(stream, mm, q.n, q.k, x, w, y, CUDA_R_16BF)?;
                    }
                    Ok(())
                })? / 4.0;
                parts += &format!(" {}={:.3}", q.name, t * 1e3);
                per.push(json!({"name": q.name, "ms": t * 1e3, "count": q.count, "gflop_s": 2.0 * mm as f64 * q.n as f64 * q.k as f64 / t / 1e9, "gb_s": (q.n * q.k * 2) as f64 / t / 1e9}));
            }
            // The whole step's GEMMs back to back, one sync.
            let t = best(&mut g, 3, |g| {
                for (q, wq) in gemms.iter().zip(&ws) {
                    for &wb in wq {
                        g.lt.gemm(stream, m.min(q.max_m), q.n, q.k, x, wb, y, CUDA_R_16BF)?;
                    }
                }
                Ok(())
            })?;
            // The same step with 8-bit operands: E4M3 -> bf16, and int8 -> i32.
            let mut alt = String::new();
            for (label, tys) in [("fp8", (CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF)), ("int8", (CUDA_R_8I, CUDA_R_8I, CUDA_R_32I))] {
                let ok = gemms.iter().all(|q| g.lt.gemm_ty(stream, m.min(q.max_m), q.n, q.k, x, w, y, tys).is_ok());
                g.sync()?;
                if !ok {
                    alt += &format!(" {label}=unsupported");
                    continue;
                }
                let t8 = best(&mut g, 3, |g| {
                    for (q, wq) in gemms.iter().zip(&ws) {
                        for &wb in wq {
                            g.lt.gemm_ty(stream, m.min(q.max_m), q.n, q.k, x, wb, y, tys)?;
                        }
                    }
                    Ok(())
                })?;
                alt += &format!(" {label}={:.3}ms", t8 * 1e3);
                for q in &gemms {
                    let mm = m.min(q.max_m);
                    let tq = best(&mut g, 5, |g| g.lt.gemm_ty(stream, mm, q.n, q.k, x, w, y, tys))?;
                    alt += &format!(" {}={:.3}", q.name, tq * 1e3);
                }
                step_rows.push(json!({"m": m, "types": label, "cublas_ms": t8 * 1e3}));
            }
            println!("{m:>5} 8-bit step:{alt}");
            let flop_ms = flops / (peak * 1e12) * 1e3;
            let bound = flop_ms.max(mem_ms);
            println!(
                "{m:>5} {flop_ms:>10.3} {mem_ms:>10.3} {:>10.3} {:>8.0}% {:>11.0} {:>11.0}  {parts}",
                t * 1e3,
                100.0 * bound / (t * 1e3),
                m as f64 / t,
                m as f64 / (bound / 1e3)
            );
            step_rows.push(json!({"m": m, "flop_ms": flop_ms, "mem_ms": mem_ms, "cublas_ms": t * 1e3, "gemms": per}));
        }
        model_info = json!({"model": spec, "gemm_weight_bytes": wbytes, "kv_bytes_per_ctx_token": kv_per_tok, "layers": c.num_hidden_layers, "full_layers": n_full});
    }
    if let Some(path) = out {
        let doc = json!({
            "sms": sms, "bandwidth": bw_rows, "read_gb_s": read_best, "copy_gb_s": copy_best, "write_gb_s": write,
            "launch_enqueue_us": enq, "launch_back_to_back_us": thru, "launch_sync_us": rt,
            "square_gemm": sq_rows, "peak_tflops": peak, "model": model_info, "step": step_rows,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc)?)?;
        println!("wrote {path}");
    }
    Ok(())
}
