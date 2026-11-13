//! Correctness check against transformers reference activations (tools/golden.py).
//!
//!     aq-check <model id or dir> <golden.safetensors>
//!
//! Compares the residual stream after every layer, the final logits, and 32 greedy tokens.

use anyhow::{Context, Result};
use aq_engine::model::{Limits, Model};
use aq_engine::weights::resolve_model;
use std::time::Instant;

fn rel_err(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
        max = max.max(d.abs());
    }
    ((num / den.max(1e-30)).sqrt(), max)
}

fn top_k(x: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
    idx.truncate(k);
    idx
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_spec = args.get(1).context("usage: aq-check <model> <golden.safetensors>")?;
    let golden_path = args.get(2).context("usage: aq-check <model> <golden.safetensors>")?;
    let dir = resolve_model(model_spec)?;

    let t0 = Instant::now();
    let mut m = Model::load(&dir, Limits { max_tokens: 2048, max_seqs: 4, max_ctx: 4096, block_size: 16 })?;
    println!(
        "load: {:.2}s total (open {:.3}s, upload {:.2}s), {:.2} GiB on device",
        t0.elapsed().as_secs_f64(),
        m.load.open_s,
        m.load.upload_s,
        m.load.weight_bytes as f64 / (1u64 << 30) as f64
    );
    println!("state per sequence: {:.1} MiB", m.cfg.state_bytes_per_seq() as f64 / (1 << 20) as f64);

    let gbytes = std::fs::read(golden_path)?;
    let g = safetensors::SafeTensors::deserialize(&gbytes)?;
    let as_f32 = |name: &str| -> Result<Vec<f32>> {
        let v = g.tensor(name)?;
        Ok(v.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let as_u32 = |name: &str| -> Result<Vec<u32>> {
        let v = g.tensor(name)?;
        Ok(v.data().chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let ids = as_u32("input_ids")?;
    let greedy_ref = as_u32("greedy_ids")?;
    let n_layers = m.cfg.num_hidden_layers;

    m.reset_slot(0)?;
    let mut dump = Vec::new();
    let t1 = Instant::now();
    let first = m.step(&m.single(&ids, 0), Some(&mut dump))?[0];
    println!("prefill {} tokens (with dumps): {:.1} ms", ids.len(), t1.elapsed().as_secs_f64() * 1e3);

    let mut worst = 0f64;
    for (l, ours) in dump.iter().enumerate() {
        let name = format!("hidden.{}", l + 1);
        // HF's last hidden_states entry is post final norm; compare it separately.
        if l + 1 == n_layers {
            break;
        }
        let refv = as_f32(&name)?;
        let (rel, max) = rel_err(ours, &refv);
        worst = worst.max(rel);
        println!("layer {l:3} ({:?}) rel_l2 {rel:.2e} max_abs {max:.3e}", m.layers[l].kind);
    }
    let last_ref = as_f32(&format!("hidden.{n_layers}"))?;
    let (rel_pre, _) = rel_err(&dump[n_layers - 1], &last_ref);
    let n = ids.len();
    let hd = m.cfg.hidden_size;
    let (rel_post, _) = rel_err(&m.final_hidden()?, &last_ref[(n - 1) * hd..n * hd]);
    println!("last hidden: rel_l2 vs pre-norm {rel_pre:.2e}, vs post-norm {rel_post:.2e}");

    let logits = m.logits()?;
    let ref_logits = as_f32("logits_last")?;
    let (rel, max) = rel_err(&logits, &ref_logits);
    let (ours5, ref5) = (top_k(&logits, 5), top_k(&ref_logits, 5));
    println!("logits rel_l2 {rel:.2e} max_abs {max:.3e}; top5 ours {ours5:?} ref {ref5:?}");

    // Greedy decode, one token per step.
    let mut out = vec![first];
    let mut pos = ids.len();
    let t2 = Instant::now();
    while out.len() < greedy_ref.len() {
        let next = m.step(&m.single(&[*out.last().unwrap()], pos), None)?[0];
        pos += 1;
        out.push(next);
    }
    let dt = t2.elapsed().as_secs_f64();
    let matched = out.iter().zip(&greedy_ref).take_while(|(a, b)| a == b).count();
    println!(
        "greedy: {matched}/{} tokens match; decode {:.1} tok/s (single sequence, M1 kernels)",
        greedy_ref.len(),
        (out.len() - 1) as f64 / dt
    );
    println!("ours: {out:?}");
    println!("ref:  {greedy_ref:?}");
    println!("worst layer rel_l2 {worst:.2e}");

    // Batched consistency: the same prompt in 3 slots at once, plus a decode, through the
    // scheduler, must produce the same greedy tokens as the single-sequence path.
    let mut eng = aq_engine::engine::Engine::new(m, vec![]);
    for id in 0..3u64 {
        eng.add(aq_engine::engine::Request { id, prompt: ids.clone(), max_new_tokens: greedy_ref.len(), ignore_eos: true })?;
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); 3];
    while eng.has_work() {
        for e in eng.step()? {
            got[e.id as usize].push(e.token);
        }
    }
    for (i, g) in got.iter().enumerate() {
        let ok = g.iter().zip(&greedy_ref).take_while(|(a, b)| a == b).count();
        println!("batched seq {i}: {ok}/{} greedy tokens match", greedy_ref.len());
    }
    Ok(())
}
