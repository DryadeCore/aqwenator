//! Correctness check against transformers reference data (tools/golden.py).
//!
//!     aq-check <model id or dir> <golden.safetensors>
//!
//! 1. Per-layer residual stream after prefill (when the golden has `hidden.*`).
//! 2. Teacher-forced logits: prefill the prompt, then feed the reference's greedy tokens one at
//!    a time through the decode path and compare every logits row with the reference. A top-1
//!    disagreement counts as a near-tie when the reference's own top-2 margin is below 0.25
//!    (one bf16 ulp at typical logit magnitudes is ~0.06-0.125); anything else is a real miss.
//! 3. Free-running greedy tokens, single sequence and 3 sequences batched through the scheduler.
//!
//! Exit status is non-zero on any real miss.

use anyhow::{Context, Result};
use aq_engine::engine::{Engine, Request};
use aq_engine::model::{Limits, Model};
use aq_engine::weights::resolve_model;
use std::time::Instant;

const NEAR_TIE: f32 = 0.25;

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

fn top2(x: &[f32]) -> (usize, f32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in x.iter().enumerate() {
        if v > v1 {
            v2 = v1;
            v1 = v;
            i1 = i;
        } else if v > v2 {
            v2 = v;
        }
    }
    (i1, v1, v2)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_spec = args.get(1).context("usage: aq-check <model> <golden.safetensors>")?;
    let golden_path = args.get(2).context("usage: aq-check <model> <golden.safetensors>")?;
    let dir = resolve_model(model_spec)?;

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
    let has_hidden = g.names().iter().any(|n| n.starts_with("hidden."));
    let has_tf = g.names().contains(&"tf_logits");

    let ctx = (ids.len() + greedy_ref.len() + 64).next_multiple_of(256);
    let t0 = Instant::now();
    let mut m = Model::load(&dir, Limits { max_tokens: 8192, max_seqs: 4, max_ctx: ctx, block_size: 16, kv_budget_bytes: 0, kv_blocks: 0 })?;
    println!(
        "load: {:.2}s, {:.2} GiB on device; prompt {} tokens, {} reference tokens",
        t0.elapsed().as_secs_f64(),
        m.load.weight_bytes as f64 / (1u64 << 30) as f64,
        ids.len(),
        greedy_ref.len()
    );
    let n_layers = m.cfg.num_hidden_layers;
    let vocab = m.cfg.vocab_size;
    let mut failed = false;

    // ---- 1. prefill (chunked by the step budget) + per-layer comparison
    m.reset_slot(0)?;
    let mut dump = Vec::new();
    let mut pos = 0;
    let mut first = 0;
    while pos < ids.len() {
        let n = (ids.len() - pos).min(8192);
        let want_dump = has_hidden && n == ids.len();
        first = m.step(&m.single(&ids[pos..pos + n], pos), if want_dump { Some(&mut dump) } else { None })?[0];
        pos += n;
    }
    if has_hidden && !dump.is_empty() {
        let mut worst = (0f64, 0usize);
        for (l, ours) in dump.iter().enumerate().take(n_layers - 1) {
            let (rel, _) = rel_err(ours, &as_f32(&format!("hidden.{}", l + 1))?);
            if rel > worst.0 {
                worst = (rel, l);
            }
        }
        println!("layers: worst rel_l2 {:.2e} (layer {})", worst.0, worst.1);
    }
    let lg = m.logits()?;
    let (rel, _) = rel_err(&lg, &as_f32("logits_last")?);
    println!("prefill logits rel_l2 {rel:.2e}");

    // ---- 2. teacher-forced decode
    let tf = if has_tf { as_f32("tf_logits")? } else { Vec::new() };
    // Reference margin (top1 - top2) at generated position k, and its own top-1.
    let ref_at = |k: usize| -> Option<(usize, f32)> {
        (has_tf && k < greedy_ref.len()).then(|| {
            let (i, a, b) = top2(&tf[k * vocab..(k + 1) * vocab]);
            (i, a - b)
        })
    };
    if has_tf {
        let inconsistent: Vec<usize> =
            (0..greedy_ref.len()).filter(|&k| ref_at(k).unwrap().0 != greedy_ref[k] as usize).collect();
        if !inconsistent.is_empty() {
            let k = inconsistent[0];
            println!(
                "reference self-check: transformers generate() and its own full forward disagree at {} positions (first k={k}, margin {:.3}); free-running comparisons after that point are against generate()",
                inconsistent.len(),
                ref_at(k).unwrap().1
            );
        }
    }
    if has_tf {
        let (mut agree, mut ties, mut misses, mut worst_rel) = (0, 0, 0, 0f64);
        let mut pos = ids.len();
        for k in 0..greedy_ref.len() {
            if k > 0 {
                m.step(&m.single(&[greedy_ref[k - 1]], pos), None)?;
                pos += 1;
            }
            let ours = if k == 0 { lg.clone() } else { m.logits()? };
            let refr = &tf[k * vocab..(k + 1) * vocab];
            let (rel, _) = rel_err(&ours, refr);
            worst_rel = worst_rel.max(rel);
            let (ro, _, _) = top2(&ours);
            let (rr, r1, r2) = top2(refr);
            if ro == rr {
                agree += 1;
            } else if r1 - r2 < NEAR_TIE {
                ties += 1;
            } else {
                misses += 1;
                println!("  MISS at k={k}: ours {ro} ref {rr} (ref margin {:.3})", r1 - r2);
            }
        }
        println!(
            "teacher-forced: top-1 {agree}/{n} agree, {ties} near-ties, {misses} real misses; worst logits rel_l2 {worst_rel:.2e}",
            n = greedy_ref.len()
        );
        failed |= misses > 0;
    }

    // ---- 3. free-running greedy: single, then 3 sequences batched. Teacher forcing advanced
    // slot 0's recurrent state, so start over from a fresh prefill.
    m.reset_slot(0)?;
    let mut pos = 0;
    while pos < ids.len() {
        let n = (ids.len() - pos).min(8192);
        first = m.step(&m.single(&ids[pos..pos + n], pos), None)?[0];
        pos += n;
    }
    let mut out = vec![first];
    while out.len() < greedy_ref.len() {
        let next = m.step(&m.single(&[*out.last().unwrap()], pos), None)?[0];
        pos += 1;
        out.push(next);
    }
    // A divergence from a sequence that matched up to k is acceptable only at a near-tie of the
    // reference (teacher-forced margin at k below NEAR_TIE).
    let classify = |name: &str, seq: &[u32], against: &[u32]| -> bool {
        match seq.iter().zip(against).position(|(a, b)| a != b) {
            None => {
                println!("{name}: {}/{} identical", seq.len().min(against.len()), against.len());
                true
            }
            Some(k) => match ref_at(k) {
                Some((_, margin)) if margin < NEAR_TIE => {
                    println!("{name}: identical for {k}/{} tokens, then diverges at a near-tie (ref margin {margin:.3}): ok", against.len());
                    true
                }
                Some((_, margin)) => {
                    println!("{name}: diverges at k={k} where the reference margin is {margin:.3}: REAL MISMATCH");
                    false
                }
                None => {
                    println!("{name}: diverges at k={k} (no reference logits to classify): FAIL");
                    false
                }
            },
        }
    };
    failed |= !classify("greedy vs reference", &out, &greedy_ref);
    let single = out.clone();

    let mut eng = Engine::new(m, vec![]);
    for id in 0..3u64 {
        eng.add(Request { id, prompt: ids.clone(), max_new_tokens: greedy_ref.len(), ignore_eos: true })?;
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); 3];
    while eng.has_work() {
        for e in eng.step()? {
            got[e.id as usize].push(e.token);
        }
    }
    for (i, gseq) in got.iter().enumerate() {
        failed |= !classify(&format!("batched seq {i} vs single"), gseq, &single);
    }

    // ---- 4. session cache: a request that extends a finished one must resume from the parked
    // state and produce the same tokens as a cold engine (session cache off) on the same prompt.
    let turn2: Vec<u32> = ids.iter().copied().chain(single[..8].iter().copied()).chain(ids[..ids.len().min(37)].iter().copied()).collect();
    let mut run_turns = |eng: &mut Engine, cached: bool| -> Result<(Vec<u32>, u64)> {
        eng.session_cache = cached;
        let mut out = Vec::new();
        for (id, prompt, n) in [(100u64, ids.clone(), 9usize), (101, turn2.clone(), 16)] {
            eng.add(Request { id, prompt, max_new_tokens: n, ignore_eos: true })?;
            while eng.has_work() {
                for e in eng.step()? {
                    if e.id == 101 {
                        out.push(e.token);
                    }
                }
            }
        }
        Ok((out, eng.cache.reused_tokens))
    };
    let (warm, reused) = run_turns(&mut eng, true)?;
    let (cold, _) = run_turns(&mut eng, false)?;
    let same = warm.iter().zip(&cold).take_while(|(a, b)| a == b).count();
    println!("session cache: turn 2 resumed from {reused} cached tokens; {same}/{} tokens identical to a cold prefill", cold.len());
    failed |= reused == 0 || same != cold.len();

    if failed {
        anyhow::bail!("correctness check FAILED");
    }
    println!("PASS");
    Ok(())
}
