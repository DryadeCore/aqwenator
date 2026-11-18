use super::*;

/// Prefills of at most this many tokens are not held back by the mixed-step prefill budget.
pub const SMALL_PREFILL: usize = 256;

impl Engine {
    pub(super) fn ensure_blocks(&mut self, si: usize, upto: usize) -> Result<()> {
        let bs = self.limits().block_size;
        while self.running[si].blocks.len() * bs < upto {
            let Some(b) = self.pool.take_page() else { bail!("KV page pool exhausted") };
            self.running[si].blocks.push(b);
        }
        Ok(())
    }

    /// Build the batch for `plan` (running index, tokens to feed). Returns the batch and the
    /// running indices that sample a token from it, in sample-row order.
    pub(super) fn build_batch(&mut self, plan: &[(usize, usize)]) -> Result<(Batch, Vec<usize>)> {
        let mb = self.limits().max_blocks_per_seq();
        let mut b = Batch::default();
        let mut sampled = Vec::new();
        for (k, &(i, n)) in plan.iter().enumerate() {
            let upto = self.running[i].computed + n;
            self.ensure_blocks(i, upto)?;
            let s = &self.running[i];
            b.seq_start.push(b.tokens.len() as u32);
            b.seq_len.push(n as u32);
            b.seq_slot.push(s.slot);
            for p in s.computed..upto {
                b.tokens.push(s.token_at(p));
                b.tok_pos.push(p as u32);
                b.tok_seq.push(k as u32);
            }
            let mut row = s.blocks.clone();
            row.resize(mb, 0);
            b.block_tables.extend_from_slice(&row);
            b.next_tok_last.push(if upto < s.known() { s.token_at(upto) } else { u32::MAX });
            if upto == s.known() {
                b.sample_rows.push(b.tokens.len() as u32 - 1);
                sampled.push(i);
            }
        }
        Ok((b, sampled))
    }

    /// Append sampled tokens, emit events; returns running indices that finished.
    pub(super) fn apply_sampled(&mut self, sampled: &[usize], toks: &[u32], events: &mut Vec<TokenEvent>) -> Vec<usize> {
        let mut done = Vec::new();
        for (&i, &tok) in sampled.iter().zip(toks) {
            let s = &mut self.running[i];
            s.out.push(tok);
            if self.trace && s.out.len() == 1 {
                eprintln!(
                    "[req] id {} prompt {} reused {} wait_ms {:.1} ttft_ms {:.1} steps {}",
                    s.req.id,
                    s.req.prompt.len(),
                    s.reused,
                    s.t_admit.duration_since(s.t_add).as_secs_f64() * 1e3,
                    s.t_add.elapsed().as_secs_f64() * 1e3,
                    s.steps
                );
            }
            let stop = !s.req.ignore_eos && self.eos.contains(&tok);
            let finished = stop || s.out.len() >= s.req.max_new_tokens;
            events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop });
            if finished {
                done.push(i);
            }
        }
        done
    }

    /// Run one step. Returns one event per sampled token.
    ///
    /// Co-execution and speculative decoding do not combine: with `spec_k > 0` the engine
    /// takes the normal (single-stream, lane 0) path, after draining any prefill batch still
    /// in flight from an earlier co-execution step.
    pub fn step(&mut self) -> Result<Vec<TokenEvent>> {
        if self.coexec {
            if self.spec_k == 0 {
                return self.step_coexec();
            }
            if self.inflight.is_some() {
                let mut events = Vec::new();
                self.collect_inflight(&mut events)?;
                if !events.is_empty() {
                    return Ok(events);
                }
            }
        }
        let t0 = Instant::now();
        let lim = self.limits();
        let mut budget = lim.max_tokens;

        // Admit waiting requests while there is room to run them.
        self.clock += 1;
        self.admit()?;

        // Background prefill: collect a finished batch, start the next long prompt.
        let mut bg_events = Vec::new();
        if self.bg_prefill > 0 {
            if self.inflight.as_ref().is_some_and(|inf| inf.lane == 0) || (self.inflight.is_some() && self.model.lane_done(1)?) {
                self.collect_inflight(&mut bg_events)?;
            }
            if self.inflight.is_none() {
                self.launch_bg()?;
                // AQ_BG_SYNC=1 (debug): wait for the batch right away, so the two lanes never overlap.
                if self.inflight.is_some() && std::env::var("AQ_BG_SYNC").is_ok() {
                    self.collect_inflight(&mut bg_events)?;
                }
            }
        }
        let bg_min = if self.bg_prefill > 0 { self.bg_prefill } else { usize::MAX };

        // Decode-path entries first (pending tokens + optional drafts), then prefill chunks in
        // admission order.
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (running index, n tokens)
        let mut drafts: Vec<Vec<u32>> = Vec::new(); // per decode entry
        let mut draft_local: Vec<usize> = Vec::new(); // leading tokens from the prompt lookup
        let mut draft_mtp: Vec<bool> = Vec::new(); // first token backed by the MTP head
        for i in 0..self.running.len() {
            let s = &self.running[i];
            if s.computed < s.req.prompt.len() || budget == 0 {
                continue;
            }
            let pending = s.known() - s.computed;
            let rem = s.req.max_new_tokens - s.out.len();
            let mut d = Vec::new();
            let mut local = 0;
            let mut from_mtp = false;
            if self.spec_k > 0 && rem > 1 {
                let k_cur = s.k_cur.max(self.spec_k);
                let kmax = k_cur.min(rem - 1).min(MAX_DECODE_ROWS_PER_SEQ.saturating_sub(pending));
                if kmax > 0 {
                    (d, local, from_mtp) = self.draft(i, kmax);
                }
            }
            if !d.is_empty() && self.running[i].spare.is_none() {
                // A spare slot for the speculative state: a free one, else the least-recently
                // used parked session's (never a shared-prefix snapshot's).
                let slot = match self.pool.slots.pop() {
                    Some(x) => Some(x),
                    None => {
                        if self.evict_lru_parked() { self.pool.slots.pop() } else { None }
                    }
                };
                self.running[i].spare = slot;
            }
            if self.running[i].spare.is_none() {
                d.clear();
            }
            // A decode entry costs the step budget what plain decoding would (one token);
            // re-fed tokens and drafts ride along in the step's row headroom
            // (`Limits::max_rows`), so speculation never changes how much prefill fits.
            let n = pending + d.len();
            plan.push((i, n));
            draft_local.push(local.min(d.len()));
            draft_mtp.push(from_mtp && !d.is_empty());
            drafts.push(d);
            budget -= 1;
        }
        let decode_seqs = plan.len();
        if decode_seqs > 0 {
            // Adaptive: while admitted prefill work is backed up past the threshold, use the
            // larger budget (throughput and TTFT under overload); otherwise the small one
            // (inter-token latency when prefill is light).
            let mixed = match self.mixed_prefill_adaptive {
                Some((hi, threshold)) => {
                    let pending: usize = self
                        .running
                        .iter()
                        .map(|s| s.req.prompt.len().saturating_sub(s.computed))
                        .sum();
                    if pending > threshold { hi } else { self.mixed_prefill_budget }
                }
                None => self.mixed_prefill_budget,
            };
            budget = budget.min(mixed);
        }
        // Prefill order: admission order, or shortest remaining prefill first (SRPT).
        // Prompts long enough for the background lane wait for it instead of taking main steps.
        let mut prefill: Vec<usize> = (0..self.running.len())
            .filter(|&i| {
                let s = &self.running[i];
                let left = s.req.prompt.len().saturating_sub(s.computed);
                left > 0 && left < bg_min && !s.in_prefill
            })
            .collect();
        // With SRPT the longest-waiting prompt would starve behind a stream of shorter ones, so
        // it goes first with a guaranteed share: half the budget, or more when the shorter
        // prompts cannot use their half.
        let mut elder: Option<(usize, usize)> = None;
        if self.policy.srpt {
            let left = |i: usize| self.running[i].req.prompt.len() - self.running[i].computed;
            prefill.sort_by_key(|&i| left(i));
            if let Some(&e) = prefill.iter().min_by_key(|&&i| self.running[i].t_add) {
                if e != prefill[0] && left(e) > SMALL_PREFILL {
                    let others: usize = prefill.iter().filter(|&&i| i != e && left(i) > SMALL_PREFILL).map(|&i| left(i)).sum();
                    elder = Some((e, (budget / 2).max(budget.saturating_sub(others))));
                    prefill.retain(|&i| i != e);
                    prefill.insert(0, e);
                }
            }
        }
        // Burst prefill: a long prompt fed 1280 tokens at a time pays the step's fixed cost
        // (and the decode rows) once per slice, and shares the slices with other prompts. Every
        // `burst_every` steps the longest-waiting prompt that needs more than the budget gets a
        // step's whole prefill allowance up to `burst_prefill` tokens instead: fewer, longer
        // steps for it, at the price of one long inter-token gap for the decoding sequences.
        let mut burst = false;
        if self.burst_prefill > 0 && decode_seqs > 0 && self.since_burst >= self.burst_every {
            let left = |i: usize| self.running[i].req.prompt.len() - self.running[i].computed;
            if let Some(&e) = prefill.iter().filter(|&&i| left(i) > budget).min_by_key(|&&i| self.running[i].t_add) {
                elder = Some((e, self.burst_prefill.max(budget)));
                prefill.retain(|&i| i != e);
                prefill.insert(0, e);
                burst = true;
            }
        }
        self.since_burst = if burst { 0 } else { self.since_burst.saturating_add(1) };
        // Checkpoints taken inside this step's prefill: (plan index, position, slot).
        let mut taps: Vec<(usize, usize, u32)> = Vec::new();
        let can_tap = self.model.supports_tap();
        // Hard cap on prefill tokens (the step's row capacity); `budget` is the soft one.
        let mut room = lim.max_tokens - decode_seqs;
        for i in prefill {
            let s = &self.running[i];
            let start = s.computed;
            let rem = s.req.prompt.len() - start;
            // Small prefills ride along whatever the mixed-step budget says: next to the
            // weights a step streams anyway they cost almost nothing, and making a short
            // turn (or the last few tokens after a checkpoint) wait behind a long prompt is
            // what the time-to-first-token tail is made of.
            let cap = match elder {
                Some((e, share)) if e == i => share,
                _ => budget,
            };
            let mut n = if rem <= SMALL_PREFILL { rem.min(room) } else { rem.min(cap).min(room) };
            if n == 0 {
                continue;
            }
            if let Some(at) = s.snapshot_at.filter(|&at| at > start) {
                n = n.min(at - start);
            }
            if let Some(cp) = s.ckpt_at.filter(|&at| at > start) {
                // The checkpoint is free when this step reaches the end of the prompt: tap
                // the state at the last 64-token chunk boundary before the backoff. Otherwise
                // stop at the page-aligned checkpoint position and take it between steps.
                let c = start + (s.req.prompt.len() - CKPT_BACKOFF - start) / GDN_TC_CHUNK * GDN_TC_CHUNK;
                if can_tap && n == rem && c > start && !self.pool.slots.is_empty() {
                    taps.push((plan.len(), c, self.pool.slots.pop().unwrap()));
                } else {
                    n = n.min(cp - start);
                }
            }
            plan.push((i, n));
            budget = budget.saturating_sub(n);
            room -= n;
        }
        if plan.is_empty() {
            // Nothing for the main lane: wait for the background batch rather than spin.
            if bg_events.is_empty() && self.inflight.is_some() {
                self.collect_inflight(&mut bg_events)?;
            }
            return Ok(bg_events);
        }

        let mb = lim.max_blocks_per_seq();
        let mut b = Batch::default();
        // Per plan entry: (first sampled row index in `toks`, number of sampled rows).
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for (k, &(i, n)) in plan.iter().enumerate() {
            let d: &[u32] = if k < decode_seqs { &drafts[k] } else { &[] };
            let real = n - d.len(); // tokens already known
            let upto = self.running[i].computed + n;
            self.ensure_blocks(i, upto)?;
            let s = &self.running[i];
            b.seq_start.push(b.tokens.len() as u32);
            b.seq_len.push(n as u32);
            b.seq_slot.push(s.slot);
            b.seq_slot_out.push(if d.is_empty() { s.slot } else { s.spare.unwrap() });
            b.seq_tap.push(match taps.iter().find(|t| t.0 == k) {
                Some(&(_, c, slot)) => (((c - s.computed) / GDN_TC_CHUNK) as u32, slot),
                None => (0, 0),
            });
            for p in s.computed..s.computed + real {
                b.tokens.push(s.token_at(p));
                b.tok_pos.push(p as u32);
                b.tok_seq.push(k as u32);
            }
            for (j, &t) in d.iter().enumerate() {
                b.tokens.push(t);
                b.tok_pos.push((s.computed + real + j) as u32);
                b.tok_seq.push(k as u32);
            }
            let mut row = s.blocks.clone();
            row.resize(mb, 0);
            b.block_tables.extend_from_slice(&row);
            b.next_tok_last.push(if upto < s.known() { s.token_at(upto) } else { u32::MAX });
            if s.computed + real == s.known() {
                // Sample at the last known token and at every draft.
                let first = b.tokens.len() - d.len() - 1;
                samples.push((b.sample_rows.len(), d.len() + 1));
                for r in first..b.tokens.len() {
                    b.sample_rows.push(r as u32);
                }
            } else {
                samples.push((0, 0));
            }
        }
        b.n_decode = decode_seqs;
        let host_us = t0.elapsed().as_micros() as u64;
        self.cache.host_us += host_us;
        self.cache.steps += 1;
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;
        if self.trace {
            // Decode rows and the KV keys they scan; prefill tokens and the keys they scan.
            let (mut rows, mut dkeys, mut ptok, mut pkeys) = (0usize, 0usize, 0usize, 0usize);
            for (k, &(i, n)) in plan.iter().enumerate() {
                let c = self.running[i].computed;
                if k < decode_seqs {
                    rows += n;
                    dkeys += n * c;
                } else {
                    ptok += n;
                    pkeys += n * (c + n / 2);
                }
            }
            eprintln!(
                "[step] dec {decode_seqs} rows {rows} dkeys {dkeys} pseqs {} ptok {ptok} pkeys {pkeys} taps {} gpu_us {gpu_us} host_us {host_us}",
                plan.len() - decode_seqs,
                taps.len()
            );
        }
        for &(i, _) in &plan {
            self.running[i].steps += 1;
        }

        // Tokens from a collected background batch come first: this step may already have
        // decoded the next token of the same sequence.
        let mut events = bg_events;
        let mut done = Vec::new();
        // (running index, sampled row) of sequences to chain MTP drafts for.
        let mut chain: Vec<(usize, usize)> = Vec::new();
        let eos = self.eos.clone();
        let mut st = self.spec;
        for (k, &(i, n)) in plan.iter().enumerate() {
            let d: &[u32] = if k < decode_seqs { &drafts[k] } else { &[] };
            let (r0, nr) = samples[k];
            let mut targets = toks[r0..r0 + nr].to_vec();
            // Accept drafts while they equal the target's greedy choice at the previous row.
            let m = d.iter().zip(&targets).take_while(|(a, b)| a == b).count();
            if self.spec_fault > 0 && m < d.len() {
                self.spec_rejects += 1;
                if self.spec_rejects % self.spec_fault == 0 {
                    targets[m] = d[m];
                }
            }
            let s = &mut self.running[i];
            // The token limit cannot fall inside a draft (see `kmax`); a stop token can.
            let ends = |t: &u32| !s.req.ignore_eos && eos.contains(t);
            let emit = if nr == 0 { 0 } else { targets[..m + 1].iter().position(ends).map_or(m + 1, |p| p + 1) };
            if k < decode_seqs && self.spec_trace {
                eprintln!(
                    "[spec] id {} out {} computed {} known {} n {n} slot {} spare {:?} local {} draft {d:?} targets {targets:?} m {m}",
                    s.req.id,
                    s.out.len(),
                    s.computed,
                    s.known(),
                    s.slot,
                    s.spare,
                    draft_local[k]
                );
            }
            // The spare slot's state is ahead of the emitted tokens: drop it like a rejection.
            let overrun = !d.is_empty() && m == d.len() && emit < m;
            if overrun {
                st.stop_in_draft += 1;
            }
            if d.is_empty() {
                s.computed += n;
            } else if m == d.len() && !overrun {
                // Every draft accepted: the spare slot holds the state after all n tokens.
                s.k_cur = if self.spec_adaptive && d.len() >= s.k_cur.max(self.spec_k) { (2 * d.len()).min(MAX_DECODE_ROWS_PER_SEQ - 1) } else { 0 };
                s.computed += n;
                let old = s.slot;
                s.slot = s.spare.unwrap();
                s.spare = Some(old);
            }
            // else: rejection. The old slot still holds the state before this step; the
            // accepted tokens become pending and are fed again next step.
            if !d.is_empty() && (m < d.len() || overrun) {
                s.k_cur = 0;
            }
            if k < decode_seqs && nr > 0 {
                st.decode_entries += 1;
                if !d.is_empty() {
                    st.spec_entries += 1;
                    st.drafted += d.len() as u64;
                    st.accepted += m as u64;
                    st.global_drafted += (d.len() - draft_local[k]) as u64;
                    st.global_accepted += m.saturating_sub(draft_local[k]) as u64;
                    if draft_mtp[k] {
                        st.mtp_drafted += 1;
                        st.mtp_accepted += (m >= 1) as u64;
                    }
                }
            }
            // The MTP head's prediction at the row whose sample is the last token emitted.
            let mtp_pred = if nr > 0 { self.model.last_mtp.get(r0 + m).copied() } else { None };
            let mut finished = false;
            for &tok in &targets[..emit] {
                s.out.push(tok);
                self.model.mtp_note([tok]);
                if self.global_draft {
                    let n = s.known() - 1;
                    let ctx: Vec<u32> = (n.saturating_sub(4)..n).map(|p| s.token_at(p)).collect();
                    self.global.observe(&ctx, tok);
                }
                if self.trace && s.out.len() == 1 {
                    eprintln!(
                        "[req] id {} prompt {} reused {} wait_ms {:.1} ttft_ms {:.1} steps {}",
                        s.req.id,
                        s.req.prompt.len(),
                        s.reused,
                        s.t_admit.duration_since(s.t_add).as_secs_f64() * 1e3,
                        s.t_add.elapsed().as_secs_f64() * 1e3,
                        s.steps
                    );
                }
                let stop = !s.req.ignore_eos && eos.contains(&tok);
                finished = stop || s.out.len() >= s.req.max_new_tokens;
                events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop });
                if k < decode_seqs {
                    st.emitted += 1;
                }
                if finished {
                    break;
                }
            }
            debug_assert!(s.computed <= s.known());
            s.mtp_next = mtp_pred.map(|t| (s.known(), vec![t]));
            if finished {
                done.push(i);
            } else if mtp_pred.is_some() {
                chain.push((i, r0 + m));
            }
        }
        self.spec = st;
        // Chained MTP drafts for the sequences that keep decoding: as many as the draft
        // budget and the KV pages they already own allow (chained rows write to the head's
        // cache at the positions the next pass overwrites).
        if self.spec_k > 1 && self.mtp_k > 1 && !chain.is_empty() {
            let bs = self.limits().block_size;
            let want = self.mtp_k.min(self.spec_k) - 1;
            let rows: Vec<(usize, u32, usize, &[u32])> = chain
                .iter()
                .map(|&(i, sample)| {
                    let s = &self.running[i];
                    let pos = s.known() - 1;
                    let rem = (s.req.max_new_tokens - s.out.len()).saturating_sub(2);
                    (sample, pos as u32, want.min(rem).min((s.blocks.len() * bs).saturating_sub(pos)), &s.blocks[..])
                })
                .collect();
            let more = self.model.mtp_chain(&rows)?;
            for (&(i, _), toks) in chain.iter().zip(more) {
                if let Some((_, t)) = self.running[i].mtp_next.as_mut() {
                    t.extend(toks);
                }
            }
        }
        // Tapped checkpoints: the tap slot holds the state at `c`; park it with the pages below.
        for &(k, c, slot) in &taps {
            self.park_prefix(plan[k].0, c, slot);
        }
        self.after_step(plan.iter().map(|&(i, _)| i))?;
        self.retire(done);
        self.last = StepStats {
            tokens: b.tokens.len(),
            decode_seqs,
            prefill_tokens: b.tokens.len() - decode_seqs,
            seqs: plan.len(),
            host_us,
            gpu_us,
        };
        Ok(events)
    }
}
