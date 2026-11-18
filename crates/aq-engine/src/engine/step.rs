use super::*;

/// Prefills of at most this many tokens are not held back by the mixed-step prefill budget.
pub const SMALL_PREFILL: usize = 256;

/// One sequence's part of a step: `n` tokens from its `computed` position, the last
/// `draft.len()` of them drafted.
pub(super) struct Entry {
    pub(super) i: usize,
    pub(super) n: usize,
    pub(super) draft: Vec<u32>,
    /// Leading draft tokens that came from the prompt lookup.
    local: usize,
    /// The first draft token is the MTP head's.
    mtp: bool,
    /// Checkpoint tapped inside this prefill: (position, slot).
    pub(super) tap: Option<(usize, u32)>,
}

impl Entry {
    /// `n` known tokens, with a checkpoint tap or without.
    pub(super) fn known(i: usize, n: usize, tap: Option<(usize, u32)>) -> Self {
        Self { i, n, draft: Vec::new(), local: 0, mtp: false, tap }
    }
}

/// Running indices of a draft-free plan's entries that sample a token, in sample-row order.
pub(super) fn sampled(plan: &[Entry], samples: &[(usize, usize)]) -> Vec<usize> {
    plan.iter().zip(samples).filter(|(_, s)| s.1 > 0).map(|(e, _)| e.i).collect()
}

/// Append `tok` to the sequence's output and report it. Returns whether the request ended.
fn emit(s: &mut Seq, tok: u32, eos: &[u32], trace: bool, events: &mut Vec<TokenEvent>) -> bool {
    s.out.push(tok);
    if trace && s.out.len() == 1 {
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
    let finished = stop || s.out.len() >= s.req.max_new_tokens;
    events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop, cached: s.reused });
    finished
}

impl Engine {
    fn ensure_blocks(&mut self, si: usize, upto: usize) -> Result<()> {
        let bs = self.limits().block_size;
        while self.running[si].blocks.len() * bs < upto {
            let Some(b) = self.pool.take_page() else { bail!("KV page pool exhausted") };
            self.running[si].blocks.push(b);
        }
        Ok(())
    }

    /// Build the batch for `plan`. Also returns, per entry, its first row among the sampled
    /// tokens and how many it samples: the last known token and every draft, or none when
    /// the entry stops short of the known tokens.
    pub(super) fn build_batch(&mut self, plan: &[Entry]) -> Result<(Batch, Vec<(usize, usize)>)> {
        let mb = self.limits().max_blocks_per_seq();
        let mut b = Batch::default();
        let mut samples = Vec::new();
        for (k, e) in plan.iter().enumerate() {
            let upto = self.running[e.i].computed + e.n;
            self.ensure_blocks(e.i, upto)?;
            let s = &self.running[e.i];
            let known = upto - e.draft.len();
            b.seq_start.push(b.tokens.len() as u32);
            b.seq_len.push(e.n as u32);
            b.seq_slot.push(s.slot);
            b.seq_slot_out.push(if e.draft.is_empty() { s.slot } else { s.spare.unwrap() });
            b.seq_tap.push(e.tap.map_or((0, 0), |(c, slot)| (((c - s.computed) / GDN_TC_CHUNK) as u32, slot)));
            for p in s.computed..upto {
                b.tokens.push(if p < known { s.token_at(p) } else { e.draft[p - known] });
                b.tok_pos.push(p as u32);
                b.tok_seq.push(k as u32);
            }
            let mut row = s.blocks.clone();
            row.resize(mb, 0);
            b.block_tables.extend_from_slice(&row);
            b.next_tok_last.push(if upto < s.known() { s.token_at(upto) } else { u32::MAX });
            if known == s.known() {
                samples.push((b.sample_rows.len(), e.draft.len() + 1));
                b.sample_rows.extend((b.tokens.len() - e.draft.len() - 1..b.tokens.len()).map(|r| r as u32));
            } else {
                samples.push((0, 0));
            }
        }
        Ok((b, samples))
    }

    /// Emit the tokens sampled for these running sequences; returns those that finished.
    pub(super) fn apply_sampled(&mut self, sampled: &[usize], toks: &[u32], events: &mut Vec<TokenEvent>) -> Vec<usize> {
        let mut done = Vec::new();
        for (&i, &tok) in sampled.iter().zip(toks) {
            if emit(&mut self.running[i], tok, &self.eos, self.trace, events) {
                done.push(i);
            }
        }
        done
    }

    /// Shorten a prefill of `n` tokens of running sequence `i` so it ends where a snapshot or
    /// a prompt checkpoint is due. A checkpoint is free when the prefill reaches the end of
    /// the prompt: with `tap_ok` the state is then tapped at the last chunk boundary before
    /// the backoff instead, into the slot returned with its position.
    pub(super) fn clamp_prefill(&mut self, i: usize, mut n: usize, tap_ok: bool) -> (usize, Option<(usize, u32)>) {
        let s = &self.running[i];
        let start = s.computed;
        let rem = s.req.prompt.len() - start;
        if let Some(at) = s.snapshot_at.filter(|&at| at > start) {
            n = n.min(at - start);
        }
        let mut tap = None;
        if let Some(cp) = s.ckpt_at.filter(|&at| at > start) {
            let c = start + (s.req.prompt.len() - CKPT_BACKOFF - start) / GDN_TC_CHUNK * GDN_TC_CHUNK;
            if tap_ok && n == rem && c > start && !self.pool.slots.is_empty() {
                tap = Some((c, self.pool.slots.pop().unwrap()));
            } else {
                n = n.min(cp - start);
            }
        }
        (n, tap)
    }

    /// Decode entries: every sequence past its prompt feeds its pending tokens and a draft.
    fn plan_decode(&mut self) -> Vec<Entry> {
        let max = self.limits().max_tokens;
        let mut plan = Vec::new();
        for i in 0..self.running.len() {
            let s = &self.running[i];
            if s.computed < s.req.prompt.len() {
                continue;
            }
            if plan.len() == max {
                break;
            }
            let pending = s.known() - s.computed;
            let rem = s.req.max_new_tokens - s.out.len();
            let (mut d, mut local, mut mtp) = (Vec::new(), 0, false);
            if self.spec_k > 0 && rem > 1 {
                // The draft stops one short of the token limit, so the limit never falls
                // inside a draft.
                let kmax = s.k_cur.max(self.spec_k).min(rem - 1).min(MAX_DECODE_ROWS_PER_SEQ.saturating_sub(pending));
                if kmax > 0 {
                    (d, local, mtp) = self.draft(i, kmax);
                }
            }
            if !d.is_empty() && self.running[i].spare.is_none() {
                // The speculative state needs a slot: never a shared-prefix snapshot's.
                if self.pool.slots.is_empty() {
                    self.evict_lru_parked();
                }
                self.running[i].spare = self.pool.slots.pop();
            }
            if self.running[i].spare.is_none() {
                d.clear();
            }
            plan.push(Entry { i, n: pending + d.len(), local: local.min(d.len()), mtp: mtp && !d.is_empty(), draft: d, tap: None });
        }
        plan
    }

    /// Add prefill entries after the `plan`'s decode entries. A decode entry costs the step
    /// budget one token: re-fed tokens and drafts ride in the row headroom
    /// (`Limits::max_rows`), so speculation never changes how much prefill fits.
    fn plan_prefill(&mut self, plan: &mut Vec<Entry>) {
        let decode_seqs = plan.len();
        // `room` is the step's row capacity, `budget` the soft cap next to decoding sequences.
        let mut room = self.limits().max_tokens - decode_seqs;
        let mut budget = room;
        if decode_seqs > 0 {
            let waiting: usize = self.running.iter().map(|s| s.req.prompt.len().saturating_sub(s.computed)).sum();
            budget = budget.min(match self.mixed_prefill_adaptive {
                Some((hi, threshold)) if waiting > threshold => hi,
                _ => self.mixed_prefill_budget,
            });
        }
        let left = |i: usize| self.running[i].req.prompt.len().saturating_sub(self.running[i].computed);
        // Prompts long enough for the background lane wait for it.
        let bg_min = if self.bg_prefill > 0 { self.bg_prefill } else { usize::MAX };
        let mut order: Vec<usize> = (0..self.running.len()).filter(|&i| left(i) > 0 && left(i) < bg_min && !self.running[i].in_prefill).collect();
        // One sequence may get a larger share than `budget`.
        let mut elder: Option<(usize, usize)> = None;
        if self.policy.srpt {
            order.sort_by_key(|&i| left(i));
            // Shortest first would starve the longest-waiting prompt: it goes first with half
            // the budget, or more when the shorter prompts cannot use their half.
            if let Some(&e) = order.iter().min_by_key(|&&i| self.running[i].t_add) {
                if e != order[0] && left(e) > SMALL_PREFILL {
                    let others: usize = order.iter().filter(|&&i| i != e && left(i) > SMALL_PREFILL).map(|&i| left(i)).sum();
                    elder = Some((e, (budget / 2).max(budget.saturating_sub(others))));
                    order.retain(|&i| i != e);
                    order.insert(0, e);
                }
            }
        }
        // Burst: a long prompt fed a budget at a time pays the step's fixed cost once per
        // slice. Every `burst_every` steps the longest-waiting one that needs more than the
        // budget gets up to `burst_prefill` tokens, at the price of one long inter-token gap.
        let mut burst = false;
        if self.burst_prefill > 0 && decode_seqs > 0 && self.since_burst >= self.burst_every {
            if let Some(&e) = order.iter().filter(|&&i| left(i) > budget).min_by_key(|&&i| self.running[i].t_add) {
                elder = Some((e, self.burst_prefill.max(budget)));
                order.retain(|&i| i != e);
                order.insert(0, e);
                burst = true;
            }
        }
        self.since_burst = if burst { 0 } else { self.since_burst.saturating_add(1) };
        let can_tap = self.model.supports_tap();
        for i in order {
            let rem = self.running[i].req.prompt.len() - self.running[i].computed;
            let cap = match elder {
                Some((e, share)) if e == i => share,
                _ => budget,
            };
            // Small prefills ignore the budget: they cost almost nothing next to the weights
            // a step reads anyway, and waiting behind a long prompt is the TTFT tail.
            let n = if rem <= SMALL_PREFILL { rem.min(room) } else { rem.min(cap).min(room) };
            if n == 0 {
                continue;
            }
            let (n, tap) = self.clamp_prefill(i, n, can_tap);
            plan.push(Entry::known(i, n, tap));
            budget = budget.saturating_sub(n);
            room -= n;
        }
    }

    /// Collect a finished background prefill and start the next one.
    fn poll_background(&mut self, events: &mut Vec<TokenEvent>) -> Result<()> {
        if self.bg_prefill == 0 {
            return Ok(());
        }
        if self.inflight.as_ref().is_some_and(|inf| inf.lane == 0) || (self.inflight.is_some() && self.model.lane_done(1)?) {
            self.collect_inflight(events)?;
        }
        if self.inflight.is_none() {
            self.launch_bg()?;
            // AQ_BG_SYNC=1 (debug): wait for the batch right away, so the two lanes never overlap.
            if self.inflight.is_some() && std::env::var("AQ_BG_SYNC").is_ok() {
                self.collect_inflight(events)?;
            }
        }
        Ok(())
    }

    /// Run one step. Returns one event per sampled token.
    ///
    /// Co-execution and speculative decoding do not combine: with `spec_k > 0` the engine
    /// takes the single-stream path, after draining a prefill batch still in flight.
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
        self.clock += 1;
        self.admit()?;
        // Tokens of a collected background batch come first: this step may already decode the
        // next token of the same sequence.
        let mut events = Vec::new();
        self.poll_background(&mut events)?;

        let mut plan = self.plan_decode();
        let decode_seqs = plan.len();
        self.plan_prefill(&mut plan);
        if plan.is_empty() {
            // Nothing for the main lane: wait for the background batch rather than spin.
            if events.is_empty() && self.inflight.is_some() {
                self.collect_inflight(&mut events)?;
            }
            return Ok(events);
        }

        let (mut b, samples) = self.build_batch(&plan)?;
        b.n_decode = decode_seqs;
        let host_us = t0.elapsed().as_micros() as u64;
        self.cache.host_us += host_us;
        self.cache.steps += 1;
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;
        if self.trace {
            self.trace_step(&plan, decode_seqs, gpu_us, host_us);
        }
        for e in &plan {
            self.running[e.i].steps += 1;
        }

        let (done, chain) = self.settle(&plan, decode_seqs, &samples, &toks, &mut events);
        self.chain_mtp(&chain)?;
        for e in &plan {
            if let Some((c, slot)) = e.tap {
                self.park_prefix(e.i, c, slot);
            }
        }
        self.after_step(plan.iter().map(|e| e.i))?;
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

    /// `[step]` line of AQ_TRACE: decode rows and prefill tokens with the keys they scan.
    fn trace_step(&self, plan: &[Entry], decode_seqs: usize, gpu_us: u64, host_us: u64) {
        let (mut rows, mut dkeys, mut ptok, mut pkeys) = (0usize, 0usize, 0usize, 0usize);
        for (k, e) in plan.iter().enumerate() {
            let (n, c) = (e.n, self.running[e.i].computed);
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
            plan.iter().filter(|e| e.tap.is_some()).count()
        );
    }

    /// Apply a step's sampled tokens: advance every entry, accept each draft while it equals
    /// the model's own choice, emit. Returns the running indices that finished and, for the
    /// sequences that keep decoding, (running index, sampled row) to chain MTP drafts from.
    fn settle(&mut self, plan: &[Entry], decode_seqs: usize, samples: &[(usize, usize)], toks: &[u32], events: &mut Vec<TokenEvent>) -> (Vec<usize>, Vec<(usize, usize)>) {
        let mut done = Vec::new();
        let mut chain: Vec<(usize, usize)> = Vec::new();
        let mut st = self.spec;
        for (k, e) in plan.iter().enumerate() {
            let (i, n, d) = (e.i, e.n, &e.draft[..]);
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
            let ends = |t: &u32| !s.req.ignore_eos && self.eos.contains(t);
            let n_emit = if nr == 0 { 0 } else { targets[..m + 1].iter().position(ends).map_or(m + 1, |p| p + 1) };
            if k < decode_seqs && self.spec_trace {
                eprintln!(
                    "[spec] id {} out {} computed {} known {} n {n} slot {} spare {:?} local {} draft {d:?} targets {targets:?} m {m}",
                    s.req.id,
                    s.out.len(),
                    s.computed,
                    s.known(),
                    s.slot,
                    s.spare,
                    e.local
                );
            }
            // The spare slot's state is ahead of the emitted tokens: drop it like a rejection.
            let overrun = !d.is_empty() && m == d.len() && n_emit < m;
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
                    st.global_drafted += (d.len() - e.local) as u64;
                    st.global_accepted += m.saturating_sub(e.local) as u64;
                    if e.mtp {
                        st.mtp_drafted += 1;
                        st.mtp_accepted += (m >= 1) as u64;
                    }
                }
            }
            // The MTP head's prediction at the row whose sample is the last token emitted.
            let mtp_pred = if nr > 0 { self.model.last_mtp.get(r0 + m).copied() } else { None };
            let mut finished = false;
            for &tok in &targets[..n_emit] {
                self.model.mtp_note([tok]);
                if self.global_draft {
                    let n = s.known() - 1;
                    let ctx: Vec<u32> = (n.saturating_sub(4)..n).map(|p| s.token_at(p)).collect();
                    self.global.observe(&ctx, tok);
                }
                finished = emit(s, tok, &self.eos, self.trace, events);
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
        (done, chain)
    }

    /// Chained MTP drafts for the sequences that keep decoding: as many as the draft budget
    /// and the KV pages they own allow (chained rows write to the head's cache at positions
    /// the next pass overwrites).
    fn chain_mtp(&mut self, chain: &[(usize, usize)]) -> Result<()> {
        if self.spec_k <= 1 || self.mtp_k <= 1 || chain.is_empty() {
            return Ok(());
        }
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
        Ok(())
    }
}
