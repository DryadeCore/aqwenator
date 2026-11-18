use super::*;

/// A prefill batch launched on lane 0 and not collected yet (co-execution). Sequences are
/// named by request id: running indices shift while decode steps retire sequences.
pub(super) struct Inflight {
    /// (request id, tokens fed) per sequence, in batch order.
    pub(super) plan: Vec<(u64, usize)>,
    /// Request ids that sample a token from this batch, in sample-row order.
    pub(super) sampled: Vec<u64>,
    pub(super) ns: usize,
    pub(super) tokens: usize,
    /// Lane the batch runs on (0: co-execution's prefill lane; 1: the background prefill lane).
    pub(super) lane: usize,
    /// Checkpoints tapped inside the batch: (request id, position, slot).
    pub(super) taps: Vec<(u64, usize, u32)>,
    pub(super) launched: Instant,
}

impl Engine {
    /// Turn on decode/prefill co-execution (creates the model's decode lane).
    pub fn enable_coexec(&mut self) -> Result<()> {
        self.model.enable_decode_lane()?;
        self.coexec = true;
        Ok(())
    }

    /// Turn on background prefill for prompts with at least `min_tokens` left to prefill.
    pub fn enable_bg_prefill(&mut self, min_tokens: usize) -> Result<()> {
        if self.coexec {
            bail!("background prefill and co-execution both need the second lane");
        }
        self.model.enable_prefill_lane()?;
        self.bg_prefill = min_tokens;
        Ok(())
    }

    /// Start the background prefill of the running sequence with the most prompt left, if it
    /// has at least `bg_prefill` tokens to go and the lane is free.
    pub(super) fn launch_bg(&mut self) -> Result<()> {
        let lim = self.limits();
        let left = |s: &Seq| s.req.prompt.len().saturating_sub(s.computed);
        let Some(i) = (0..self.running.len())
            .filter(|&i| !self.running[i].in_prefill && left(&self.running[i]) >= self.bg_prefill)
            .max_by_key(|&i| left(&self.running[i]))
        else {
            return Ok(());
        };
        let n = left(&self.running[i]).min(lim.max_tokens);
        let (n, tap) = self.clamp_prefill(i, n, self.model.supports_tap());
        let plan = [Entry::known(i, n, tap)];
        let (b, samples) = self.build_batch(&plan)?;
        let sampled = sampled(&plan, &samples);
        // Whatever the main lane still has queued for this sequence (slot reset or copy, a
        // page copy) must have run before the other lane reads it.
        self.model.gpu.sync()?;
        // AQ_BG_LANE=0 (debug): run the batch on the main lane through the same launch/collect path.
        let lane: usize = std::env::var("AQ_BG_LANE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        self.model.set_lane(lane);
        let ns = self.model.launch(&b);
        self.model.set_lane(0);
        let ns = ns?;
        self.running[i].in_prefill = true;
        let id = self.running[i].req.id;
        self.inflight = Some(Inflight {
            plan: vec![(id, n)],
            sampled: sampled.iter().map(|&k| self.running[k].req.id).collect(),
            ns,
            tokens: n,
            lane,
            taps: tap.map(|(c, slot)| (id, c, slot)).into_iter().collect(),
            launched: Instant::now(),
        });
        Ok(())
    }

    /// Co-execution step: keep one prefill batch in flight on lane 0 and run a decode step for
    /// every other ready sequence on the high-priority lane 1. Decode tokens never wait for a
    /// prefill chunk; a prefill batch is collected once its lane-0 event completes (or waited
    /// for when nothing is decoding).
    ///
    /// Ordering: admission (slot reset/copy) and snapshots are enqueued on lane 0, the lane of
    /// every sequence until its prefill completes; a sequence moves to lane 1 only after the
    /// host has observed its prefill batch complete, and back to lane 0 (a later turn) only
    /// after it finished decoding, which is synchronous. KV pages and state slots are per
    /// sequence, so the two lanes never write the same memory.
    pub(super) fn step_coexec(&mut self) -> Result<Vec<TokenEvent>> {
        let t0 = Instant::now();
        let lim = self.limits();
        self.model.set_lane(0);
        self.clock += 1;
        self.admit()?;
        let mut events = Vec::new();

        // 1. Launch a prefill batch on lane 0 if none is in flight.
        let mut prefill_tokens = 0;
        if self.inflight.is_none() {
            let mut budget = lim.max_tokens;
            let mut plan = Vec::new();
            for i in 0..self.running.len() {
                if budget == 0 {
                    break;
                }
                let s = &self.running[i];
                if !s.in_prefill && s.computed < s.req.prompt.len() {
                    let n = (s.req.prompt.len() - s.computed).min(budget);
                    let (n, _) = self.clamp_prefill(i, n, false);
                    plan.push(Entry::known(i, n, None));
                    budget -= n;
                }
            }
            if !plan.is_empty() {
                let (b, samples) = self.build_batch(&plan)?;
                let sampled = sampled(&plan, &samples);
                let ns = self.model.launch(&b)?;
                prefill_tokens = b.tokens.len();
                for e in &plan {
                    self.running[e.i].in_prefill = true;
                }
                self.inflight = Some(Inflight {
                    plan: plan.iter().map(|e| (self.running[e.i].req.id, e.n)).collect(),
                    sampled: sampled.iter().map(|&i| self.running[i].req.id).collect(),
                    ns,
                    tokens: b.tokens.len(),
                    lane: 0,
                    taps: Vec::new(),
                    launched: Instant::now(),
                });
            }
        }

        // 2. One decode step on lane 1 for every ready sequence outside the in-flight batch.
        let dplan: Vec<Entry> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.in_prefill && s.computed >= s.req.prompt.len() && s.known() > s.computed)
            // Normally one pending token; more after a rejected speculative step (spec_k
            // switched off since), which the decode path also handles.
            .map(|(i, s)| Entry::known(i, (s.known() - s.computed).min(MAX_DECODE_ROWS_PER_SEQ), None))
            .collect();
        let decode_seqs = dplan.len();
        if decode_seqs > 0 && self.inflight.is_some() && !self.coexec_pace.is_zero() {
            // Pacing: give the in-flight prefill the GPU until the next decode step is due.
            if let Some(last) = self.last_decode {
                let due = last + self.coexec_pace;
                while Instant::now() < due {
                    if self.model.lane_done(0)? {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
        let t1 = Instant::now();
        if decode_seqs > 0 {
            self.last_decode = Some(t1);
            if self.inflight.is_some() {
                self.overlap_steps += 1;
            }
            let (mut b, samples) = self.build_batch(&dplan)?;
            let sampled = sampled(&dplan, &samples);
            b.n_decode = decode_seqs;
            self.model.set_lane(1);
            let toks = self.model.step(&b, None);
            self.model.set_lane(0);
            let toks = toks?;
            for e in &dplan {
                self.running[e.i].computed += e.n;
            }
            let done = self.apply_sampled(&sampled, &toks, &mut events);
            self.retire(done);
        }

        // 3. Collect the prefill batch when it is done; wait for it if nothing else can run.
        if self.inflight.is_some() && (decode_seqs == 0 || self.model.lane_done(0)?) {
            prefill_tokens = prefill_tokens.max(self.collect_inflight(&mut events)?);
        }
        self.last = StepStats {
            tokens: decode_seqs + prefill_tokens,
            decode_seqs,
            prefill_tokens,
            seqs: decode_seqs,
            host_us: t1.duration_since(t0).as_micros() as u64,
            gpu_us: t1.elapsed().as_micros() as u64,
        };
        Ok(events)
    }

    /// Wait for the in-flight prefill batch (co-execution), apply it and emit its sampled tokens.
    /// Returns the batch's token count (0 if none was in flight).
    pub(super) fn collect_inflight(&mut self, events: &mut Vec<TokenEvent>) -> Result<usize> {
        let Some(inf) = self.inflight.take() else { return Ok(0) };
        let back = self.model.lane();
        self.model.set_lane(inf.lane);
        let toks = self.model.collect(inf.ns);
        self.model.set_lane(back);
        let toks = toks?;
        if self.trace {
            eprintln!("[bg] lane {} tokens {} ms {:.1}", inf.lane, inf.tokens, inf.launched.elapsed().as_secs_f64() * 1e3);
        }
        for &(id, n) in &inf.plan {
            let i = self.index_of(id);
            self.running[i].computed += n;
            self.running[i].in_prefill = false;
            self.running[i].steps += 1;
        }
        for &(id, c, slot) in &inf.taps {
            let i = self.index_of(id);
            self.park_prefix(i, c, slot);
        }
        let idx: Vec<usize> = inf.plan.iter().map(|&(id, _)| self.index_of(id)).collect();
        self.after_step(idx.into_iter())?;
        let sampled: Vec<usize> = inf.sampled.iter().map(|&id| self.index_of(id)).collect();
        let mut done = self.apply_sampled(&sampled, &toks, events);
        done.extend(inf.plan.iter().map(|&(id, _)| self.index_of(id)).filter(|&i| self.running[i].cancelled));
        self.retire(done);
        Ok(inf.tokens)
    }
}
