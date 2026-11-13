//! Continuous-batching scheduler over [`Model::step`], with a session cache.
//!
//! Every step packs, in order: one decode token for each running sequence that has finished
//! its prompt, then prefill chunks (oldest first) until the token budget is spent.
//!
//! Admission reserves every KV page a request can need (prompt + max new tokens) up front, so
//! a running sequence never has to be preempted.
//!
//! Session cache: a finished sequence is *parked* instead of freed. Its state slot (DeltaNet
//! recurrent + conv state) and KV pages stay as they are, keyed by the exact tokens that were
//! fed through the model. A new request whose prompt strictly extends a parked token list
//! resumes from it and prefills only the new suffix, which is exact: the state is the state
//! after those tokens. Parked sessions are evicted least-recently-used when a new request
//! needs their slot or pages.
//!
//! Shared prefixes: when a new request shares a long page-aligned prefix with a known sequence
//! (e.g. the system prompt every agent starts with), its prefill pauses at that boundary and
//! the engine snapshots the recurrent state into a spare slot, pinning the prefix's KV pages
//! with reference counts. Later requests with that prefix fork from the snapshot: a state-slot
//! copy plus shared read-only pages. Exact, because no sequence ever writes below its own
//! `computed` position, so shared pages are never written.

use crate::model::{Batch, Limits, MAX_DECODE_ROWS_PER_SEQ, Model};
use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub ignore_eos: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenEvent {
    pub id: u64,
    pub token: u32,
    pub index: usize,
    pub finished: bool,
}

struct Seq {
    req: Request,
    out: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    /// Tokens already fed through the model.
    computed: usize,
    /// Pause prefill at this position and snapshot the state there (shared prefix).
    snapshot_at: Option<usize>,
    /// Spare state slot a speculative step writes into (swapped in on full acceptance).
    spare: Option<u32>,
}

impl Seq {
    fn known(&self) -> usize {
        self.req.prompt.len() + self.out.len()
    }
    fn token_at(&self, i: usize) -> u32 {
        let p = self.req.prompt.len();
        if i < p { self.req.prompt[i] } else { self.out[i - p] }
    }
}

/// Recurrent state + KV pages at a shared prefix boundary (page aligned).
struct Snapshot {
    tokens: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    last_used: u64,
    forks: u64,
}

/// Shared prefixes shorter than this are not worth a snapshot slot.
pub const SNAPSHOT_MIN_TOKENS: usize = 1024;

/// A finished sequence kept for reuse.
struct Parked {
    /// Tokens fed through the model; the slot's state and the KV pages cover exactly these.
    tokens: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    last_used: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub admitted: u64,
    pub hits: u64,
    pub forks: u64,
    pub snapshots: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub evictions: u64,
}

/// Speculative decoding counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Decode entries that carried at least one draft token.
    pub spec_entries: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Decode entries that emitted tokens (with or without drafts) and tokens they emitted.
    pub decode_entries: u64,
    pub emitted: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StepStats {
    pub tokens: usize,
    pub decode_seqs: usize,
    pub prefill_tokens: usize,
    pub seqs: usize,
    pub host_us: u64,
    pub gpu_us: u64,
}

pub struct Engine {
    pub model: Model,
    waiting: VecDeque<Request>,
    running: Vec<Seq>,
    free_slots: Vec<u32>,
    free_blocks: Vec<u32>,
    parked: Vec<Parked>,
    snapshots: Vec<Snapshot>,
    /// Reference count per KV page (owners: running/parked sequences and snapshots).
    refcnt: Vec<u32>,
    /// Max snapshot slots kept at once.
    pub max_snapshots: usize,
    clock: u64,
    eos: Vec<u32>,
    pub last: StepStats,
    pub cache: CacheStats,
    /// Keep finished sequences for prefix reuse (on by default).
    pub session_cache: bool,
    /// Max sequences running at once; state slots beyond this hold parked sessions.
    pub max_running: usize,
    /// Speculative decoding: max draft tokens per decode step (0 = off). Greedy requests only.
    /// Exact: drafts are verified through the normal decode-path kernels, the state is written
    /// to a spare slot, and on a rejection the accepted tokens are simply fed again.
    pub spec_k: usize,
    /// Prompt-lookup drafting: longest suffix n-gram (down to `spec_min_ngram`) matched against
    /// the sequence's own earlier tokens.
    pub spec_max_ngram: usize,
    pub spec_min_ngram: usize,
    pub spec: SpecStats,
    /// Max prefill tokens in a step that also carries decode tokens. Bounds how long a running
    /// sequence waits for its next token while new prompts are prefilled (ITL tail), at some
    /// cost to TTFT. `usize::MAX` = no separate cap (only the step budget).
    pub mixed_prefill_budget: usize,
}

impl Engine {
    pub fn new(model: Model, eos: Vec<u32>) -> Self {
        let lim = model.limits;
        Self {
            model,
            waiting: VecDeque::new(),
            running: Vec::new(),
            free_slots: (0..lim.max_seqs as u32).rev().collect(),
            free_blocks: (0..lim.num_blocks() as u32).rev().collect(),
            parked: Vec::new(),
            snapshots: Vec::new(),
            refcnt: vec![0; lim.num_blocks()],
            max_snapshots: 4,
            clock: 0,
            eos,
            last: StepStats::default(),
            cache: CacheStats::default(),
            session_cache: true,
            max_running: lim.max_seqs,
            spec_k: 0,
            spec_max_ngram: 4,
            spec_min_ngram: 2,
            spec: SpecStats::default(),
            mixed_prefill_budget: usize::MAX,
        }
    }

    pub fn limits(&self) -> Limits {
        self.model.limits
    }

    pub fn add(&mut self, req: Request) -> Result<()> {
        let lim = self.limits();
        if req.prompt.is_empty() || req.prompt.len() + req.max_new_tokens > lim.max_ctx {
            bail!("request {} needs {} tokens of context (max {})", req.id, req.prompt.len() + req.max_new_tokens, lim.max_ctx);
        }
        self.waiting.push_back(req);
        Ok(())
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Drop every parked session and snapshot (their slots and pages return to the pool).
    pub fn clear_cache(&mut self) {
        while self.evict_lru() {}
    }

    pub fn num_parked(&self) -> usize {
        self.parked.len()
    }

    /// Drop one owner reference to each page; pages with no owner left go back to the pool.
    fn release(&mut self, blocks: Vec<u32>) {
        for b in blocks {
            let r = &mut self.refcnt[b as usize];
            *r -= 1;
            if *r == 0 {
                self.free_blocks.push(b);
            }
        }
    }

    /// Drop the least-recently-used parked session (or, if none, the least-recently-used
    /// snapshot), returning its slot and pages.
    fn evict_lru(&mut self) -> bool {
        if let Some(i) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used) {
            let p = self.parked.swap_remove(i);
            self.free_slots.push(p.slot);
            self.release(p.blocks);
            self.cache.evictions += 1;
            return true;
        }
        if let Some(i) = (0..self.snapshots.len()).min_by_key(|&i| self.snapshots[i].last_used) {
            let sn = self.snapshots.swap_remove(i);
            self.free_slots.push(sn.slot);
            self.release(sn.blocks);
            self.cache.evictions += 1;
            return true;
        }
        false
    }

    /// Take `n` free KV pages (refcount 1), evicting cached sessions as needed.
    fn take_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        while self.free_blocks.len() < n {
            if !self.evict_lru() {
                return None;
            }
        }
        let out = self.free_blocks.split_off(self.free_blocks.len() - n);
        for &b in &out {
            self.refcnt[b as usize] = 1;
        }
        Some(out)
    }

    /// Longest page-aligned prefix that `prompt` shares with any running or parked sequence.
    fn shared_prefix(&self, prompt: &[u32]) -> usize {
        let bs = self.limits().block_size;
        let lcp = |t: &mut dyn Iterator<Item = u32>| prompt.iter().zip(t).take_while(|(a, b)| **a == *b).count();
        let mut best = 0;
        for p in &self.parked {
            best = best.max(lcp(&mut p.tokens.iter().copied()));
        }
        for s in &self.running {
            best = best.max(lcp(&mut (0..s.known()).map(|k| s.token_at(k))));
        }
        // Keep at least one token to feed, and align down to a page.
        (best.min(prompt.len().saturating_sub(1)) / bs) * bs
    }

    /// Take a snapshot of running sequence `i` at its current position (page aligned).
    fn snapshot(&mut self, i: usize) -> Result<()> {
        if self.snapshots.len() >= self.max_snapshots {
            if let Some(j) = (0..self.snapshots.len()).min_by_key(|&j| self.snapshots[j].last_used) {
                let sn = self.snapshots.swap_remove(j);
                self.free_slots.push(sn.slot);
                self.release(sn.blocks);
            }
        }
        let Some(slot) = self.free_slots.pop() else { return Ok(()) };
        let bs = self.limits().block_size;
        let s = &self.running[i];
        let n = s.computed;
        let tokens: Vec<u32> = (0..n).map(|k| s.token_at(k)).collect();
        let blocks: Vec<u32> = s.blocks[..n / bs].to_vec();
        let src = s.slot as usize;
        self.model.copy_slot(slot as usize, src)?;
        for &b in &blocks {
            self.refcnt[b as usize] += 1;
        }
        self.snapshots.push(Snapshot { tokens, slot, blocks, last_used: self.clock, forks: 0 });
        self.cache.snapshots += 1;
        Ok(())
    }

    /// Admit the request at the head of the queue, resuming a parked session when one is a
    /// strict prefix of its prompt. Returns false if resources are not available yet.
    fn admit_one(&mut self) -> Result<bool> {
        let bs = self.limits().block_size;
        let req = self.waiting.front().unwrap();
        let need = (req.prompt.len() + req.max_new_tokens).div_ceil(bs);
        let best = if self.session_cache {
            (0..self.parked.len())
                .filter(|&i| {
                    let t = &self.parked[i].tokens;
                    t.len() < req.prompt.len() && req.prompt[..t.len()] == t[..]
                })
                .max_by_key(|&i| self.parked[i].tokens.len())
        } else {
            None
        };
        let fork = if best.is_none() && self.session_cache {
            (0..self.snapshots.len())
                .filter(|&i| {
                    let t = &self.snapshots[i].tokens;
                    t.len() < req.prompt.len() && req.prompt[..t.len()] == t[..]
                })
                .max_by_key(|&i| self.snapshots[i].tokens.len())
        } else {
            None
        };
        let mut snapshot_at = None;
        let (slot, mut blocks, computed) = if let Some(i) = best {
            let p = self.parked.swap_remove(i);
            (p.slot, p.blocks, p.tokens.len())
        } else {
            let slot = match self.free_slots.pop() {
                Some(s) => s,
                None => {
                    if !self.evict_lru() {
                        return Ok(false);
                    }
                    self.free_slots.pop().unwrap()
                }
            };
            if let Some(i) = fork.filter(|&i| i < self.snapshots.len()) {
                // Fork from the shared-prefix snapshot: copy its state, share its pages.
                let clock = self.clock;
                let sn = &mut self.snapshots[i];
                sn.last_used = clock;
                sn.forks += 1;
                let (src, blocks, n) = (sn.slot as usize, sn.blocks.clone(), sn.tokens.len());
                self.model.copy_slot(slot as usize, src)?;
                for &b in &blocks {
                    self.refcnt[b as usize] += 1;
                }
                self.cache.forks += 1;
                (slot, blocks, n)
            } else {
                self.model.reset_slot(slot as usize)?;
                if self.session_cache {
                    let req = self.waiting.front().unwrap();
                    let shared = self.shared_prefix(&req.prompt);
                    if shared >= SNAPSHOT_MIN_TOKENS {
                        snapshot_at = Some(shared);
                    }
                }
                (slot, Vec::new(), 0)
            }
        };
        if blocks.len() > need {
            let extra = blocks.split_off(need);
            self.release(extra);
        }
        let more = need - blocks.len();
        match self.take_blocks(more) {
            Some(b) => blocks.extend(b),
            None => {
                // Not enough pages even after evicting everything parked: put the session
                // back (or release the fresh slot) and wait for running sequences to finish.
                if best.is_some() {
                    let tokens = self.waiting.front().unwrap().prompt[..computed].to_vec();
                    self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock });
                } else {
                    self.free_slots.push(slot);
                    self.release(blocks);
                }
                return Ok(false);
            }
        }
        let req = self.waiting.pop_front().unwrap();
        self.cache.admitted += 1;
        if computed > 0 {
            self.cache.hits += 1;
            self.cache.reused_tokens += computed as u64;
        }
        self.cache.prefilled_tokens += (req.prompt.len() - computed) as u64;
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed, snapshot_at, spare: None });
        Ok(true)
    }

    /// Prompt-lookup draft: find the most recent earlier occurrence of the longest suffix
    /// n-gram of the sequence's known tokens and propose the tokens that followed it.
    fn draft(&self, s: &Seq, k: usize) -> Vec<u32> {
        let n_known = s.known();
        let tok = |i: usize| s.token_at(i);
        let mut n = self.spec_max_ngram.min(n_known.saturating_sub(1));
        while n >= self.spec_min_ngram.max(1) {
            let suf0 = n_known - n;
            // scan backwards for an earlier occurrence of tokens[suf0..n_known]
            let mut start = suf0;
            while start > 0 {
                start -= 1;
                if (0..n).all(|j| tok(start + j) == tok(suf0 + j)) {
                    let from = start + n;
                    let to = (from + k).min(n_known);
                    if to > from {
                        return (from..to).map(tok).collect();
                    }
                }
            }
            n -= 1;
        }
        Vec::new()
    }

    fn ensure_blocks(&mut self, si: usize, upto: usize) -> Result<()> {
        let bs = self.limits().block_size;
        while self.running[si].blocks.len() * bs < upto {
            let Some(b) = self.free_blocks.pop() else { bail!("KV page pool exhausted") };
            self.running[si].blocks.push(b);
        }
        Ok(())
    }

    /// Run one step. Returns one event per sampled token.
    pub fn step(&mut self) -> Result<Vec<TokenEvent>> {
        let t0 = Instant::now();
        let lim = self.limits();
        let mut budget = lim.max_tokens;

        // Admit waiting requests (FIFO) while there is room to run them.
        self.clock += 1;
        while !self.waiting.is_empty() && self.running.len() < self.max_running.min(lim.max_seqs) {
            if !self.admit_one()? {
                break;
            }
        }

        // Decode-path entries first (pending tokens + optional drafts), then prefill chunks in
        // admission order.
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (running index, n tokens)
        let mut drafts: Vec<Vec<u32>> = Vec::new(); // per decode entry
        for i in 0..self.running.len() {
            let s = &self.running[i];
            if s.computed < s.req.prompt.len() || budget == 0 {
                continue;
            }
            let pending = s.known() - s.computed;
            let rem = s.req.max_new_tokens - s.out.len();
            let mut d = Vec::new();
            if self.spec_k > 0 && rem > 1 {
                let kmax = self.spec_k.min(rem - 1).min(MAX_DECODE_ROWS_PER_SEQ.saturating_sub(pending));
                if kmax > 0 {
                    d = self.draft(s, kmax);
                }
            }
            if !d.is_empty() && self.running[i].spare.is_none() {
                // A spare slot for the speculative state, only if one is free: speculation is
                // opportunistic and never evicts cached sessions or snapshots.
                self.running[i].spare = self.free_slots.pop();
            }
            if self.running[i].spare.is_none() {
                d.clear();
            }
            let n = (pending + d.len()).min(budget);
            if n < pending + d.len() {
                d.clear();
            }
            let n = n.min(pending + d.len());
            plan.push((i, n));
            drafts.push(d);
            budget -= n;
        }
        let decode_seqs = plan.len();
        if decode_seqs > 0 {
            budget = budget.min(self.mixed_prefill_budget);
        }
        for (i, s) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if s.computed < s.req.prompt.len() {
                let mut n = (s.req.prompt.len() - s.computed).min(budget);
                if let Some(at) = s.snapshot_at.filter(|&at| at > s.computed) {
                    n = n.min(at - s.computed);
                }
                plan.push((i, n));
                budget -= n;
            }
        }
        if plan.is_empty() {
            return Ok(Vec::new());
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
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;

        let mut events = Vec::new();
        let mut done = Vec::new();
        let eos = self.eos.clone();
        let mut st = self.spec;
        for (k, &(i, n)) in plan.iter().enumerate() {
            let d: &[u32] = if k < decode_seqs { &drafts[k] } else { &[] };
            let (r0, nr) = samples[k];
            let targets = &toks[r0..r0 + nr];
            // Accept drafts while they equal the target's greedy choice at the previous row.
            let m = d.iter().zip(targets).take_while(|(a, b)| a == b).count();
            let s = &mut self.running[i];
            if d.is_empty() {
                s.computed += n;
            } else if m == d.len() {
                // Every draft accepted: the spare slot holds the state after all n tokens.
                s.computed += n;
                let old = s.slot;
                s.slot = s.spare.unwrap();
                s.spare = Some(old);
            }
            // else: rejection. The old slot still holds the state before this step; the
            // accepted tokens become pending and are fed again next step.
            if k < decode_seqs && nr > 0 {
                st.decode_entries += 1;
                if !d.is_empty() {
                    st.spec_entries += 1;
                    st.drafted += d.len() as u64;
                    st.accepted += m as u64;
                }
            }
            let mut finished = false;
            for &tok in &targets[..if nr > 0 { m + 1 } else { 0 }] {
                s.out.push(tok);
                finished = s.out.len() >= s.req.max_new_tokens || (!s.req.ignore_eos && eos.contains(&tok));
                events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished });
                if k < decode_seqs {
                    st.emitted += 1;
                }
                if finished {
                    break;
                }
            }
            if finished {
                done.push(i);
            }
        }
        self.spec = st;
        for &(i, _) in &plan {
            if self.running[i].snapshot_at == Some(self.running[i].computed) {
                self.running[i].snapshot_at = None;
                self.snapshot(i)?;
            }
        }
        done.sort_unstable_by(|a, b| b.cmp(a));
        for i in done {
            let s = self.running.swap_remove(i);
            if let Some(sp) = s.spare {
                self.free_slots.push(sp);
            }
            if self.session_cache {
                let tokens = (0..s.computed).map(|k| s.token_at(k)).collect();
                self.parked.push(Parked { tokens, slot: s.slot, blocks: s.blocks, last_used: self.clock });
            } else {
                self.free_slots.push(s.slot);
                self.release(s.blocks);
            }
        }
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
