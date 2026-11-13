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

use crate::draft::DraftIndex;
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
    /// The request ended on an end-of-sequence token (rather than its token limit).
    pub stop: bool,
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
    /// Part of the prefill batch currently in flight on lane 0 (co-execution).
    in_prefill: bool,
    /// Cancelled while its prefill batch was in flight; retired when that batch is collected.
    cancelled: bool,
    /// Prompt-lookup index over the known tokens, built on the first speculative step.
    draft: Option<DraftIndex>,
    /// Turns served since this session last became resident (cache hits keep counting).
    turns: u32,
    /// Session id: parked entries left by the same chain of turns share it.
    sid: u64,
    /// Pause prefill here and take the prompt checkpoint (page aligned, a few tokens before
    /// the prompt's end); `None` once taken, or when there is nothing to checkpoint.
    ckpt_at: Option<usize>,
}

/// A prefill batch launched on lane 0 and not collected yet (co-execution). Sequences are
/// named by request id: running indices shift while decode steps retire sequences.
struct Inflight {
    /// (request id, tokens fed) per sequence, in batch order.
    plan: Vec<(u64, usize)>,
    /// Request ids that sample a token from this batch, in sample-row order.
    sampled: Vec<u64>,
    ns: usize,
    tokens: usize,
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

/// `t` is a strict prefix of `prompt`. Cheap rejects first (last and middle token), so the
/// full comparison usually runs only for the request's own session.
#[inline]
fn is_strict_prefix(t: &[u32], prompt: &[u32]) -> bool {
    let n = t.len();
    n < prompt.len() && (n == 0 || (t[n - 1] == prompt[n - 1] && t[n / 2] == prompt[n / 2])) && prompt[..n] == t[..]
}

/// Which parked session to evict when pages or slots run out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evict {
    /// Least recently parked first (classic LRU).
    Lru,
    /// Most recently parked first. With bounded tool think time, the session parked longest
    /// ago is the one most likely to come back next, so LRU evicts exactly the wrong one.
    Mru,
    /// Idle-aware: sessions idle longer than `dead_after_ms` are presumed finished and go
    /// first (oldest first); among the live ones, most recently parked first.
    Idle,
}

/// Memory and scheduling policy knobs (all default to the original behavior).
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub evict: Evict,
    /// Never evict shared-prefix snapshots to make room (only `max_snapshots` replacement),
    /// and do not create a snapshot identical to an existing one.
    pub pin_snapshots: bool,
    /// Working-set load control: a request that is not a cache hit may not evict parked
    /// sessions; it waits for free pages unless nothing runs or it has waited `max_wait_ms`.
    pub load_control: bool,
    /// Shortest-remaining-uncached-work first, for admission and prefill chunk order, with
    /// aging past `max_wait_ms`.
    pub srpt: bool,
    pub max_wait_ms: u64,
    /// `Evict::Idle`: idle time after which a parked session is presumed finished.
    pub dead_after_ms: u64,
    /// Load control residency quantum: a waiting non-hit request may evict a live parked
    /// session once that session has been served this many turns since it became resident
    /// (0 = never; only finished sessions).
    pub quantum: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            evict: Evict::Lru,
            pin_snapshots: false,
            load_control: false,
            srpt: false,
            max_wait_ms: 2000,
            dead_after_ms: 6000,
            quantum: 0,
        }
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

/// Tokens between a prompt checkpoint and the end of its prompt, at least (see
/// `Engine::prompt_checkpoints`).
pub const CKPT_BACKOFF: usize = 4;

/// Shared prefixes shorter than this are not worth a snapshot slot.
pub const SNAPSHOT_MIN_TOKENS: usize = 1024;

/// A finished sequence kept for reuse.
struct Parked {
    /// Tokens fed through the model; the slot's state and the KV pages cover exactly these.
    tokens: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    last_used: u64,
    parked_at: Instant,
    turns: u32,
    sid: u64,
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
    pub checkpoints: u64,
    pub snapshot_evictions: u64,
    pub deferred: u64,
    /// Scheduler host time (admission + batch building) and steps, cumulative.
    pub host_us: u64,
    pub steps: u64,
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
    waiting: VecDeque<(Request, Instant)>,
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
    /// Decode/prefill co-execution (AQ_COEXEC=1): prefill batches run asynchronously on lane 0
    /// while decode steps for the other sequences run on a high-priority lane 1.
    pub coexec: bool,
    inflight: Option<Inflight>,
    /// Decode steps that ran while a prefill batch was in flight (co-execution).
    pub overlap_steps: u64,
    /// Minimum time between decode steps while a prefill batch is in flight (AQ_COEXEC_PACE_US):
    /// leaves the GPU to prefill between decode steps, trading inter-token latency for TTFT.
    pub coexec_pace: std::time::Duration,
    last_decode: Option<Instant>,
    /// (larger budget, pending-prefill threshold in tokens): switch mixed steps to the larger
    /// prefill budget while more than `threshold` admitted prompt tokens await prefill.
    pub mixed_prefill_adaptive: Option<(usize, usize)>,
    /// Prompt checkpoints: keep a copy of the state at the last page boundary at least
    /// `CKPT_BACKOFF` tokens before a request's prompt ends (a parked entry sharing the
    /// prompt's KV pages). A later request that extends the *prompt* then resumes from it even
    /// if it does not extend the tokens this engine generated, which is what agent harnesses
    /// send: the previous prompt plus the assistant turn as the chat template renders it.
    /// The backoff is there because templates end a prompt with a generation header whose
    /// last token re-tokenizes once the turn's text follows it (Qwen3.5: `<think>\n` becomes
    /// `<think>\n\n`), so consecutive prompts diverge just before the previous one's end.
    pub prompt_checkpoints: bool,
    pub policy: Policy,
    /// Frame pacing: target duration of a step that carries decode tokens (µs).
    pub itl_target_us: Option<f64>,
    pace: Pace,
    /// Set during an admission that may only evict sessions presumed finished.
    dead_only: bool,
}

/// Online step-cost model for frame pacing.
#[derive(Debug, Clone, Copy)]
struct Pace {
    decode_us: f64,
    prefill_us_per_tok: f64,
}

/// Smallest prefill chunk frame pacing will schedule next to decodes (keeps prefill moving).
const PACE_MIN_CHUNK: usize = 256;

impl Engine {
    pub fn new(model: Model, eos: Vec<u32>) -> Self {
        let lim = model.limits;
        let mut eng = Self {
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
            coexec: false,
            inflight: None,
            overlap_steps: 0,
            coexec_pace: std::time::Duration::from_micros(
                std::env::var("AQ_COEXEC_PACE_US").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            ),
            last_decode: None,
            mixed_prefill_adaptive: None,
            prompt_checkpoints: std::env::var("AQ_NO_CKPT").is_err(),
            policy: Policy::default(),
            itl_target_us: None,
            pace: Pace { decode_us: 5000.0, prefill_us_per_tok: 30.0 },
            dead_only: false,
        };
        if std::env::var("AQ_COEXEC").as_deref() == Ok("1") {
            eng.enable_coexec().expect("could not create the decode lane for AQ_COEXEC=1");
        }
        eng
    }

    /// Turn on decode/prefill co-execution (creates the model's decode lane).
    pub fn enable_coexec(&mut self) -> Result<()> {
        self.model.enable_decode_lane()?;
        self.coexec = true;
        Ok(())
    }

    pub fn limits(&self) -> Limits {
        self.model.limits
    }

    pub fn add(&mut self, req: Request) -> Result<()> {
        let lim = self.limits();
        if req.prompt.is_empty() || req.prompt.len() + req.max_new_tokens > lim.max_ctx {
            bail!("request {} needs {} tokens of context (max {})", req.id, req.prompt.len() + req.max_new_tokens, lim.max_ctx);
        }
        // A request that needs more KV pages than the whole pool can never be admitted; left
        // in the queue it would block every request behind it forever.
        let need = (req.prompt.len() + req.max_new_tokens).div_ceil(lim.block_size);
        if need > lim.num_blocks() {
            bail!(
                "request {} needs {} KV pages ({} tokens) but the pool has {}; lower max_tokens or raise the KV budget",
                req.id,
                need,
                req.prompt.len() + req.max_new_tokens,
                lim.num_blocks()
            );
        }
        self.waiting.push_back((req, Instant::now()));
        Ok(())
    }

    /// Stop a request early, e.g. because its client went away. A waiting request is dropped.
    /// A running one is retired as if it had finished (parked for reuse when the session cache
    /// is on), or, if its prefill batch is in flight, as soon as that batch is collected.
    /// Returns false if the request is not known (already finished).
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(i) = self.waiting.iter().position(|(r, _)| r.id == id) {
            self.waiting.remove(i);
            return true;
        }
        let Some(i) = self.running.iter().position(|s| s.req.id == id) else { return false };
        if self.running[i].in_prefill {
            self.running[i].cancelled = true;
        } else {
            self.retire(vec![i]);
        }
        true
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

    /// Drop the least-recently-used parked session only (never a snapshot).
    fn evict_lru_parked(&mut self) -> bool {
        let Some(i) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used) else { return false };
        let p = self.parked.swap_remove(i);
        self.free_slots.push(p.slot);
        self.release(p.blocks);
        self.cache.evictions += 1;
        true
    }

    fn is_dead(&self, i: usize, now: Instant) -> bool {
        now.duration_since(self.parked[i].parked_at).as_millis() as u64 >= self.policy.dead_after_ms
    }

    /// May a restricted admission (load control, non-hit) evict parked session `i`? Sessions
    /// presumed finished always; with a residency quantum, also sessions that have been served
    /// `quantum` turns since they last became resident (working-set rotation).
    fn evictable_restricted(&self, i: usize, now: Instant) -> bool {
        self.is_dead(i, now) || (self.policy.quantum > 0 && self.parked[i].turns >= self.policy.quantum)
    }

    /// Evict one parked session chosen by the policy (or, if none and snapshots are not
    /// pinned, the least-recently-used snapshot), returning its slot and pages.
    /// With `self.dead_only`, only sessions presumed finished may go.
    fn evict_lru(&mut self) -> bool {
        if self.dead_only {
            // Restricted: finished sessions first (oldest first), then the sessions that have
            // used up their residency quantum (most served first).
            let now = Instant::now();
            let dead = (0..self.parked.len()).filter(|&i| self.is_dead(i, now)).min_by_key(|&i| self.parked[i].parked_at);
            let Some(i) = dead.or_else(|| {
                (0..self.parked.len()).filter(|&i| self.evictable_restricted(i, now)).max_by_key(|&i| self.parked[i].turns)
            }) else {
                return false;
            };
            let p = self.parked.swap_remove(i);
            self.free_slots.push(p.slot);
            self.release(p.blocks);
            self.cache.evictions += 1;
            return true;
        }
        let victim = match self.policy.evict {
            Evict::Lru => (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used),
            Evict::Mru => (0..self.parked.len()).max_by_key(|&i| self.parked[i].last_used),
            Evict::Idle => {
                let now = Instant::now();
                let dead = |i: usize| now.duration_since(self.parked[i].parked_at).as_millis() as u64 >= self.policy.dead_after_ms;
                (0..self.parked.len())
                    .filter(|&i| dead(i))
                    .min_by_key(|&i| self.parked[i].parked_at)
                    .or_else(|| (0..self.parked.len()).max_by_key(|&i| self.parked[i].parked_at))
            }
        };
        if let Some(i) = victim {
            let p = self.parked.swap_remove(i);
            self.free_slots.push(p.slot);
            self.release(p.blocks);
            self.cache.evictions += 1;
            return true;
        }
        if self.policy.pin_snapshots {
            return false;
        }
        if let Some(i) = (0..self.snapshots.len()).min_by_key(|&i| self.snapshots[i].last_used) {
            let sn = self.snapshots.swap_remove(i);
            self.free_slots.push(sn.slot);
            self.release(sn.blocks);
            self.cache.evictions += 1;
            self.cache.snapshot_evictions += 1;
            return true;
        }
        false
    }

    /// Test hook: permanently remove up to `n` free pages from the pool (a smaller GPU).
    pub fn withhold_pages(&mut self, n: usize) {
        let keep = self.free_blocks.len() - n.min(self.free_blocks.len());
        self.free_blocks.truncate(keep);
    }

    pub fn free_pages(&self) -> usize {
        self.free_blocks.len()
    }

    /// Test hook: evict every parked session.
    pub fn evict_all_parked(&mut self) {
        while !self.parked.is_empty() {
            self.evict_lru();
        }
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

    /// Prompt checkpoint of running sequence `i` at its current (page-aligned) position: a
    /// parked entry with a copy of the state and shared references to the KV pages below it.
    /// Skipped when no state slot is free.
    fn checkpoint(&mut self, i: usize) -> Result<()> {
        let Some(slot) = self.free_slots.pop() else { return Ok(()) };
        let bs = self.limits().block_size;
        let s = &self.running[i];
        let n = s.computed;
        let blocks: Vec<u32> = s.blocks[..n.div_ceil(bs)].to_vec();
        let (src, sid, turns) = (s.slot as usize, s.sid, s.turns);
        let tokens = s.req.prompt[..n].to_vec();
        self.model.copy_slot(slot as usize, src)?;
        for &b in &blocks {
            self.refcnt[b as usize] += 1;
        }
        self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns, sid });
        self.cache.checkpoints += 1;
        Ok(())
    }

    /// Take checkpoints and shared-prefix snapshots that became due for these running sequences.
    fn after_step(&mut self, idx: impl Iterator<Item = usize>) -> Result<()> {
        for i in idx {
            let s = &self.running[i];
            if s.snapshot_at == Some(s.computed) {
                self.running[i].snapshot_at = None;
                self.snapshot(i)?;
            }
            if self.running[i].ckpt_at == Some(self.running[i].computed) {
                self.running[i].ckpt_at = None;
                self.checkpoint(i)?;
            }
        }
        Ok(())
    }

    /// Drop every parked entry of session `sid` (stale once one of them has been resumed).
    fn drop_session(&mut self, sid: u64) {
        let mut k = 0;
        while k < self.parked.len() {
            if self.parked[k].sid == sid {
                let p = self.parked.swap_remove(k);
                self.free_slots.push(p.slot);
                self.release(p.blocks);
            } else {
                k += 1;
            }
        }
    }

    /// Take a snapshot of running sequence `i` at its current position (page aligned).
    fn snapshot(&mut self, i: usize) -> Result<()> {
        if self.policy.pin_snapshots {
            let s = &self.running[i];
            let n = s.computed;
            if self.snapshots.iter().any(|sn| sn.tokens.len() == n && (0..n).all(|k| sn.tokens[k] == s.token_at(k))) {
                return Ok(());
            }
        }
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

    fn find_parked(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.parked.len())
            .filter(|&i| is_strict_prefix(&self.parked[i].tokens, prompt))
            .max_by_key(|&i| self.parked[i].tokens.len())
    }

    fn find_snapshot(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.snapshots.len())
            .filter(|&i| is_strict_prefix(&self.snapshots[i].tokens, prompt))
            .max_by_key(|&i| self.snapshots[i].tokens.len())
    }

    /// Tokens of `prompt` that already have state (parked session or snapshot).
    fn cached_len(&self, prompt: &[u32]) -> usize {
        let p = self.find_parked(prompt).map(|i| self.parked[i].tokens.len()).unwrap_or(0);
        let s = self.find_snapshot(prompt).map(|i| self.snapshots[i].tokens.len()).unwrap_or(0);
        p.max(s)
    }

    /// Admit waiting requests while there is room, per the policy.
    fn admit(&mut self) -> Result<()> {
        let cap = self.max_running.min(self.limits().max_seqs);
        let pol = self.policy;
        if !pol.load_control && !pol.srpt {
            // Original behavior: strict FIFO, stop at the first request that does not fit.
            while !self.waiting.is_empty() && self.running.len() < cap {
                if !self.try_admit(0, true)? {
                    break;
                }
            }
            return Ok(());
        }
        let now = Instant::now();
        let aged = |t: Instant| now.duration_since(t).as_millis() as u64 >= pol.max_wait_ms;
        // Candidate order: aged requests first, then by remaining uncached work (SRPT) or FIFO.
        let mut order: Vec<(bool, usize, usize, u64)> = self
            .waiting
            .iter()
            .enumerate()
            .map(|(i, (r, t))| {
                let rem = if pol.srpt { r.prompt.len() - self.cached_len(&r.prompt) } else { 0 };
                (!aged(*t), rem, i, r.id)
            })
            .collect();
        order.sort();
        for (not_aged, _, _, id) in order {
            if self.running.len() >= cap {
                break;
            }
            let Some(wi) = self.waiting.iter().position(|(r, _)| r.id == id) else { continue };
            let hit = self.find_parked(&self.waiting[wi].0.prompt).is_some();
            // Hits may evict freely unless a residency quantum is set (then everyone rotates
            // through the same restricted victims); anyone waiting past max_wait may.
            let hit_ok = hit && pol.quantum == 0;
            let allow_evict = !pol.load_control || hit_ok || !not_aged || self.running.is_empty();
            // Without permission to evict live sessions, sessions presumed finished may still go.
            self.dead_only = !allow_evict;
            let ok = self.try_admit(wi, allow_evict);
            self.dead_only = false;
            if !ok? {
                self.cache.deferred += 1;
            }
        }
        Ok(())
    }

    /// Admit waiting request `wi`: resume a parked session that is a strict prefix of its
    /// prompt, else fork a shared-prefix snapshot, else start fresh. With `allow_evict` false
    /// it only uses free slots and pages plus those of sessions presumed finished (the caller
    /// sets `dead_only` to match). Checks feasibility first and returns false, changing
    /// nothing, when the request cannot fit even after the evictions it is allowed.
    fn try_admit(&mut self, wi: usize, allow_evict: bool) -> Result<bool> {
        let bs = self.limits().block_size;
        let req = &self.waiting[wi].0;
        let need = (req.prompt.len() + req.max_new_tokens).div_ceil(bs);
        let best = self.find_parked(&req.prompt);
        let fork = if best.is_none() { self.find_snapshot(&req.prompt) } else { None };
        {
            // Feasibility before touching anything: free pages, plus (if eviction is allowed)
            // the pages evicting other parked sessions would actually return (refcount 1).
            // Without this, a request that cannot fit evicts the whole cache and then fails.
            let have = match (best, fork) {
                (Some(i), _) => self.parked[i].blocks.len().min(need),
                (None, Some(i)) => self.snapshots[i].blocks.len().min(need),
                _ => 0,
            };
            let mut avail = self.free_blocks.len();
            let mut slots = self.free_slots.len();
            let now = Instant::now();
            for (k, p) in self.parked.iter().enumerate() {
                if Some(k) == best || (!allow_evict && !self.evictable_restricted(k, now)) {
                    continue;
                }
                avail += p.blocks.iter().filter(|&&b| self.refcnt[b as usize] == 1).count();
                slots += 1;
            }
            let slot_ok = best.is_some() || slots > 0;
            // One more page if the resumed entry ends inside a shared page (copied below).
            let cow = match best {
                Some(i) => {
                    let p = &self.parked[i];
                    let n = p.tokens.len();
                    (n % bs != 0 && self.refcnt[p.blocks[n / bs] as usize] > 1) as usize
                }
                None => 0,
            };
            if !slot_ok || avail < need - have + cow {
                return Ok(false);
            }
        }
        if best.is_none() && fork.is_none() && self.session_cache {
            // A running sequence is about to snapshot the prefix this request shares with it:
            // wait for that snapshot and fork it, rather than prefill the same prefix again
            // (agents that start together would otherwise each compute the system prompt).
            let shared = self.shared_prefix(&self.waiting[wi].0.prompt);
            if shared >= SNAPSHOT_MIN_TOKENS && self.running.iter().any(|s| s.snapshot_at == Some(shared)) {
                return Ok(false);
            }
        }
        let mut snapshot_at = None;
        let mut forked = false;
        let mut resident_turns = 0;
        let mut sid = self.waiting[wi].0.id;
        let (slot, mut blocks, computed) = if let Some(i) = best {
            let p = self.parked.swap_remove(i);
            resident_turns = p.turns;
            sid = p.sid;
            // The session's other entries (its checkpoint or its generation end) are stale now.
            self.drop_session(sid);
            (p.slot, p.blocks, p.tokens.len())
        } else {
            let slot = match self.free_slots.pop() {
                Some(s) => s,
                None => {
                    if !self.evict_lru() {
                        return Ok(false);
                    }
                    match self.free_slots.pop() {
                        Some(s) => s,
                        None => return Ok(false),
                    }
                }
            };
            // Eviction may have removed or reordered snapshots: look the fork up again by
            // content, never by a stale index.
            let fork = if fork.is_some() { self.find_snapshot(&self.waiting[wi].0.prompt) } else { None };
            if let Some(i) = fork {
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
                forked = true;
                (slot, blocks, n)
            } else {
                self.model.reset_slot(slot as usize)?;
                if self.session_cache {
                    let shared = self.shared_prefix(&self.waiting[wi].0.prompt);
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
        // Copy-on-write: the sequence will write the rest of the page that holds position
        // `computed`. If another owner still references that page, continue on a private copy.
        let cow = computed % bs != 0 && self.refcnt[blocks[computed / bs] as usize] > 1;
        let more = need - blocks.len() + cow as usize;
        match self.take_blocks(more) {
            Some(mut b) => {
                if cow {
                    let fresh = b.pop().unwrap();
                    let old = std::mem::replace(&mut blocks[computed / bs], fresh);
                    self.model.copy_page(fresh as usize, old as usize)?;
                    self.release(vec![old]);
                }
                blocks.extend(b)
            }
            None => {
                // Not enough pages even after evicting everything parked: put the session
                // back (or release the fresh slot) and wait for running sequences to finish.
                if best.is_some() {
                    let tokens = self.waiting[wi].0.prompt[..computed].to_vec();
                    self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns: resident_turns, sid });
                } else {
                    self.free_slots.push(slot);
                    self.release(blocks);
                }
                return Ok(false);
            }
        }
        let (req, _) = self.waiting.remove(wi).unwrap();
        let ckpt_at = if self.session_cache && self.prompt_checkpoints {
            Some(req.prompt.len().saturating_sub(CKPT_BACKOFF) / bs * bs).filter(|&at| at > computed)
        } else {
            None
        };
        self.cache.admitted += 1;
        if forked {
            self.cache.forks += 1;
        }
        if computed > 0 {
            self.cache.hits += 1;
            self.cache.reused_tokens += computed as u64;
        }
        self.cache.prefilled_tokens += (req.prompt.len() - computed) as u64;
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed, snapshot_at, spare: None, in_prefill: false, cancelled: false, draft: None, turns: resident_turns, sid, ckpt_at });
        Ok(true)
    }

    /// Prompt-lookup draft for running sequence `i`: what followed the most recent earlier
    /// occurrence of the longest suffix n-gram of its known tokens (see [`DraftIndex`]).
    fn draft(&mut self, i: usize, k: usize) -> Vec<u32> {
        let (min_n, max_n) = (self.spec_min_ngram, self.spec_max_ngram);
        let s = &mut self.running[i];
        let mut idx = match s.draft.take() {
            Some(idx) if idx.matches(min_n, max_n) => idx,
            _ => DraftIndex::new(min_n, max_n),
        };
        for p in idx.len()..s.known() {
            idx.push(s.token_at(p));
        }
        let d = idx.draft(k);
        s.draft = Some(idx);
        d
    }

    fn ensure_blocks(&mut self, si: usize, upto: usize) -> Result<()> {
        let bs = self.limits().block_size;
        while self.running[si].blocks.len() * bs < upto {
            let Some(b) = self.free_blocks.pop() else { bail!("KV page pool exhausted") };
            self.refcnt[b as usize] = 1;
            self.running[si].blocks.push(b);
        }
        Ok(())
    }

    /// Build the batch for `plan` (running index, tokens to feed). Returns the batch and the
    /// running indices that sample a token from it, in sample-row order.
    fn build_batch(&mut self, plan: &[(usize, usize)]) -> Result<(Batch, Vec<usize>)> {
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
            if upto == s.known() {
                b.sample_rows.push(b.tokens.len() as u32 - 1);
                sampled.push(i);
            }
        }
        Ok((b, sampled))
    }

    /// Append sampled tokens, emit events; returns running indices that finished.
    fn apply_sampled(&mut self, sampled: &[usize], toks: &[u32], events: &mut Vec<TokenEvent>) -> Vec<usize> {
        let mut done = Vec::new();
        for (&i, &tok) in sampled.iter().zip(toks) {
            let s = &mut self.running[i];
            s.out.push(tok);
            let stop = !s.req.ignore_eos && self.eos.contains(&tok);
            let finished = stop || s.out.len() >= s.req.max_new_tokens;
            events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop });
            if finished {
                done.push(i);
            }
        }
        done
    }

    /// Park (session cache) or free finished sequences, given their running indices.
    fn retire(&mut self, mut done: Vec<usize>) {
        let bs = self.limits().block_size;
        done.sort_unstable_by(|a, b| b.cmp(a));
        done.dedup();
        for i in done {
            let mut s = self.running.swap_remove(i);
            if let Some(sp) = s.spare {
                self.free_slots.push(sp);
            }
            // Nothing computed (cancelled before its first chunk): nothing worth keeping.
            if self.session_cache && s.computed > 0 {
                let tokens = (0..s.computed).map(|k| s.token_at(k)).collect();
                // Admission reserved pages for the whole token limit. Keep only the pages that
                // hold computed tokens: a request that stops early (EOS well before a large
                // max_tokens) must not pin the unused reservation while it is parked.
                let keep = s.computed.div_ceil(bs).min(s.blocks.len());
                let extra = s.blocks.split_off(keep);
                self.release(extra);
                self.parked.push(Parked { tokens, slot: s.slot, blocks: s.blocks, last_used: self.clock, parked_at: Instant::now(), turns: s.turns + 1, sid: s.sid });
            } else {
                self.free_slots.push(s.slot);
                self.release(s.blocks);
            }
        }
    }

    fn index_of(&self, id: u64) -> usize {
        self.running.iter().position(|s| s.req.id == id).expect("in-flight sequence missing")
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
    fn step_coexec(&mut self) -> Result<Vec<TokenEvent>> {
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
            for (i, s) in self.running.iter().enumerate() {
                if budget == 0 {
                    break;
                }
                if !s.in_prefill && s.computed < s.req.prompt.len() {
                    let mut n = (s.req.prompt.len() - s.computed).min(budget);
                    for at in [s.snapshot_at, s.ckpt_at].into_iter().flatten().filter(|&at| at > s.computed) {
                        n = n.min(at - s.computed);
                    }
                    plan.push((i, n));
                    budget -= n;
                }
            }
            if !plan.is_empty() {
                let (b, sampled) = self.build_batch(&plan)?;
                let ns = self.model.launch(&b)?;
                prefill_tokens = b.tokens.len();
                for &(i, _) in &plan {
                    self.running[i].in_prefill = true;
                }
                self.inflight = Some(Inflight {
                    plan: plan.iter().map(|&(i, n)| (self.running[i].req.id, n)).collect(),
                    sampled: sampled.iter().map(|&i| self.running[i].req.id).collect(),
                    ns,
                    tokens: b.tokens.len(),
                });
            }
        }

        // 2. One decode step on lane 1 for every ready sequence outside the in-flight batch.
        let dplan: Vec<(usize, usize)> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.in_prefill && s.computed >= s.req.prompt.len() && s.known() > s.computed)
            // Normally one pending token; more after a rejected speculative step (spec_k
            // switched off since), which the decode path also handles.
            .map(|(i, s)| (i, (s.known() - s.computed).min(MAX_DECODE_ROWS_PER_SEQ)))
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
            let (mut b, sampled) = self.build_batch(&dplan)?;
            b.n_decode = decode_seqs;
            self.model.set_lane(1);
            let toks = self.model.step(&b, None);
            self.model.set_lane(0);
            let toks = toks?;
            for &(i, n) in &dplan {
                self.running[i].computed += n;
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
    fn collect_inflight(&mut self, events: &mut Vec<TokenEvent>) -> Result<usize> {
        let Some(inf) = self.inflight.take() else { return Ok(0) };
        let toks = self.model.collect(inf.ns)?;
        for &(id, n) in &inf.plan {
            let i = self.index_of(id);
            self.running[i].computed += n;
            self.running[i].in_prefill = false;
        }
        let idx: Vec<usize> = inf.plan.iter().map(|&(id, _)| self.index_of(id)).collect();
        self.after_step(idx.into_iter())?;
        let sampled: Vec<usize> = inf.sampled.iter().map(|&id| self.index_of(id)).collect();
        let mut done = self.apply_sampled(&sampled, &toks, events);
        done.extend(inf.plan.iter().map(|&(id, _)| self.index_of(id)).filter(|&i| self.running[i].cancelled));
        self.retire(done);
        Ok(inf.tokens)
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
                    d = self.draft(i, kmax);
                }
            }
            if !d.is_empty() && self.running[i].spare.is_none() {
                // A spare slot for the speculative state: a free one, else the least-recently
                // used parked session's (never a shared-prefix snapshot's).
                let slot = match self.free_slots.pop() {
                    Some(x) => Some(x),
                    None => {
                        if self.evict_lru_parked() { self.free_slots.pop() } else { None }
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
            // Frame pacing: size the prefill part so the whole step fits the ITL target,
            // from online estimates of the decode cost and the per-token prefill cost.
            if let Some(target) = self.itl_target_us {
                let room = target - self.pace.decode_us.min(target * 0.8);
                let tokens = (room / self.pace.prefill_us_per_tok.max(1e-3)) as usize;
                budget = budget.min(tokens.max(PACE_MIN_CHUNK));
            }
        }
        // Prefill order: admission order, or shortest remaining prefill first (SRPT).
        let mut prefill: Vec<usize> = (0..self.running.len()).filter(|&i| self.running[i].computed < self.running[i].req.prompt.len()).collect();
        if self.policy.srpt {
            prefill.sort_by_key(|&i| self.running[i].req.prompt.len() - self.running[i].computed);
        }
        for i in prefill {
            let s = &self.running[i];
            if budget == 0 {
                break;
            }
            {
                let mut n = (s.req.prompt.len() - s.computed).min(budget);
                for at in [s.snapshot_at, s.ckpt_at].into_iter().flatten().filter(|&at| at > s.computed) {
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
        self.cache.host_us += host_us;
        self.cache.steps += 1;
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;
        // Update the pacing model: pure-decode steps give the decode cost, mixed steps give
        // the per-token prefill cost on top of it.
        let prefill_tokens = b.tokens.len() - decode_seqs;
        if prefill_tokens == 0 {
            self.pace.decode_us = 0.9 * self.pace.decode_us + 0.1 * gpu_us as f64;
        } else if prefill_tokens >= 64 {
            let base = if decode_seqs > 0 { self.pace.decode_us } else { 0.0 };
            let per = ((gpu_us as f64 - base).max(0.0)) / prefill_tokens as f64;
            self.pace.prefill_us_per_tok = 0.8 * self.pace.prefill_us_per_tok + 0.2 * per;
        }

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
            if finished {
                done.push(i);
            }
        }
        self.spec = st;
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
