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

use crate::model::{Batch, Limits, Model};
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
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub evictions: u64,
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
    clock: u64,
    eos: Vec<u32>,
    pub last: StepStats,
    pub cache: CacheStats,
    /// Keep finished sequences for prefix reuse (on by default).
    pub session_cache: bool,
    /// Max sequences running at once; state slots beyond this hold parked sessions.
    pub max_running: usize,
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
            clock: 0,
            eos,
            last: StepStats::default(),
            cache: CacheStats::default(),
            session_cache: true,
            max_running: lim.max_seqs,
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

    pub fn num_parked(&self) -> usize {
        self.parked.len()
    }

    /// Drop the least-recently-used parked session, returning its slot and pages.
    fn evict_lru(&mut self) -> bool {
        let Some(i) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used) else { return false };
        let p = self.parked.swap_remove(i);
        self.free_slots.push(p.slot);
        self.free_blocks.extend(p.blocks);
        self.cache.evictions += 1;
        true
    }

    /// Take `n` free KV pages, evicting parked sessions as needed.
    fn take_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        while self.free_blocks.len() < n {
            if !self.evict_lru() {
                return None;
            }
        }
        Some(self.free_blocks.split_off(self.free_blocks.len() - n))
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
            self.model.reset_slot(slot as usize)?;
            (slot, Vec::new(), 0)
        };
        if blocks.len() > need {
            self.free_blocks.extend(blocks.split_off(need));
        }
        let more = need - blocks.len();
        match self.take_blocks(more) {
            Some(b) => blocks.extend(b),
            None => {
                // Not enough pages even after evicting everything parked: put the session
                // back (or release the fresh slot) and wait for running sequences to finish.
                if computed > 0 {
                    let tokens = self.waiting.front().unwrap().prompt[..computed].to_vec();
                    self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock });
                } else {
                    self.free_slots.push(slot);
                    self.free_blocks.extend(blocks);
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
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed });
        Ok(true)
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

        // Decode first (1 token each), then prefill chunks in admission order.
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (running index, n tokens)
        for (i, s) in self.running.iter().enumerate() {
            if s.computed >= s.req.prompt.len() && s.known() - s.computed == 1 && budget > 0 {
                plan.push((i, 1));
                budget -= 1;
            }
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
                let n = (s.req.prompt.len() - s.computed).min(budget);
                plan.push((i, n));
                budget -= n;
            }
        }
        if plan.is_empty() {
            return Ok(Vec::new());
        }

        let mb = lim.max_blocks_per_seq();
        let mut b = Batch::default();
        let mut sampled: Vec<usize> = Vec::new(); // running indices that sample this step
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
        b.n_decode = decode_seqs;
        let host_us = t0.elapsed().as_micros() as u64;
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;

        for &(i, n) in &plan {
            self.running[i].computed += n;
        }
        let mut events = Vec::with_capacity(sampled.len());
        let mut done = Vec::new();
        for (&i, &tok) in sampled.iter().zip(&toks) {
            let s = &mut self.running[i];
            s.out.push(tok);
            let finished = s.out.len() >= s.req.max_new_tokens || (!s.req.ignore_eos && self.eos.contains(&tok));
            events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished });
            if finished {
                done.push(i);
            }
        }
        done.sort_unstable_by(|a, b| b.cmp(a));
        for i in done {
            let s = self.running.swap_remove(i);
            if self.session_cache {
                let tokens = (0..s.computed).map(|k| s.token_at(k)).collect();
                self.parked.push(Parked { tokens, slot: s.slot, blocks: s.blocks, last_used: self.clock });
            } else {
                self.free_slots.push(s.slot);
                self.free_blocks.extend(s.blocks);
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
