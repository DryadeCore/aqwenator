use super::*;

/// Where an admitted request starts from.
struct Start {
    slot: u32,
    blocks: Vec<u32>,
    /// Tokens the slot's state and the pages already cover.
    computed: usize,
    sid: u64,
    turns: u32,
    snapshot_at: Option<usize>,
    forked: bool,
}

impl Engine {
    pub(super) fn admit(&mut self) -> Result<()> {
        self.reserve_slots();
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

    /// Whether a request needing `need` pages can be admitted: free pages and slots, plus
    /// what evicting parked entries would return (all of them, or with `allow_evict` false
    /// only those a restricted admission may evict). Without this check a request that cannot
    /// fit would evict the whole cache and then fail.
    fn fits(&self, need: usize, best: Option<usize>, fork: Option<usize>, allow_evict: bool) -> bool {
        let bs = self.limits().block_size;
        let have = match (best, fork) {
            (Some(i), _) => self.parked[i].blocks.len().min(need),
            (None, Some(i)) => self.snapshots[i].blocks.len().min(need),
            _ => 0,
        };
        let mut avail = self.pool.pages.len();
        let mut slots = self.pool.slots.len();
        let now = Instant::now();
        // A page comes back only when every owner is an evictable parked entry (a session's
        // checkpoint and its generation end share the prompt's pages). Counted only when the
        // free pages do not cover the request.
        let short = avail < need - have + 1;
        let mut held: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for (k, p) in self.parked.iter().enumerate() {
            if Some(k) == best || (!allow_evict && !self.evictable_restricted(k, now)) {
                continue;
            }
            if short {
                for &b in &p.blocks {
                    *held.entry(b).or_insert(0) += 1;
                }
            }
            slots += 1;
        }
        avail += held.iter().filter(|&(&b, &n)| self.pool.refs(b) == n).count();
        // One more page if the resumed entry ends inside a shared page (see `own_pages`).
        let cow = best.is_some_and(|i| {
            let p = &self.parked[i];
            let n = p.tokens.len();
            n % bs != 0 && self.pool.refs(p.blocks[n / bs]) > 1
        });
        (best.is_some() || slots > 0) && avail >= need - have + cow as usize
    }

    /// Resume parked entry `i` for a prompt of `prompt_len` tokens.
    fn resume(&mut self, i: usize, prompt_len: usize) -> Result<Start> {
        let bs = self.limits().block_size;
        let (turns, sid, computed) = (self.parked[i].turns, self.parked[i].sid, self.parked[i].tokens.len());
        // A turn too short to leave a checkpoint of its own must not consume the one it
        // resumes from, or the session's next turn finds nothing: copy it when a slot is free.
        let no_new_ckpt = self.prompt_checkpoints && prompt_len.saturating_sub(CKPT_BACKOFF) / bs * bs <= computed;
        let (slot, blocks) = match no_new_ckpt.then(|| self.pool.slots.pop()).flatten() {
            Some(slot) => {
                let (src, blocks) = (self.parked[i].slot as usize, self.parked[i].blocks.clone());
                self.parked[i].last_used = self.clock;
                self.model.copy_slot(slot as usize, src)?;
                self.pool.share(&blocks);
                // The session's other entries are stale: drop them, keep this one.
                self.parked[i].sid = u64::MAX;
                self.drop_session(sid);
                let k = self.parked.iter().position(|p| p.sid == u64::MAX).unwrap();
                self.parked[k].sid = sid;
                (slot, blocks)
            }
            None => {
                let p = self.parked.swap_remove(i);
                self.drop_session(sid);
                (p.slot, p.blocks)
            }
        };
        Ok(Start { slot, blocks, computed, sid, turns, snapshot_at: None, forked: false })
    }

    /// Start in a free slot: from the shared-prefix snapshot of `prompt` if there is one,
    /// else from nothing. `None` when no slot can be had.
    fn start(&mut self, prompt: &[u32], id: u64, fork: bool) -> Result<Option<Start>> {
        if self.pool.slots.is_empty() && !self.evict_lru() {
            return Ok(None);
        }
        let Some(slot) = self.pool.slots.pop() else { return Ok(None) };
        let mut st = Start { slot, blocks: Vec::new(), computed: 0, sid: id, turns: 0, snapshot_at: None, forked: false };
        // The eviction above may have removed or reordered snapshots: look it up by content.
        if let Some(i) = self.find_snapshot(prompt).filter(|_| fork) {
            let sn = &mut self.snapshots[i];
            sn.last_used = self.clock;
            sn.forks += 1;
            let src = sn.slot as usize;
            (st.blocks, st.computed, st.forked) = (sn.blocks.clone(), sn.tokens.len(), true);
            self.model.copy_slot(slot as usize, src)?;
            self.pool.share(&st.blocks);
        } else {
            self.model.reset_slot(slot as usize)?;
            if self.session_cache {
                st.snapshot_at = Some(self.shared_prefix(prompt)).filter(|&n| n >= SNAPSHOT_MIN_TOKENS);
            }
        }
        Ok(Some(st))
    }

    /// Give `st` the `need` pages of its request: drop surplus ones, take the missing ones
    /// (evicting as allowed), and continue on a private copy of the page that holds position
    /// `computed` if another owner still references it. False, with `st` unchanged apart from
    /// the surplus, when the pages cannot be had.
    fn own_pages(&mut self, st: &mut Start, need: usize) -> Result<bool> {
        let bs = self.limits().block_size;
        if st.blocks.len() > need {
            let extra = st.blocks.split_off(need);
            self.pool.release(extra);
        }
        let at = st.computed / bs;
        let cow = st.computed % bs != 0 && self.pool.refs(st.blocks[at]) > 1;
        let Some(mut fresh) = self.take_blocks(need - st.blocks.len() + cow as usize) else { return Ok(false) };
        if cow {
            let copy = fresh.pop().unwrap();
            let old = std::mem::replace(&mut st.blocks[at], copy);
            self.model.copy_page(copy as usize, old as usize)?;
            self.pool.release([old]);
        }
        st.blocks.extend(fresh);
        Ok(true)
    }

    /// Admit waiting request `wi`: resume a parked session that is a strict prefix of its
    /// prompt, else fork a shared-prefix snapshot, else start fresh. With `allow_evict` false
    /// it only uses free slots and pages plus those of sessions presumed finished (the caller
    /// sets `dead_only` to match). Returns false, with the cache as it was, when the request
    /// cannot fit.
    pub(super) fn try_admit(&mut self, wi: usize, allow_evict: bool) -> Result<bool> {
        let bs = self.limits().block_size;
        let req = &self.waiting[wi].0;
        let need = (req.prompt.len() + req.max_new_tokens).div_ceil(bs);
        let best = self.find_parked(&req.prompt);
        let fork = if best.is_none() { self.find_snapshot(&req.prompt) } else { None };
        if !self.fits(need, best, fork, allow_evict) {
            return Ok(false);
        }
        if best.is_none() && fork.is_none() && self.session_cache {
            // A running sequence is about to snapshot the prefix this request shares with it:
            // wait and fork it rather than prefill the same prefix again (agents that start
            // together would otherwise each compute the system prompt).
            let shared = self.shared_prefix(&req.prompt);
            if shared >= SNAPSHOT_MIN_TOKENS && self.running.iter().any(|s| s.snapshot_at == Some(shared)) {
                return Ok(false);
            }
        }
        let (prompt, id) = (req.prompt.clone(), req.id);
        let mut st = match best {
            Some(i) => self.resume(i, prompt.len())?,
            None => match self.start(&prompt, id, fork.is_some())? {
                Some(st) => st,
                None => return Ok(false),
            },
        };
        if !self.own_pages(&mut st, need)? {
            // Wait for running sequences to finish: the session goes back as it was.
            if best.is_some() {
                let tokens = prompt[..st.computed].to_vec();
                self.parked.push(Parked { tokens, slot: st.slot, blocks: st.blocks, last_used: self.clock, parked_at: Instant::now(), turns: st.turns, sid: st.sid });
            } else {
                self.pool.slots.push(st.slot);
                self.pool.release(st.blocks);
            }
            return Ok(false);
        }
        let (req, t_add) = self.waiting.remove(wi).unwrap();
        let Start { slot, blocks, computed, sid, turns, snapshot_at, forked } = st;
        let ckpt_at = Some(req.prompt.len().saturating_sub(CKPT_BACKOFF) / bs * bs).filter(|&at| self.session_cache && self.prompt_checkpoints && at > computed);
        self.cache.admitted += 1;
        self.cache.forks += forked as u64;
        if computed > 0 {
            self.cache.hits += 1;
            self.cache.reused_tokens += computed as u64;
        }
        self.cache.prefilled_tokens += (req.prompt.len() - computed) as u64;
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed, snapshot_at, spare: None, in_prefill: false, cancelled: false, draft: None, turns, sid, ckpt_at, t_add, t_admit: Instant::now(), steps: 0, reused: computed, k_cur: 0, mtp_next: None });
        Ok(true)
    }
}
