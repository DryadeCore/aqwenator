use super::*;

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

    /// Admit waiting request `wi`: resume a parked session that is a strict prefix of its
    /// prompt, else fork a shared-prefix snapshot, else start fresh. With `allow_evict` false
    /// it only uses free slots and pages plus those of sessions presumed finished (the caller
    /// sets `dead_only` to match). Checks feasibility first and returns false, changing
    /// nothing, when the request cannot fit even after the evictions it is allowed.
    pub(super) fn try_admit(&mut self, wi: usize, allow_evict: bool) -> Result<bool> {
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
            let mut avail = self.pool.pages.len();
            let mut slots = self.pool.slots.len();
            let now = Instant::now();
            // A page comes back when every reference to it is held by an evictable parked
            // entry (a session's checkpoint and its generation end share the prompt's pages).
            // Counted only when the free pages alone do not cover the request.
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
            let slot_ok = best.is_some() || slots > 0;
            // One more page if the resumed entry ends inside a shared page (copied below).
            let cow = match best {
                Some(i) => {
                    let p = &self.parked[i];
                    let n = p.tokens.len();
                    (n % bs != 0 && self.pool.refs(p.blocks[n / bs]) > 1) as usize
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
            resident_turns = self.parked[i].turns;
            sid = self.parked[i].sid;
            let n = self.parked[i].tokens.len();
            // A turn too short to produce a checkpoint of its own must not consume the one it
            // resumes from, or the session's next turn would find nothing: fork it instead
            // (copy the state, share the pages) when a slot is free.
            let prompt_len = self.waiting[wi].0.prompt.len();
            let no_new_ckpt = self.prompt_checkpoints && prompt_len.saturating_sub(CKPT_BACKOFF) / bs * bs <= n;
            if no_new_ckpt && !self.pool.slots.is_empty() {
                let slot = self.pool.slots.pop().unwrap();
                let (src, blocks) = (self.parked[i].slot as usize, self.parked[i].blocks.clone());
                self.parked[i].last_used = self.clock;
                self.model.copy_slot(slot as usize, src)?;
                self.pool.share(&blocks);
                // Drop the session's other entries, keep this one.
                let keep = std::mem::replace(&mut self.parked[i].sid, u64::MAX);
                self.drop_session(keep);
                let k = self.parked.iter().position(|p| p.sid == u64::MAX).unwrap();
                self.parked[k].sid = keep;
                (slot, blocks, n)
            } else {
                let p = self.parked.swap_remove(i);
                // The session's other entries (its checkpoint or its generation end) are stale now.
                self.drop_session(sid);
                (p.slot, p.blocks, n)
            }
        } else {
            let slot = match self.pool.slots.pop() {
                Some(s) => s,
                None => {
                    if !self.evict_lru() {
                        return Ok(false);
                    }
                    match self.pool.slots.pop() {
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
                self.pool.share(&blocks);
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
            self.pool.release(extra);
        }
        // Copy-on-write: the sequence will write the rest of the page that holds position
        // `computed`. If another owner still references that page, continue on a private copy.
        let cow = computed % bs != 0 && self.pool.refs(blocks[computed / bs]) > 1;
        let more = need - blocks.len() + cow as usize;
        match self.take_blocks(more) {
            Some(mut b) => {
                if cow {
                    let fresh = b.pop().unwrap();
                    let old = std::mem::replace(&mut blocks[computed / bs], fresh);
                    self.model.copy_page(fresh as usize, old as usize)?;
                    self.pool.release(vec![old]);
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
                    self.pool.slots.push(slot);
                    self.pool.release(blocks);
                }
                return Ok(false);
            }
        }
        let (req, t_add) = self.waiting.remove(wi).unwrap();
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
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed, snapshot_at, spare: None, in_prefill: false, cancelled: false, draft: None, turns: resident_turns, sid, ckpt_at, t_add, t_admit: Instant::now(), steps: 0, reused: computed, k_cur: 0, mtp_next: None });
        Ok(true)
    }
}
