use super::*;

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
    /// Draft tokens that came from the cross-session model (the rest: prompt lookup).
    pub global_drafted: u64,
    pub global_accepted: u64,
    /// Drafts whose first token came from the MTP head, and how many of those were accepted.
    pub mtp_drafted: u64,
    pub mtp_accepted: u64,
    /// Requests that stopped on a token inside a fully accepted draft.
    pub stop_in_draft: u64,
}

impl Engine {
    /// Prompt-lookup draft for running sequence `i`: what followed the most recent earlier
    /// occurrence of the longest suffix n-gram of its known tokens (see [`DraftIndex`]).
    /// Returns the draft and how many of its leading tokens came from the prompt lookup; the
    /// rest were added by the cross-session model.
    pub(super) fn draft(&mut self, i: usize, k: usize) -> (Vec<u32>, usize, bool) {
        let (min_n, max_n) = (self.spec_min_ngram, self.spec_max_ngram);
        let s = &mut self.running[i];
        let mut idx = match s.draft.take() {
            Some(idx) if idx.matches(min_n, max_n) => idx,
            _ => DraftIndex::new(min_n, max_n),
        };
        for p in idx.len()..s.known() {
            idx.push(s.token_at(p));
        }
        let mut d = idx.draft(k);
        s.draft = Some(idx);
        let mut local = d.len();
        // The MTP head's prediction for the next token outranks a lookup that disagrees with
        // it: the head has seen the whole context, the lookup only the last few tokens.
        let n_known = s.known();
        let mtp = s.mtp_next.as_ref().filter(|(at, t)| *at == n_known && !t.is_empty()).map(|(_, t)| t);
        if let Some(t) = mtp {
            if d.first() != Some(&t[0]) {
                d.clear();
                d.extend_from_slice(&t[..t.len().min(k)]);
                local = 0;
            }
        }
        let from_mtp = mtp.is_some();
        if self.global_draft && d.len() < k {
            let n = s.known();
            let tail: Vec<u32> = (n.saturating_sub(4)..n).map(|p| s.token_at(p)).collect();
            self.global.extend(&tail, &mut d, k);
        }
        (d, local, from_mtp)
    }
}
