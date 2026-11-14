//! Prompt-lookup draft index for speculative decoding.
//!
//! A draft is the continuation of the most recent earlier occurrence of the sequence's longest
//! suffix n-gram (`max_n` down to `min_n` tokens). The engine used to find it by scanning the
//! whole context backwards on every decode step, once per n-gram length: at 32k tokens with
//! no match that is ~100k token compares per sequence per step, which is milliseconds of host
//! time per step at 16+ sequences, i.e. comparable to the GPU step itself.
//!
//! This index makes a lookup O(1) amortized, LZ77-style: a hash table maps the `min_n` tokens
//! that end at a position to the most recent such position, and `prev` chains older ones.
//! Every earlier occurrence of an n-gram suffix (n >= `min_n`) is on the chain of the current
//! suffix, so a lookup walks that chain most-recent-first and extends each candidate
//! backwards to measure its match length.
//!
//! Drafts never affect outputs (every draft is verified by the model); the index only decides
//! how many tokens a step can accept. Std only, so it is unit-testable without CUDA.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Candidates examined per lookup. Lookups stop early at a full-length (`max_n`) match, so
/// the cap only matters when a short n-gram repeats very often without ever extending.
const MAX_PROBES: usize = 256;
const NONE: u32 = u32::MAX;

/// The keys are already mixed 64-bit hashes; hashing them again is wasted work.
#[derive(Default)]
struct Passthrough(u64);

impl Hasher for Passthrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed")
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

pub struct DraftIndex {
    toks: Vec<u32>,
    /// Hash of the `min_n` tokens ending at `e` (exclusive) -> the most recent such `e`.
    head: HashMap<u64, u32, BuildHasherDefault<Passthrough>>,
    /// `prev[e]`: the previous end position with the same hash as `e`, or `NONE`.
    prev: Vec<u32>,
    min_n: usize,
    max_n: usize,
}

impl DraftIndex {
    pub fn new(min_n: usize, max_n: usize) -> Self {
        Self { toks: Vec::new(), head: HashMap::default(), prev: Vec::new(), min_n: min_n.max(1), max_n }
    }

    /// Tokens indexed so far.
    pub fn len(&self) -> usize {
        self.toks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Whether this index was built for these n-gram bounds.
    pub fn matches(&self, min_n: usize, max_n: usize) -> bool {
        self.min_n == min_n.max(1) && self.max_n == max_n
    }

    /// Hash of the `min_n` tokens ending at `end` (exclusive).
    fn key(&self, end: usize) -> u64 {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;
        for &t in &self.toks[end - self.min_n..end] {
            h = (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01B3);
            h ^= h >> 29;
        }
        h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Append the next token of the sequence.
    pub fn push(&mut self, tok: u32) {
        // The n-gram ending at the current end becomes an *earlier* occurrence only now that
        // a token follows it, so it is registered here rather than when its last token came.
        let end = self.toks.len();
        self.prev.push(NONE); // prev[end]
        if end >= self.min_n {
            let k = self.key(end);
            if let Some(old) = self.head.insert(k, end as u32) {
                self.prev[end] = old;
            }
        }
        self.toks.push(tok);
    }

    /// Up to `k` draft tokens: what followed the most recent earlier occurrence of the longest
    /// suffix n-gram (`max_n` down to `min_n`). Empty when there is none.
    pub fn draft(&self, k: usize) -> Vec<u32> {
        let len = self.toks.len();
        if k == 0 || self.max_n < self.min_n || len <= self.min_n {
            return Vec::new();
        }
        let Some(&first) = self.head.get(&self.key(len)) else { return Vec::new() };
        let (mut best_e, mut best_n) = (0usize, 0usize);
        let mut e = first;
        let mut probes = 0;
        while e != NONE && probes < MAX_PROBES {
            let end = e as usize;
            // Match length of the n-gram ending at `end` against the suffix, capped at max_n.
            // It has to be checked from the first token: the chain links equal hashes only.
            let mut n = 0;
            while n < self.max_n && n < end && self.toks[end - 1 - n] == self.toks[len - 1 - n] {
                n += 1;
            }
            if n >= self.min_n && n > best_n {
                (best_e, best_n) = (end, n);
                if n == self.max_n {
                    break;
                }
            }
            e = self.prev[end];
            probes += 1;
        }
        if best_n == 0 {
            return Vec::new();
        }
        self.toks[best_e..(best_e + k).min(len)].to_vec()
    }
}

/// Cross-session next-token model for drafting text a sequence has not produced before but
/// the server has: tool-call syntax, section headers, stock phrases. Agents repeat these
/// across turns and sessions, and the per-sequence prompt lookup cannot draft them the first
/// time they appear in a session.
///
/// For every context of `min_n..=max_n` generated tokens it keeps a majority-vote candidate
/// for the token that follows (Boyer-Moore: the candidate's lead over everything else), so
/// memory is one entry per distinct context. A prediction is offered only when the candidate
/// leads by `MIN_LEAD` and has won most of its contests, because a draft row is not free: at
/// long context it costs KV traffic for every sequence in the step.
pub struct NgramModel {
    /// One table per context length, `min_n` first.
    tables: Vec<HashMap<u64, Vote, BuildHasherDefault<Passthrough>>>,
    min_n: usize,
    max_n: usize,
}

#[derive(Clone, Copy)]
struct Vote {
    tok: u32,
    /// The candidate's lead (majority-vote counter).
    lead: u32,
    /// Times this context was seen, and times the current candidate followed it.
    seen: u32,
    hits: u32,
}

const MIN_LEAD: u32 = 2;

impl NgramModel {
    pub fn new(min_n: usize, max_n: usize) -> Self {
        let min_n = min_n.max(1);
        Self { tables: (min_n..=max_n.max(min_n)).map(|_| HashMap::default()).collect(), min_n, max_n: max_n.max(min_n) }
    }

    fn key(ctx: &[u32]) -> u64 {
        let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ ctx.len() as u64;
        for &t in ctx {
            h = (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01B3);
            h ^= h >> 29;
        }
        h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Record that `next` followed `ctx` (the tokens before it; only the last `max_n` matter).
    pub fn observe(&mut self, ctx: &[u32], next: u32) {
        for n in self.min_n..=self.max_n.min(ctx.len()) {
            let v = self.tables[n - self.min_n].entry(Self::key(&ctx[ctx.len() - n..])).or_insert(Vote { tok: next, lead: 0, seen: 0, hits: 0 });
            v.seen += 1;
            if v.tok == next {
                v.lead += 1;
                v.hits += 1;
            } else if v.lead == 0 {
                *v = Vote { tok: next, lead: 1, seen: v.seen, hits: 1 };
            } else {
                v.lead -= 1;
            }
        }
    }

    /// The token expected after `ctx`, from the longest context that has a confident vote.
    pub fn predict(&self, ctx: &[u32]) -> Option<u32> {
        for n in (self.min_n..=self.max_n.min(ctx.len())).rev() {
            if let Some(v) = self.tables[n - self.min_n].get(&Self::key(&ctx[ctx.len() - n..])) {
                // A longer context that has been seen decides, whether or not it is confident:
                // falling back to a shorter one would ignore what the longer one learned.
                return (v.lead >= MIN_LEAD && 2 * v.hits > v.seen).then_some(v.tok);
            }
        }
        None
    }

    /// Extend `draft` (which follows `tail`, the sequence's last tokens) to at most `k` tokens.
    pub fn extend(&self, tail: &[u32], draft: &mut Vec<u32>, k: usize) {
        let mut ctx: Vec<u32> = tail[tail.len().saturating_sub(self.max_n)..].to_vec();
        ctx.extend_from_slice(draft);
        while draft.len() < k {
            let from = ctx.len().saturating_sub(self.max_n);
            let Some(t) = self.predict(&ctx[from..]) else { break };
            draft.push(t);
            ctx.push(t);
        }
    }

    pub fn contexts(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan the engine used before this index existed (the specification).
    fn reference(toks: &[u32], k: usize, min_n: usize, max_n: usize) -> Vec<u32> {
        let n_known = toks.len();
        let mut n = max_n.min(n_known.saturating_sub(1));
        while n >= min_n.max(1) {
            let suf0 = n_known - n;
            let mut start = suf0;
            while start > 0 {
                start -= 1;
                if (0..n).all(|j| toks[start + j] == toks[suf0 + j]) {
                    let from = start + n;
                    let to = (from + k).min(n_known);
                    if to > from {
                        return toks[from..to].to_vec();
                    }
                }
            }
            n -= 1;
        }
        Vec::new()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self, m: u32) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % m as u64) as u32
        }
    }

    #[test]
    fn matches_the_backward_scan() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        // Small vocabularies force repeats, overlaps and long chains; large ones force misses.
        for &(vocab, len) in &[(2u32, 300usize), (3, 400), (7, 600), (50, 800), (100_000, 500)] {
            for &(min_n, max_n) in &[(2usize, 4usize), (1, 4), (2, 2), (3, 6), (1, 1)] {
                let mut idx = DraftIndex::new(min_n, max_n);
                let mut toks = Vec::new();
                for _ in 0..len {
                    let t = rng.next(vocab);
                    toks.push(t);
                    idx.push(t);
                    for k in [1usize, 4, 8] {
                        assert_eq!(idx.draft(k), reference(&toks, k, min_n, max_n), "vocab {vocab} n {min_n}..{max_n} len {} k {k}", toks.len());
                    }
                }
            }
        }
    }

    #[test]
    fn ngram_model_learns_boilerplate_and_stays_quiet_when_unsure() {
        let mut m = NgramModel::new(2, 4);
        let call = [900u32, 901, 902, 903, 904, 905];
        // The same tool-call header generated in three sessions, after different text.
        for prefix in [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]] {
            let seq: Vec<u32> = prefix.iter().chain(&call).copied().collect();
            for i in 1..seq.len() {
                m.observe(&seq[..i], seq[i]);
            }
        }
        // A fourth session reaches the start of the header: the rest is drafted.
        let mut d = Vec::new();
        m.extend(&[50, 51, 900, 901], &mut d, 4);
        assert_eq!(d, vec![902, 903, 904, 905]);
        // After the header nothing was ever generated: no draft.
        let mut d = Vec::new();
        m.extend(&[903, 904, 905], &mut d, 4);
        assert!(d.is_empty());
        // A context seen once is not trusted.
        let mut d = Vec::new();
        m.extend(&[1, 2, 3], &mut d, 4);
        assert!(d.is_empty());
        // A context followed by different tokens each time is not trusted either.
        let mut m = NgramModel::new(2, 4);
        for next in [10u32, 11, 12, 13, 10, 11] {
            m.observe(&[7, 7, 7], next);
        }
        assert_eq!(m.predict(&[7, 7, 7]), None);
        // An existing draft is extended, not replaced.
        let mut m = NgramModel::new(2, 4);
        for _ in 0..3 {
            m.observe(&[1, 2], 3);
            m.observe(&[1, 2, 3], 4);
        }
        let mut d = vec![2];
        m.extend(&[0, 1], &mut d, 3);
        assert_eq!(d, vec![2, 3, 4]);
    }

    #[test]
    fn quotes_an_earlier_span() {
        let mut idx = DraftIndex::new(2, 4);
        for t in [10, 11, 12, 13, 14, 15, 99, 98, 11, 12, 13] {
            idx.push(t);
        }
        assert_eq!(idx.draft(4), vec![14, 15, 99, 98]);
        assert_eq!(idx.draft(1), vec![14]);
    }

    #[test]
    fn nothing_to_draft() {
        let mut idx = DraftIndex::new(2, 4);
        assert!(idx.draft(4).is_empty());
        for t in [1, 2, 3, 4, 5] {
            idx.push(t);
            assert!(idx.draft(4).is_empty());
        }
        // min > max: the old loop never ran.
        let mut idx = DraftIndex::new(5, 4);
        for t in [1, 1, 1, 1, 1, 1, 1, 1] {
            idx.push(t);
        }
        assert!(idx.draft(4).is_empty());
    }

    #[test]
    fn a_degenerate_repeat_stays_cheap_and_exact() {
        // One token repeated: every position is on the chain, and the first probe is a full
        // match, so the lookup must not walk it.
        let mut idx = DraftIndex::new(2, 4);
        let toks = vec![7u32; 50_000];
        for &t in &toks {
            idx.push(t);
        }
        assert_eq!(idx.draft(4), reference(&toks, 4, 2, 4));
    }
}
