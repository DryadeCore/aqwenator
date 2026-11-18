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

use crate::draft::{DraftIndex, NgramModel};
use crate::model::{Batch, GDN_TC_CHUNK, Limits, MAX_DECODE_ROWS_PER_SEQ, Model};
use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::time::Instant;

mod admit;
mod cache;
mod lanes;
mod pool;
mod spec;
mod step;

pub use cache::{CKPT_BACKOFF, CacheStats, Evict, Policy, SNAPSHOT_MIN_TOKENS};
use cache::{Parked, Snapshot};
use lanes::Inflight;
use pool::Pool;
pub use spec::SpecStats;
pub use step::SMALL_PREFILL;

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
    /// Draft tokens this sequence may carry next step: `spec_k` normally, doubled (up to the
    /// row limit) after a step that accepted a full draft, because copy runs are long and a
    /// longer draft only pays while they last.
    k_cur: usize,
    /// The MTP head's predictions for the tokens after the last known one (the first from the
    /// main pass, the rest chained), with the number of known tokens they were made at.
    mtp_next: Option<(usize, Vec<u32>)>,
    /// When the request was submitted and admitted, and the steps it took part in before its
    /// first token (AQ_TRACE request log).
    t_add: Instant,
    t_admit: Instant,
    steps: u32,
    reused: usize,
    /// Pause prefill here and take the prompt checkpoint (page aligned, a few tokens before
    /// the prompt's end); `None` once taken, or when there is nothing to checkpoint.
    ckpt_at: Option<usize>,
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
    pool: Pool,
    parked: Vec<Parked>,
    snapshots: Vec<Snapshot>,
    /// Max snapshot slots kept at once.
    pub max_snapshots: usize,
    clock: u64,
    pub eos: Vec<u32>,
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
    /// Cross-session draft model: extends (or supplies) drafts with what the server's own
    /// earlier generations most often produced after the same few tokens. On unless
    /// AQ_NO_GLOBAL_DRAFT is set.
    pub global_draft: bool,
    global: NgramModel,
    /// Grow a sequence's draft length while its drafts are fully accepted (off with
    /// AQ_NO_SPEC_ADAPT).
    pub spec_adaptive: bool,
    /// Draft tokens taken from the MTP head per step: the first comes with the main pass,
    /// the rest from chained passes of the head alone (AQ_MTP_K, default 3). Chaining paid
    /// off once decode attention read K/V once per sequence instead of once per row: on the
    /// DGX Spark's real traces 3.37 -> 3.97 tokens per step and +10% turns per minute; it is
    /// neutral on the B200.
    pub mtp_k: usize,
    /// Burst prefill (AQ_BURST_PREFILL=tokens, 0 = off) and the minimum number of steps
    /// between two bursts (AQ_BURST_EVERY, default 3).
    pub burst_prefill: usize,
    pub burst_every: usize,
    since_burst: usize,
    /// AQ_SPEC_TRACE: log every decode entry's draft, targets and acceptance.
    spec_trace: bool,
    /// AQ_SPEC_FAULT=N: test hook for the correctness check itself. Every Nth rejected draft
    /// is emitted as if it had been accepted, which the check must report as a mismatch.
    spec_fault: u64,
    spec_rejects: u64,
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
    /// AQ_TRACE=1: log one line per step (`[step]`: composition and GPU time) and one per
    /// request at its first token (`[req]`: queue wait, steps, time to first token).
    pub trace: bool,
    /// Background prefill (AQ_BG_PREFILL=tokens, 0 = off): a prompt with at least this many
    /// tokens left to prefill runs as one asynchronous batch on a second lane, while steps for
    /// everyone else (decode, speculation, short prefills) continue. Keeps a long tool result
    /// from stalling the sequences that are generating. Not combined with co-execution.
    pub bg_prefill: usize,
    pub policy: Policy,
    /// Set during an admission that may only evict sessions presumed finished.
    dead_only: bool,
}

impl Engine {
    pub fn new(model: Model, eos: Vec<u32>) -> Self {
        let lim = model.limits;
        let mut eng = Self {
            model,
            waiting: VecDeque::new(),
            running: Vec::new(),
            pool: Pool::new(lim.max_seqs, lim.num_blocks()),
            parked: Vec::new(),
            snapshots: Vec::new(),
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
            global_draft: std::env::var("AQ_NO_GLOBAL_DRAFT").is_err(),
            global: NgramModel::new(2, 4),
            spec_adaptive: std::env::var("AQ_NO_SPEC_ADAPT").is_err(),
            mtp_k: std::env::var("AQ_MTP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            burst_prefill: std::env::var("AQ_BURST_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            burst_every: std::env::var("AQ_BURST_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            since_burst: 0,
            spec_trace: std::env::var("AQ_SPEC_TRACE").is_ok(),
            spec_fault: std::env::var("AQ_SPEC_FAULT").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            spec_rejects: 0,
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
            trace: std::env::var("AQ_TRACE").as_deref() == Ok("1"),
            bg_prefill: 0,
            // AQ_SRPT=1: test hook, so aq-check can run its suite under the SRPT prefill order.
            policy: Policy { srpt: std::env::var("AQ_SRPT").as_deref() == Ok("1"), ..Policy::default() },
            dead_only: false,
        };
        // The MTP head keeps its own KV over the whole context, so it runs on every step of
        // every lane once it is loaded.
        eng.model.mtp_on = eng.model.has_mtp();
        if let Some(n) = std::env::var("AQ_BG_PREFILL").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
            eng.enable_bg_prefill(n).expect("could not create the background prefill lane");
        } else if std::env::var("AQ_COEXEC").as_deref() == Ok("1") {
            eng.enable_coexec().expect("could not create the decode lane for AQ_COEXEC=1");
        }
        eng
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
        self.model.mtp_note(req.prompt.iter().copied());
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

    pub(super) fn index_of(&self, id: u64) -> usize {
        self.running.iter().position(|s| s.req.id == id).expect("in-flight sequence missing")
    }
}
