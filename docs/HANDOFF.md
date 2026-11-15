# Update 2026-11-15: branch `mythos-review` (not merged to main)

Everything below this section describes main as of 2026-11-13. Since then all work has gone to
`mythos-review` (worktree `.claude/worktrees/mythos-review`), which contains main, the
memory-policy lane, and:

- **Real-trace benchmark** (`tools/trace_bench.py`, `tools/trace_sweep.sh`): replay of recorded
  OpenHands trajectories. This replaced the synthetic agentic sweep as the benchmark that
  matters. Traces: `~/aq/traces-9b-40.json` on the Spark and the B200.
- **Serving**: prompt checkpoints, in-kernel state tap, wider prefill attention tiles,
  background prefill lane (`AQ_BG_PREFILL`), `AQ_TRACE` step/request logs, SRPT prefill order
  with an elder share (`--srpt true`, no measurable gain on real traces).
- **Speculation**: cross-session n-gram draft model and adaptive draft length; the model's own
  MTP draft head (`AQ_MTP=1`, opt-in; `AQ_MTP_SHORT` shortlist size, `AQ_MTP_K` chained drafts,
  default 1 because chaining did not raise throughput). ~90% first-draft acceptance on real
  traces, 2.89 -> 3.78 tokens per step.
- **Cold start**: parallel pinned-memory weight loader, 9B in ~8 s warm / ~18 s cold page cache.
- **Numerics**: cuBLASLt algorithms used to be picked by timing three runs, so a busy GPU gave
  each process different rounding; at a bifurcation point of repetitive text (9b-long, sequence
  1, token 223) that alone moves logits by 2-3 and showed up as a rare speculation "REAL
  MISMATCH". The pick is now stable (heuristic's first candidate unless another is 3% faster,
  best of 5), and `aq-check` judges a divergence against how far its two plain paths (decode,
  prefill) disagree there. Diagnostics: `AQ_GEMM_PICK=seed` (random plans), `AQ_GEMM_MIN_M`,
  `AQ_SPEC_FAULT=N` (fault injection: the check must fail), `AQ_SPEC_TRACE`,
  `AQ_CHECK_DRIFT_AT=seq:tok`, `AQ_CHECK_ROWDEP=N`, `AQ_CHECK_TF_ONLY`, `AQ_CHECK_SPEC_REPEAT=N`.

Results (Qwen3.5-9B, real traces):

| Machine | Agents | Engine | Turns/min | Tok/s | TTFT p50/p99 (ms) |
|---|---|---|---|---|---|
| Spark | 16 | aqwenator (cross-session drafts, SRPT) | 78.6 | 182 | 411 / 3834 |
| Spark | 16 | vLLM 0.33, n-gram speculation | 47.6 | 127 | 722 / 2292 |
| Spark | 16 | vLLM 0.33, MTP speculation (3) | 62.4 | 137 | 977 / 3729 |
| B200 | 16 | aqwenator, no MTP | 319.4 | 842 | 67 / 1898 |
| B200 | 16 | aqwenator, MTP head (1 draft) | 338.8 | 896 | 72 / 1688 |

Spark, quiet window, 24 GB KV pool, 5 min per leg (2026-11-15):

| Agents | Build | Turns/min | Tok/s | TPOT p50/p99 (ms) | TTFT p50/p99 (ms) | Tok/step |
|---|---|---|---|---|---|---|
| 8 | no MTP | 50.8 | 117 | 47.3 / 127.9 | 342 / 3975 | 2.62 |
| 8 | MTP head | 58.4 | 151 | 36.7 / 74.0 | 337 / 2426 | 3.47 |
| 8 | MTP + background prefill lane | 54.0 | 135 | 41.6 / 96.6 | 364 / 3171 | 3.45 |
| 16 | no MTP | 73.2 | 168 | 73.2 / 161.1 | 547 / 4444 | 2.62 |
| 16 | MTP head | 80.0 | 193 | 58.4 / 189.6 | 520 / 6850 | 3.47 |
| 16 | MTP + background prefill lane | 78.0 | 188 | 59.9 / 157.2 | 533 / 6287 | 3.45 |

So the MTP head is on by default when the checkpoint has one (`AQ_MTP=0` disables it). The
background prefill lane (`AQ_BG_PREFILL=4096`) stays opt-in: on the B200 it cuts the 16-agent
TTFT p99 from 1236 ms to 351-378 ms (the tail there is head-of-line blocking behind 20-30k-token
first turns), but on the Spark, which has no spare GPU capacity for a second stream, it costs
throughput and does not help the tail. `tools/trace_tail.py` dissects the tail from AQ_TRACE logs.

Same-window head-to-head on the Spark (both with MTP speculation, 5 min per leg, 2026-11-15):

| Agents | Engine | Turns/min | Tok/s | TPOT p50 (ms) | TTFT p50/p90/p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|---|
| 8 | aqwenator | 54.0 | 138 | 39 | 348 / 1428 / 3444 | 395 |
| 8 | vLLM 0.33 | 44.6 | 102 | 58 | 665 / 1583 / 2601 | 489 |
| 16 | aqwenator (FIFO prefill) | 81.4 | 195 | 60 | 597 / 2633 / 5791 | 492 |
| 16 | aqwenator (shortest prefill first) | 82.0 | 197 | 60 | 469 / 1506 / 5631 | 464 |
| 16 | vLLM 0.33 | 57.4 | 122 | 108 | 1022 / 1882 / 2909 | 763 |

`aq-serve` now orders prefill shortest-first by default (`--srpt false` for admission order).
A larger per-step prefill budget (2560) pushed ITL p99 to ~800 ms without helping.

Burst prefill (default in `aq-serve`: `--burst-prefill 4096 --burst-every 3`, 0 = off; engine
knobs `AQ_BURST_PREFILL` / `AQ_BURST_EVERY`): every few steps the longest-waiting prompt that
needs more than the per-step budget gets a whole step of up to N prefill tokens to itself.
Spark, 16 agents, 5 min, turns with 3,000+ new tokens (p50/p90 TTFT) and overall:

| Setting | Turns/min | TTFT p50/p90/p99 (ms) | 3k+ turns p50/p90 (ms) | ITL p99 (ms) |
|---|---|---|---|---|
| no burst (shortest first) | 82.0 | 469 / 1506 / 5631 | 3409 / 6478 | 464 |
| burst 4096 every 3 | 85.8 | 455 / 1565 / 4128 | 2355 / 4266 | 482 |
| burst 3072 every 2 | 83.2 | 533 / 1861 / 3741 | 2566 / 3861 | 511 |
| burst 6144 every 4 | 79.4 | 552 / 1864 / 3745 | 1725 / 3288 | 496 |
| vLLM 0.33, MTP | 57.4 | 1022 / 1882 / 2909 | 1347 / 2909 | 763 |

Bursts are rare enough that ITL p99 hardly moves. Larger bursts close more of the gap on big
turns but make small turns wait a little longer (p90 for <1k-token turns 658-805 ms without,
782-1054 with 4096, 1138-1439 with 6144; vLLM 1707).

10-minute run of the final defaults (tag `final-defaults`): at 8 agents 53.9 turns/min, 147
tok/s, TTFT p50/p90/p99 331/964/2546 ms, ITL p99 340 ms, and every turn-size bucket ahead of
vLLM (44.6 turns/min, 665/1583/2601, 489), including turns with 3k+ new tokens (p50/p90
1954/2835 vs 2434/3133 ms). The 16-agent leg ran while other jobs were on the GPU
(91% busy before the start) and came out at 70.1 turns/min, TTFT p99 7.0 s, ITL p99 833 ms: not
a valid confirmation of the 5-minute quiet-window numbers above; rerun it on a quiet machine.

Open: at 16 agents vLLM still has the better TTFT p99 (2.9 s vs 3.7-4.1 s in quiet 5-minute
runs) and the better p90 for turns with 1k+ new tokens. Also: the GEMM plan fix has not been re-verified on
the Spark with 9B; merge to main.

Gate status: `aq-check` on the B200, 6 goldens x {default, small budget, SRPT}, passes with
and without `AQ_MTP=1`.

Remote jobs: launch detached with a DONE marker and a timeout per process; never a wait loop
keyed on log lines only (one sat dead for 11 hours).

---

# Handoff: aqwenator state as of 2026-11-13

## FIRST: known correctness bug on main (fix exists, unmerged)
The memory-policy lane (branch `worktree-agent-ada1630c33961379c`) found that a shared-prefix
**fork admission that evicts to get a slot reuses a stale snapshot index**, so after an eviction it
can fork the *wrong prefix state* (silent wrong output under memory pressure). Fixed in `da01b7c`
(snapshot re-looked-up by content), together with a feasibility fix (`2d0338c`: an admission that
cannot fit no longer evicts parked sessions one by one and then fails) and a host-time fix
(`527332c`: prefix matching was O(waiting × parked × 18k tokens) per step). That branch is based on
c6f9b86 (before the speculation and co-execution merges): rebase onto main, run the full aq-check
suite (its new memory-pressure test + `AQ_CHECK_POLICY`), and merge these fixes first. Until then,
avoid memory-pressure workloads with session cache + snapshots on main.

Lane findings worth keeping: snapshot pinning + dedupe (151 snapshots → 1 at 32 agents) and
`--prewarm` (tune all cuBLASLt plans at startup; mid-run autotune is the likely cause of multi-second
ITL p99.9 spikes) should become defaults; idle-aware eviction, load control with `--quantum`, and
`--mixed-prefill-adaptive` as opt-in; MRU eviction, SRPT (until re-measured on the fixed build) and
25 ms frame pacing are not worth it. Best variants still trail vLLM at 16/32 agents (163 vs 189,
92 vs 188 turns/min); the remaining gap is long-context decode attention (~1 TB/s effective, ITL p50
11.5 vs 5.8 ms at 16 agents / 18k ctx) and re-prefill throughput. Its gate auto-runs on the B200 when
the driver recovers: `~/aq/logs/mem_gate.log`; sweeps: `tools/agentic_sweep_mem.sh`.

## What exists (main @ 7771327, all gated)
- Rust workspace: `aq-kernels` (cuda-oxide `#[cuda_module]` kernels), `aq-engine` (model, scheduler,
  session cache, aq-check, aq-bench), `aq-server` (`aq-serve`, OpenAI `/v1/completions`, SSE).
- Model family: `qwen3_5` dense (Qwen3.5 0.8B–27B, Qwen3.6-27B, Qwen3.8-27B), bf16 only (no FP8 yet).
- Kernels: tensor-core prefill attention (`attention_tc`), split-K decode attention with coalesced
  128-bit loads, tensor-core chunked gated delta rule (`gdn_tc_*`), single-token `gdn_decode`,
  vectorized norm/SwiGLU, two-phase argmax. cuBLASLt for GEMMs (M-bucketed, autotuned plans).
- Engine: continuous batching + chunked prefill, paged KV (refcounted pages), DeltaNet state slots,
  CUDA graphs for decode, session cache (park/resume exact prefixes, LRU), shared-prefix snapshots
  (fork state + share pages), KV pool sized from `--kv-gb`.
- Exact speculative decoding (prompt-lookup drafts, k=4 default in `aq-serve`): token-identical to
  greedy; +25–62% on random prompts, 2.3–4× on quote-back workloads.
- Decode/prefill co-execution (`AQ_COEXEC=1`, off by default): ~10× better ITL tail on B200 at equal
  throughput, worse TTFT; hurts throughput on the Spark (needs green contexts / SM partitioning).

## Correctness (the #1 rule)
`aq-check <model> <golden>` compares against transformers references in `~/aq/golden/*.safetensors`
(made by `tools/golden.py`): teacher-forced logits at every generated position, free-running and
batched greedy, session resume, shared-prefix fork. Divergences are classified by top-2 margin
(near-tie < 0.25). Always run the full suite: 6 goldens × {default, `AQ_CHECK_BUDGET=1000
AQ_CHECK_CHUNK=777`} (the odd budget found a real upload/compute stream race), plus `AQ_SPEC_K=4`.
Byte-identity comparisons need `AQ_GEMM_HEURISTIC=1` (cuBLASLt plans are picked by timing).

## Open branches (not merged)
- `worktree-agent-ae9e174bba1af161f` @ d7e410c — lossless KV compression (400 B rows, 1.28× tokens,
  logits byte-identical to raw). Rebased on 7771327, opt-in (`AQ_KV_COMPRESS=1`). **Gate not yet run**
  (B200 driver hang): run `tools/kvc_gate.sh` before merging. Decode attention is ~2.3× slower than
  raw, so it only pays when capacity avoids evictions (memory-tight agentic runs, the Spark).
- `worktree-agent-ada1630c33961379c` — memory policy / scheduler lane; **contains the stale-fork-index correctness fix** (see top). Eviction policies, snapshot
  pinning, load control, SRPT). See its final report / commit messages; results in
  `~/aq/results/agentic/mem-*` on the B200.
- `worktree-agent-a6d76573bade9b155` — skinny GEMM experiments (did not beat cuBLASLt; keep unmerged).

## Benchmarks (where things stand)
- B200, 9B, 1024/256: we win at c ≤ 4 (e.g. c=1 252 vs 235 tok/s), lose at c ≥ 16 (prefill speed).
- DGX Spark, 9B, 1024/256: 93–108% of vLLM throughput; ITL p99 better at c ≤ 16; vLLM's 2048-token
  mixed steps are ~30% cheaper on GB10 (profile Spark prefill next). Cold start 6 s vs vLLM 163 s.
- Agentic (B200, 9B, 32 GB budget): we win at 4 agents, lose at 8–32 (memory policy; 32 agents
  collapses). Spark agentic head-to-head was running at handoff: results in
  `~/aq/results/agentic/spark-*` on the Spark (`scripts/spark.sh`), log `~/aq/logs/run_spark_queue.log`.
  It uses a pre-speculation build (`aq-serve-current` = session cache only).
- Progress page: https://claude.ai/artifact/MNkgXru1gcTGbZz3Zw3zwN (data files under `docs/progress/`,
  regenerate `bench.json`/`agentic.json` with `tools/collect.py`, `AQ_GPU_LABEL=GB10` on the Spark).

## Machines and gotchas
- B200 (shared, `scripts/b200.sh`, `scripts/sync.sh` with `AQ_REMOTE_DIR` for a separate checkout):
  no sudo/docker; toolchain in `~/aq/env`; `source ~/aq/src/scripts/remote/env.sh`; build
  `cargo oxide build --arch sm_100a -- --release -p aq-engine -p aq-server --bins`. GPU driver was
  hung at handoff (nvidia-smi in D state). Never `pkill`/`pgrep -f` (self-matches, shared users);
  kill only your own PIDs; use `grep "[p]attern"` in wait loops.
- DGX Spark (`ssh $AQ_REMOTE`, `scripts/spark.sh`, `scripts/spark_sync.sh`):
  also runs other services; build with
  `--arch sm_121a`; `source ~/aq/src/scripts/spark/env.sh`.
- Nsight Compute counters are blocked on the B200 (`ERR_NVGPUCTRPERM`); use `AQ_PROFILE=1|decode|prefill`
  (CUDA events, works inside graphs) and `AQ_KINFO=1` (registers/spills per kernel).
- vLLM baselines: `tools/baseline_vllm.sh` (readiness wait 900 s, `--temperature 0`),
  `tools/agentic_sweep.sh`, `tools/agentic_bench.py`.

## Research
`docs/research/sota-survey.md`, `docs/research/moonshots.md`, `docs/research/compression-and-skinny-gemm.md`.
Top unexplored ideas: prompt-end snapshots + runahead for chat-template re-renders, cooperative scan
over shared prefixes, batch-invariant numerics, NVMe session tier (safetensors I/O), green-context
SM partitioning on the Spark, FP8 weights for Qwen3.8-27B.
