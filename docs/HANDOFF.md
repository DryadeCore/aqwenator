# Handoff: aqwenator state as of 2026-11-13

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
- `worktree-agent-ada1630c33961379c` — memory policy / scheduler lane (eviction policies, snapshot
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
