# Roadmap (agreed 2026-11-16)

The competitors for the local goal are Strata, llama.cpp, gufo and the hardware ceiling itself;
vLLM stays as the server baseline. In order:

1. **int8-weight GEMM kernel** (`gemm_i8w`, in progress): trace-golden KL against the expansion
   path, load test, full gate, then merge.
2. **Longer drafts: measured on the B200, no gain there.** Chained MTP drafts (`AQ_MTP_K=3`) and
   lookup drafts (`--spec-k 4`, the total draft budget) are on by default and give 4.22 tokens
   per decode pass on the traces (72% of drafted tokens accepted, the MTP head's first draft
   90%). Budget 8 with 3 / 5 / 7 chained drafts gives 4.38 / 4.51 / 4.54 tokens per pass and
   435 / 428 / 425 turns/min at 16 agents against 433 (budget 16: 412): the extra rows cost what
   the extra tokens save. Chaining was +10% on the Spark and neutral on the B200, so repeat this
   on a bandwidth-limited device before changing the defaults.
3. **Longer prompt-lookup drafts when a request decodes alone** (others verify up to 64 copied
   tokens in one pass). Same caveat as 2: measure on a bandwidth-limited device.
4. **8-bit attention cache** (then 8-bit keys / 4-bit values). Decode attention is 30% of the fp8
   step and bandwidth-bound.
5. **Session checkpoints in host RAM and on disk**: evicted sessions and restarts resume from the
   longest exact prefix instead of re-running prefill. Safetensors `main` has a prefetch engine
   (still changing) worth using for the disk tier.
6. **6-bit or mixed dense weights.** Measure quality first: others report 4-bit dense at +9%
   perplexity against +0.2% for 8-bit.
7. **Benchmarks against Strata, llama.cpp and gufo** on the same traces and machine.
8. **Qwen3.8-Flash-Next on small hardware** (worth it only if it runs on small cards). New pieces:
   MoE with 512 experts per layer and 10 active, hyper-connections, sparse attention, a 51 GB
   n-gram embedding table read from disk. Known lessons: one preallocated expert slot pool with
   routing remapped to slots so the grouped GEMM is unchanged; LRU is within 3 points of the
   optimal cache policy; router read-ahead wastes about 40% of its reads; the fastest design keeps
   all experts in pinned host RAM, the hot ones in VRAM, and computes misses on the CPU.
9. **Research: zml `attnd`** ("sparse logarithmic attention, on CPU, over UDP"). Only the
   announcement was read; the mechanism is in its video.

# Update 2026-11-16: branch `mythos-review` (main has everything up to the Docker/CI commits; FP8 is branch-only)

Everything below this section describes main as of 2026-11-13. Since then all work has gone to
`mythos-review` (worktree `.claude/worktrees/mythos-review`), which contains main, the
memory-policy lane, and:

- **Real-trace benchmark** (`scripts/trace_bench.py`, `scripts/trace_sweep.sh`): replay of recorded
  OpenHands trajectories. This replaced the synthetic agentic sweep as the benchmark that
  matters. Traces: `~/aq/traces-9b-40.json` on the Spark and the B200.
- **Serving**: prompt checkpoints, in-kernel state tap, wider prefill attention tiles,
  background prefill lane (`AQ_BG_PREFILL`), `AQ_TRACE` step/request logs, SRPT prefill order
  with an elder share (`--srpt true`, no measurable gain on real traces).
- **Speculation**: cross-session n-gram draft model and adaptive draft length; the model's own
  MTP draft head (`AQ_MTP=1`, opt-in; `AQ_MTP_SHORT` shortlist size, `AQ_MTP_K` chained drafts,
  default 1 because chaining did not raise throughput). ~90% first-draft acceptance on real
  traces, 2.89 -> 3.78 tokens per step.
- **Cold start**: weight loader with direct I/O (O_DIRECT reads into pinned staging buffers,
  overlapped with device allocation). 9B on the Spark: weights resident at 2.4 s with a warm or
  cold page cache, first token at 3.2 s (the mmap loader it replaced: 2.1 s warm, 16.7 s cold).
  The drive reads at ~10 GB/s, so 1.8 s is the floor for 17.7 GiB; allocation is ~0.9 s.
- **Numerics**: cuBLASLt algorithms used to be picked by timing three runs, so a busy GPU gave
  each process different rounding; at a bifurcation point of repetitive text (9b-long, sequence
  1, token 223) that alone moves logits by 2-3 and showed up as a rare speculation "REAL
  MISMATCH". A stabilized timing pick (first candidate unless another is 3% faster, best of 5)
  still let one gate leg in eleven fail on the B200 (9b-long, `AQ_MTP=0`: two or three different
  token paths across identical runs), so there is no timing any more: always the heuristic's
  first candidate. Six runs of that leg then take one path, token-identical to plain decoding.
  Speed on real traces, 16 agents: Spark 92.2 and 93.2 turns/min vs 93.0 timed; B200 434.4 vs
  438.6 (noise). `aq-check` judges a divergence against how far its two plain paths (decode,
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
throughput and does not help the tail. `scripts/trace_tail.py` dissects the tail from AQ_TRACE logs.

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

## Hardware ceilings (2026-11-15, uncommitted at the time of writing: commit signing needs the user)

`aq-ceiling [<model>] [--gb N] [--distinct] [--out f.json]` (crates/aq-engine/src/bin/aq-ceiling.rs)
measures what the device can do and what that means for one step of a model: read/copy/write
bandwidth, peak bf16 GEMM rate, launch overhead, and per row count M two lower bounds on the
step's GEMM time (every weight byte read once; every multiply-add at the peak rate) next to
what cuBLASLt takes for the same GEMMs. `--distinct` gives each GEMM its own weight buffer
(the engine's situation; costs the model's weight footprint in memory).

| | DGX Spark (GB10, 48 SMs) | B200 (148 SMs) |
|---|---|---|
| Read bandwidth | 237 GB/s | 6.7 TB/s |
| Peak bf16 GEMM | 98 TFLOP/s | 2.1 PFLOP/s |
| Launch (enqueue / launch + sync) | 1.9 / 4.9 us | 1.7 / 6.9 us |
| 9B weights read once (15.9 GB) | 67 ms | 2.4 ms (3.0 ms with distinct buffers) |
| KV read, 16 seqs x 20k ctx (10.5 GB) | 44 ms | 1.6 ms |

Qwen3.5-9B: 15.9 GB of GEMM weights per step (lm_head alone is 2.0 GB), 32 KiB of attention
KV per context token, 49 MiB of recurrent state per sequence.

Where the engine stands (decode step time against the bandwidth floor = weights + KV + state):
- Spark: 84 ms single stream (floor 70 ms), 141 ms at 16 seqs x 16k context (floor about
  110 ms). About 80% of the ceiling; cuBLASLt itself runs at 92-99% of the weight-read bound
  for M <= 64. The Spark is bandwidth-bound: more throughput has to come from more tokens per
  weight pass (drafts, batch) or fewer bytes per pass (quantized weights, not exact).
- B200: 4.1 ms single stream (floor 2.4-3.0 ms), 5.6 ms at 16 seqs, 10.7 ms at 64 seqs x 4k
  context (floor 4.7 ms). 45-58% of the ceiling before the attention work below.
- B200 prefill at 16k context: `attn_prefill` is 31% of the step and far below both ceilings
  (a block scans the whole context serially; no split-K in prefill). Open.

### Decode attention on tensor cores (`attn_decode_tq`; the split kernel it replaced is deleted)

One block per (sequence tile, KV head, slice) serves every row of the sequence (the token plus
its drafts) and all query heads of the GQA group: K/V is fetched once per sequence instead of
once per row and head. Slices come from a fixed table of absolute key bounds (32 fine slices
of `AQ_DEC_TC_SLICE` = 128 keys, then 32 coarse ones up to max_ctx), so a row's result does not
depend on the batch or on its drafts (this replaces `AQ_ATTN_SPLIT_KEYS`, `AQ_ATTN_FILL` and the
row-local slicing mode, all removed).

Findings on the way:
- The K/V page lookup did two 64-bit divisions per 16-byte copy; that was most of a tile's
  time. `kv_tile_cp_async` does one 32-bit division per thread and tile. `attention_tc`
  (prefill), `attn_prep` and the reference kernels still divide per element.
- Per 16-key tile on the B200: about 1.75 us fetching K/V, 0.5 us of MMAs. Double buffering
  and fetching four tiles per round trip did not help (more shared memory, fewer blocks).

B200, 9B, decode attention per step: 64 seqs x 4k ctx 4.77 -> 1.61 ms (floor 1.36), 16 seqs
1.31 -> 0.60 ms, single stream 0.17 -> 0.28 ms. End to end: 64 streams 2267 -> 2547 tok/s (ITL p50
10.75 -> 7.69 ms), 16 streams 1576 -> 1672, single stream 227 -> 224; with k=4 speculation at 16k
context 291 -> 322 (1 stream) and 521 -> 567 tok/s (16 streams).

Spark, real traces (2026-11-15, 24 GB KV pool): quiet 10-minute 16-agent run of the previous
defaults 82.4 turns/min, 206 tok/s, TTFT p50/p90/p99 459/1673/3596 ms, ITL p99 489 ms (vLLM:
57.4, 122, 1022/1882/2909, 763). With the new decode attention, 5-minute legs: 8 agents 59.0
turns/min, 161 tok/s, TTFT 309/924/2344, ITL p99 300 (before: 53.9, 147, 331/964/2546, 340);
16 agents 84.4 turns/min, 199 tok/s against 83.2 and 200 for the split kernel in the same
session (an earlier A/B of a first version gave 91.6 against 80.8: trace runs differ by several
percent between sessions). Per-op profile, 16 seqs x 16k context: decode attention 38.6 ms
against 40.1 ms with almost no drafts in flight (the floor is 36 ms); the gain grows with
the number of draft rows per sequence.

Not at the ceiling yet (largest first): B200 prefill attention (20% of its bandwidth floor;
one block scans the whole context serially); B200 small-kernel overhead in decode (about
1.1 ms of a 4.2 ms single-stream step, ~300 launches); GDN decode at high concurrency (1.46 ms
against a 0.94 ms floor at 64 streams); cuBLASLt at 128-512 rows (67-80% of its bound). On the
Spark: lm_head reads 2 GB per step (13% of the weight bytes) and prefill has not been
measured against the FLOP ceiling.

### Ceiling-gap investigation and chained drafts (2026-11-15, afternoon)

Verdicts on the gaps listed above:
- Prefill attention (B200): per tile, fetch 28%, MMAs 43%, softmax and bookkeeping 25%. The
  kernel needs 252 registers per thread, so one block (8 warps) per SM. Double-buffered K/V
  gave nothing. On the Spark it is 8% of a prefill step. No cheap win found.
- Small kernels in decode (B200 only): on real traces about 1.9 ms of a 7.9 ms step at ~8 us
  each. Fusing conv1d with the GDN step would save about 3%; not done.
- GDN decode: at its bandwidth floor on the Spark, 65% on the B200 (at most 6% of a step).
- cuBLASLt at 128-512 rows: not a gap. Workspace size (4 MB to 1 GB) changes nothing; it is
  the range where the bandwidth and FLOP limits both bind.
- lm_head: an exact shortcut (low-precision screen, exact recheck) would save 3-5% of a decode
  step; not worth the machinery.
- Spark prefill (16 seqs x 16k, 4096 rows per step, 950 ms): GEMMs 666 ms against a 591 ms
  FLOP floor (88%); attention 77 ms, GDN 83 ms, silu 40 ms and norm 35 ms (both at their
  bandwidth floor), conv1d 18 ms.
- Tried and dropped: 32-bit divisions in silu_mul and attn_prep (no change), padded instead of
  XOR-swizzled shared tiles in attn_decode_tq (170 -> 130 registers, 225 -> 220 us per call),
  four-tile K/V fetches.

Real-trace profile (`AQ_PROFILE=decode` with aq-serve, B200, 16 agents, mean context 20.9k):
decode step 7.9 ms; attention 2.0 ms (floor 1.64), gate_up 1.4, down 0.84, gdn_in 0.62,
gdn 0.58, norm 0.54, lm_head 0.36.

Chained MTP drafts are now the default (`AQ_MTP_K=3`, was 1). With K/V read once per sequence
the extra draft rows are nearly free on a bandwidth-bound machine. Spark, 16 agents, 5 min:
K=1 82.8 turns/min, 195 tok/s, 3.37 tok/step; K=2 90.6, 221, 3.77; K=3 91.4, 222, 3.97;
K=4 89.2, 215, 4.12; `--spec-k 8` alone 85.0, 203; K=3 with `--spec-k 8` 90.8, 220. B200:
437.6 / 435.4 / 441.2 / 443.2 turns/min for K=1 / K=3 / K=3 + k8 / k8 (neutral). Gate on the
B200: 30/30 PASS (6 goldens x default, SRPT + burst, small budget, AQ_MTP=0, AQ_MTP_K=1), fault
injection fails as required.

Spark, 10-minute runs with K=3 (tags mtpk-k3-16, mtpk-k3-8): 16 agents 91.9 turns/min, 229
tok/s, TTFT p50/p90/p99 483/1573/3864 ms, ITL p99 466 ms; 8 agents 57.4 turns/min, 157 tok/s,
TTFT 365/981/2445, ITL p99 360. vLLM (5 min): 57.4, 122, 1022/1882/2909, 763 at 16 agents;
44.6, 102, 665/1583/2601, 489 at 8. B200, 16 agents: 437.6 turns/min, 1146 tok/s (368.6 and
973 in the morning).

### Where the time goes on the Spark (2026-11-15, evening)

16 agents on real traces, K=3 (tag prof-k3-16: 93.4 turns/min, 232 tok/s; the GPU is busy for
the whole run). A least-squares fit of the `[step]` lines of `AQ_TRACE`: a decode-only step
costs 78 ms + 0.25 ms per row + 153 ms per million keys read (once per sequence); a prefill
token costs 0.22 ms. Mean step: 11 decoding sequences, 75 rows (6.7 per sequence: 2.6 pending,
4.1 drafts), 20.7k keys per sequence.

| Share of GPU time | What | Against its limit |
|---|---|---|
| 40% | weights read once per step | 78 ms vs a 67 ms floor |
| 33% | prefill | GEMMs at 88% of the FLOP ceiling |
| 18% | K/V read by decode attention | 214 GB/s vs 237 GB/s measured peak |
| 9% | per-row compute | a third of the rows are rejected drafts or re-fed tokens |

Draft acceptance by position (`AQ_SPEC_TRACE`, 16.8k decode entries): 90% / 85% / 87% / 86%
at positions 1-4 and 93-95% at 5-8 (only n-gram copies get that deep, 22% of entries). Mean
4.15 drafted, 3.03 accepted. Deeper head chains do not pay (K=4 measured no better than K=3):
the head feeding on its own guesses is what limits positions 2+, not the row budget.

Dead ends on the K/V traffic:
- Lossless KV compression: 1.28x on these values, at most 4% of step time before the cost of
  decompressing.
- Skipping the V read of key tiles with negligible attention weight: an instrumented kernel
  counted tiles where no query slot of the block has a score within 2^-20 / 2^-30 / 2^-40 of
  its running maximum. On real traces 99.9996% of 625M tiles are needed even at 2^-20, so
  there is nothing to skip.

What is left on the Spark with exact bf16 is a handful of items worth <= 3% each (base step
78 -> 67 ms, wasted draft rows, non-GEMM prefill ops). The step change is fewer bytes per
weight (8-bit weights would halve the 40% share), which gives up exact outputs.

### Release and 8-bit weights (2026-11-16)

**Release.** `Dockerfile` (CUDA 13.0.3 devel ->
base + `libcublas-13-0`, 1.23 GB), `ci.yml` (image build on amd64 and arm64), `release.yml`
(tag `vX.Y.Z` -> `ghcr.io/<owner>/aqwenator` + binaries on a GitHub release; no tag pushed
yet). One build per CPU type: kernels target `sm_80` and the driver compiles them for the
device (Spark: checks pass, 19.9 vs 19.5 tok/s for the `sm_121a` build). The arm64 image's
`aq-check` passes on the Spark; the amd64 image has not run on a GPU yet.

**8-bit weights: what the hardware offers** (`aq-ceiling` now times the 9B step's GEMMs with
8-bit operands; Spark, ms per step):

| Rows M | bf16 | FP8 E4M3 x E4M3 -> bf16 | int8 x int8 -> i32 |
|---|---|---|---|
| <= 128 | 75-82 | 38-42 | 38-46 |
| 512 | 107 | 55 | 81 |
| 4096 | 648-666 | 304-310 | 407 |

cuBLASLt 13.1 has no 8-bit weights x bf16 activations, and its per-row scale modes
(`OUTER_VEC_32F`) are rejected for this problem (heuristic status 7).

**`AQ_FP8=1|2`** (experimental, opt-in, not exact; `AQ_FP8_SET` picks the GEMMs): E4M3
weights with a power-of-two scale per output channel, quantized as each weight lands and the
bf16 copy freed (`gpu::Weight`); activations quantized per row before each GEMM
(`quant_e4m3`), output rescaled after (`scale_outer`). 2 = output head too.
- Speed, Spark, real traces, 16 agents, 5 min: bf16 83.6 turns/min 204 tok/s (a low run; 92-93
  is usual); FP8 layers 103.6 / 278; layers + head 106.8 / 286. Synthetic 16 x 16k: 25.1 vs
  20.0 out tok/s.
- Memory, 9B with a 4096-token KV pool: 11.70 GiB (layers) vs 18.05 GiB. Weights resident at
  4.4 s vs 2.9 s (uploads drained per weight; a pre-quantized checkpoint would fix it).
- Exact gate: 9B golden passes; resume-vs-cold identity checks fail on 0.8B and 9B-long
  (the decode and prefill paths now differ by quantization noise, not ulps).

**Tiers on real traces** (Spark, 16 agents, 5 min each, same binary, 2026-11-16 02:10-02:36;
`AQ_QUANT=fp8|int8` replaced `AQ_FP8`):

| Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|
| bf16 | 92.6 | 224 | 450 / 6040 | 50.9 / 126.2 |
| bf16 (second run) | 93.2 | 231 | 478 / 5146 | 50.5 / 110.4 |
| fp8 (layers) | 106.2 | 293 | 397 / 3748 | 40.7 / 98.0 |
| int8 (layers, whole-weight expansion) | 70.6 | 150 | 709 / 5371 | 84.5 / 181.3 |

**Final matrix** (Spark, 10 min each, one binary with int8 expanded 1024 rows at a time,
2026-11-16 03:10-04:25, 0 errors; a rebuild used CPU during the first leg):

| Agents | Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 16 | bf16 | 91.5 | 228 | 491 / 4550 | 50.6 / 117.9 |
| 16 | fp8 | 113.7 | 295 | 374 / 3374 | 38.1 / 80.3 |
| 16 | int8 | 71.6 | 182 | 619 / 3946 | 67.0 / 132.6 |
| 8 | bf16 | 60.6 | 166 | 342 / 2780 | 32.9 / 65.0 |
| 8 | fp8 | 79.3 | 209 | 256 / 2075 | 23.4 / 49.9 |
| 8 | int8 | 47.7 | 124 | 456 / 3383 | 46.1 / 105.1 |

fp8 is +24% (16 agents) and +31% (8) turns/min over bf16 with better latency everywhere; int8
is -22% and -21% (chunked expansion took its tok/s from 150 to 182 at 16 agents).

**vLLM on the same 8-bit scheme** (vLLM 0.33, `RedHatAI/Qwen3.5-9B-FP8-dynamic`, MTP
speculation 3 tokens, same replay, 10 min each, 05:16-05:40, 0 errors):

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 16 | vLLM FP8 | 51.9 | 131 | 1052 / 3739 | 101.2 / 135.8 |
| 16 | aqwenator fp8 | 113.7 | 295 | 374 / 3374 | 38.1 / 80.3 |
| 8 | vLLM FP8 | 47.4 | 132 | 480 / 1719 | 45.1 / 60.4 |
| 8 | aqwenator fp8 | 79.3 | 209 | 256 / 2075 | 23.4 / 49.9 |

2.2x (16 agents) and 1.7x (8) turns/min. vLLM keeps the better TTFT p99 at 8 agents (1719 vs
2075 ms). vLLM gained little from FP8 at 16 agents (bf16 earlier: 57.4 turns/min, 122 tok/s).

**Scaling points** (4 min each, 06:15-06:37, 0 errors):

| Agents | Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 4 | bf16 | 35.2 | 90 | 258 / 5203 | 26.8 / 57.6 |
| 4 | fp8 | 49.8 | 132 | 197 / 1938 | 17.7 / 33.2 |
| 32 | bf16 | 137.5 | 247 | 964 / 10992 | 103.7 / 180.7 |
| 32 | fp8 | 148.2 | 297 | 808 / 9102 | 84.5 / 151.4 |

fp8 tok/s is flat from 16 to 32 agents (295 -> 297): the machine is saturated at 16, and more
agents only add queueing (TTFT p99 9 s). The fp8 gain shrinks with load (+41% turns/min at 4
agents, +8% at 32) as attention over the cached context, which fp8 does not touch, takes over.

**Extra tier points** (16 agents, 5 min each, 04:31-05:13; short runs, turns/min moves a few
percent with the sessions drawn):

| Variant | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|
| fp8 + FP8 output head (`AQ_QUANT_HEAD=1`) | 108.6 | 293 | 422 / 4546 | 39.1 / 87.4 |
| fp8, MLP only (`AQ_QUANT_SET=ud`) | 103.4 | 273 | 450 / 4286 | 43.1 / 93.8 |
| int8, MLP only | 81.0 | 193 | 580 / 5415 | 61.1 / 125.1 |

The FP8 head buys nothing measurable (293 vs 295 tok/s) and is not worth its quality cost.
fp8 on the MLP alone keeps about two thirds of the tok/s gain at 0.018 nats. int8 on the MLP
alone is a little faster than int8 everywhere (193 vs 182 tok/s); its memory was not measured.

**Where the step goes** (`AQ_PROFILE=1`, 16 agents, last report window, us per call):

| Op | bf16 | fp8 | int8 |
|---|---|---|---|
| `gemm_gate_up` | 1244 | 870 | 1892 |
| `gemm_down` | 645 | 461 | 1190 |
| `gemm_gdn_in` | 645 | 451 | 1029 |
| `gemm_qkv` | 520 | 403 | 732 |
| `gemm_lm_head` (bf16 in all three) | 3308 | 3408 | 3401 |
| `attn_dec` | 4774 | 4573 | 4918 |

Decode-only step fits from the 10-minute runs: base 77 ms (bf16), 62 (fp8), 127 (int8); the
per-key attention term is unchanged (138-163 ms per million keys). With fp8, decode attention
is the largest single op (30% of step time) and the bf16 head is 8%: those two, not the layer
GEMMs, are the next targets. The profiled GEMM gain (~30%) is well under the 2x the bare
cuBLASLt timing shows, because the quantize and rescale passes sit around each small GEMM
(fusing them is open).

**Quality** is measured teacher-forced against transformers on three real trace turns (8k,
16k, 26k prompt tokens, 256 greedy tokens each; `scripts/golden.py --ids-file`, goldens
`~/aq/golden/qwen35-9b-trace-{a,b,c}.safetensors` on the Spark; `aq-check` prints
KL(ref||ours)). Mean KL in nats over the three, and top-1 agreement out of 768:

| Variant | Mean KL | Top-1 |
|---|---|---|
| bf16 | 0.002 | 764 |
| FP8, every layer GEMM | 0.028 | 739 |
| FP8, MLP only | 0.018 | 750 |
| FP8, one GEMM kind only (down / GDN in / qkv / gate-up) | 0.007 / 0.010 / 0.011 / 0.015 | |
| bf16 GEMMs, weights rounded through E4M3 (`AQ_QSIM=w:f`) | 0.022 | 751 |
| bf16 GEMMs, activations rounded through E4M3 (`a:f`) | 0.022 | 740 |
| bf16 GEMMs, weights int8 per output row (`w:i`) | 0.009 | 749 |
| bf16 GEMMs, activations int8 per row (`a:i`) | 0.141 | 710 |
| bf16 GEMMs, weights int8 with a scale per 32 (`w:b`) | 0.003 | 763 |
| bf16 GEMMs, weights int8 with row x column power-of-two scales (one pass; code not kept) | 0.059 | 739 |

So: E4M3's ~4 significant bits are the limit (weights and activations cost the same, no GEMM
is the culprit); linear int8 activations are unusable; block-scaled int8 weights with bf16
activations are at the noise floor. The fast FP8 path therefore costs ~0.03 nats and ~3% of
top-1 tokens, some at reference margins of 2-5 logits (often message / tool-call terminators).

**Open design question** (the user's call): how to multiply 8-bit weights by 16-bit
activations. Candidates: (a) int8 GEMM with each activation row split into two int8 digit
rows: decode ~50-58 ms of GEMM vs 78, but the i32 output pass makes it a loss for prefill, and
a per-32 scale along K cannot be factored out, which leaves per-row scales (0.009) since
one-pass row x column scaling measured worse (0.059);
(b) dequantize each weight to a bf16 scratch buffer and use the bf16 GEMM: clean quality and
half the memory, but ~2.3x the decode weight traffic; (c) our own tensor-core GEMM reading
int8 weights. `AQ_QSIM` and the `qsim` kernel are diagnostics to delete once this is settled.

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
the driver recovers: `~/aq/logs/mem_gate.log`; sweeps: `a local sweep script`.

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
(made by `scripts/golden.py`): teacher-forced logits at every generated position, free-running and
batched greedy, session resume, shared-prefix fork. Divergences are classified by top-2 margin
(near-tie < 0.25). Always run the full suite: 6 goldens × {default, `AQ_CHECK_BUDGET=1000
AQ_CHECK_CHUNK=777`} (the odd budget found a real upload/compute stream race), plus `AQ_SPEC_K=4`.
(2026-11-15: cuBLASLt plans are no longer picked by timing, so runs are repeatable without a switch.)

## Open branches (not merged)
- `worktree-agent-ae9e174bba1af161f` @ d7e410c — lossless KV compression (400 B rows, 1.28× tokens,
  logits byte-identical to raw). Rebased on 7771327, opt-in (`AQ_KV_COMPRESS=1`). **Gate not yet run**
  (B200 driver hang): run `scripts/kvc_gate.sh` before merging. Decode attention is ~2.3× slower than
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
  `~/aq/results/agentic/spark-*` on the Spark, log `~/aq/logs/run_spark_queue.log`.
  It uses a pre-speculation build (`aq-serve-current` = session cache only).
- Progress page: https://claude.ai/artifact/MNkgXru1gcTGbZz3Zw3zwN (data files under `docs/progress/`,
  regenerate `bench.json`/`agentic.json` with `scripts/collect.py`, `AQ_GPU_LABEL=GB10` on the Spark).

## Machines and gotchas
- Two machines were used: a B200 box (x86_64, `sm_100`, shared with other users) and a DGX Spark
  (aarch64, `sm_121`, unified memory). How to reach them is not in this repo; ssh wrappers and
  sync scripts live in an untracked `.local/` directory.
- Both use the user-space toolchain from `scripts/setup.sh` under `~/aq`; `source scripts/env.sh`,
  then `cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins` (the
  numbers in this file were measured with `--arch sm_100a` / `sm_121a` builds; `sm_80` runs at
  the same speed).
- On a shared machine: never `pkill`/`pgrep -f` (self-matches, other users); kill only your own
  PIDs; use `grep "[p]attern"` in wait loops; pick an idle GPU first.
- Nsight Compute counters may be blocked (`ERR_NVGPUCTRPERM`); use `AQ_PROFILE=1|decode|prefill`
  (CUDA events, works inside graphs) and `AQ_KINFO=1` (registers/spills per kernel).
- vLLM baselines: `scripts/baseline_vllm.sh` (readiness wait 900 s, `--temperature 0`),
  `scripts/agentic_sweep.sh`, `scripts/agentic_bench.py`.
- Correctness gate: `scripts/gate.sh <aq-check> <reference dir>`.

## Research
`docs/research/sota-survey.md`, `docs/research/moonshots.md`, `docs/research/compression-and-skinny-gemm.md`.
Top unexplored ideas: prompt-end snapshots + runahead for chat-template re-renders, cooperative scan
over shared prefixes, batch-invariant numerics, NVMe session tier (safetensors I/O), green-context
SM partitioning on the Spark, FP8 weights for Qwen3.8-27B.
