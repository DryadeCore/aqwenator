# Benchmarks

## Method

The benchmark is a replay of recorded agent sessions
([`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)):
40 sessions, mean context around 20k tokens. Each simulated agent walks one session: for every
assistant turn it sends everything before it as token ids, asks for as many tokens as the
recorded turn had, then pauses 0.5 to 3 s. The next prompt extends the previous prompt with
the recorded assistant message, not with what the server generated, so a server that can only
resume from its own output misses its cache on every turn. Decoding is greedy with stop
tokens ignored.

```sh
scripts/trace_sweep.sh aq   <model> <traces.json> <tag>     # AGENTS="4 8 16" DURATION=240
scripts/trace_sweep.sh vllm <model> <traces.json> <tag>
scripts/baseline_llamacpp.sh <model dir> <traces.json> <tag>
```

Reported per run: completed turns per minute, generated tokens per second, time to first
token (TTFT), time per output token (TPOT), inter-token latency and the latency of a whole
turn. At one and four agents the replay itself bounds turns per minute for the fast engines,
so read the latencies there.

## Results

Results are in [results](results.md), a page generated from `docs/results.jsonl`. They are
not written by hand anywhere else, so a number always has a date, a device and a run length
next to it. To add a run:

```sh
python scripts/trace_bench.py run --traces traces.json --model <model> --agents 16 \
    --engine "aqwenator 0.3.0 (bf16)" --device "RTX 3070" --record docs/results.jsonl
python scripts/trace_bench.py report docs/results.jsonl > docs/results.md
```

The page shows the latest run of each engine label, model, device and agent count, and under
each table the largest recorded load at which each engine still meets a latency target
(`report --ttft-p99 1000 --tpot-p99 50` by default). `--repeats N` measures N windows back
to back and reports their spread.

Caveats: a row is a single run of 4 to 10 minutes, turns per minute moves by a few percent
between runs, the B200 is a shared machine, and the other engines run with settings taken
from their documentation, not tuned by their authors.

## What the hardware allows

`aq-ceiling` measures a device and derives lower bounds for one step of a model: every weight
byte read once, every multiply-add at the peak rate.

| | DGX Spark (GB10, 48 SMs) | B200 (148 SMs) |
|---|---|---|
| Read bandwidth | 237 GB/s | 6.7 TB/s |
| Peak bf16 GEMM | 98 TFLOP/s | 2.1 PFLOP/s |
| 9B weights read once (15.9 GB) | 67 ms | 2.4 ms |
| Attention cache read, 16 sequences of 20k tokens (10.5 GB) | 44 ms | 1.6 ms |

The 9B reads 15.9 GB of weights per step (the output head alone is 2.0 GB), 32 KiB of
attention cache per context token, and keeps 49 MiB of recurrent state per sequence.

- **Spark:** a decode step takes 84 ms single-stream against a 70 ms floor, and 141 ms at 16
  sequences of 16k tokens against about 110 ms. More throughput has to come from more tokens
  per weight pass (drafts, batching) or fewer bytes per pass (quantized weights).
- **B200:** 45–58% of the bound before the decode attention kernel; no later measurement
  against the bound.

## Where the time goes (November 2026)

**9B on the Spark, 16 agents** (share of GPU time):

| Share | What | Against its limit |
|---|---|---|
| 40% | weights read once per step | 78 ms against a 67 ms floor |
| 33% | prefill | GEMMs at 88% of the FLOP bound |
| 18% | attention cache read by decode | 214 GB/s against 237 GB/s |
| 9% | per-row compute | a third of the rows are rejected drafts or re-fed tokens |

**9B on the B200, 32 agents.** Steps that carry prefill are about half of the GPU time, and
decoding sequences wait through them: that is the TPOT gap to vLLM. In those steps prefill
attention is 50%. It runs at about 6% of the card's GEMM rate: one warp per 16 tokens and
head scans the whole context serially. Splitting the key range across blocks with a combine
pass, as the decode kernel does, is the open item.

**35B-A3B on the B200.**

- One agent: the expert passes take about 100 us per layer of about 285 us, and vLLM's whole
  layer is about 77 us. The gate/up pass fits `38 us + 0.12 us per MB` of expert weights: a
  fixed cost per block of walking its ring, then about 8 TB/s.
- 32 agents: decode attention 38–40%, expert gate/up 22–24%, expert down 12–13%. The expert
  passes then read most of the 64 GB every step and are near the card's bandwidth.

## Known gaps (November 2026)

- On a B200 with the 9B, vLLM with MTP speculation is ahead from 32 agents and keeps scaling
  to at least 96 (1432 turns per minute, within the latency target), while aqwenator levels
  off near 850 and leaves the target after 64. Not profiled at that load yet. The rows from
  32 agents up use 160 recorded sessions and up to 128 running sequences; the rows below use
  40 sessions and 32, so the two sets are not one curve.
- vLLM is ahead at every load with the 35B-A3B when it speculates.
- vLLM has the better TTFT p99 in most configurations.
- Nothing here was measured on a consumer card. The 8-bit and 4-bit kernels and the 8-bit
  attention cache were built for bandwidth-bound devices and have only run on a B200, where
  they are slower than bf16.
- Strata and gufo have not been run under this replay.
