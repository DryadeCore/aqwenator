# Architecture

aqwenator is three crates:

| Crate | What it holds |
|---|---|
| `aq-kernels` | Device kernels, written in Rust and compiled to PTX by [cuda-oxide](https://github.com/NVIDIA/cuda-rust) |
| `aq-engine` | Model loading, the forward pass, the scheduler and session cache, and the tools `aq-check`, `aq-bench`, `aq-ceiling` |
| `aq-server` | `aq-serve`: the HTTP front end |

The engine runs on one OS thread that owns the CUDA context and loops over steps. HTTP handlers
submit requests over a channel and receive token events back; text never reaches the engine
thread.

## A step

A step is one forward pass over a batch of rows. Rows are prefill tokens (a prompt is cut into
chunks of at most `--max-tokens`), the next token of each decoding sequence, and its drafted
tokens. Sequences join and leave between steps (continuous batching).

Every buffer of a step lives in one preallocated arena, so addresses are stable and a decode
step of a given shape is captured once as a CUDA graph and replayed.

Per layer, a step runs:

- **Gated DeltaNet layers** (three of every four): input projection (GEMM), a width-4 causal
  convolution, the gated delta rule, a gated norm, output projection (GEMM).
- **Attention layers**: q/k/v projection (GEMM), norm and partial RoPE, attention over the
  sequence's cached keys and values, output projection (GEMM).
- **MLP**: gate/up GEMM, SiLU and product, down GEMM; or the expert block of a
  mixture-of-experts model (see [models](models.md)).

## Kernels

GEMMs go through cuBLASLt, with one plan per shape and row-count bucket. Plans are the
heuristic's first candidate, never picked by timing: timing made rounding differ between
processes on a busy GPU. Everything else is a kernel of ours:

| Kernel | Role |
|---|---|
| `attention_tc` | Prefill attention on tensor cores: one block per (query tile, KV head), online softmax, K/V tiles fetched with `cp.async`, two buffer pairs where shared memory allows |
| `attn_decode_tq` | Decode attention: one block per (sequence, KV head, slice of the context) serves the token, its drafts and every query head of the group, so K/V is read once per sequence |
| `gdn_tc_prep` / `gdn_tc_scan` / `gdn_tc_out` | Chunked gated delta rule for prefill, on tensor cores |
| `gdn_decode` | Gated delta rule for one token |
| `gemm_i8w` | GEMM that reads 8-bit or 4-bit weights as stored (see [quantization](quantization.md)) |
| `moe_route`, `moe_bucket`, `moe_gemm`, `moe_combine` | Mixture-of-experts block |
| `add_rmsnorm`, `silu_mul`, `conv1d_*`, `argmax_*` | Vectorized element-wise work |

Decode attention splits the context into slices at fixed absolute key positions, so a row's
result does not depend on the batch it is in or on its drafts.

## Session state

A sequence owns:

- **Attention cache pages.** Keys and values live in a pool of 16-token pages, reference
  counted, sized by `--kv-gb`.
- **One state slot** holding the recurrent state of every Gated DeltaNet layer (49 MiB for the
  9B). `--max-seqs` is the number of slots.

When a request ends its session is parked, not freed. A later request whose prompt extends a
parked session resumes from it and only runs its new tokens. Three mechanisms make that hit
more often:

- **Prompt checkpoints**: during prefill the state is also saved at a point inside the chunk,
  so a follow-up that shares most of the prompt resumes from there.
- **Shared-prefix snapshots**: when several prompts share a long prefix, its state is
  snapshotted once and the others fork from it, sharing its pages.
- **A slot reserve**: before admission, parked sessions are evicted (least recently used
  first) until an eighth of the slots is free. Without it a large cache pool let parked
  sessions fill every slot and new checkpoints were skipped; on a B200 at 32 agents that cost
  a third of the throughput.

Eviction is least recently used; `--evict`, `--pin-snapshots` and `--load-control` change the
policy.

## Scheduling

Prefill is ordered shortest first. Every few steps the longest-waiting long prompt gets a
whole step to itself (`--burst-prefill`, `--burst-every`), which bounds the wait of large
turns without moving the inter-token latency of the others.

Two modes are opt-in because they only pay on a card with spare capacity: a background
prefill lane on a second stream (`AQ_BG_PREFILL`), and decode/prefill co-execution
(`AQ_COEXEC`).

## Speculative decoding

Drafts come from three sources: lookup in the sequence's own prompt, an n-gram model shared
across sessions, and the model's own draft head (MTP) when the checkpoint has one, chained up
to three tokens (`AQ_MTP_K`). The head scores a shortlist of 32,768 tokens
(`AQ_MTP_SHORT`). `--spec-k` is the draft budget per sequence.

Drafted rows run in the same step as the token they follow, and a draft is accepted only if
it equals the model's own choice, so output is token-identical to plain greedy decoding.
`aq-check` verifies that on every run. On recorded agent traces the 9B emits 4.2 tokens per
decode pass.

On a dense model extra rows are nearly free where the card is bound by memory bandwidth: the
weights are read once per step whatever the row count. On a mixture-of-experts model each row
brings its own experts.

## Loading

Weights are read with direct I/O into pinned staging buffers while device memory is being
allocated. The 9B on a DGX Spark has its weights resident 2.4 s after launch with a warm or
cold page cache, and emits its first token at 3.2 s.

## Tried and dropped

- Lossless compression of the attention cache: 1.28 times the tokens per GB, decode attention
  2.3 times slower.
- Skipping the value read of key tiles with negligible attention weight: on agent traces
  99.9996% of tiles are needed even at a threshold of 2^-20.
- Custom "skinny" decode GEMMs in bf16: did not beat cuBLASLt.
- Deeper draft chains: the head feeding on its own guesses limits positions 2 and later, not
  the row budget.
- Fusing small kernels in decode, an exact output-head shortcut: 3–5% of a step each, not
  worth the code.
