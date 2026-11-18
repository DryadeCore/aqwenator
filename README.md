# aqwenator

An inference server written for one model family: Qwen3.5 hybrid (Gated DeltaNet + attention)
models, on NVIDIA GPUs. Rust host, kernels in Rust compiled to PTX with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), GEMMs through cuBLASLt. It targets
agentic workloads (many long multi-turn sessions) on a single GPU, from a datacenter card
down to a DGX Spark, with a 4-bit path meant for 8 GB cards.

Outputs are checked token for token against `transformers` (`aq-check`).

The aim is good models at good speed on hardware that should not be able to run them, with
many agents sharing one box. The focus is Qwen models for the moment; other models get added
when they fit the existing kernels.

## What is in it

- Sessions stay on the device between turns: paged attention cache, recurrent state slots,
  prompt checkpoints, eviction of the least recently used session. A returning agent only
  pays for its new tokens.
- Speculative decoding with the model's own draft head (MTP, chained) and prompt-lookup drafts.
- Tensor-core kernels for prefill and decode attention and for integer-weight GEMMs.
- Weight tiers: bf16 (exact), `fp8`, `int8`, and 4-bit AutoRound / GPTQ checkpoints.
- Small-card options: 8-bit attention cache, 8-bit output head, embedding table in host memory.

## Numbers

Qwen3.5-9B, replay of recorded OpenHands trajectories, 10 minutes per run, DGX Spark:

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) |
|---|---|---|---|---|
| 16 | aqwenator | 91.9 | 229 | 483 / 3864 |
| 16 | vLLM 0.33, MTP speculation | 57.4 | 122 | 1022 / 2909 |
| 8 | aqwenator | 57.4 | 157 | |
| 8 | vLLM 0.33, MTP speculation | 44.6 | 102 | |

Same model and replay on one B200, turns/min:

| Agents | aqwenator | vLLM 0.33, MTP speculation | vLLM 0.33 | llama.cpp 1.0-dev (f16) |
|---|---|---|---|---|
| 1 | 28.2 | 27.5 | 24.2 | 20.5 |
| 4 | 114.8 | 105.8 | 91.8 | 51.2 |
| 16 | 432.8 | 384.5 | 313.0 | 80.2 |
| 32 | 738.5 | 795.2 | 576.2 | 79.5 |

vLLM is ahead at 32 agents, and has the better TTFT p99 at 1, 16 and 32.

Cold start, 9B on the Spark: first token 3.2 s after launch, page cache warm or cold.

### 8-bit weights (opt-in)

`AQ_QUANT=fp8|int8` quantizes the layer weights at load. Same replay, same machine, 16 agents:

| Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | Weights + 4k-token pool | KL vs transformers (nats) |
|---|---|---|---|---|---|
| bf16 (default, exact) | 91.5 | 228 | 491 / 4550 | 18.1 GiB | 0.002 |
| `fp8` (E4M3 weights and activations) | 113.7 | 295 | 374 / 3374 | 11.7 GiB | 0.028 |
| `int8` (block-scaled weights, bf16 activations) | 71.6 | 182 | 619 / 3946 | 12.2 GiB | 0.003 |

vLLM 0.33 serving a published FP8 checkpoint of the same model (MTP speculation) does 51.9
turns/min and 131 tok/s on this replay (TTFT 1052 / 3739 ms).

`fp8` is the fast one and changes about 3% of greedy tokens; `int8` is near the bf16 noise
floor and trades speed for memory.

### 4-bit weights (pre-quantized checkpoint)

A 4-bit AutoRound / GPTQ checkpoint is loaded as it is after one conversion (needs `torch`
and `safetensors`, and both checkpoints on disk):

```sh
hf download Qwen/Qwen3.5-9B
hf download Intel/Qwen3.5-9B-int4-AutoRound
python scripts/convert_autoround.py <Intel/Qwen3.5-9B-int4-AutoRound dir> <Qwen/Qwen3.5-9B dir> ~/models/qwen35-9b-int4
```

The engine multiplies the 4-bit codes directly, stores the output head in 8 bits and leaves
the embedding table in host memory. For a small card, give it few sessions, a small attention
cache and short prefill chunks:

```sh
docker run --gpus all -p 8000:8000 -v ~/models:/models:ro \
  ghcr.io/<owner>/aqwenator /models/qwen35-9b-int4 \
  --max-seqs 4 --max-running 2 --kv-gb 1 --max-tokens 1024
```

| 9B, 4-bit, settings above | Engine allocates |
|---|---|
| with the draft head | 7.0 GiB |
| with the draft head, `-e AQ_KV=int8 --kv-gb 0.5` (about the same context) | 6.5 GiB |
| without the draft head (`-e AQ_MTP=0`) | 6.0 GiB |

plus the CUDA context (0.7 GiB on a B200). `-e AQ_KV=int8` stores the attention cache in 8
bits: the same `--kv-gb` then holds 1.9 times the context, at no measurable cost in quality. Measured on a B200 only so far, where it does 339
turns/min at 16 agents (bf16: 433); an 8 GB card has not been tried yet.

Quality is the checkpoint's: on 20k-token agent traces about 92% of greedy tokens agree with
the bf16 model (KL 0.08–0.15 nats), against 99% for the 8-bit tiers. That is the price of 4
bits on this model, calibrated or not; the engine adds nothing to it.

Method, more machines and the history are in [docs/HANDOFF.md](docs/HANDOFF.md).

## Run

The model must be in the local Hugging Face cache (`hf download Qwen/Qwen3.5-9B`), or pass a
directory. Needs the NVIDIA container toolkit and a driver for CUDA 13.

```sh
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/aq/.cache/huggingface:ro \
  ghcr.io/<owner>/aqwenator Qwen/Qwen3.5-9B --kv-gb 24
```

```sh
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

Endpoints: `/v1/completions` (text or token-id prompts, `stream`), `/v1/models`, `/health`.
`aq-serve <model> --help`-style options are listed in `crates/aq-server/src/main.rs`; the ones
that matter are `--max-seqs`, `--max-ctx`, `--max-tokens` (prefill tokens per step) and
`--kv-gb` (attention cache pool). Environment: `AQ_QUANT=fp8|int8` (weight tier),
`AQ_KV=int8` (8-bit attention cache), `AQ_MTP=0` (no draft head).

## Build

```sh
docker build -t aqwenator .                                         # image
docker build --target binaries --output type=local,dest=bin .       # binaries only
```

Without Docker: CUDA 13 toolkit, clang 21, Rust 1.99.1 and `cargo-oxide` (see
`scripts/setup.sh`), then

```sh
source scripts/env.sh
cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins
```

Binaries: `aq-serve` (server), `aq-check` (correctness against reference data from
`scripts/golden.py`), `aq-bench` (synthetic load), `aq-ceiling` (hardware limits: memory
bandwidth, GEMM rate, and what they allow per step).

## Release

Pushing a tag `vX.Y.Z` builds amd64 and arm64 images, publishes
`ghcr.io/<owner>/aqwenator:X.Y.Z` and `:latest`, and attaches the binaries to a GitHub release.

## License

Apache-2.0, see [LICENSE](LICENSE).
