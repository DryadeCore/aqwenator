# aqwenator

An inference server written for one model family: Qwen3.5 hybrid (Gated DeltaNet + attention)
models, on NVIDIA GPUs. Rust host, kernels in Rust compiled to PTX with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), GEMMs through cuBLASLt. It targets
agentic workloads (many long multi-turn sessions) on a single GPU, down to a DGX Spark.

Outputs are checked token for token against `transformers` (`aq-check`).

## Numbers

Qwen3.5-9B, replay of recorded OpenHands trajectories, 10 minutes per run, DGX Spark:

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) |
|---|---|---|---|---|
| 16 | aqwenator | 91.9 | 229 | 483 / 3864 |
| 16 | vLLM 0.33, MTP speculation | 57.4 | 122 | 1022 / 2909 |
| 8 | aqwenator | 57.4 | 157 | |
| 8 | vLLM 0.33, MTP speculation | 44.6 | 102 | |

Cold start, 9B on the Spark: first token 3.2 s after launch, page cache warm or cold.

Method, more machines and the history are in [docs/HANDOFF.md](docs/HANDOFF.md).

## Run

The model must be in the local Hugging Face cache (`hf download Qwen/Qwen3.5-9B`), or pass a
directory. Needs the NVIDIA container toolkit and a driver for CUDA 13.

```sh
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/<owner>/aqwenator Qwen/Qwen3.5-9B --kv-gb 24
```

```sh
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

Endpoints: `/v1/completions` (text or token-id prompts, `stream`), `/v1/models`, `/health`.
`aq-serve <model> --help`-style options are listed in `crates/aq-server/src/main.rs`; the ones
that matter are `--max-seqs`, `--max-ctx`, `--max-tokens` (prefill tokens per step) and
`--kv-gb` (attention cache pool).

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
