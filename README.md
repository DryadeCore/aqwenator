# aqwenator

An inference server written for one model family: Qwen3.5 hybrid (Gated DeltaNet + attention)
models, on NVIDIA GPUs. Rust host, kernels in Rust compiled to PTX with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), GEMMs through cuBLASLt. It targets
agentic workloads (many long multi-turn sessions) on a single GPU, down to a DGX Spark.

Outputs are checked token for token against `transformers` (`aq-check`).

The aim is good models at good speed on hardware that should not be able to run them, with
many agents sharing one box. The focus is Qwen models for the moment; other models get added
when they fit the existing kernels.

## Numbers

Qwen3.5-9B, replay of recorded OpenHands trajectories, 10 minutes per run, DGX Spark:

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) |
|---|---|---|---|---|
| 16 | aqwenator | 91.9 | 229 | 483 / 3864 |
| 16 | vLLM 0.33, MTP speculation | 57.4 | 122 | 1022 / 2909 |
| 8 | aqwenator | 57.4 | 157 | |
| 8 | vLLM 0.33, MTP speculation | 44.6 | 102 | |

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

A 4-bit AutoRound / GPTQ checkpoint is loaded as it is after one conversion:

```sh
python scripts/convert_autoround.py <Intel/Qwen3.5-9B-int4-AutoRound dir> <Qwen/Qwen3.5-9B dir> qwen35-9b-int4
aq-serve qwen35-9b-int4
```

4.1 bits per layer weight; on long agent traces about 92% of greedy tokens agree with the
bf16 model (KL 0.08–0.15 nats), which is that checkpoint's quality, not something the engine
adds. The embedding table and the output head are stored in 8 bits. The 9B then needs 7.3 to
8.6 GB of device memory depending on the settings (see docs/HANDOFF.md): not yet comfortable
on an 8 GB card.

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

## License

Apache-2.0, see [LICENSE](LICENSE).
