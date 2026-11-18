# aqwenator

An inference server written for one model family: Qwen3.5 hybrid models (Gated DeltaNet +
attention), dense and mixture-of-experts, on NVIDIA GPUs. The host is Rust, the kernels are
Rust compiled to PTX with [cuda-oxide](https://github.com/NVIDIA/cuda-rust), and GEMMs go
through cuBLASLt.

It is built for agents: many long, multi-turn sessions sharing one GPU. The aim is good
models at good speed on hardware that should not be able to run them.

Outputs are checked token for token against `transformers`.

## Quickstart

Needs the NVIDIA container toolkit and a driver for CUDA 13. The model must be in the local
Hugging Face cache (`hf download Qwen/Qwen3.5-9B`), or pass a directory.

```sh
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/aq/.cache/huggingface:ro \
  ghcr.io/mcpatate/aqwenator Qwen/Qwen3.5-9B --kv-gb 24
```

```sh
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

Endpoints: `/v1/completions` (text or token-id prompts, streaming), `/v1/models`, `/health`.
Decoding is greedy. Options are in [docs/development.md](docs/development.md#server-options).

## What is in it

- **Sessions stay on the device between turns.** Paged attention cache, recurrent state
  slots, prompt checkpoints and shared-prefix snapshots: a returning agent only pays for its
  new tokens.
- **Exact speculative decoding** with the model's own draft head and prompt-lookup drafts:
  4.2 tokens per decode pass on agent traces, token-identical to plain greedy decoding.
- **Tensor-core kernels** for prefill and decode attention, the gated delta rule,
  mixture-of-experts blocks and integer-weight GEMMs.
- **Weight tiers**: bf16 (exact), `fp8`, `int8`, and published 4-bit checkpoints.
- **Small-card options**: 8-bit attention cache, 8-bit output head, embedding table in host
  memory.

## Models

| Model | Status |
|---|---|
| Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-9B | Supported; quantized tiers measured on the 9B |
| Qwen/Qwen3.6-35B-A3B (mixture of experts) | Supported, bf16 |

Details in [docs/models.md](docs/models.md).

## Results

On a replay of recorded agent sessions with Qwen3.5-9B (November 2026): on a DGX Spark,
1.4 to 1.6 times the turns per minute of vLLM with MTP speculation at 8 and 16 agents; on a
B200, ahead of vLLM up to 16 agents and behind it at 32; several times llama.cpp everywhere.
vLLM usually has the better TTFT p99. Every run, with its date and device, is in
[docs/results.md](docs/results.md); the method is in [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

| Page | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Steps, kernels, session cache, scheduling, speculation |
| [Models](docs/models.md) | Supported models, the mixture-of-experts block |
| [Quantization](docs/quantization.md) | `fp8`, `int8`, 4-bit checkpoints, 8-bit cache, small cards |
| [Benchmarks](docs/benchmarks.md) | Method, hardware bounds, where the time goes |
| [Results](docs/results.md) | Every published run, generated |
| [Correctness](docs/correctness.md) | Reference files, `aq-check`, the gate |
| [Development](docs/development.md) | Build, options, scripts, kernel notes, release |
| [Roadmap](docs/roadmap.md) | Direction and what is next |

## Build

```sh
docker build -t aqwenator .
```

Building without Docker is described in [docs/development.md](docs/development.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
