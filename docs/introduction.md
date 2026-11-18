# aqwenator

An inference server written for one model family: Qwen3.5 hybrid models (Gated DeltaNet +
attention), dense and mixture-of-experts, on NVIDIA GPUs. The host is Rust, the kernels are
Rust compiled to PTX with [cuda-oxide](https://github.com/NVIDIA/cuda-rust), and GEMMs go
through cuBLASLt.

It is built for agents: many long, multi-turn sessions sharing one GPU. The aim is good
models at good speed on hardware that should not be able to run them. Outputs are checked
token for token against `transformers`.

```sh
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/aq/.cache/huggingface:ro \
  ghcr.io/mcpatate/aqwenator Qwen/Qwen3.5-9B --kv-gb 24
```

The source, releases and issue tracker are on
[GitHub](https://github.com/McPatate/aqwenator).

| Page | Contents |
|---|---|
| [Architecture](architecture.md) | Steps, kernels, session cache, scheduling, speculation |
| [Models](models.md) | Supported models, the mixture-of-experts block |
| [Quantization](quantization.md) | `fp8`, `int8`, 4-bit checkpoints, 8-bit cache, small cards |
| [Benchmarks](benchmarks.md) | Method, hardware bounds, where the time goes |
| [Results](results.md) | Every published run, generated |
| [Correctness](correctness.md) | Reference files, `aq-check`, the gate |
| [Development](development.md) | Build, options, scripts, kernel notes, release |
| [Roadmap](roadmap.md) | Direction and what is next |
