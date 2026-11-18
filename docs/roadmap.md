# Roadmap

## Direction

The target is the moment of "this should not run on this hardware", at good speed:
high-quality models on constrained machines, using the whole machine (VRAM, RAM, SSD) and
the structure of the model to move bytes to where they fit. Speed stays a requirement, not a
trade.

The edge to build on is many agents on one box: the session cache, prompt checkpoints and
exact speculation are where the engine is measurably ahead of llama.cpp and, on
bandwidth-bound hardware, of vLLM.

Scope is Qwen models for the moment. Other models are added while they reuse the existing
kernels and scheduler; one that needs significant rework is a decision, not a default yes.
The code stays lean: the fastest path only.

The engines to compare against are Strata, llama.cpp and gufo, with vLLM as the server
baseline. Where another engine is already good (one user, the largest model on a gaming PC),
measure it before building the same thing.

## Next

1. **Experts across VRAM, host RAM and disk** under a device-memory budget, for
   Qwen3.6-35B-A3B on a small card. Each token uses 8 of 256 experts per layer, so the hot
   ones can live on the device and the rest be uploaded or computed on a miss. Published
   4-bit checkpoints of the model exist; fewer bytes per expert is also the only lever left at
   high load on a large card.
2. **A measurement on a consumer card.** The 4-bit 9B is sized for 8 GB and has only run on a
   B200.
3. **An expert kernel for small steps** that reads weights from global memory with every warp
   of the device, where the tensor-core kernel is bound by the fixed cost of its ring.
4. **Parked sessions in host RAM, then on SSD**, so evicted sessions and restarts resume from
   the longest exact prefix instead of running prefill again.
5. **Scaling past 32 agents on a large card**, where vLLM keeps going and aqwenator levels
   off: profile at 64 agents first; prefill attention split across blocks is the known item.
6. **Strata under the multi-agent replay**, on a model and machine it supports.

## Later

- Publish the kernels (gated delta rule, attention) on the Hugging Face Hub through
  `kernel-builder`, which supports kernels written with the same compiler; transformers
  already looks there for a gated delta rule kernel.
- More measurements: memory and energy per token at the operating point, a task-level
  quality score for the tiers that are not exact, open-loop load, prefill rate reported on
  its own.
- Qwen3.8-Flash-Next (hyper-connections, sparse attention, an n-gram table on disk).
- Quantized tiers for mixture-of-experts models.
- Draft lengths re-tuned on a bandwidth-bound device.

