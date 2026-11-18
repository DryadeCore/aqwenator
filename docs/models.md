# Models

aqwenator runs the Qwen3.5 hybrid family: three Gated DeltaNet layers for every attention
layer, zero-centered RMSNorm, SwiGLU MLPs, an optional multi-token-prediction (MTP) draft head.
The focus is Qwen models for the moment; others are added when they fit the existing kernels.

| Model | Config type | Status |
|---|---|---|
| Qwen/Qwen3.5-0.8B | `qwen3_5` | In the correctness gate |
| Qwen/Qwen3.5-9B | `qwen3_5` | In the correctness gate; the model most numbers are measured on |
| Qwen/Qwen3.6-35B-A3B | `qwen3_5_moe` | In the correctness gate; bf16 only |
| Other dense `qwen3_5` checkpoints (up to 27B) | `qwen3_5` | Same layout, not run |

The model is given as a Hugging Face id found in the local cache, or as a directory. The
vision tower of a checkpoint is ignored: text only.

What the kernels assume, checked at load: 256-wide attention heads, 1, 2, 4 or 8 query heads
per KV head, 128-wide Gated DeltaNet heads, a width-4 convolution, and for a
mixture-of-experts model 256 experts with 8 per token and a shared expert of the experts'
shape.

## Mixture of experts

`qwen3_5_moe` replaces the MLP with 256 experts of width 512, 8 per token, plus a shared
expert behind a sigmoid gate. Qwen3.6-35B-A3B is 67 GiB in bf16, of which the experts are
about 64.

Per layer, with nothing going back to the host:

1. A router GEMM gives 256 expert logits and the shared expert's gate logit per token.
2. `moe_route` takes the softmax, keeps the top 8 and renormalizes them.
3. `moe_bucket` groups the step's (token, expert) rows by expert into jobs of up to 64 rows.
   The shared expert is treated as expert 256 and takes every token.
4. `moe_gemm` runs each job against its expert's weights on tensor cores: once for gate/up,
   with SiLU and the product in its epilogue, once for down.
5. `moe_combine` scales and adds the 8 rows of each token, then the shared expert's.

The launch grid is the same whatever the routing (unused jobs return at once), so steps are
still captured as graphs.

### Sensitivity to rounding

Router logits are bf16 and often tie at the eighth place, so a one-ulp difference upstream
changes which experts run. This is a property of the model, not of an engine: transformers
differs from itself between its two attention implementations by 0.0081 nats of mean KL on a
1.8k-token prompt, against 0.00015 for the 9B. aqwenator sits at 0.0008, 0.0064 and 0.0037
nats against transformers on prompts of 19, 1.8k and 16k tokens.

Two consequences:

- On this model the tokens of a request can depend on what else is in the step, as with any
  batching server. Rows of one step are bit-identical to each other, and speculative decoding
  stays token-identical to plain decoding.
- `aq-check` runs mixture-of-experts models with every GEMM pinned to one plan and a wider
  near-tie margin (see [correctness](correctness.md)).

### Speed and what limits it

See [results](results.md). In short: level with vLLM without
speculation, behind vLLM with MTP speculation. A drafted row is not nearly free on this
model, because each row reads its own 8 experts (6.3 MB each in bf16), and at 32 agents the
expert passes read most of the 64 GB every step.

Quantized tiers are not available for this model yet.
