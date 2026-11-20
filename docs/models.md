# Models

aqwenator runs the Qwen3.5 hybrid family: three Gated DeltaNet layers for every attention
layer, zero-centered RMSNorm, SwiGLU MLPs, an optional multi-token-prediction (MTP) draft head.
The focus is Qwen models for the moment; others are added when they fit the existing kernels.

| Model | Config type | Status |
|---|---|---|
| Qwen/Qwen3.5-0.8B | `qwen3_5` | In the correctness gate |
| Qwen/Qwen3.5-9B | `qwen3_5` | In the correctness gate; the model most numbers are measured on |
| Qwen/Qwen3.6-35B-A3B | `qwen3_5_moe` | In the correctness gate in bf16; 4-bit checkpoints, also with the experts paged ([quantization](quantization.md)) |
| Qwen/Qwen3.8-Flash-Next | `qwen4_exp` | First steps, 4-bit checkpoint only: see below |
| Other dense `qwen3_5` checkpoints (up to 27B) | `qwen3_5` | Same layout, not run |

The model is given as a Hugging Face id found in the local cache, or as a directory. The
vision tower of a checkpoint is ignored: text only.

What the kernels assume, checked at load: 256-wide attention heads, 128-wide Gated DeltaNet
heads, a width-4 convolution, and for a mixture-of-experts model at most 512 experts with at
most 16 per token and a shared expert of the experts' shape. The tensor-core attention
kernels take 1, 2, 4 or 8 query heads per KV head; another ratio runs on the reference
kernel, which is slow.

## Qwen3.8-Flash-Next

180B parameters, 6B of them used per token. What it adds to the Qwen3.5 layout and where it
stands here:

| Part | State |
|---|---|
| Four residual streams with a gated mix before each block | done |
| n-gram embedding table (95 GiB, 16 rows read per token, never on the device) | done; read from the checkpoint's own shards, or from an extract of the rows some prompts need |
| 512 experts, 10 per token | done, through the expert pool |
| Attention with 12 query heads per KV head | reference kernel only |
| Sparse attention (the 512 best blocks of 4 tokens) | not yet: contexts over 2048 tokens are wrong |
| Draft head | not loaded |

With `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` converted (only its routed experts are
4-bit) and every expert on the device, 71 GiB: against transformers on the same weights, a
19-token prompt gives 32 of 32 greedy tokens and a mean KL of 0.005 nats teacher-forced.
Running it with the experts paged has not been tried yet.

## Mixture of experts

`qwen3_5_moe` replaces the MLP with 256 experts of width 512, 8 per token, plus a shared
expert behind a sigmoid gate. Qwen3.6-35B-A3B is 67 GiB in bf16, of which the experts are
about 64; from a [4-bit checkpoint](quantization.md#4-bit-checkpoints) it loads in 18 GiB.

Per layer, with nothing going back to the host:

1. A router GEMM gives 256 expert logits and the shared expert's gate logit per token.
2. `moe_route` takes the softmax, keeps the top 8 and renormalizes them.
3. `moe_bucket` groups the step's (token, expert) rows by expert into jobs of up to 64 rows.
4. `moe_gemm` runs each job against its expert's weights on tensor cores: once for gate/up,
   with SiLU and the product in its epilogue, once for down. 4-bit experts are expanded as
   they are staged in shared memory.
5. The shared expert takes every token, so it runs as a dense MLP.
6. `moe_combine` scales and adds the 8 rows of each token, then the shared expert's.

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
