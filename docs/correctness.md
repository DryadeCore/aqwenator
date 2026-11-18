# Correctness

Correctness comes first: a change that is faster and wrong is not a change. The default
configuration (bf16 weights and cache) is checked against transformers, and every faster path
is checked against the plain one.

## Reference files

`scripts/golden.py` runs a checkpoint in transformers and writes one safetensors file: the
prompt, the residual stream after every layer, the logits at the last prompt position, a
greedy continuation, and the teacher-forced logits at each of its positions.

```sh
python -I scripts/golden.py --model Qwen/Qwen3.5-9B --out golden/qwen35-9b.safetensors
python -I scripts/golden.py --model Qwen/Qwen3.5-9B --long --out golden/qwen35-9b-long.safetensors
python -I scripts/golden.py --model Qwen/Qwen3.5-9B --len 16384 --no-hidden --attn sdpa \
    --out golden/qwen35-9b-16k.safetensors
```

The gate uses three prompts per model: 19 tokens, about 1.8k and about 16k.

## `aq-check`

```sh
aq-check <model id or dir> <reference.safetensors>
```

It exits non-zero on any real miss. What it runs:

1. **Layers**: the residual stream after prefill, layer by layer.
2. **Teacher-forced logits**: the reference's greedy tokens fed one at a time through the
   decode path, every logits row compared (top-1, relative error, KL).
3. **Free-running greedy** against the reference, single sequence, then three sequences
   batched and staggered against the single run.
4. **Session cache**: a second turn resumed from the cache against a cold prefill, prompt
   checkpoints, shared-prefix snapshots and forks, and the same under memory pressure.
5. **Speculative decoding**: output with drafts on must be token-identical to drafts off,
   single and batched.

A top-1 disagreement is a near-tie, not a miss, when the top-2 margin at that position is
below 0.25 logits (one bf16 ulp at typical logit magnitudes is 0.06–0.125).

**Mixture-of-experts models** amplify rounding (see [models](models.md#sensitivity-to-rounding)),
so for them the identity checks run with every GEMM pinned to one plan, and the near-tie
margin is one logit. Comparison against the reference is unchanged in kind: teacher-forced
logits and KL.

## The gate

```sh
scripts/gate.sh <aq-check binary> <directory of reference files>
```

It runs `aq-check` on every reference file in six configurations: the default, shortest-first
prefill with bursts, a small step budget with an odd chunk size, the draft head off, a single
chained draft, and the 8-bit attention cache. A last leg injects a wrong speculative token
and must fail. Reference files are named after the model
(`qwen35-9b-long.safetensors` for `Qwen/Qwen3.5-9B`), which must be in the local Hugging Face
cache.

The gate runs before every commit that touches the engine or the kernels. It needs a GPU, so
CI does not run it: CI builds the image.

Bugs the gate has caught that no benchmark would have: a race between upload and compute
streams (found by the odd step budget), a fork that reused a stale snapshot index under
memory pressure, and a read past a cache page in the first 8-bit cache kernel.

## Quantized tiers

They are not exact, so they are measured instead of gated: teacher-forced KL and top-1
agreement against transformers on turns of recorded agent traces (`golden.py --ids-file`).
Results are in [quantization](quantization.md). Their internal consistency (batched, resumed
and speculative paths against the plain one) is still checked by `aq-check`.

## Diagnostics

| Variable | Effect |
|---|---|
| `AQ_CHECK_TF_ONLY=1` | Teacher-forced comparison only |
| `AQ_CHECK_ROWDEP=N` | N copies of one prompt in the same steps; reports any difference between rows |
| `AQ_CHECK_SPEC_REPEAT=N` | Repeat the speculation test N times |
| `AQ_CHECK_DRIFT_AT=seq:token` | Compare the decode and prefill paths at one position |
| `AQ_SPEC_FAULT=N` | Corrupt a speculative token; the check must fail |
| `AQ_SPEC_TRACE=1` | Log every draft and its acceptance |
| `AQ_GEMM_MIN_M=N` | Pin GEMMs below N rows to one plan |
| `AQ_ATTN_NAIVE=1` | Reference attention kernels |
