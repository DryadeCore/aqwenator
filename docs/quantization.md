# Quantization

The default is the checkpoint's bf16 weights and a bf16 attention cache, checked token for
token against transformers. Everything on this page is opt-in and not exact.

| Tier | How to get it | Bytes per weight | Greedy tokens matching bf16 | Mean KL (nats) |
|---|---|---|---|---|
| bf16 | default | 2 | reference | 0.002 |
| `fp8` | `AQ_QUANT=fp8` | 1 | 96–97% | 0.028 |
| `int8` | `AQ_QUANT=int8` | 1.06 | 99% | 0.003 |
| 4-bit | a converted AutoRound / GPTQ checkpoint | 0.53 | about 92% | 0.08–0.15 |
| 8-bit attention cache | `AQ_KV=int8` | (cache: 272 bytes per row instead of 512) | 99–100% | 0.001–0.003 |

Quality is measured teacher-forced against transformers on three turns of recorded agent
traces (8k, 16k and 26k prompt tokens, 256 positions each) with the 9B; bf16's own figure is
the noise floor of that comparison. `fp8` applies to dense models only; `int8` also takes
the weights of a mixture-of-experts model outside its experts, alone or next to 4-bit
experts (for Qwen3.8-Flash-Next, 2.3 GiB less on the device at the same quality).

## `fp8`

E4M3 weights with a power-of-two scale per output channel, quantized at load. Activations
are quantized per row before each GEMM and the output rescaled after, so cuBLASLt multiplies
8-bit by 8-bit. It is the fast tier on a bandwidth-bound device: on a DGX Spark it is a
quarter to a third faster than bf16, with better latency throughout (see
[results](results.md)).

The cost is E4M3's four significant bits: about 3% of greedy tokens change, some at reference
margins of 2–5 logits. Weights and activations contribute equally and no single GEMM is the
culprit. `AQ_QUANT_SET` restricts it to some GEMMs (`ud` = MLP only: two thirds of the gain
at 0.018 nats). An FP8 output head (`AQ_QUANT_HEAD=1`) buys nothing measurable.

## `int8`

int8 weights with one scale per 32 values, bf16 activations: at the noise floor, half the
memory. Two execution paths:

- **Kernel** (`gemm_i8w`): for GEMMs of at most 96 rows on large weights, the weight bytes go
  straight to the integer tensor cores. Activations are split into two int8 digit planes with
  a scale per 32-wide block.
- **Expansion**: otherwise 1024 weight rows at a time are expanded to bf16 and multiplied by
  the bf16 GEMM.

On a B200 it is slower than bf16: that card is not short of bandwidth. The kernel exists for devices that are, where it reads half the
bytes; that has not been measured yet.

## 4-bit checkpoints

aqwenator has no quantizer of its own for 4 bits. It loads published AutoRound / GPTQ
checkpoints (symmetric, group size 128) after one conversion:

```sh
hf download Qwen/Qwen3.5-9B
hf download Intel/Qwen3.5-9B-int4-AutoRound
python scripts/convert_autoround.py <quantized dir> <original dir> ~/models/qwen35-9b-int4
```

The original checkpoint is needed for the draft head, which the quantized one lacks.
`Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` converts the same way (20 GiB; its router and
shared expert stay bf16). With such a checkpoint the engine also:

- multiplies the 4-bit codes directly in `gemm_i8w` (0.5 bytes read per weight), and
  expands those of the experts inside `moe_gemm`;
- stores the output head in 8 bits;
- leaves the embedding table in host memory and uploads only the rows a step needs.

Quality is the checkpoint's. About 0.12 nats is the price of 4 bits on this model on agent
traces, calibrated or not, and the error grows with context length (0.005 on a short prompt,
0.022 at 16k tokens).

The 4-bit 35B is checked against transformers running the same weights expanded to bf16:
mean KL 0.002 to 0.005 nats on the three reference prompts, the range the bf16 model has
against its own reference. What the 4 bits cost against the original model has not been
measured yet.

### Fitting a small card

```sh
docker run --gpus all -p 8000:8000 -v ~/models:/models:ro \
  ghcr.io/<owner>/aqwenator /models/qwen35-9b-int4 \
  --max-seqs 4 --max-running 2 --kv-gb 1 --max-tokens 1024
```

Device memory of the 4-bit 9B with those settings, measured on a B200 (the CUDA context is
0.7 GiB of it on that card):

| Variant | Device memory |
|---|---|
| as above | 7.9 GB |
| `-e AQ_KV=int8 --kv-gb 0.5` (about 27k tokens of context) | 7.4 GB |
| the same without the draft head (`-e AQ_MTP=0`) | 6.4 GB |

On an RTX 3070 laptop (8 GB) the second row loads and serves: 6.5 GiB allocated by the
engine, 6.9 GB in use on the card, about 43 generated tokens per second for one request.

### A model larger than the card

The experts of a 4-bit mixture-of-experts checkpoint need not all be on the device. With
`AQ_EXPERT_GB=N` the device keeps N GiB of them, the least recently used ones being
replaced, and the others wait in page-locked host memory; with `AQ_EXPERT_RAM_GB` host
memory is a cache too and the rest is read from the checkpoint file with direct I/O. A
token uses 8 of a layer's 256 experts, so most steps find most of what they need. Each
layer also runs the next layer's router on its own input, which names three quarters of
the experts that layer will select, and uploads those while it computes
(`AQ_EXPERT_AHEAD`, experts named per token, 8 by default). Outputs are those of the model
with every expert on the device.

The 4-bit Qwen3.6-35B-A3B (18 GiB with everything on the device) on the same 8 GB laptop:

```sh
docker run --gpus all -p 8000:8000 -v ~/models:/models:ro -e AQ_EXPERT_GB=4.2 -e AQ_MTP=0 \
  ghcr.io/<owner>/aqwenator /models/qwen36-35b-a3b-int4 \
  --max-seqs 2 --max-running 1 --kv-gb 0.5 --max-ctx 8192 --spec-k 0
```

| Expert pool | Drafts | Device memory | Tokens per second, one request |
|---|---|---|---|
| 4.2 GiB | off, no draft head | 7.6 GB | 30 |
| 3 GiB | off, no draft head | 6.3 GB | 26 |
| 3.8 GiB | on | 7.4 GB | 25 |

That machine uploads at 6.7 GiB/s and is bound by it: drafts, which pay on a B200, cost
more uploads than they save steps here. On a B200 with a 3 GiB pool the model keeps
about three quarters of its speed (40 against 48 tokens per second without drafts). Reading
experts from the checkpoint is much slower for now (15 tokens per second on a B200 with
4 GiB of host memory): each layer waits for its reads.

## 8-bit attention cache

`AQ_KV=int8` stores each key and value row as 256 int8 and one scale, so a pool holds 1.9
times the tokens. Quality is at the noise floor. On a B200 it is slower, because expanding a
tile is compute; it is meant for memory and for bandwidth-bound devices.

## What the measurements ruled out

Mean KL on the same traces, for schemes that were emulated or tried and not kept:

| Scheme | Mean KL (nats) |
|---|---|
| int8 weights, one scale per output row | 0.009 |
| int8 weights, row and column power-of-two scales | 0.059 |
| int8 activations, one scale per row | 0.141 |
| 6-bit uniform, one scale per 32 | 0.015–0.045 |
| 5-bit uniform | 0.11–0.14 |
| 4-bit uniform | 0.12–0.21 |
| 4-bit, NF4 levels | 0.15–0.19 |
| 2-bit uniform | 0.5–1.0 |

- Linear int8 activations are unusable; int8 weights need a scale finer than the row.
- At 4 bits no layer group is the culprit, so mixing 4-bit and 8-bit groups does not rescue
  it, and clipping the scale makes it worse.
- cuBLASLt has no 8-bit weight by bf16 activation GEMM, which is why `int8` has its own
  kernel.
