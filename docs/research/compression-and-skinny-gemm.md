# Lossless compression and skinny decode GEMMs: results

B200 (GPU 7), Qwen3.5-9B, 2026-11-13. Tools: `tools/compress_study.py`, `AQ_SKINNY`.

## 1. Compressibility (12-bit PFOR-style bf16 format)

The format, per group of G values: an 8-bit plane holding sign plus 7-bit mantissa, a b-bit exponent delta from the group's maximum exponent (all-ones means an escape), an 8-bit base per group, and a 24-bit patch per escape. Every roundtrip below is **bit-exact**.

**Weights**: all 352 bf16 tensors of the text model, 9.197B weights.

| | value |
|---|---|
| Exponent entropy | 2.61 bits |
| b=4, G=128: escape rate | 0.011% |
| b=4, G=128: bits/weight | **12.065 (1.326x)** |
| b=3, G=128: escape rate | 2.75% |
| b=3, G=128: bits/weight | 11.72 (1.365x) |

The worst tensors are the conv1d weights (b=4: 0.25% escapes; b=3: 31%). Those are tiny.

**KV**: 17.6k-token prompt, all 8 full-attention layers, K and V. The KV data is exponent-coded the same way.

| Grouping | b | Bits/value | Escapes |
|---|---|---|---|
| Per (token, head) row of 256 | 4 | **12.04–12.05 (1.33x)** | 0.01–0.07% |
| Half rows of 128 | 4 | 12.07 | 0.01–0.06% |
| Half rows of 128 | 3 | 11.9–13.6 | 3–10% |
| Per (head, channel) over 64-token pages | 4 | 12.13 | ~0.01% |
| Per (head, channel) over 64-token pages | 3 | 11.5–12.0 | 1.7–3.5% |

Exponent entropy is 2.7–3.0 bits.

The verdict: weights and KV both compress losslessly by about 1.33x with the simplest layout (b=4, groups aligned to the natural read unit). A 3-bit delta buys almost nothing more.

## 2. Skinny decode GEMM vs cuBLASLt (bf16)

`skinny_gemm_m{1,2,4}`: mma.sync m16n8k16, M <= 64, split-K with a fixed-order reduce, and an optional fused SwiGLU epilogue. Four versions:

| Version | Design | Outcome |
|---|---|---|
| v1 | 64x64 cp.async tiles, 4 stages, block barrier per stage | ~2x slower than cuBLASLt |
| v2 | Fragments loaded straight from global (16-byte vectors, k permuted identically for A and B), no smem | ~1.6x slower (latency-bound, 144 registers) |
| v3 | v2 + specialization per M-tile count, deeper K unroll, explicit unroll markers | Per-call parity with cuBLASLt (gate_up + SwiGLU 45 µs vs 39 + 6), but slower end to end |
| v4 | Per-warp cp.async ring, 256 B contiguous per row per instruction | Slower than v3 (spills, extra latency) |

End to end, aq-bench 9B at 1024/256, ITL p50:

| Concurrency | cuBLASLt | v3, all GEMMs (`AQ_SKINNY=1`) | v3, fused SwiGLU only (`AQ_SKINNY=2`) |
|---|---|---|---|
| 1 | 3.76–3.79 ms | 4.23 ms | 3.90 ms |
| 8 | 4.00–4.04 ms | 4.71 ms | 4.22 ms |
| 32 | 5.08 ms | 8.47 ms | 6.23 ms |

cuBLASLt (autotuned, 32 candidates) wins at every batch size. Removing the `silu_mul` round trip does not pay back the slower GEMM.

The correctness gate passed with `AQ_SKINNY=1`: aq-check on 0.8b, 0.8b-long, 9b and 9b-long, in default mode and with `AQ_CHECK_BUDGET=1000 AQ_CHECK_CHUNK=777`.

## 3. Recommendation

- **Keep cuBLASLt for weight GEMMs.** The skinny kernel stays behind `AQ_SKINNY` (default off) as a base for experiments.
- **12-bit weights (W12) are not worth building on top of v3.** v3 reaches about 60–75% of cuBLASLt's effective bandwidth. W12's 25% byte saving would at best get back to parity, and it would add decode work and escape handling.
- **Spend the compression effort on KV inside our own attention kernels.** There cuBLAS is not the competitor, and long-context decode attention and KV capacity scale directly with the 1.33x.
  - The per-(token, head) row layout, b=4, matches the split-K decode kernel's read unit: 256 values per (token, KV head).
  - The format needs one byte of base per row, 4-bit deltas, and an escape fallback. At 0.01–0.07% escapes, a per-row "raw" flag is enough: rows with any escape are stored uncompressed.
