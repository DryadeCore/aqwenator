# SOTA survey: serving Qwen3.5 hybrids for agentic multi-turn on RTX 5090 / DGX Spark

Date: 2026-11-13. Scope: LLM inference serving work from 2024 to Nov 2026, filtered for one engine (aqwenator), one model family (`qwen3_5`: 3 Gated DeltaNet layers to 1 gated GQA layer), one workload (many concurrent agent sessions with 16k–128k context, 1–8k new tokens and a few hundred output tokens per turn, seconds idle between turns) and two GPUs (RTX 5090 sm_120, GB10 sm_121).

Sources: local vLLM checkout (tag `v0.33.0`, plus `main` at 18c7d9bc), GitHub PRs and issues, arXiv, engineering blogs. Numbers I derived myself are marked *(derived)*. Numbers from a single PR author or vendor blog are labelled as such; I did not reproduce them.

---

## 0. The arithmetic that drives every ranking below

From the HF configs ([9B](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [27B](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json); both set `mamba_ssm_dtype: float32`):

| | 9B (32 layers: 8 attn, 24 GDN) | 27B (64 layers: 16 attn, 48 GDN) |
|---|---|---|
| KV per token (bf16) | 32 KiB | 64 KiB |
| KV at 16k / 64k / 128k | 0.5 / 2 / 4 GiB | 1 / 4 / 8 GiB |
| GDN state per seq (fp32) + conv state | 48 MiB + 1.1 MiB | 144 MiB + 2.8 MiB |
| State expressed as KV tokens | ≈1.6k tokens | ≈2.3k tokens |
| bf16 weights | ≈18–19 GB | ≈54 GB (FP8 official ckpt ≈28 GB) |

Consequences *(derived)*:

1. **At 16k–128k context, KV dominates both memory and decode bandwidth; the recurrent state is only 2–10% of a session's bytes.** The hybrid architecture already removed 3/4 of the KV. What remains is still the capacity problem.
2. **On an RTX 5090 with 9B in bf16, roughly 10–11 GB is left for cache. That holds about 5 sessions at 64k.** 27B does not fit in bf16. The official FP8 27B checkpoint leaves about 2–3 GB, which is under one 64k session. "Many concurrent agents" on a 5090 therefore means most sessions must live **off-GPU while idle**.
3. **A cache miss is catastrophic.** Re-prefilling 64k tokens of 9B costs about 18 GFLOP/token × 64k for the linear layers plus about 0.3 PFLOP of attention. At a realistic 120–140 TFLOPS on a 5090 (209 TFLOPS dense BF16 peak), that is **about 10–12 s**. Restoring the same session from pinned host RAM is 2 GiB of KV plus 50 MB of state over PCIe 5 x16 (about 50 GB/s in practice), so **about 40–60 ms**. p99 TTFT is therefore decided by the cache-hit rate, not by kernels.
4. **Decode on a 5090 is weight-bandwidth-bound**, because capacity keeps the batch small. The 9B weights are 18 GB / 1.8 TB/s ≈ 10 ms per step, and resident KV adds at most about 6 ms. **On a DGX Spark** (273 GB/s) the weights alone are about 66 ms per step for 9B bf16, and each decoding 64k session adds about 7.7 ms of KV reads. Speculative decoding amortizes both weight and KV reads, which makes it the main ITL lever on both machines.
5. **GDN prefill FLOPs are tiny.** On an RTX PRO 6000 (sm_120), FlashQLA runs an 8k GDN chunk in about 0.4 ms per layer, about 10 ms for all 24 layers ([fla#1094](https://github.com/fla-org/flash-linear-attention/pull/1094)). The 8k-token GEMMs take about 1.2 s. For an 8k turn against a 64k cached context, **prefill attention is about 30–40% of prefill FLOPs** (8 layers × 4·8k·64k·256·16 ≈ 70 TFLOP). Prefill kernel effort should go to hd256 attention and the GEMMs, not to GDN.

---

## (a) Executive summary: top 10, ranked by expected impact for this workload

"vLLM 0.33?" means whether vLLM 0.33 already does it for Qwen3.5 (code paths in section b).

### 1. A session-aware hybrid prefix cache with exact, template-aware state checkpoints

**What:**
- Use a radix/hash prefix cache over token IDs that shares system-prompt and tool-definition KV across sessions.
- Write GDN plus conv-state checkpoints only at *semantic* boundaries:
  - (i) the end of the last user or tool message of each request, *before* the `<|im_start|>assistant\n<think>\n` generation header (see §c1.3 for why it must be before the header);
  - (ii) the end of decode, when the turn ended in a tool call (the Qwen3.5 template keeps thinking inside a tool loop);
  - (iii) the shared-prefix junction (end of system prompt plus tools);
  - (iv) optionally, every 16–32k tokens as insurance against partial KV eviction.
- Anchor the GDN chunk grid at absolute multiples of 64. Store the checkpoint at `floor64(boundary)` and recompute at most 63 tokens on a hit. A cache hit is then **bitwise identical** to a cold prefill, given deterministic kernels.
- Keep a **separate state-checkpoint pool sized for idle sessions, not running ones.**
- Eviction must never drop a checkpoint while its KV is kept. Rank entries by recency × (prefill FLOPs saved / bytes), as Marconi does. Pin entries for a TTL while a tool call is outstanding, as Continuum does.

**Expected gain:** this is the dominant lever for p99 TTFT. A warm turn costs a prefill of only the 1–8k new tokens plus up to a few hundred replayed tokens: about 0.2–1.5 s on a 5090, against 10 s or more on a miss *(derived)*.
- In SGLang's own agent benchmark (GLM-5.3-Flash, 48 sessions, B300), sizing the state pool for idle sessions and checkpointing lazily took the hit rate from 0.056 to 0.888 and p50 TTFT from 54.8 s to 0.15 s ([sglang#49886](https://github.com/sgl-project/sglang/pull/49886), PR author's numbers).
- Marconi reports up to 34× token hit rate and up to 71% lower P95 TTFT over naive hybrid caching ([arXiv 2411.19379](https://arxiv.org/abs/2411.19379)).

**Effort:** medium. Scheduler and allocator work, plus a "write state at offset X" output in the prefill kernel.
**Exactness:** lossless (bit-exact with grid alignment).
**vLLM 0.33?** Partially.
- `mamba_cache_mode="align"` is on by default with block-granular hits (about 530–580 tokens for 9B, 784 for 27B).
- It keeps a replay-boundary checkpoint at the prompt end, and the junction checkpoint only behind a flag plus EAGLE/MTP.
- It does **not** cache decode-produced state.
- It uses one LRU list for all groups, with no session or TTL awareness.

### 2. An idle-session tier: pinned host RAM on the 5090; NVMe and priority on Spark

**What:**
- When a turn ends, write the session's KV pages and its boundary checkpoint to pinned host memory asynchronously. Keep the GPU copy until the space is needed (write-back, not move).
- When the next request for that session arrives, start an H2D prefetch immediately, layer by layer.
- Overlap the restore with the prefill of the new 1–8k tokens. Layer L's restore must complete before layer L's attention runs.
- On GB10, memory is unified, so "offload" is just eviction priority, plus an optional NVMe tier for very old sessions. antirez's ds4 does disk KV snapshots for the same reason.

**Expected gain:** this decides **concurrency** on the 5090.
- With 128–256 GB of host RAM, resident capacity grows from about 5 to about 60–120 sessions at 64k, and a reload takes about 40 ms against about 10 s to recompute *(derived)*.
- CachedAttention reports up to 87% lower TTFT ([ATC'24, arXiv 2403.19708](https://arxiv.org/abs/2403.19708)). Tokencake, which offloads during tool calls and prefetches before the predicted return, reports over 47% lower end-to-end latency vs vLLM ([arXiv 2510.18586](https://arxiv.org/abs/2510.18586)).

**Effort:** medium. Pinned pool, a copy engine on its own stream, per-layer events.
**Exactness:** lossless.
**vLLM 0.33?** Partially. `--kv-offloading-size` (native CPU tier, LRU/ARC) and the LMCache and Mooncake connectors support `align`-mode mamba groups. There is no session-aware prefetch on arrival and no TTL. The vLLM AgentX blog lists prefetch-during-idle and soft-pinning as *planned* ([vllm.ai/blog/2026-09-08-vllm-agentx](https://vllm.ai/blog/2026-09-08-vllm-agentx)).

### 3. MTP speculative decoding with ReplaySSM-style GDN verification and adaptive k

**What:**
- Use the shipped MTP head (one layer, applied recursively) with k = 1–3, chosen per step from batch size and per-position acceptance estimates.
- Do not give each draft position its own 48–144 MB state slot. Keep an fp32 checkpoint plus a ring of per-token GDN inputs. For GDN, store `u = β(v − S k)` rather than raw `v`, plus `k, g, β`. Compute verify outputs from the checkpoint and the ring.
- Rollback becomes a cursor move. A periodic exact fold commits the ring into the checkpoint ([ReplaySSM, Dao AI Lab + NVIDIA](https://dao-lab.ai/blog/2026/replayssm/)).

**Expected gain:**
- MTP on Qwen3.5-27B: acceptance length 3.25–3.62 at k=3; 2.5–2.8× at concurrency 1 and 2.0–2.3× at concurrency 32 on a B200 ([z-lab card](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash), SGLang).
- DGX Spark with llama.cpp, Qwen3.6-27B Q8_0, batch 1: 7.1 → 14–19 tok/s ([llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673)).
- ReplaySSM shrinks spec scratch memory from 11.48 to 1.81 GB and adds 15–30% throughput at concurrency 8–32 on Qwen3.5-122B ([sglang#28695](https://github.com/sgl-project/sglang/pull/28695)). It also saves 3 × 48 MB per running 9B sequence at k=3.
- Expect **about 2× ITL on Spark and 1.5–2× on the 5090** at the small batches capacity forces *(derived; verify at 64k context, where verify attention is wider)*.

**Effort:** medium-high. Drafter forward, verify kernels, the GDN ring kernel, device-side commit, graph capture per k.
**Exactness:** lossless under standard rejection sampling, up to float error. Keep the checkpoint fp32 and use the exact sequential refold: sglang#28695 found that an approximate fold caused long-output loops.
**vLLM 0.33?** MTP yes (`qwen3_5_mtp.py`), but with k+1 fp32 state slots per request. There is no ReplaySSM for GDN: `supports_replayssm` is only on NemotronH and Kimi-K3, and the GDN version, vllm PR #62004, is still open.

### 4. Suffix / prompt-lookup drafting combined with MTP

**What:**
- Keep a per-session suffix tree (or a cheap n-gram index) over the prompt plus prior outputs.
- When the match is long, draft from it. Tool outputs quoted back, file contents and diffs are long exact copies. Otherwise, use MTP.
- This is model-free and fits in the same verify path as item 3.

**Expected gain:**
- SWE-bench, Llama-3.1-8B, batch 1: 7.8 accepted tokens/step and 2.45× (prompt lookup: 1.50×). End-to-end OpenHands: 1.8–4.5× ([SuffixDecoding, arXiv 2411.04975](https://arxiv.org/html/2411.04975v3)).
- An independent SE study finds only 1.34× on average across SE tasks (2.27× on repair) and notes that verification goes compute-bound at about 16k ([arXiv 2604.26469](https://arxiv.org/abs/2604.26469)).
- Incremental over MTP: likely +10–40% on copy-heavy turns, roughly zero elsewhere.

**Effort:** low-medium.
**Exactness:** lossless.
**vLLM 0.33?** `method: suffix` and `method: mtp` both exist as separate methods. I did not find a combined MTP+suffix mode.

### 5. A long-context decode-attention kernel built for this shape, with shared-prefix (cascade) decode

**What:**
- Split-KV (flash-decoding / stream-K) with **fixed split size**, not a fixed split count, so results are batch-invariant (see Thinking Machines, §c6).
- Pack the GQA query heads (4 for 9B, 6 for 27B) and the k+1 speculative query tokens into the mma M dimension.
- Leave the KV tile loader ready for FP8.
- Add a shared-prefix pass: attend the shared system-prompt/tools prefix once for every session in the batch (Hydragen / FlashInfer cascade), then merge the log-sum-exp results.

**Expected gain:** decode attention is the *only* term that grows with context.
- At 64k, KV reads per decoding session (2 GiB) are 20× its state traffic. A kernel at 85–90% of DRAM bandwidth vs 60% is worth about 1.3–1.5× on the attention share of each step *(derived)*.
- Cascade saves (sessions − 1) × shared_len × 32 KiB per step. That is about 25% of attention bytes with a 16k shared prefix at 64k context. Hydragen reports up to 32× end-to-end in prefix-heavy batch settings ([arXiv 2402.05099](https://arxiv.org/html/2402.05099v2)); our ratio is far smaller.

**Effort:** medium (you already have split-K decode).
**Exactness:** lossless mathematically. Bitwise reproducibility needs a fixed reduction order. Cascade on/off changes the order, so make it a deterministic function of the request rather than the batch, or accept tolerance-level matching.
**vLLM 0.33?** Split-KV decode yes, through FlashInfer/FA. **Cascade attention was removed** in vLLM in Oct 2026 ([#61126](https://github.com/vllm-project/vllm/pull/61126), Woosuk Kwon; it had been disabled by default since #36318). This is a place where we can beat vLLM.

### 6. An SLO-aware, agent-aware scheduler

**What:**
- **Chunked prefill with a token budget per step** derived from a target ITL and a measured cost model (Sarathi-Serve, SOLA).
- **Order prefills** by (new tokens to compute, session's attained service), as Autellix does with program-level least-attained-service. A session coming back from a tool with a 2k-token tool result should not queue behind a cold 100k prefill.
- **Never preempt by recompute.** Preempt by moving the session to the host tier (item 2).
- **Admission control** that counts the bytes of idle checkpoints and pinned TTLs, not just running KV.
- Optionally, POD-style fused prefill+decode attention launches.

**Expected gain:**
- Sarathi-Serve: 2.6× capacity under tail-latency SLOs ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)).
- Autellix: 4–15× program throughput at equal latency vs vLLM ([arXiv 2502.13965](https://arxiv.org/abs/2502.13965)).
- Continuum: >8× average job completion time on SWE-bench/BFCL/OpenHands traces ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230)).
- For us, expect large p99 TTFT gains under bursty tool returns and little gain on mean throughput.

**Effort:** low-medium, since it is scheduler code only.
**Exactness:** lossless.
**vLLM 0.33?** Partially. It has chunked prefill, `fcfs`/`priority`, `long_prefill_token_threshold`, `watermark` and `prefill_schedule_interval`. It has no ITL-driven budget, no program/session priority, and preempts by recompute.

### 7. FP8 KV cache (opt-in; lossy)

**What:** E4M3 KV with per-head scales, dequantized in registers in both the prefill and decode attention kernels, with fp32 or two-level accumulation.

**Expected gain:** about 2× KV capacity, which means 2× resident sessions or 2× the host-tier hit window, and about 2× decode attention bandwidth at long context.
- vLLM's study: Qwen3.5-27B reasoning within −0.7 points, MRCR up to 1M matching baseline, *after* fixing an accumulation bug that had dropped 128k NIAH to 13% ([vllm.ai/blog/2026-04-22-fp8-kvcache](https://vllm.ai/blog/2026-04-22-fp8-kvcache)).
- The same post advises staying on BF16 for hd256 when prefill speed matters.

**Effort:** medium (two kernels, a scale pipeline).
**Exactness:** **lossy.** Ship as an opt-in mode, never the default.
**vLLM 0.33?** Yes (`--kv-cache-dtype fp8`, also `nvfp4`).

### 8. Prefill throughput at hd256: attention first, then GEMMs

**What:** make the FA2-style mma.sync prefill kernel hit 85–95% of the 5090's 209 TFLOPS at head_dim 256 for the long-context case (an 8k-token query chunk against 64–128k of cached KV):
- swizzled smem;
- a 2-stage cp.async pipeline;
- ldmatrix.x4;
- tile shapes that fit 99 KB of smem with hd256 (for example 64×64 KV tiles, with Q kept in registers).

Then check cuBLASLt algorithm choice for M = 1–8k on sm_120/121. GDN prefill kernel work is low priority: it is under 2% of prefill time, see §0.5.

**Expected gain:** 10–30% TTFT on warm turns *(derived from the FLOP split in §0)*. A from-scratch 5090 FA2 study went from 143 to 198 TFLOPS against a 209.5 peak at hd128 ([gau-nernst](https://gau-nernst.github.io/fa-5090/)).

**Effort:** medium.
**Exactness:** lossless.
**vLLM 0.33?** It uses FlashInfer/FA2 on sm_120. Upstream FA's sm_120 port was tested only up to D=128, with no paged, varlen or split-KV support ([flash-attention#2329](https://github.com/Dao-AILab/flash-attention/pull/2329)).

### 9. Small-M decode GEMMs and fusion (and only then megakernels)

**What:**
- With 4–32 sessions × (k+1) verify tokens, M is 4–128. Measure cuBLASLt against a hand-written split-K skinny GEMM at those M values on sm_120 and on GB10; I found no published data, so measure it.
- Fuse RMSNorm into the following GEMM prologue, fuse the GDN decode chain (conv1d + gates + L2-norm + recurrence + gated norm), and update the state in place by slot, with no gather or scatter.
- A persistent megakernel is only worth trying for bs ≤ 4.

**Expected gain:** 5–15% decode step time *(derived)*.
- Megakernels report 1.0–1.7× over CUDA graphs only at bs 1–16 for 1–8B models ([MPK arXiv 2512.22219](https://arxiv.org/html/2512.22219v1); [Hazy No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). For a 9B model at about 10 ms per step, launch overhead under graphs is about 1 ms at most.
- FlashInfer's in-place K-last GDN decode ran 2.8× faster than Triton at batch 32, but gave only about 2% end-to-end ([sglang#19150](https://github.com/sgl-project/sglang/pull/19150)).

**Effort:** medium.
**Exactness:** lossless (batch-invariance caveats in §c6).
**vLLM 0.33?** Partially (fused CUDA GDN decode is optional via `VLLM_GDN_DECODE_KERNEL=cuda`).

### 10. Lossy capacity knobs, opt-in only, behind flags and evals

**What:**
- (a) **Store-once int8 or bf16 for *idle* cached checkpoints.** Quantize once at checkpoint time, dequantize once on restore, and keep the active state fp32. SGLang keeps cached states in int8 with per-(head, k-channel) scales (`--enable-int8-mamba-checkpoint`). That is 12.5 MB per sequence on 9B instead of 50.
- (b) NVFP4 KV: −1.6 points on SWE-bench Verified vs FP8 KV ([LMSYS](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache)).
- (c) The **official** Qwen FP8 checkpoints (27B and up). This is "exact" only if you define the FP8 checkpoint as the reference.

**Expected gain:** (a) is minor, because state is small next to KV. (b) gives about 1.8× capacity over FP8.

**Exactness:** all **lossy**.
- **Never** store the active state at low precision every step. LeapQuant measured Qwen3.5-9B AIME falling from 87.9 to 72.1 with per-step bf16 state, and INT8/FP8 per-step collapsing to 7–15 (as cited in [vllm#64725](https://github.com/vllm-project/vllm/issues/64725)).
- **Never** use token eviction (SnapKV, H2O, StreamingLLM) here. On SCBench multi-turn it drops 48.7 to 15–21 ([arXiv 2412.10319](https://arxiv.org/html/2412.10319v2)).

**vLLM 0.33?** It has `--mamba-ssm-cache-dtype` for global (per-step) state precision, which is the dangerous kind. It has no store-once checkpoint quantization.

### What I deliberately ranked low

- GDN prefill kernel rewrites. FlashQLA shows 2.1–2.6× over Triton on sm_120 ([fla#1094](https://github.com/fla-org/flash-linear-attention/pull/1094)), but it is under 2% of prefill time for us.
- Green-context or MPS SM partitioning for prefill/decode (semi-PD, Nexus, MuxWise). The gains are real on datacenter GPUs, but the complexity is high and items 1, 2 and 6 attack the same tail first.
- DFlash drafters. They have better acceptance than MTP: 3.5–4.8× at concurrency 1 on Qwen3.5-27B per z-lab. The cost is an extra 6-layer model plus drafter KV, and there are no long-context agentic numbers. Revisit after item 3 lands.

---

## (b) What vLLM 0.33 and SGLang do (and don't) for hybrid Qwen3.5 in agentic multi-turn

### vLLM 0.33.0 (tag `v0.33.0`, 2026-10-31; read from local checkout)

| Topic | Behavior | Code |
|---|---|---|
| Prefix caching on hybrid | On by default. `enable_prefix_caching=True` forces `mamba_cache_mode="align"` and `mamba_block_size = block_size`, and requires chunked prefill. | `vllm/model_executor/models/config.py` `MambaModelConfig.verify_and_update_config`; `vllm/config/cache.py` L304 |
| Block size | Attention block size is inflated until one attention page ≥ one GDN layer's state, then the mamba page is padded to match. For 9B: 2.0 MiB state / 4 KiB per token per layer gives about 528 tokens, or 576 when rounded to the 64-token FlashInfer checkpoint alignment *(derived)*. LMCache docs report 544 (0.8B) and 784 (27B). The pool is unified with the same page size for both groups. | `vllm/platforms/interface.py` `update_block_size_for_backend` (L1020–1120); `docs/design/hybrid_kv_cache_manager.md` "Case 4" |
| Hit granularity | Default hash unit = GCD of group block sizes = block size, so hits land on block boundaries. `--prefix-match-unit 64` allows sub-block hits, but only the final aligned state is stored. | `vllm/v1/core/kv_cache_utils.py` `resolve_kv_cache_block_sizes`; `docs/features/automatic_prefix_caching.md` |
| Which states are kept | `prefix_cache_retention_interval=0` (default) keeps only "semantic" checkpoints: the replay boundary at `num_prompt−1`, floored to the hash unit, plus detected shared-prefix junctions (Marconi-style: needs the prefix to have appeared before). The junction checkpoint is stored mid-block only with `--enable-mamba-shared-prefix-checkpoint` **and** EAGLE/MTP **and** a small `prefix_match_unit`. | `single_type_kv_cache_manager.py` `MambaManager.reachable_block_mask`; `sched/scheduler.py` `_mamba_block_aligned_split` |
| Decode-produced state | **Not cached.** `_mamba_block_aligned_split` says: "Past the prompt the manager writes nothing". A tool-loop continuation hits at the previous prompt's boundary and recomputes the previous output and the new tool result. | `sched/scheduler.py` L620–660 |
| Mid-prefill checkpoints | Yes, when the GDN prefill backend is FlashInfer. That is the default on sm_120 with CUDA ≥ 13 and head_k_dim 128; on GB10 (12.1) via the `is_device_capability_family(120)` check. Alignment is 64 tokens. | `qwen_gdn_linear_attn.py` `_resolve_gdn_prefill_backend`; `v1/attention/backends/gdn_attn.py` `FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64` |
| SSM state dtype | fp32, taken from the HF config `mamba_ssm_dtype`. A user override wins with a warning. | `models/config.py` `Qwen3_5ForConditionalGenerationConfig` |
| Eviction | A single LRU list shared across groups. No session, TTL or FLOP-aware policy. | `docs/design/hybrid_kv_cache_manager.md` "cache eviction policy" |
| Offload | Native CPU tier (`--kv-offloading-size`, LRU/ARC), LMCache, Mooncake store. All support `align` mamba groups, with hits rounded to the offload chunk. | `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` L140–170; `mooncake/store/scheduler.py` L187 |
| Spec decode on GDN | MTP / EAGLE-3 / DFlash / ngram / suffix / FLy (lossy) are available. GDN verify writes the state after every draft token into its own slot (`num_speculative_blocks = num_speculative_tokens`) and reads slot `num_accepted−1` next step. Chains only, no trees. | `layers/mamba/abstract.py` `get_kv_cache_spec`; `qwen_gdn_linear_attn.py` L1755–1780 (`fused_sigmoid_gating_delta_rule_update(..., ssm_state_indices, num_accepted_tokens)`) |
| ReplaySSM | Mamba2 (NemotronH) and Kimi-K3 KDA only. **Not Qwen3.5 GDN.** FlashInfer ReplaySSM spec is also incompatible with prefix caching. | `models/interfaces.py` `SupportsReplaySSM` users; `config/vllm.py` `validate_mamba_cached_kernel` |
| Adaptive spec | `num_speculative_tokens_per_batch_size` (static ranges), `enable_adaptive_verification` (cost-budgeted, needs full graphs). | `config/speculative.py` L524, L585 |
| Cascade / shared-prefix attention | **Removed** (#61126, 2026-10-12). `disable_cascade_attn` is a no-op. | `config/model.py` L316 |
| Preemption | By recompute. Cached blocks may survive in the prefix cache, but nothing prioritises them. | `sched/scheduler.py` ~L930 |

Behaviour on our workload *(derived from the above)*:
- **Inside a tool loop:** each new request hits at the previous request's prompt boundary, floored to about 576 tokens. It recomputes ≤ 576 + previous output + tool result tokens. That costs a few hundred extra tokens per turn, which is acceptable.
- **At a new user message:** the hit falls back to before the first assistant message of the previous episode (§c1.3). If a session's tail blocks or its single checkpoint are LRU-evicted while it idles, the hit falls back to the system-prompt junction. In that case the whole history is recomputed.
- The memory cost of `align` is about one extra 50 MB state per running request (plus k per request with MTP).

### SGLang (main, Nov 2026; from subagent code reading plus PRs)

- **MambaRadixCache.** Two LRU lists: KV is evicted leaf-to-root, and a state can be dropped at any node. A hit copies the state into a private slot ([PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)).
- **Strategy flag:** `--mamba-radix-cache-strategy {auto,no_buffer,extra_buffer,extra_buffer_lazy}`. `extra_buffer` uses ping-pong state slots and per-chunk state output ([#14792](https://github.com/sgl-project/sglang/pull/14792)). Decode is tracked every `mamba_track_interval` (256) tokens.
- **`--hybrid-cache-checkpoints-level` (merged 2026-11-13, [#49886](https://github.com/sgl-project/sglang/pull/49886)).**
  - Level 2 (default) checkpoints only at fork points and request ends.
  - Qwen3.8-27B host writes were halved (66.7 → 33.4 GB) with the same 0.993 continuation hit rate.
  - Cost: a late fork with no checkpoint drops from a 1.0 to a 0.0 hit.
- **Session-aware eviction.** A `session_id` re-ranks eviction without pinning memory, and `/close_session` releases it. On Qwen3.5-397B SWE-bench, device hit rate went from about 5% to 34% ([LMSYS unified radix cache](https://lmsys.org/blog/2026-08-11-unified-radix-cache/)).
- **Int8 cached checkpoints:** `--enable-int8-mamba-checkpoint` keeps idle cached states in int8 in a pool separate from the fp32 active pool (`mem_cache/mamba_checkpoint_pool.py`).
- **HiCache** has a mamba host tier.
- **GDN ReplaySSM spec:** `--enable-gdn-replayssm-spec` ([#28695](https://github.com/sgl-project/sglang/pull/28695)), chain-only.
- **Tree speculation on GDN** (`topk>1`) works by storing one state per tree node.
- **Pool sizing docs** ([#50033](https://github.com/sgl-project/sglang/pull/50033), open): 40 sessions on Qwen3.5-0.8B gave 0% warm reuse with 8 state slots, 60% with 64 and 100% with 96. **The state pool must be sized for idle sessions.**
- **Known bugs:**
  - #39342: mixed-chunk corrupts checkpoints.
  - #37834: ReplaySSM forces no_buffer and inflates TTFT up to 4.7×.
  - #41351: logprob drift on repeated branch scoring.

**Gaps in both engines that we can exploit:**
- There is no prefetch-on-arrival from host for idle agent sessions; vLLM lists it as planned.
- There is no shared-prefix decode attention (vLLM removed it).
- vLLM has no GDN ReplaySSM.
- Neither engine places checkpoints with knowledge of the chat template; vLLM uses block-granular boundaries.
- Neither engine is batch-invariant by default.

### llama.cpp (for reference on our hardware)

- `server-context.cpp` checkpoints the full state at user-message starts and near the prompt end: `--ctx-checkpoints` (32) and `--checkpoint-min-step` (8192). The host prompt cache is `--cache-ram`, per slot, with no cross-session sharing.
- GDN prefill is token-recurrent, with no chunked kernel ([#19504](https://github.com/ggml-org/llama.cpp/pull/19504), [#20391](https://github.com/ggml-org/llama.cpp/pull/20391)).
- MTP is supported, with GDN intermediate-state rollback ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673)).

---

## (c) Per-topic findings

### c1. KV / state cache management for multi-turn agents

**c1.1 Hybrid prefix-caching designs**
- **Marconi** ([arXiv 2411.19379](https://arxiv.org/abs/2411.19379), MLSys'25).
  - Admission: checkpoint state only at radix branch points, found by speculative insertion, and after the last decoded token.
  - Eviction: recency + FLOPs-saved/byte.
  - This is the right mental model. vLLM's "junction" and SGLang's level-2 checkpoints are both Marconi derivatives.
- **Jenga** ([arXiv 2503.18292](https://arxiv.org/abs/2503.18292)). An LCM-page-size allocator for heterogeneous layers. vLLM's unified hybrid allocator implements this idea, which is why its blocks are about 500–800 tokens.
- **For aqwenator:** you don't need a unified page size. Keep two pools:
  - a KV page pool with 16–64-token pages;
  - a state-slot pool with whole per-sequence states, split into active (fp32) and checkpoint (fp32, or optionally int8).

  Prefix matching then uses KV pages, and the state checkpoint is an attribute on a radix node. vLLM's coupling of KV block size to state size is an artifact of its allocator, not a requirement.

**c1.2 Session tiering and agent-aware retention**
- **CachedAttention / AttentionStore** ([arXiv 2403.19708](https://arxiv.org/abs/2403.19708)): per-session host/disk KV with layer-wise preload overlapped with compute. Up to 87% lower TTFT and 7.8× prefill throughput.
- **Pensieve** ([arXiv 2312.05516](https://arxiv.org/abs/2312.05516)): a two-tier GPU/CPU cache per conversation. 1.14–3.0× throughput.
- **Continuum** ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230)): pins KV with a TTL derived from predicted tool duration and reload cost, plus program-level FCFS. Over 8× average JCT.
- **KVFlow** ([arXiv 2507.07400](https://arxiv.org/abs/2507.07400)): steps-to-execution eviction plus prefetch. 1.83–2.19× vs SGLang HiCache.
- **Tokencake** ([arXiv 2510.18586](https://arxiv.org/abs/2510.18586)): offload during tool calls, re-upload before the predicted return.
- **LMCache with hybrids:** works only with vLLM `align` mode, and treats the state as an opaque page. CacheGen and CacheBlend do not apply to state. An external check found an off-by-one on full-prompt hits (the state was restored for N tokens while N−1 were credited) ([arXiv 2609.15030](https://arxiv.org/abs/2609.15030)). **Lesson:** on a full hit you still need the logits for the last prompt token. Either recompute at least one token, as vLLM's replay boundary does, or cache the last hidden state.
- **CacheBlend and InfiniGen** (selective KV reuse / prefetch) have no recurrent-state equivalent. Skip them.

**c1.3 Template-induced prefix divergence (critical for Qwen3.5 agents)**

The Qwen3.5 template ([chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/chat_template.jinja)) works as follows:
- It computes `last_query_index` as the last `user` message that is not a `<tool_response>`.
- Assistant messages *after* it are rendered with `<think>…</think>`; assistant messages *at or before* it lose their reasoning.
- The generation prompt is `<|im_start|>assistant\n<think>\n`.

Consequences:
- **Inside a tool loop**, history is stable, and a checkpoint at the end of decode can hit, subject to tokenization round-trip.
- **When a new user message arrives**, every assistant message in the previous episode is re-rendered without thinking. The prefix diverges right after the previous episode's `<|im_start|>assistant\n`. Everything from there on is recomputed, typically the whole tool-loop episode, which can be tens of thousands of tokens. This is inherent to the reference template. Changing it would change model inputs, which would break the exactness requirement unless the client opts in.
- **A checkpoint at "prompt end" (after `<think>\n`) is 2–3 tokens past the divergence point**, so it is useless for the next episode. Place the checkpoint at the end of the last user or tool message, before the assistant header. vLLM gets this boundary somewhat by accident, through block flooring. llama.cpp checkpoints at user-message starts for this reason.
- **Tokenization round-trip:**
  - String tool arguments are re-emitted verbatim.
  - Mapping and list arguments go through `tojson`, which can reformat compact JSON.
  - `content|trim` and `\n\n` separators can alter whitespace.
  - BPE re-tokenization of sampled text may differ from the sampled IDs.

  Match on token IDs after rendering, and expect occasional early divergence. The vLLM forum confirms there is no server-side fix under default templates ([discuss.vllm.ai #2955](https://discuss.vllm.ai/t/decode-kv-writeback-can-multi-turn-chat-preserve-cache-reuse-with-default-templates/2955)).
- **Shared prefix:** keep the system prompt and tool definitions byte-stable. Manus calls KV hit rate "the single most important metric" and reports input:output ratios of about 100:1 ([Manus blog](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).

**c1.4 Exactness of cache hits for GDN**

The chunked GDN algorithm's rounding depends on where chunk boundaries fall. If a cold prefill runs chunks at 0, 64, 128, …, and a warm prefill resumes from a checkpoint at an arbitrary token p, the two are *not* bitwise equal. Two options:
- (a) Checkpoint only at multiples of 64, and recompute the 0–63-token tail on hits. This is what vLLM's 64-token FlashInfer alignment gives you.
- (b) Accept tolerance-level equality.

Option (a) costs nothing measurable and makes "prefix hit == cold run" a testable invariant. The same holds for the conv state: store it together with the checkpoint, and do not reconstruct it.

### c2. KV / state compression

**Lossless options:** none worth their cost for KV. Entropy coding of bf16 KV yields about 1.2–1.4×; CacheGen is lossy. A DGX Spark engine (DGPP) uses a lossless 12-bit weight format for a 1–12% decode gain ([forum](https://forums.developer.nvidia.com/t/dgpp-a-gb10-optimized-c-cuda-inference-engine/383406)). That is an interesting idea for Spark's weight-bound decode, but small.

**FP8 KV (lossy, mild):** see item 7. Use per-head scales and fp32 or two-level accumulation for long context.

**NVFP4 KV (lossy):**
- SGLang supports it on sm_120, dequantizing to BF16 in registers. SWE-bench Verified is −1.6 points vs FP8 KV ([LMSYS](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache)).
- NVIDIA claims under 1% loss including RULER-64K ([NVIDIA blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)).
- sm_120a has block-scaled mma.sync (`kind::mxf4` / nvf4) usable without tcgen05 ([llama.cpp#17906](https://github.com/ggml-org/llama.cpp/pull/17906)).

**Recurrent-state precision:**
- *Per-step* low precision of the active state is clearly harmful on Qwen3.5. Per-step bf16 dropped AIME from 87.9 to 72.1; int8/fp8 collapse. Per-*window* bf16 (ReplaySSM with a 16-bit checkpoint) gave 87.8 vs 87.9 (LeapQuant, as cited in [vllm#64725](https://github.com/vllm-project/vllm/issues/64725)). Nemotron-3-Super saw FP16 state inflate output lengths by 37–40% without stochastic rounding ([arXiv 2604.12374 §4.3](https://arxiv.org/html/2604.12374v1)).
- Rule: **active state fp32.** Quantize only idle checkpoints, once, and opt-in.

**Token eviction / compression (lossy; reject for this product):**
- SCBench multi-turn, Llama-3.1-8B (full cache 48.7): StreamingLLM 15.5, SnapKV 20.9, PyramidKV 21.0 at a 1/32 budget; KIVI 32.0 at 1/8 ([arXiv 2412.10319](https://arxiv.org/html/2412.10319v2)).
- Query-agnostic methods are better: KVzip 3–4× with "negligible" loss ([arXiv 2505.23416](https://arxiv.org/abs/2505.23416)); Expected Attention about 50% on NIAH ([arXiv 2510.00636](https://arxiv.org/abs/2510.00636)). None of them is exact, and multi-turn reuse of a compressed prefix is precisely their documented failure mode.

### c3. Scheduling for tail latency and goodput

- **Sarathi-Serve** ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)): chunked prefill plus stall-free batching. 2.6–3.7× capacity under tail SLOs. You have chunked prefill; the missing piece is a budget derived from the ITL SLO and a measured step-cost model:

  `t_step ≈ (W + Σ KV_i + Σ 2·S_i) / BW + prefill_tokens · c_prefill(ctx)`

- **SOLA** ([MLSys'25](https://mlsys.org/virtual/2025/3345)): SLO-aware dynamic budget. Attainment 45.5% → 99.4%.
- **Intra-GPU P/D separation:** semi-PD ([arXiv 2504.19867](https://arxiv.org/html/2504.19867v1)), Nexus ([arXiv 2507.06608](https://arxiv.org/html/2507.06608v5)), MuxWise/Drift with green contexts ([arXiv 2504.14489](https://arxiv.org/abs/2504.14489)), Bullet ([arXiv 2504.19516](https://arxiv.org/html/2504.19516v4)) and DuetServe ([arXiv 2511.04791](https://arxiv.org/html/2511.04791v1)) report 1.3–2.2× under SLO.
  - Green contexts on 9.0+ partition in multiples of 8 SMs, and partitions are not guaranteed to run concurrently ([CUDA docs](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)).
  - Treat this as phase 2.
- **POD-Attention** ([arXiv 2410.18038](https://arxiv.org/abs/2410.18038)): fused prefill+decode attention on the same SMs. Up to 59% faster attention and 22% throughput in Sarathi. It fits a hand-written kernel stack well.
- **NanoFlow** ([arXiv 2408.12757](https://arxiv.org/abs/2408.12757)): nano-batch overlap gave 1.91× on A100. It needs a compute-bound regime, which we are rarely in.
- **Agent-level:**
  - Autellix ([arXiv 2502.13965](https://arxiv.org/abs/2502.13965)): program-level least-attained-service. 4–15×.
  - Continuum (TTL pinning).
  - Session-sticky routing beats load-based routing (vLLM AgentX). Single-node, this simply means "don't evict the active session".
- **Fairness:** VTC ([arXiv 2401.00588](https://arxiv.org/abs/2401.00588)).
- **Overlap scheduling:** SGLang's zero-overhead scheduler prepares batch N+1 on the CPU while the GPU runs batch N, for 1.1–1.3× ([LMSYS v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)). This matters with spec decoding, where acceptance must be committed on-device (ReplaySSM's "commit without sync").

### c4. Speculative decoding for agents and hybrids

- **MTP (shipped, one layer):**
  - Qwen3.5-27B acceptance length 3.25–3.62 at k=3 (z-lab, SGLang, B200).
  - Qwen3-Next per-position acceptance 0.897 / 0.719 / 0.476 at k=3 on GSM8K ([Red Hat](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)).
  - Agentic and coding text accepts more than chat.
  - **Pitfall:** host-side per-step bookkeeping can make MTP slower than no-spec ([vllm#47277](https://github.com/vllm-project/vllm/issues/47277): 0.86× at 83% acceptance). Keep everything on-GPU and inside graphs.
  - *(Unverified)* the MTP layer is a full-attention layer with its own KV. Budget about one extra attention layer of KV (+12.5% on 9B).
- **DFlash** ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036); z-lab drafters exist for Qwen3.5-4B/9B/27B/35B-A3B and Qwen3.6-27B): acceptance length 4.6–6.2, 2.1–3.0× at concurrency 32 on 27B. It is a separate model with sliding-window KV. In vLLM it is not mainline (vllm#40898 closed); the vLLM config lists `dflash`, so check the status before relying on it.
- **EAGLE-3:** no official Qwen3.5-27B head. DFlash ≥ EAGLE-3 in LMSYS comparisons ([LMSYS DFlash v2](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2)).
- **Suffix decoding:** see item 4. Also SAM-decoding ([arXiv 2411.10666](https://arxiv.org/abs/2411.10666)): +3–11% on top of EAGLE-2.
- **Hybrid state and speculation:**
  - vLLM: k+1 state slots, chains only.
  - SGLang: an intermediate state cache `[layers, slots+1, draft_tokens, HV, V, K]`, which supports trees.
  - ReplaySSM: checkpoint + ring.
    - Qwen3.5 kernel 1.43–1.64× and end-to-end 1.20–1.27× on *plain* decode at batch 256.
    - Spec verify at T=6 costs 1.27–1.72× a standard decode step, against 4.85× for the baseline.
    - Max concurrency under a fixed HBM budget is 3.0–3.3× the baseline ([ReplaySSM blog](https://dao-lab.ai/blog/2026/replayssm/); paper arXiv 2610.13778).
    - For our small batches at long context, the plain-decode gain shrinks to a few percent, because KV reads dominate. The spec-memory gain stays.
  - STree ([arXiv 2505.14969](https://arxiv.org/abs/2505.14969)): tree verification via transition-matrix accumulation, which assumes diagonal decay. GDN's (I − βkkᵀ) is not diagonal. Tree speculation on GDN is practical only by brute force (one state per node), so stay with chains.
- **Adaptive k:**
  - vLLM's `enable_adaptive_verification` admits draft slots by survival probability under a profiled step-cost budget ([docs](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/adaptive_verification.md)).
  - SmartSpec/TurboSpec do goodput-based control ([arXiv 2406.14066](https://arxiv.org/abs/2406.14066)).
  - On a 5090 with a capacity-limited batch you will almost always want k ≥ 1. On Spark, always.
- **Lossy verification:** FLy (in vLLM 0.33, `rejection_sample_method="fly"`) is explicitly lossy; 96–98% quality retention. Do not use it.

### c5. Kernel-level

- **GDN prefill.**
  - FLA chunk-64 WY/UT kernels have fused kkt+solve_tril, exp2 gates and native GVA ([FLA releases](https://github.com/fla-org/flash-linear-attention/releases)).
  - FlashQLA (TileLang, Qwen) runs on sm_120 forward bf16 at 2.1–2.6× over Triton for 8k–32k, with max |Δo| = 1e-3 vs the naive reference ([fla#1094](https://github.com/fla-org/flash-linear-attention/pull/1094)). It is restricted to cc 12.0 by #1098, so not GB10 ([Alibaba blog](https://alibabacloud.com/blog/flashqla-cp-bwd-friendly-fused-linear-attention-kernels-for-gdn_603084)).
  - FlashInfer `chunk_gated_delta_rule` has native SM120 kernels and emits mid-prefill state checkpoints ([docs](https://docs.flashinfer.ai/api/gdn_prefill.html)).
  - The one feature we need from our own kernel is **"emit state at offsets {p₁, p₂, …} (multiples of 64)"**, for item 1.
- **GDN decode.**
  - FlashInfer bf16 K-last "pretranspose" decode is in-place by slot index with no gather/scatter: 20.9 µs vs 59.1 µs Triton at batch 32.
  - FlashInfer also has `gdn_fused_decode_step`, which fuses the projection GEMV, conv1d, head split and recurrence ([docs](https://docs.flashinfer.ai/_sources/api/gdn_fused_decode.rst.txt)).
  - The llama.cpp warp-sharded state-column layout removed register spills at S_v=128 ([#20391](https://github.com/ggml-org/llama.cpp/pull/20391)).
- **Attention on sm_120.**
  - FA3 (wgmma) and FA4 (tcgen05) are not portable.
  - FA's own sm_120 path is an Sm80 subclass using mma.sync m16n8k16 with 99 KB smem, tested only at D ≤ 128 ([#2329](https://github.com/Dao-AILab/flash-attention/pull/2329)).
  - The gau-nernst 5090 study is the best public recipe ([link](https://gau-nernst.github.io/fa-5090/)).
  - For long-context decode, LeanAttention's stream-K split ([arXiv 2405.10480](https://arxiv.org/html/2405.10480v1)) and FlashDecoding++'s unified-max trick ([arXiv 2311.01282](https://arxiv.org/abs/2311.01282)) are the relevant ideas.
  - Shared prefix: Hydragen ([arXiv 2402.05099](https://arxiv.org/html/2402.05099v2)) and FlashInfer cascade ([blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)).
- **Megakernels / persistent kernels.**
  - Hazy "No Bubbles": Llama-1B bs=1 under 1 ms on H100, 2.5× vs vLLM; launch overhead is 1.3 µs per kernel even with graphs ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)).
  - Hazy throughput megakernel: +22% vs SGLang on 8×H100 70B ([blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)).
  - MPK: 1.0–1.7×, decode only, bs 1–16 ([arXiv 2512.22219](https://arxiv.org/html/2512.22219v1)).
  - The gain shrinks as weights grow, because step time is set by weight bytes. For 9B and 27B it is low priority.
- **Small-batch GEMM.**
  - Marlin W4A16 is near-ideal up to bs 16–32 ([arXiv 2408.11743](https://arxiv.org/abs/2408.11743)). It is lossy unless the reference is a 4-bit checkpoint.
  - On sm_120, vLLM NVFP4 often falls back to Marlin ([vllm#31085](https://github.com/vllm-project/vllm/issues/31085)).
  - No public data on cuBLASLt efficiency at M=1–64 on sm_120/121; benchmark it yourself.
- **Hardware notes.**
  - 5090: 209.5 TFLOPS dense BF16 with FP32 accumulate per the RTX Blackwell whitepaper (secondary sources conflict).
  - GB10: I found no reliable BF16 TFLOPS figure.
  - Prebuilt Spark images without sm_121 cubins ran Qwen3.5-35B-A3B at 13.3 tok/s, against 49 tok/s with a native build ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083)). Always build for sm_121 explicitly.

### c6. Practitioners: what matters for us

- **Horace He / Thinking Machines, "Defeating nondeterminism"** ([blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). Output differences under load come from batch-size-dependent reductions: split-K GEMMs, split-KV decode with a split *count*, and RMSNorm reduction strategy. The fix uses fixed configs and fixed split *sizes*. vLLM's run time went from 26 s to 42 s.
  - Directly relevant to "correctness is non-negotiable".
  - Recommendation: make the decode attention split size fixed now (cheap). Offer a batch-invariant mode, with a single GEMM config per (N, K), for golden testing rather than as the default.
- **Tri Dao / Dao AI Lab:** ReplaySSM is the most relevant 2026 hybrid-serving result.
- **Lianmin Zheng / SGLang:** MambaRadixCache, session-aware eviction and checkpoint levels. The lesson from their agent benchmark: **pool sizing for idle sessions matters more than any kernel.**
- **Woosuk Kwon / vLLM:** removed cascade attention (Oct 2026). vLLM AgentX lists prefetch-during-idle and soft-pinning as future work.
- **Zihao Ye / FlashInfer:** GDN prefill/decode kernels with checkpoint export; cascade inference.
- **Georgi Gerganov / llama.cpp:** semantic checkpoints at user-message boundaries with a minimum spacing. The chunked-vs-recurrent GDN prefill crossover is "device-dependent". I found no substantive write-up of Spark bottlenecks.
- **antirez / ds4** ([GitHub](https://github.com/antirez/ds4)): a deliberately narrow engine for one model with Spark as the primary CUDA target, and on-disk KV session snapshots so resumes skip prefill. It validates both aqwenator's "one family only" strategy and the session-tier idea.
- **Hazy Research:** megakernels. Big wins for 1B models at bs=1, diminishing for ours.
- **Kimi / Mooncake** ([arXiv 2407.00079](https://arxiv.org/abs/2407.00079)): KV-centric tiering. Mooncake store supports vLLM hybrid `align` groups.
- **DeepSeek open-infra** ([DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [profile-data](https://github.com/deepseek-ai/profile-data)): shape-specialised JIT GEMMs and two-microbatch overlap. The transferable idea is shape-specialised kernels for our fixed model shapes, which cuda-oxide makes cheap.

---

## (d) Open questions (measure before committing)

1. **What is the real session distribution?** Turns per user episode, the share of turns that are tool-loop continuations vs new user messages, tool idle time, and shared-prefix length. Item 1's value differs by 10× between tool-loop-heavy and chat-heavy traces. Build a trace replayer first; a SWE-bench/OpenHands-style trace through our own chat template is enough.
2. **Do warm turns actually hit?** Measure the fraction of tool-loop turns where re-rendered history token-matches the sampled tokens. If JSON-argument re-serialisation breaks it often, the decode-end checkpoint (item 1-ii) is not worth its memory.
3. **PCIe throughput on the target 5090 box.** Is the slot x16 Gen5? What is the pinned H2D bandwidth with concurrent kernels? That sets whether host restore fits inside the prefill of 1–8k new tokens.
4. **GB10 specifics:**
   - Achievable DRAM bandwidth with concurrent CPU traffic.
   - BF16 mma.sync throughput on sm_121.
   - Whether `cudaMallocManaged`/host-registered memory costs anything vs device allocations.
   - NVMe read bandwidth for a cold-session tier.
5. **MTP acceptance on our agent traces at 64k context**, and verify cost at k=3 when the verify attention reads 64k of KV for 4 query rows. It should still be memory-bound; confirm.
6. **The bit-exactness contract.** Is "matching the reference" defined as greedy-token agreement on a golden set, logprob tolerance, or bitwise equality across batch compositions? This decides whether batch-invariant kernels are a requirement (about 20–60% slower GEMMs in Thinking Machines' measurements) or a test mode.
7. **27B on a 5090.** With the official FP8 checkpoint, about 2–3 GB remains for KV. Is 27B on the 5090 a goal at all for long-context agents, or is that Spark's job? A sharper question: does FP8 count as "the reference"?
8. **Status checks:**
   - Is SGLang `--enable-int8-mamba-checkpoint` accuracy published?
   - Does vLLM GDN ReplaySSM (#62004) land, and with prefix caching?
   - Is DFlash supported on vLLM main for Qwen3.5?
9. **Qwen3.6/3.8 template drift.** Check whether newer templates add a `preserve_thinking`-style kwarg. If so, item 1-ii becomes much more valuable for those models.
