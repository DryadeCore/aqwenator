# Moonshots: ideas from other fields for serving many long-context agents per GPU

Date: 2026-11-13. Status: research only; nothing here is implemented.
Companion to [`sota-survey.md`](sota-survey.md). The survey covers ML serving prior art, so this note only cites prior art to say what is new.

Scope: aqwenator serving `qwen3_5` (3 Gated DeltaNet layers per gated GQA layer) on an RTX 5090 or a DGX Spark, for many concurrent agent sessions with 16k–128k context.

Exactness labels used throughout:

- **L0 (bit-exact):** output bits are identical to the engine's plain path.
- **L1 (math-exact):** no approximation; only the floating-point reduction order changes. This is the same class of difference as running on another GPU or with another split-K.
- **Lossy:** the model is approximated.

Numbers marked *(measured)* were measured for this note. Numbers marked *(derived)* are back-of-envelope calculations, with their assumptions stated. Everything else cites a source.

---

## 0. The physics

### 0.1 Constants

Model constants come from the HF configs plus my own parameter count. All bf16 unless stated.

| | Qwen3.5-9B | Qwen3.5-27B |
|---|---|---|
| Layers | 32 (24 GDN + 8 attention), hidden 4096, MLP 12288 | 64 (48 GDN + 16 attention), hidden 5120, MLP 17408 |
| Attention | 16 Q heads / 4 KV heads, head dim 256 → **GQA group 4** | 24 Q heads / 4 KV heads → **GQA group 6** |
| GDN | 16 k-heads, 32 v-heads, 128×128 fp32 state | 16 k-heads, 48 v-heads |
| Weights read per decode step (no embedding table) | **15.9 GB** (6.91B body + 1.02B LM head) | ≈25.6 GB (FP8 body) |
| Embedding table (untied) | 2.03 GB (only gathered, never streamed) | 2.5 GB |
| MTP head | 0.49 GB, **one full-attention layer with its own KV** (+4 KiB/token) | ≈0.5 GB, +4 KiB/token |
| KV per token | 32 KiB (+4 KiB MTP) | 64 KiB (+4 KiB MTP) |
| GDN + conv state per session | 50.4 MB | ≈154 MB |
| Prefill FLOPs per new token | 13.8 GFLOP dense (no LM head) + 131k·ctx attention: **22.4 GFLOP at 64k**, 31 at 128k | 48.6 GFLOP dense + 393k·ctx attention |
| GDN prefill FLOPs | ≈0.1 GFLOP/token (negligible) | ≈0.3 |

Hardware constants:

| | RTX 5090 | DGX Spark (GB10) |
|---|---|---|
| DRAM bandwidth | 1.79 TB/s (assume 85% achieved) | 273 GB/s (85%) |
| Dense bf16, fp32 accumulate | 209.5 TFLOPS ([forum](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)). The **fp16-accumulate rate is 2× that** (see 4.2). | ≈100 TFLOPS measured ([StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops)) |
| Roofline ridge | ≈117 FLOP/B | ≈366 FLOP/B |
| Host link | PCIe 5 x16, ≈50 GB/s achieved | unified memory, no link |

### 0.2 Consequences that drive the rankings *(derived)*

**F1. Decode at long context is bound by KV bytes, and the ceiling is hard.**
- Aggregate decode tokens/s ≤ BW_eff / (KV bytes per token × ctx / (a · c)), where a = accepted tokens per verify step and c = lossless compression ratio.
- For 9B at 64k, a 5090 tops out at **≈650 tok/s total** for a = c = 1, however many sessions are resident. Spark tops out at **≈100 tok/s total**.
- At context ≥ 64k there are only three lossless multipliers: **speculation (a), compression (c), and shared-prefix scans.**
- On Spark with a 50 ms ITL target, 9B at 64k can batch only **2 sessions**, even with MTP speculation at a ≈ 2.

**F2. Agentic turns are prefill-heavy.**
- A 3k-token tool result at 64k context costs 67 TFLOP. That is 0.46 s on a 5090 and 0.9 s on Spark.
- Decoding 800 tokens costs about the same GPU time once decode is batched.
- So **prefill FLOPs and decode bytes compete for the same GPU-second**, but they stress different resources: compute versus bandwidth. That is the opening for co-scheduling (M2).

**F3. Swapping beats recomputing by roughly 200× on the 5090.**
- A 64k session is 2.36 GB, about 47 ms over PCIe.
- Recomputing it is 1.19 PFLOP, about 8 s.
- On Spark there is no link, and memory (≈95 GB of KV room) is not the constraint. Bandwidth is.

**F4. Swap-in can hide completely behind the turn's own prefill.**
- Loading one attention layer's KV at 64k is 206 MB compressed (268 MB raw), about 4–5 ms.
- Prefill of n new tokens through 1/8 of the model takes n × 19 µs.
- So streaming KV layer by layer is free once **n ≳ 220 tokens at 64k (≈320 at 128k)**. Every tool result clears that bar. **On the 5090, idle sessions do not need to occupy VRAM at all.**

**F5. The Qwen3.5 chat template hides a large re-prefill.** This comes from reading `chat_template.jinja`.
- Inside a tool loop (every message after the last real user query), reasoning is re-rendered as `<think>\n{reasoning.trim}\n</think>\n\n{content}`.
- When a new user message arrives, `last_query_index` moves, and **every earlier assistant message loses its thinking**.
- Token positions shift from the first assistant message of the previous episode onward. So the whole previous tool loop is re-prefilled.
- Example: 15 tool turns × 2.5k tokens is about 37k tokens to re-prefill. That is **≈0.85 PFLOP: 5.8 s TTFT on a 5090 and 11 s on Spark**, paid on the turn where a human is waiting.
- Tool-call arguments are also re-serialized (`tojson` for mappings), so even inside a loop the re-render does not always match the generated tokens byte for byte.
- The survey (§c1.3) identifies the fallback point. Nobody hides its cost (see M1).

**F6. Weight exponents carry about 2.6 bits.** *(measured)*
- I range-fetched about 100M weights from 17 tensors of Qwen3.5-9B (LM head, embedding, MLP layers 0/31, GDN and attention projections, MTP).
- H(exponent) = 2.55–2.67 bits. H(sign+mantissa) = 7.97 bits. So the entropy bound is ≈10.55 bits/weight (1.52×).
- With a 4-bit exponent code relative to the per-128-weight group maximum (code 15 = escape), only **0.009–0.020% of weights escape**. Including the group base, that is ≈12.1 bits (1.32×), with exceptions in ~1.4% of groups.
- A 3-bit window escapes 2.5–4%. That matches ZipServ's 7-exponent window at ≈11.3 bits.
- Script: Appendix A.

**F7. Speculation is nearly free below the ridge.**
- Decode GEMMs stay bandwidth-bound until about 117 (5090) or 366 (Spark) tokens per step.
- So 10 sessions × 5 verified tokens on a 5090, or 32 × 5 on Spark, costs no more time than plain decode.

**F8. GQA group × verify width fits the tensor-core tile.**
- 9B: 4 Q heads per KV head × (k+1 = 4) = **16 rows, exactly one `mma.m16n8k16` M-tile**. Each K/V element loaded once feeds 16 rows with zero padding.
- 27B: 6 × (k+1 = 8) = 48 = 3 tiles.
- Choose k from the tile, not only from acceptance.

---

## 1. Candidate transfers

"Gain" is for 9B at 64k unless stated. Effort: S ≈ days, M ≈ 1–3 weeks, L ≈ a month or more.

| # | Source idea (field) | Mechanism in aqwenator | Why it fits this workload and hardware | Expected gain (BoE) | Exact | Effort | LLM prior art |
|---|---|---|---|---|---|---|---|
| 1 | **Runahead execution** (CPU, Mutlu HPCA'03) + DB background writer | While a session is idle, use spare GPU time to prefill the *predicted* next prompt prefix: the re-rendered assistant turn, the `<tool_response>` wrapper and, after a final answer, the thinking-stripped history (F5). Check the prediction by token comparison when the request arrives. | Agents idle for seconds and humans for tens of seconds. The template is deterministic, and the server applies it. | Removes 0.1–0.3 s per tool turn and **5–11 s per new user message** from TTFT (F5). | L0 (checked) | M | None for template re-render. Continuum/Tokencake prefetch KV, but do not compute ahead. |
| 2 | **Write-through buffer pool + GreedyDual-Size + Belady with predicted next use** (DB, Cao & Irani'97, Hawkeye ISCA'16) | Immutable full KV pages are written to host as they fill, so GPU eviction is free (always clean). The victim maximizes predicted-next-use / (recompute-cost / bytes). Next use is predicted per tool name (#22). | Tool idle time is predictable per tool. Pages are append-only and immutable. | Eviction stalls drop to 0. Only decoding sessions are resident (F4): **2–3× more sessions per GB of VRAM** than TTL pinning. | L0 | M | vLLM tiered offload (demand-driven LRU/ARC), Continuum (TTL), CachedAttention |
| 3 | **ARIES checkpoint + redo tail** (DB WAL) | GDN checkpoints at template boundaries, stored at `floor64(b)`, plus a ≤63-token redo replay on a canonical chunk grid. | GDN state is a checkpointable summary. Divergence points are message boundaries. | Hits at exact message boundaries. Covered in survey item 1. | L0 | M | Marconi, vLLM `align`, SGLang #49886 |
| 4 | **Prefetch pipelining, layer-streamed** (OS readahead) | On resume, load GDN state (3 × 1.6 MB) for layers 0–2 first, then stream each attention layer's KV one layer ahead of compute. | F4: hidden for ≥220 new tokens. | Swap-in adds 0 ms to TTFT, instead of 32–47 ms. | L0 | S–M | CachedAttention (layer-wise preload) |
| 5 | **Banker's algorithm / working-set load control** (OS, Dijkstra'65, Denning'68) | A session enters the decode set only if its KV plus a max-new-token reservation fits. Overflow pauses (swaps) the youngest decoder for ~40 ms. Never preempt by recompute. | Recompute preemption at 64k costs 8 s (F3), which is a p999 cliff. | Removes recompute storms. p999 TTFT is bounded by swap time, not prefill time. | L0 | S | vLLM preempts by recompute |
| 6 | **PFOR lightweight compression** (MonetDB/X100, Zukowski ICDE'06) applied to weights | 12-bit bf16: 8-bit sign+mantissa plane, 4-bit exponent delta from a per-128 group base, patch list for escapes. Decoded in registers inside the decode GEMV; prefill decompresses one layer into scratch, then runs cuBLASLt. | Decode is bandwidth-bound. Spark has compute to spare. 9B bf16 is all-bf16, so the whole 25% applies. | Weights 15.9 → 12.1 GB. **Frees 3.9 GB of VRAM** (≈1.6 sessions at 64k) and makes the weight-read part of each step 1.3× faster. | L0 | M | ZipServ (fused, 11.3 b), DFloat11, DGPP bf12 on GB10 (+5–12% on MoE/FP8 mixes) |
| 7 | **PFOR for KV, read in-kernel, with an LSM memtable** (LSM trees, O'Neil'96) | The raw tail page (≤64 tokens) is the memtable. On flush, encode the page (per token-head exponent base, 4-bit deltas). Decode attention reads the compressed history plus the raw tail. | F1: KV bytes *are* decode throughput at 64k+. Every lossless 1.3× is 1.3× on tokens/s and on capacity. | KV ≈1.25–1.33× (SplitZip measured ≈1.3× on KV transfer). Capacity and throughput gain the same. | L0 | M–L | SplitZip / ExANS (transfer only, not in the attention kernel) |
| 8 | **Bit-plane split + entropy coding** (columnar formats, Blosc bitshuffle) | Host tier stores exponent planes with ANS, mantissa planes raw. | PCIe and host RAM are the 5090's second tier. | Host bytes ≈1.5× smaller (ExANS reports 1.51×). Swap ≈32 ms → ≈25 ms. | L0 | S (nvCOMP) | ExANS |
| 9 | **Demand paging / cold-data placement** (OS) | Keep the 2.03 GB embedding table in pinned host memory and gather rows zero-copy (8 KB/token). Never load `model.visual.*`. | The embedding table is gathered, never streamed. Prefill of 4k tokens gathers 32 MB, about 0.7 ms over PCIe. | **+2 GB of VRAM** for KV (≈0.85 more 64k sessions). | L0 | S | Not seen |
| 10 | **Cooperative scans** (DB, Zukowski VLDB'07; QPipe) | Decode attention over the shared system-plus-tools prefix runs once for all sessions' queries (M = sessions × 4 × (k+1) rows), then merges with each session's suffix partial in a fixed order. | Agent harnesses ship 10–20k-token shared prefixes. | KV read −23% at 64k, **−47% at 32k** (15k shared). In the model this is +60% decode throughput on Spark at 32k. | L1, or L0 with fixed split points | M | Hydragen, FlashInfer cascade (vLLM removed it in #61126) |
| 11 | **Multiprogramming of I/O-bound and CPU-bound jobs; Little's law for memory parallelism** (OS, queueing) | Run the bandwidth-bound decode graph on a few SMs (enough bytes in flight to saturate DRAM) and the compute-bound prefill on the rest, at the same time. See M2. | F2: agentic work is about half and half. Decode leaves roughly 55–75% of tensor throughput idle. | **Turn GPU-time −24% (5090, 64k) and −34% (128k); +30–60% agents** in the model. | L0 if kernels are SM-count-invariant | M | NanoFlow, POD-Attention, Bullet, semi-PD |
| 12 | **Store buffer / retire-time commit** (CPU ROB) + redo log | Spec verification reads the GDN state but never writes it. The next step's kernel replays the m accepted (k, v, β, g) updates **in recurrent order** in registers, then verifies. See M4. | GDN state cannot be rolled back. vLLM writes k state copies per step. | GDN traffic per step = 1 read + 1 write, independent of k. **Bit-identical to non-speculative decode.** | **L0** | M | ReplaySSM (chunked replay, L1), SpecLA (delayed update) |
| 13 | **Tournament branch predictor** (McFarling'93) | A per-session chooser between MTP drafts and a suffix-automaton draft over the session's own tokens and tool outputs, with saturating counters keyed by context (in thinking, in tool-call args, in a code block). | Tool args and code edits quote the context. Thinking is novel text. | a ≈2.0 → 2.6 (assumed). SuffixDecoding reports up to 3.9–5.3× on agentic traces. | L0 (drafts are checked) | M | SuffixDecoding, ArcticInference hybrid |
| 14 | **Pipeline gating and confidence estimation** (Manne ISCA'98) | Choose k per session per step from roofline headroom (F7) × draft confidence. Drop to k = 0 when compute-bound. | 5090 at batch 10: k ≤ 10 is free. Spark: k ≤ 11 at batch 32. | Avoids wasted verification at high load. | L0 | S | vLLM adaptive verification |
| 15 | **Value predictors may be wrong** (CPU value prediction) | Everything on the draft path may be lossy: give the MTP layer a 4k sliding window plus sink KV and FP8 weights. | The MTP KV is 4 KiB/token, 11% of a session's KV, and only affects draft quality. | **−11% KV bytes and capacity**; the MTP layer drops 0.49 → 0.25 GB. | L0 for outputs (drafts are checked) | S | Not seen for MTP KV |
| 16 | **Tile shaping / data reuse** (HPC GEMM blocking) | Verify-attention tiles packed as (GQA group × verify tokens) = 16 rows (F8). | Long-context attention is bytes-bound. Each KV byte should feed a full mma. | Verification attention costs about the same as decode attention. | L0 | S | Implicit in FlashInfer/FA3 decode |
| 17 | **Counter-based RNG** (HPC, Random123/Philox, Salmon SC'11) + Gumbel-max sampling | The sample at position t is argmax(logits/T + G(seed, t)) after top-k/top-p. A draft is accepted iff it equals the sample. | Agents need reproducible runs. Deterministic sampling makes speculation exact *per seed*. | Same acceptance rate as rejection sampling with a point-mass draft. The output stream depends only on (prompt, seed). | **L0** | S | Not seen in serving engines |
| 18 | **Serializable isolation for numerics** (DB) | Batch-invariant GEMMs (fixed tile and K order per (N,K), no M-dependent split-K), fixed-split attention, a canonical 64-token GDN grid. Cache hits, swaps, batching and speculation then become invisible in the output bits. | The #1 requirement is exactness. It also makes aggressive caching safe to test. | Cost: maybe 0–20% on decode GEMMs (to measure). Benefit: correctness ships with the optimizations. | L0 | M | Thinking Machines batch-invariant ops (their GEMMs were 20–60% slower) |
| 19 | **SRPT with aging, cache-aware sizes** (queueing, Schrage'68; SOAP, Scully'18) | Order prefill chunks by remaining *uncached* FLOPs, with aging so elephants (cold 60k sessions, F5 re-prefills) cannot starve. Preempt at 64-token chunk boundaries. | Prefill sizes are known on arrival and vary by about 100× (300 tokens to 60k). | At ρ = 0.7 with C² ≈ 3, FCFS mean wait is about 4.7 E[S]. SRPT puts typical tool turns at ≈1.1 S. **p99 TTFT for ordinary turns 5–10× lower.** | L0 | S | FastServe (MLFQ), Autellix (PLAS) |
| 20 | **Frame pacing** (game engines) + **EDF** (real-time) | Each engine step has a time budget equal to the tightest decoder's ITL deadline, minus a predicted decode cost from a roofline model (#24). Leftover time goes to prefill chunks sized to fit. | aqwenator m3b has ITL p99 = 82 ms vs p50 = 3.7 ms at c=32. vLLM has 6.5 ms. That gap is prefill interference. | ITL p99 ≤ 1.5× p50. | L0 | S | Sarathi-Serve (static token budget) |
| 21 | **Cut-through switching** (networking, Kermani & Kleinrock'79) + **madvise hints** (OS) | Optional session API: `POST /sessions/{id}/append` streams tool stdout while the tool runs, so prefill proceeds incrementally, holding back the last pre-token for BPE safety. Plus hints: `idle_expected_ms`, `closed`. | Long tool outputs (tests, builds) arrive over seconds. | TTFT after a slow 8k-token tool ≈ last chunk only (≈0.1 s instead of 1.2 s). | L0 | M (API) | Not seen |
| 22 | **Quantile sketches** (XGBoost weighted quantile sketch; KLL, Karnin'16) | A per-(harness, tool name) KLL sketch of idle duration. Prefetch at the p10 of the predicted return time; evict past the p90. | Tool names appear in the generated tool call. | Prefetch hit rate; makes #2 and #4 proactive. | n/a | S | Continuum (predicted TTL) |
| 23 | **Copy-on-write fork / MVCC snapshots** (OS, DB) | Subagent spawn and parallel-branch retries share refcounted KV pages and copy the 50 MB GDN state once. | Agents fork for subagents and retries. | Fork cost 50 MB instead of a full re-prefill. | L0 | S | RadixAttention (KV only) |
| 24 | **Cost-based optimizer with a learned residual** (DB; CatBoost oblivious trees for branch-free scoring) | A roofline step-time model plus a tiny oblivious-tree residual predicts step latency from (Σctx, rows, prefill tokens). It feeds #11, #14, #19 and #20. | Every scheduler decision above needs a 1 µs, accurate cost model. | Enables the rest (±5% step prediction). | n/a | S | Not seen |
| 25 | **Morsel-driven parallelism** (HyPer, Leis SIGMOD'14) | Split-KV decode attention with fixed 2k-token morsels, pulled from a device-side queue by a persistent kernel. Graph-compatible, and fixed morsel size keeps it batch-invariant. | Contexts range from 16k to 128k in one batch, so static grids load-imbalance. | Long-context decode attention at 85–90% of bandwidth instead of 60–70% with skew. | L0 | M | FlashInfer balanced scheduling, FlashDecoding++ |
| 26 | **Trace compilation / megakernel** (JIT, HyPer query compilation) | A persistent decode megakernel that prefetches the next layer's weights while the current layer runs. | Small-batch 5090 decode loses ~6–8% to launch gaps. | +5–15% at batch ≤ 8. | L0 | L | Mirage MPK, Hazy megakernels |
| 27 | **CPU attention over cold KV** (heterogeneous HPC) | On EPYC/Threadripper hosts (300+ GB/s DRAM), compute attention partials for host-resident KV on the CPU and merge with log-sum-exp. | Only on workstation hosts. Desktop DDR5 (~90 GB/s) adds 5%. | +15–25% decode bandwidth on 8–12-channel hosts. | L1 | L | Neo, FastDecode |
| W1 | Fp16-accumulate with fp32 promotion per k16 (DeepSeek-V3 FP8 promotion; prescale by powers of two) | Prefill GEMMs on GeForce run at 419 instead of 209.5 TFLOPS. bf16 → fp16 conversion is exact for \|x\| ≥ 2^-17, given power-of-two prescaling. | Prefill dominates agentic GPU time (F2). | **≈1.8× prefill on the 5090.** No gain on GB10 (to check). | **Lossy** (partial sums keep 11 bits, not 24) | M | DeepSeek-V3 (FP8) |
| W2 | FP8/NVFP4 KV | See survey items 7 and 10. | F1 | 2× / 3.5× on KV bytes | **Lossy** | M | many |

**Rejected after estimating** *(derived)*:
- **Ozaki-scheme exact GEMM on int8 tensor cores.** bf16 values span about 14 exponents within a 128-group (F6), so it needs 3 slices per operand, which is 9 int8 GEMMs. At ≤4× the int8 rate that is slower than bf16.
- **Exact LM-head pruning by bounds** (VA-file filter-and-refine, WAND). Rigorous Hölder/Cauchy–Schwarz error bounds are about √4096 = 64× looser than real logit gaps, so the candidate sets explode. With top_k = 20 the sampling side *would* be exact, but the bounds kill it.
- **A layer-0 token→projection table** (partial evaluation). 6.1 GB of table to save 0.6% of FLOPs and bytes.
- **Recomputing KV instead of swapping it** on the 5090. 200× worse (F3).
- **Gorilla XOR-delta between GDN snapshots.** The state decays and churns over hundreds of tokens, so XOR deltas are not sparse.
- **Reusing KV for repeated tool text at a different position** (CacheBlend-like). Not exact, so it fails rule #1.

---

## 2. Top 5 moonshots

The ranking is by (gain on concurrent agents at p99 SLO) × P(success) / effort, restricted to lossless ideas. M1, M2 and M4 are new relative to what the survey found in vLLM and SGLang. M3 and M5 are new mainly in where they are applied.

### M1. Runahead for idle sessions: spend idle GPU time on the next prompt

**Idea.** A CPU in runahead mode keeps executing past a cache miss, to warm caches for work it will need. Our "miss" is the tool call or the human. During that time the server knows, with high probability, most of the next prompt's tokens, because **the server itself renders the chat template**. Everything that can be computed ahead should be, at idle priority, and checked by token comparison when the request arrives.

**Design (Qwen3.5-specific).**

1. **Predict the next prefix.** At the end of each generation, build the predicted next-turn token sequence:
   - **Tool call ended the turn** (we are inside the loop): `history + render(assistant{reasoning, content, tool_calls}) + "<|im_end|>\n<|im_start|>user\n<tool_response>\n"`.
   - **Final answer ended the turn:** the whole history re-rendered with `last_query_index` moved to a hypothetical next user message. That strips thinking from every assistant message of the episode (F5). Append `"<|im_start|>user\n"`.
   - **Per-session harness policy, learned like a branch predictor** from the first two turns: does the harness send back `reasoning_content`? Does it reorder JSON keys? Does it drop `content` when there is a tool call? A 2-bit counter per policy bit picks which variant to precompute. Both variants can be run when GPU slack allows.
2. **Find the divergence point.** Take the longest common prefix of the predicted tokens with the cached tokens (generated tokens included). Resume from the latest GDN checkpoint at or before it. Checkpoint policy (shared with survey item 1): keep {end of shared prefix, generation start of the first turn after the last user query, latest generation start, latest generation end}. That is 4 × 50 MB per session, kept in host memory on the 5090.
3. **Run ahead.** Prefill the predicted delta as an *idle-class* job: lowest priority in the M5 scheduler, preemptible every 64-token chunk. Results go on a **copy-on-write branch** (#23): new KV pages plus a new GDN state. The real history is never overwritten.
4. **Canonicalize while idle (strict mode).** Decode-produced GDN states differ in rounding from a chunked prefill of the same tokens. In idle time, re-prefill the generated tail in canonical 64-chunks from the generation-start checkpoint. The next turn then is bit-identical to a cold prefill. That costs about 800 × 22 GFLOP ≈ 18 TFLOP per turn, spent only when idle.
5. **Commit or discard.** When the request arrives, tokenize it and match it against the radix tree; the speculative branch is just another path. A match makes the branch real. A mismatch discards it, leaving only wasted idle cycles.
6. **Prefetch** (5090). A KLL sketch per (harness, tool name) gives a return-time distribution. At p10 minus the transfer time, start the layer-0..2 state copy. The rest streams behind prefill (F4).

**Why it is a moonshot.** The new-user-message turn is the turn with a human waiting, and its 5–11 s of re-prefill is hidden behind that human's 10–60 s of reading and typing.
- For tool turns it saves 0.1–0.3 s (re-render deltas plus wrapper).
- More importantly, idle prefill fills the troughs of a bursty load. So the M5 queue sees less work exactly when agents bunch up.
- It does not save GPU-seconds overall (except where it fills idle troughs). It moves them off the critical path.

**Cheap experiment (CPU only, 1–2 days).**
- Record 50+ agent episodes: mini-swe-agent or OpenHands against vLLM 0.33 serving Qwen3.5-9B, logging the raw message JSON and the rendered prompt token IDs per request.
- Replay offline through our tokenizer and template and measure:
  - (a) the divergence-position distribution per turn type;
  - (b) the runahead hit rate (predicted prefix == actual prefix up to the tool payload) per harness;
  - (c) tokens moved off the critical path per episode;
  - (d) the F5 tax in tokens per user message.
- **Kill** if (b) < 70% or (c) is under 10% of total prefill tokens.
- **Go** if the F5 tax is over 20k tokens per user message for harnesses that send back reasoning.

**Failure modes.**
- Harnesses that edit history (compaction, truncating old tool outputs, injecting timestamps or todo lists). The divergence then moves early and runahead cannot help. The trace study measures this directly.
- Runahead stealing GPU time at high load. It must be strictly idle-class, never admitted when queue slack is under one chunk.
- Branch memory blowup. Cap at one live branch per session; the branch's KV pages count against the host-tier budget.
- BPE merges across the predicted/actual boundary. Predict only up to the last special token or newline; the token check catches the rest.

### M2. Little's-law co-scheduling: bandwidth-bound decode and compute-bound prefill on the same GPU, at the same time

**Idea.** The classic argument for multiprogramming is to overlap I/O-bound and CPU-bound jobs. Here:
- Decode at long context is DRAM-bound with roughly 55–75% of tensor throughput idle (F1, F7).
- Agentic prefill is compute-bound with almost no DRAM traffic. Weights are amortized over 2k-token chunks, and history KV is read once per chunk: 2 GB at 64k, about 1.3 ms, against about 300 ms of compute.

Little's law sets how few SMs saturate DRAM: bytes in flight = BW × latency.
- 5090: 1.79 TB/s × ~0.7 µs ≈ 1.3 MB in flight. With cp.async 4-stage pipelines (~64 KB per SM in flight) that is **about 20 of 170 SMs**.
- GB10: 273 GB/s × ~1 µs ≈ 270 KB, about **5–8 of 48 SMs**.
- Decode also needs compute: rows × (15.9 GFLOP dense + 131k·ctx attention) per step. At 9 sessions × 5 rows and 64k, that is 0.7 TFLOP dense + 0.4 TFLOP attention per 16 ms step, about 45% of the 5090's achievable compute. On Spark at 12 sessions × 5 rows it is ~40%. So decode needs ~45% of SMs for its FLOPs, and fewer still to saturate DRAM.
- The rest of the tensor throughput is free for prefill.

**Design.**
- Two persistent CTA pools, or two streams with fixed CTA counts:
  - **Decode**, high priority: the CUDA-graph decode step, every kernel launched with a grid capped at S_d SMs. The memory-bound kernels are written as persistent loops over tiles, so grid size never changes results (L0).
  - **Prefill**, low priority: our own mma.sync GEMM and attention kernels with grids capped at S_p = SMs − S_d. cuBLASLt cannot be capped reliably. Use it only when no decode is running, or use green contexts if they turn out to co-run on GeForce (the survey notes partitions are "not guaranteed to run concurrently"; that needs a test).
- S_d is chosen per step by the cost model (#24): the smallest S_d whose predicted decode time meets the frame budget (#20).
- Keep prefill's L2 footprint in check. Tile prefill GEMMs to stream weights with an evict-first cache hint (`ld.global.L1::no_allocate` / `cp.async ... .L2::evict_first`). Decode's KV stream is evict-first as well. Neither will hit in a 96 MB L2 at 64k anyway.
- Numerics: no kernel may derive its reduction split from the SM count. Splits come from fixed morsel sizes (#25). Then co-running is L0.

**Expected gain** *(derived, model in §3.3, assuming 80% of decode's idle compute is recoverable)*:

| Device / context | Turn GPU-time, serial | Turn GPU-time, co-run | Agents (9B) |
|---|---|---|---|
| 5090, 64k | 1.02 s | 0.78 s | +31% |
| 5090, 128k | 1.87 s | 1.24 s | +51% |
| Spark, 32k | 1.97 s | 1.30 s | +51% |

The survey's objection, that NanoFlow "needs a compute-bound regime, which we are rarely in", concerns overlapping inside *one* batch's operators. Here the two job classes come from different sessions, and agentic load always contains both.

**Cheap experiment (1 day on the B200, then the 5090 and Spark).**
- Capture the existing decode graph (batch 8, synthetic 64k contexts). On a second stream, loop a bf16 GEMM of 2048 × 12288 × 4096 with the grid capped at S_p.
- Sweep S_d.
- Measure decode step inflation and GEMM TFLOPS retained.
- **Kill** if no S_d gives ≤ 15% decode inflation with ≥ 50% prefill throughput retained.

**Failure modes.**
- The hardware CTA scheduler does not honor stream priority well enough: long-running prefill CTAs block decode CTAs, and ITL spikes. Mitigations: persistent decode kernels that grab SMs first and keep them, or partitioning inside a fused kernel (POD-Attention style).
- DRAM row-buffer and L2 interference costs more than modeled.
- Writing our own prefill GEMM to near cuBLASLt speed on sm_120/121 is required, because cuBLASLt grids cannot be capped. This is the main effort risk.

### M3. PFOR everywhere: lightweight, in-register, lossless compression of weights and KV

**Idea.** X100/Vectorwise showed that when the CPU is fast and memory is slow, you compress with *lightweight* codecs (PFOR, PDICT) that decompress in registers at memory speed, never with general-purpose codecs. GB10 is the extreme case: a 366 FLOP/B ridge.

The bf16 exponent is the compressible field. F6 measured it on our model: 4-bit frame-of-reference codes escape on 0.01% of weights.

**Design.**
- **Weights, decode path.** Use two planes per tile, in the same 16×16 fragment order that `ldmatrix` would produce:
  - an 8-bit sign+mantissa plane;
  - a 4-bit exponent-delta plane, plus a 1-byte group base per 128 weights along K.
  - Escapes go to a per-tile patch list (PFOR "exceptions"), applied after the vector decode as a scatter into registers. Only ~1.4% of 128-groups have any escape.
  - Reconstruction is `bf16 = (s_m & 0x80) << 8 | (base - d) << 7 | (s_m & 0x7F)`: about 3 `lop3`/`prmt` operations per pair using packed 16-bit math. That is under 10% of ALU throughput at full DRAM rate on either GPU.
  - Same arithmetic in the same order, so the result is **L0**. The decode GEMV kernel already loads through registers.
- **Weights, prefill path.** Decompress the next layer into a bf16 scratch buffer one layer ahead (0.5 GB/layer → 0.3 ms on the 5090, overlapped), then call cuBLASLt. Extra traffic is under 1% of a compute-bound prefill.
- **KV.**
  - The current page (64 tokens) is raw: the memtable.
  - On page fill, a flush kernel encodes it with a per-(token, KV head) exponent base over the 256-wide row (1 byte per 256 values), using the same 12-bit layout.
  - Decode attention loads compressed K/V tiles, decodes into registers and builds `mma` B-fragments directly. Each decoded element feeds 16 rows (F8), so decode cost is amortized 16×.
- **GDN state:** leave it fp32 and uncompressed while active (survey §c2).
- **Host tier (5090):** ANS on the exponent plane only (nvCOMP), which reaches ≈10.6 bits.

**Expected gain** *(derived; KV ratio assumed 1.3, to be measured)*:
- **5090, 9B.** VRAM for KV grows from 9.4 GB to 16.6 GB, together with #9 and #15 (embedding in host, MTP window). Resident 64k sessions: 4.3 → 9.7. Decode step bytes per session-token drop 23%.
- **Spark, 9B, 64k, under a 50 ms ITL target.** The decodable batch grows from 2 to 6 sessions (weights and KV both cheaper). Agents grow 2.2 → 6.1 *with no change to speculation* (MTP at a = 2 in both cases).
- **Weight-only ratio:** 1.32× (12.1 bits). ZipServ-style 3-bit windows reach ≈1.42× but need 2.5–4% escapes. Measure both.

**Cheap experiment.**
- (a) **Done here:** weight exponent stats (F6).
- (b) **1 hour on the B200:** run a 64k-token forward of 9B and dump K (post-RoPE, post-k_norm) and V for all 8 attention layers. Run the Appendix A script on the dump. **Kill the KV half** if the escape rate is over 0.5% at 4 bits, or the ratio is under 1.2×.
- (c) **2–3 days:** write a decode GEMV for M ∈ {1, 8, 16, 32, 64} with and without exponent decode on Spark and the 5090. **Kill the weight half** if it is under 1.15× at M ≤ 32 on Spark.
- DGPP's +5–12% on GB10 is a warning sign. Their models were MoE/FP8 with only part of the bytes in bf16, and Qwen hyper-connection weights were left out. For a dense all-bf16 9B the compressible share is ~100% of weight bytes.

**Failure modes.**
- ALU-bound decode at M = 64 on the 5090. Mitigation: switch codecs by batch size; keep raw weights in host memory as an alternative.
- KV exponents spread wider than weight exponents (RoPE'd keys with large norms). Mitigation: per-head-dim bases.
- Kernel complexity in the attention path, which is the hardest kernel to keep at roofline.

### M4. An out-of-order speculative core that does not change a single output bit

**Idea.** A CPU speculates aggressively but never makes speculation architecturally visible. Results retire in order, stores sit in a store buffer until commit, and mispredictions cost only time. Our version has four parts:
- predictors (tournament of MTP and suffix automaton, #13);
- a confidence-gated issue width (#14);
- tile-shaped verification (#16);
- a store buffer for GDN state with **recurrent-order replay**, plus a deterministic sampler (#17).

Together these make speculative decoding **bit-identical** to plain decoding, for greedy decoding *and* for seeded sampling.

**Design.**
1. **Verify.** For k drafts per session, run all layers on k+1 rows.
   - GEMMs are batch-invariant (#18), so each row's bits equal what single-token decode would produce.
   - Attention uses fixed 2k-token morsels and a fixed merge order. Rows are packed GQA-group × (k+1) = 16 (9B, k = 3).
   - GDN: one kernel per (layer, head) loads S (64 KB fp32) into registers and shared memory once. It steps the *same device function* as the decode kernel k+1 times and emits the k+1 outputs. **It does not write S back.** It writes the per-token (k, v, β, g) and conv inputs to a small ring: about 1 KB per token per head.
2. **Accept.** Sampling at position t: `x_t = argmax_i(logit_i/T + Gumbel(Philox(seed, session, t, i)))`, applied after the reference's top-k/top-p masking. The noise depends only on (seed, t), so the sample is a pure function of logits_t. Accept draft d_t iff d_t == x_t. The acceptance probability is p(d_t), the same as standard rejection sampling with a point-mass draft. The first mismatch position yields x_t as the bonus token.
3. **Commit (store-buffer drain).** The *next* step's GDN kernel loads S, replays the m accepted updates from the ring in recurrent order (the same function, so the same bits as if decoded one by one), then verifies the new drafts.
   - Per step, S is read once and written once, whatever k is.
   - vLLM writes a state copy per draft token (survey §b). ReplaySSM rebuilds with a chunked solve, which is L1, not L0.
   - The conv state works the same way: keep the last 3 committed inputs plus pending ones.
4. **Choose k** per session per step: `k = argmax E[accepted(k)] / cost(k)`. Cost comes from the roofline model and is flat up to the ridge (F7). Accepted(k) comes from per-context chooser confidence. k ∈ {0, 1, 3, 7} so that GQA × (k+1) tiles cleanly.
5. **Drafters:**
   - MTP with a 4k sliding-window KV and FP8 weights (#15; lossy is fine on the draft path).
   - A per-session suffix automaton over prompt plus generated tokens, updated on the CPU as tokens stream (O(1) amortized per token). It is shared across sessions only for the global shared prefix.
   - Chooser: 2-bit counters indexed by (section type: think / content / tool-args / code-fence, last-hit-source).

**Expected gain** *(derived)*.
- a = 2.0 → 2.6 assumed with both drafters. To be measured; SuffixDecoding reports much higher for agentic tool-arg and code-heavy traces.
- At fixed ITL, decode throughput scales with a.
- In the composite model this is the single largest lever on Spark: 64k agents 7.0 → 10.9, and at 32k 16.3 → 23.3.
- GDN state traffic per committed token falls by a (from 2 × 50 MB per token to 2 × 50 MB / a). That matters at short context, where the profile shows GDN decode at 39% of decode time.

**Cheap experiment.**
- (a) **CPU only:** on the M1 traces, simulate the suffix-automaton drafter on generated outputs. Measure expected accepted length per section type. **Kill the suffix half** if mean a_suffix < 1.3 on tool-args and code sections.
- (b) **Exactness gate:** greedy and Philox-seeded runs on 1000 prompts × 512 tokens. Speculative versus k = 0 must give **identical token streams and identical logits bits**. Any difference is a bug.
- (c) **MTP acceptance** on the B200 with the existing engine.

**Failure modes.**
- Thinking text is novel, so a there is MTP-only (~2). Spec helps less in long thinking spans.
- At large batch on the 5090, compute-bound verification forces k → 0. That is handled by #14, but the gain shrinks at 32k.
- Batch-invariant GEMM cost (#18). If it is over 15% on decode, keep L0 as a test mode and ship L1.
- The Philox/Gumbel sampler is not bit-compatible with vLLM's sampler, though it is the same distribution. Reference tests must compare distributions, or use greedy.

### M5. A control plane from queueing, real-time and OS theory: SRPT with aging, frame-paced steps, banker's admission

**Idea.** p99 and p999 are queueing phenomena, so use the policies that are provably good for the job-size structure we actually have:
- **Prefill sizes are known on arrival** (uncached FLOPs after radix matching) and vary about 100×. SRPT is optimal for mean response time with known sizes. Aging bounds the worst case (Scully's SOAP framework gives the analysis).
- **Decode has deadlines** (ITL SLO), so use EDF frame pacing.
- **Memory over-commit causes thrashing** (recompute storms). Use Denning load control: admit only when the working set fits. Use banker's-style reservation for growth.

**Design.**
- **Job classes:** (1) decode steps; (2) prefill chunks of *foreground* turns; (3) runahead and canonicalization (M1); (4) cold sessions above X tokens. Classes 3 and 4 are admitted only into slack.
- **Each engine step is a frame:**
  1. budget = min over decoding sessions of (deadline − now), capped at the ITL target;
  2. decode cost comes from the cost model (#24);
  3. the remaining budget is filled with prefill chunks in SRPT order on remaining FLOPs, with priority = remaining_FLOPs / (1 + wait/τ);
  4. chunks are multiples of 64 tokens (canonical grid).
  - With M2, "fill the budget" means fill the prefill SM partition, not serialize.
- **Admission:** a session joins the decode set iff resident KV + reservation (min(max_tokens, predicted output p95) × 36 KiB) fits. Otherwise it waits in the host tier, which on the 5090 is cheap (F4).
  - Under memory pressure the action is **pause-and-swap** (≈40 ms) of the session with the most slack. Never preempt by recompute.
  - Hard overload (predicted TTFT > SLO for new sessions) → return 429 with Retry-After for *new* sessions only. Sessions with sunk cache investment keep priority (program-level fairness, as in Autellix and Continuum).
- **Fairness across tenants:** stride scheduling (Waldspurger) on GPU-ms charged per session, so one 128k agent cannot starve ten 16k agents.

**Expected gain** *(derived from M/G/1)*.
- With ρ = 0.7 and service times that are mostly 0.2–0.5 s plus rare 5–10 s elephants (C² ≈ 3), FCFS gives every turn a mean wait of about 4.7 E[S] (≈1.5–2 s) and a p99 above 8 s.
- Under SRPT, ordinary tool turns wait about ρ_small/(1 − ρ_small) of their own size: under 0.1 s.
- **p99 TTFT for ordinary turns 5–10× better. Elephants slow down by under 1.5×** (bounded by aging).
- ITL p99 drops from 82 ms (m3b today) toward ≈1.5× p50.

**Cheap experiment.**
- A discrete-event simulator (≈300 lines of Rust or Python) driven by the M1 traces, with kernel costs from the existing bench/profile. Compare FCFS (vLLM-like) with SRPT+aging+frames.
- **Kill** if p99 TTFT for non-elephant turns improves by < 1.5× at ρ = 0.6–0.8.
- Then implement frame pacing alone in the engine (S effort) and re-run the c=32 bench: ITL p99 should drop to ≤ 2× p50.

**Failure modes.**
- Size misestimates for decode length. Only prefill uses SRPT; decode uses deadlines.
- Pathological aging parameters. τ must be tuned on traces.
- Harness timeouts on elephant turns. Those are made rarer by M1, which moves most of them to idle time.

---

## 3. Composite design: "sessions are processes, VRAM is a cache, turns are transactions"

### 3.1 Architecture

```
                 ┌──────────────────────── control plane (CPU) ─────────────────────────┐
 HTTP / SSE ───▶ │ tokenizer + template ─▶ radix match ─▶ SRPT+aging queue ─▶ frame planner │
 (+ session      │      ▲ runahead predictor (M1)   ▲ KLL tool-time sketches (#22)        │
  hints, #21)    │      │ harness-policy counters   │ cost model: roofline + tree (#24)   │
                 └──────┼───────────────────────────┼──────────────────────────────────────┘
                        │                           │ per frame: decode set + prefill chunks + idle work
 ┌──────────── GPU (one device, two SM partitions, M2) ─────────────────────────────────────┐
 │ decode partition (S_d SMs, hi-prio, CUDA graph)   │ prefill partition (S_p SMs, lo-prio)  │
 │  • PFOR-weight GEMV (M3)                          │  • layer-ahead weight decompress (M3) │
 │  • morsel split-KV attention over PFOR KV (#25,M3)│  • canonical 64-grid GDN chunks        │
 │  • cooperative scan of shared prefix (#10)        │  • hd256 flash attention, PFOR KV read │
 │  • GDN store-buffer verify/commit (M4)            │  • runahead branches (M1, idle class)  │
 │  • Philox-Gumbel sampler, tournament drafts (M4)  │  • layer-streamed swap-in (#4)         │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
 Memory hierarchy (all lossless; pages immutable once full; write-through):
   L0  VRAM raw memtable pages (≤64 tok/session/layer) + active GDN states (fp32)
   L1  VRAM PFOR-12 KV pages (decoding sessions only on 5090; everything on Spark)
   L2  pinned host: ANS-exponent pages + GDN checkpoints {junction, episode-start, gen-start, gen-end}
   L3  NVMe (sessions idle > minutes)
   Eviction: GreedyDual-Size × predicted-next-use (#2); admission: banker's reservation (M5)
   Embedding table: host-resident, zero-copy gather (#9). Vision tower: not loaded.
```

### 3.2 One turn, 5090, 9B, a 64k session, 3k-token tool result

1. **Turn ends with a tool call.** The gen-end checkpoint (D2D copy of 50 MB, ≈70 µs) and the last partial KV page are written through to host.
   - The session's GPU pages are now clean, so they are evicted for free if the space is needed.
   - The runahead predictor queues "re-rendered assistant delta + `<tool_response>\n` wrapper" as idle work on a CoW branch.
2. **Tool runs (≈5 s).** If any idle frame exists, the runahead branch is prefilled (≈50–300 tokens, ≈10–60 ms of GPU).
   - At p10 of the KLL-predicted return time, layers 0–2 GDN state is staged on the GPU.
3. **Request arrives.** Tokenize, then radix-match. The match ends at the runahead branch tip; otherwise at the gen-start checkpoint plus up to 63 replay tokens.
   - The remaining prefill is the tool payload only: ≈3k tokens.
   - SRPT places it ahead of any cold 60k session. Attention-layer KV streams in from host one layer ahead (F4: hidden).
   - TTFT ≈ queue + 0.46 s × co-run slowdown (≤1.15).
4. **Decode.** Banker's check, then the session joins the decode set. k ∈ {3, 7} by confidence and roofline.
   - ITL ≈ 6–7 ms per token at ~9 decoding sessions (model). Output streams are bit-identical to k = 0.
5. **New user message after a final answer.** The thinking-stripped history was already re-prefilled during the human's think time, so TTFT is only the user message.

### 3.3 VRAM budget (5090, 9B)

| | Baseline (vLLM-like) | Composite |
|---|---|---|
| Decode-streamed weights | 15.9 GB | 12.1 GB (PFOR, M3) |
| Embedding table | 2.0 GB | 0 (host, #9) |
| MTP | 0.5 GB | 0.25 GB (FP8 draft path, #15) |
| Workspace / context | ≈3 GB | ≈2 GB |
| **KV + state pool** | **≈9.4 GB** | **≈16.6 GB** |
| Bytes per 64k session | 2.47 GB (KV + MTP KV + state) | 1.70 GB (PFOR KV, windowed MTP) |
| Resident 64k sessions | ≈4 | ≈10 (decode set only; idle sessions live in L2) |

### 3.4 Projected concurrent agents (model output, not measured)

Assumptions:
- Qwen3.5-9B, 3k new tokens per turn, 800 output tokens, 5 s tools, 15k shared prefix.
- ITL target 30 ms (5090) or 50 ms (Spark).
- Bandwidth 85% and compute 70–75% achieved; co-run recovers 80% of decode's idle compute.
- Baseline = "vLLM-like": perfect prefix hits, MTP with a = 2.0, no compression, no co-run, no shared-prefix scan.
- The baseline is *generous*: it ignores F5 re-prefills and cache-miss storms.

Model: `agents = cycle_time / GPU_seconds_per_turn`, with the decode batch limited by the ITL target and by resident capacity.

| Stack (cumulative) | 5090 32k | 5090 64k | 5090 128k | Spark 32k | Spark 64k | Spark 128k |
|---|---|---|---|---|---|---|
| Baseline | 10.1 | 5.8 | 3.2 | 5.3 | 2.2 | 1.2 |
| + embedding in host, MTP-KV window (#9, #15) | 12.4 | 6.8 | 3.2 | 6.2 | 3.3 | 1.2 |
| + PFOR weights (M3) | 14.4 | 8.7 | 4.4 | 8.9 | 4.3 | 2.2 |
| + PFOR KV (M3) | 16.6 | 10.2 | 5.5 | 11.4 | 6.1 | 3.2 |
| + cooperative shared-prefix scan (#10) | 16.4 | 10.3 | 5.5 | 16.3 | 7.0 | 3.2 |
| + tournament drafts, a = 2.6 (M4) | 16.2 | 10.5 | 5.8 | 23.3 | 10.9 | 5.1 |
| + prefill/decode co-run (M2) | **17.1** | **13.8** | **8.8** | **35.2** | **14.0** | **5.9** |
| Ratio vs baseline | 1.7× | 2.4× | 2.75× | 6.6× | 6.4× | 4.9× |

How to read this:
- **On the 5090**, capacity (M3, #9) and co-run (M2) matter most. At 32k the 5090 becomes prefill-compute-bound, and only W1 (lossy fp16 accumulate) would move it further.
- **On Spark**, every lossless KV-byte multiplier compounds, because the ITL target caps the decode batch (F1).
- **27B on Spark** follows the Spark pattern at about half the agent counts (2× KV per token, 2.2× FLOPs). 27B on the 5090 is not viable for long context: the FP8 weights leave under 4 GB.
- M1 and M5 do not show up in these averages. They act on p99 TTFT (5–10× lower for ordinary turns) and on the F5 tax (5–11 s removed from user-message turns).

### 3.5 Build order (each step is useful alone)

1. **M5 frame pacing + SRPT** (S): fixes today's ITL p99 of 82 ms. **#9 embedding in host** (S): +2 GB.
2. **Write-through host tier + layer-streamed resume + semantic checkpoints** (survey item 1 + #2, #4). Then **M1 trace study**, then M1 runahead.
3. **M4 store-buffer GDN verify + MTP**, then the suffix drafter. The exactness gate (identical streams) goes into CI with step 3.
4. **M3**: KV exponent measurement (1 h), then the PFOR decode GEMV, then PFOR KV in attention.
5. **M2** once our own prefill GEMM is close to cuBLASLt; the co-run microbench decides.
6. Cooperative scan (#10), morsel attention (#25), megakernel (#26) as kernel work allows.

---

## 4. Lossy options, kept separate

These are for an opt-in "fast" profile behind evals, never the default.

- **W1, fp16 accumulate with per-k16 fp32 promotion.** ≈1.8× prefill on the 5090 *only*, the biggest single lever for prefill-bound 5090 agents at ≤32k.
  - Inputs are exact after power-of-two prescaling. Each partial sum of 16 products keeps 11 mantissa bits instead of 24.
  - Expect about 10–25% of bf16 outputs to flip by 1 ulp. That is above cross-GPU noise (to measure).
  - Gate it on greedy agreement and logprob KL over long agent traces against the bf16 path.
- **FP8 KV:** 2× KV bytes, which is decisive on Spark at 128k. **NVFP4 KV:** ≈3.5×. See survey items 7 and 10.
- **Anything on the draft path** (#15) is free, because drafts are checked.

---

## 5. Experiments, cheapest first

| # | Experiment | Cost | Decides |
|---|---|---|---|
| E0 | Weight exponent stats (done: F6) | done | M3 weights viable: 1.32× at 4-bit FOR, 0.01% escapes |
| E1 | KV exponent dump at 64k on the B200 + Appendix A script | 1 h | M3 KV half |
| E2 | Agent-trace capture + offline replay: divergence points, runahead hit rate, F5 tax, suffix-draft acceptance, tool-time sketches, shared-prefix length | 1–2 days, CPU | M1, M4 drafter, M5 simulator inputs |
| E3 | Co-run microbench (decode graph + SM-capped GEMM) | 1 day | M2 |
| E4 | PFOR decode GEMV vs cuBLASLt, M ≤ 64, on Spark and the 5090 | 2–3 days | M3 weights |
| E5 | Spec vs k = 0 identical-stream gate (greedy + Philox) | 1 day after M4 | M4 exactness |
| E6 | Discrete-event scheduler sim on E2 traces | 2 days | M5 |
| E7 | Layer-streamed resume TTFT at 16k/64k/128k with 300/3k/8k new tokens | 1 day on a 5090 | #4 / F4 |

---

## Appendix A: exponent statistics script (used for F6)

It range-fetches tensors straight from the HF safetensors shards, so no full download is needed:

```python
# uv run --with numpy python -I expo.py <shard> <name-substrings,comma-separated> <max_elems>
import sys, json, struct, urllib.request, numpy as np
BASE = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/"
def rng(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "research-script"})
    return urllib.request.urlopen(r).read()
def H(c):
    p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())
shard, pats, nmax = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
url = BASE + shard; n = struct.unpack("<Q", rng(url, 0, 7))[0]
hdr = json.loads(rng(url, 8, 7 + n)); off = 8 + n
for k, v in sorted(hdr.items()):
    if k == "__metadata__" or v["dtype"] != "BF16" or not any(p in k for p in pats): continue
    a, b = v["data_offsets"]; raw = rng(url, off + a, off + a + min(b - a, 2 * nmax) - 1)
    u = np.frombuffer(raw, np.uint16); e = ((u >> 7) & 0xFF).astype(np.int64)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.int64)
    g = e[: len(e) // 128 * 128].reshape(-1, 128); d = g.max(1, keepdims=True) - g
    print(k, f"H(exp)={H(np.bincount(e, minlength=256)):.2f} H(sm)={H(np.bincount(sm, minlength=256)):.2f}",
          f"esc4={(d > 14).mean():.5f} esc3={(d > 6).mean():.4f}")
```

Results on Qwen3.5-9B (≈6–8M weights per tensor, 17 tensors):
- H(exp) = 2.55–2.67 bits; H(sign+mantissa) = 7.97 bits.
- 4-bit FOR/128 escapes: 0.009–0.020%. 3-bit escapes: 2.5–4.0%.
- Top-15 exponents cover 99.96–99.99% of weights.
- No tensor type is an outlier: LM head, embedding, MLP layers 0/31, GDN in_proj_qkv/z/out_proj, attention q/k/v/o and MTP MLP all look the same.

For KV, run the same statistics on a raw dump of the K/V pages (dtype bf16, row = one token × one KV head × 256).

---

## Sources

Classic systems literature (by field):
- **Databases.**
  - Zukowski, Héman, Nes, Boncz, *Super-Scalar RAM-CPU Cache Compression* (PFOR), ICDE 2006.
  - Zukowski et al., *Cooperative Scans*, VLDB 2007.
  - O'Neil et al., *LRU-K*, SIGMOD 1993; Megiddo & Modha, *ARC*, FAST 2003; Cao & Irani, *GreedyDual-Size*, USITS 1997.
  - Mohan et al., *ARIES*, TODS 1992; O'Neil et al., *The LSM-tree*, Acta Informatica 1996.
  - Leis et al., *Morsel-driven parallelism*, SIGMOD 2014; Neumann, *Efficiently compiling efficient query plans*, VLDB 2011.
  - Pelkonen et al., *Gorilla*, VLDB 2015.
- **Operating systems.** Denning, *The working set model*, CACM 1968; Dijkstra, banker's algorithm, 1965; Waldspurger & Weihl, *Lottery/stride scheduling*, OSDI 1994.
- **CPU microarchitecture.**
  - Mutlu et al., *Runahead execution*, HPCA 2003.
  - Lipasti et al., *Value locality and load value prediction*, ASPLOS 1996.
  - McFarling, *Combining branch predictors*, DEC WRL TN-36, 1993.
  - Manne, Klauser, Grunwald, *Pipeline gating*, ISCA 1998.
  - Jain & Lin, *Back to the future: Belady's algorithm (Hawkeye)*, ISCA 2016.
- **HPC.** Salmon et al., *Parallel random numbers: as easy as 1, 2, 3* (Philox), SC 2011; Williams, Waterman, Patterson, *Roofline*, CACM 2009; Ozaki et al., error-free matrix multiplication, 2012.
- **Queueing and scheduling.** Schrage, SRPT optimality, 1968; Scully & Harchol-Balter, *SOAP*, SIGMETRICS 2018.
- **Sketches and boosting.** Karnin, Lang, Liberty, *KLL sketch*, FOCS 2016; Chen & Guestrin, *XGBoost* (weighted quantile sketch, cache-aware blocks), KDD 2016; Prokhorenkova et al., *CatBoost* (oblivious trees), NeurIPS 2018.
- **Networking.** Kermani & Kleinrock, *Virtual cut-through*, 1979.

LLM serving prior art checked for novelty:
- **Weight and KV compression.**
  - [ZipServ (arXiv 2603.17435)](https://arxiv.org/html/2603.17435v1): fused lossless bf16 GEMM; RTX 5090 1.34–1.87× at batch 32; 11.3 bits/weight.
  - [SplitZip (arXiv 2605.01708)](https://arxiv.org/html/2605.01708): 4-bit exponent codebook for KV transfer, ≈1.3×.
  - [ExANS (Show HN)](https://hn.svelte.dev/item/49185576).
  - DFloat11 (arXiv 2504.11651).
  - [DGPP bf12 on GB10](https://forums.developer.nvidia.com/t/dgpp-a-gb10-optimized-c-cuda-inference-engine/383406).
- **Hybrid caching and agent scheduling.**
  - [Marconi (arXiv 2411.19379)](https://arxiv.org/abs/2411.19379).
  - [Continuum (arXiv 2511.02230)](https://arxiv.org/abs/2511.02230).
  - CachedAttention / AttentionStore (ATC 2024, arXiv 2403.19708).
  - [vLLM tiered KV offloading](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading): demand-driven, LRU/ARC, mamba state supported, no compression, no agent prefetch.
  - [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).
- **Speculative decoding.**
  - [SuffixDecoding (arXiv 2411.04975)](https://arxiv.org/abs/2411.04975).
  - [ReplaySSM (Tri Dao, 2026)](https://tridao.me/blog/2026/replayssm/).
  - [SpecLA (arXiv 2607.16673)](https://arxiv.org/abs/2607.16673).
- **Co-scheduling and batching.** NanoFlow (OSDI 2025, arXiv 2408.12757); POD-Attention (ASPLOS 2025, arXiv 2410.18038); Sarathi-Serve (OSDI 2024); Hydragen (arXiv 2402.05099).
- **Determinism.** Thinking Machines, *Defeating nondeterminism in LLM inference* (2025).
- **Hardware figures.**
  - [RTX 5090 peak BF16 (NVIDIA forum)](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543).
  - [DGX Spark review, StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops) (≈100 TFLOPS bf16 measured).
- **Model files.** Model configs, weight index and `chat_template.jinja` from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B).
