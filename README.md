# aqwenator

An inference server written for one model family: Qwen3.5 hybrid (Gated DeltaNet + attention)
models, on NVIDIA GPUs. Rust host, kernels in Rust compiled to PTX with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), GEMMs through cuBLASLt. It targets
agentic workloads (many long multi-turn sessions) on a single GPU, down to a DGX Spark.

Outputs are checked token for token against `transformers` (`aq-check`).

## Numbers

Qwen3.5-9B, replay of recorded OpenHands trajectories, 10 minutes per run, DGX Spark:

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) |
|---|---|---|---|---|
| 16 | aqwenator | 91.9 | 229 | 483 / 3864 |
| 16 | vLLM 0.33, MTP speculation | 57.4 | 122 | 1022 / 2909 |
| 8 | aqwenator | 57.4 | 157 | |
| 8 | vLLM 0.33, MTP speculation | 44.6 | 102 | |

Cold start, 9B on the Spark: first token 3.2 s after launch, page cache warm or cold.

### 8-bit weights (opt-in)

`AQ_QUANT=fp8|int8` quantizes the layer weights at load. Same replay, same machine, 16 agents:

| Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | Weights + 4k-token pool | KL vs transformers (nats) |
|---|---|---|---|---|---|
| bf16 (default, exact) | 91.5 | 228 | 491 / 4550 | 18.1 GiB | 0.002 |
| `fp8` (E4M3 weights and activations) | 113.7 | 295 | 374 / 3374 | 11.7 GiB | 0.028 |
| `int8` (block-scaled weights, bf16 activations) | 71.6 | 182 | 619 / 3946 | 12.2 GiB | 0.003 |

vLLM 0.33 serving a published FP8 checkpoint of the same model (MTP speculation) does 51.9
turns/min and 131 tok/s on this replay (TTFT 1052 / 3739 ms).

`fp8` is the fast one and changes about 3% of greedy tokens; `int8` is near the bf16 noise
floor and trades speed for memory.

Method, more machines and the history are in [docs/HANDOFF.md](docs/HANDOFF.md).

## Run

The model must be in the local Hugging Face cache (`hf download Qwen/Qwen3.5-9B`), or pass a
directory. Needs the NVIDIA container toolkit and a driver for CUDA 13.

```sh
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/home/aq/.cache/huggingface:ro \
  ghcr.io/<owner>/aqwenator Qwen/Qwen3.5-9B --kv-gb 24
```

```sh
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

```sh
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Name the capital of France."}]}'
```

See "OpenAI-compatible serving" below for the fields, the flags and the limits.

## OpenAI-compatible serving

`aq-serve` speaks the part of the OpenAI HTTP API that clients written for vLLM or for
OpenAI use. `scripts/serve_check.py` checks everything in this section against a running
server; `scripts/cache_replay.py` measures the prompt cache.

Endpoints: `/v1/completions` (text or token-id prompts), `/v1/chat/completions` (the model's
chat template), `/v1/models`, `/health` (503 once the engine has stopped; the process then
exits non-zero for a supervisor to restart it), `/metrics` (Prometheus, `aq_*` names;
`--vllm-metrics true` adds vLLM's names for the request gauges and token counters).

Request fields, both endpoints unless noted:

| Field | Behaviour |
|---|---|
| `max_tokens`, `max_completion_tokens` (chat) | Default: the rest of the context. |
| `temperature`, `top_p`, `top_k`, `seed` | Default: the model's `generation_config.json` values, else temperature 1 (`--default-temperature T` overrides; 0 = greedy). Temperature 0 is the engine's greedy path. A seeded request repeats as far as its logits do (they can move by a rounding step with what else shares the batch). |
| `stop`, `stop_token_ids`, `ignore_eos`, `min_tokens` | The stop string or token is not part of the output; `stop_reason` names it. |
| `logit_bias`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Applied to the logits before sampling. |
| `logprobs` + `top_logprobs` (chat), `logprobs` N (completions) | Up to 20 alternatives. Log-probabilities of the model after bias and penalties, before temperature, `top_k`, `top_p` and grammar masks. |
| `stream`, `stream_options.include_usage` | Server-sent events, `[DONE]` last. |
| `response_format` | `json_object` and `json_schema`: the output is constrained to the schema (after the reasoning block, if the request thinks). |
| `tools`, `tool_choice`, `parallel_tool_calls` (chat) | `auto`: calls the model writes are parsed into `tool_calls`. `required` or a named function: the output is constrained to a valid call. Either way the wire shape is vLLM's: `tool_calls[].function.arguments` is a JSON string, `finish_reason` is `tool_calls` (`stop` for a named function, as in vLLM). `none`: the output is content as written. |
| `chat_template_kwargs` (chat) | Passed to the template (`enable_thinking`). Reasoning comes back in `reasoning_content` (and `reasoning`). |
| `reasoning_effort` (chat) | `none` turns the template's thinking off, any other level on, unless `chat_template_kwargs.enable_thinking` says. |
| `user`, `store`, `metadata`, `service_tier`, `prompt_cache_key`, `safety_identifier` | Accepted and ignored: they change nothing on a server like this one. `--strict-requests true` makes them (and `reasoning_effort`) a 400. |
| `n` > 1, `best_of` > 1, `echo`, `suffix`, `min_p`, `truncate_prompt_tokens`, `continue_final_message`, `add_generation_prompt` false, image or audio content | 400 that names the field. So is any field the server does not know, and a `model` other than the served one is a 404. Nothing is dropped unseen. |

`usage.prompt_tokens_details.cached_tokens` reports, on both endpoints and in the final
streamed usage chunk, how many prompt tokens the request did not have to compute.

Flags that matter (the full list is at the top of `crates/aq-server/src/main.rs`):

| Flag | Meaning |
|---|---|
| `--host`, `--port`, `--served-model-name` | Listens on 127.0.0.1 unless `--host` (or `AQ_HOST`) says otherwise; the image sets `AQ_HOST=0.0.0.0`. |
| `--max-seqs N`, `--max-running N` | Sequences in one step (sizes the step buffers), and how many requests run at once. |
| `--max-ctx N`, `--max-tokens N` | Context per sequence; prefill tokens per step. |
| `--kv-gb X`, `--kv-reserve N` | Attention-cache pool, and the output tokens a request reserves in it up front (default 4096; 0 = its whole limit). Past its reservation a request takes pages as it generates; if the pool is dry it waits for the next request to end (new requests queue behind it), and only when every running request is out of pages does one end early with `finish_reason` `length` (`aq_requests_cut_total`). |
| `--cache-slots N` | State slots for the prompt cache beyond `--max-seqs` (see below). One slot is the recurrent state only: 49 MiB on Qwen3.5-9B, 19 MiB on 0.8B. |
| `--pin-snapshots`, `--max-snapshots N`, `--evict lru\|mru\|idle`, `--session-cache false` | Cache policy; the defaults are described below. |
| `--default-temperature T`, `--strict-requests true` | See the table above. |

### Prompt cache

The model carries a recurrent state next to its attention cache, and that state exists only
at the positions where it was saved. A request can therefore resume from a saved position
that is a prefix of its prompt, never from the middle of one. Three kinds of entries are
kept, each in one state slot, sharing attention-cache pages:

- the end of a turn: reused by a client that sends back exactly the tokens it received;
- a prompt checkpoint, a few tokens before the end of each prompt: reused by a chat client,
  whose next request renders the conversation again (the previous prompt, then the assistant
  turn as the template writes it);
- a shared-prefix snapshot: when a new prompt shares at least 1024 tokens with another
  sequence (and 1024 more than any snapshot it could fork), the engine saves the state at the
  end of the shared part. Every later request with that prefix forks it. The system prompt
  and tool schemas of an agent fleet are computed twice (the second sighting takes the
  snapshot), then never again; a group of agents that also share a longer prefix gets a
  second, deeper snapshot.

Slots: `--max-seqs` plus `--cache-slots`. A running request holds one, and a second while it
decodes with drafts; what is left holds the cache. When none is free, a new checkpoint or
snapshot takes the slot of a parked entry: first the kind clients have not been coming back
to (turn ends, for chat clients), in a conversation that holds both, then the least recently
used conversation. Snapshots are never evicted for a conversation entry; at most
`--max-snapshots` (4) exist, the least recently forked one is replaced, and
`--pin-snapshots true` keeps them from being evicted when running requests need slots.

What to expect: with fewer cache slots than conversations, every request still reuses the
shared prefix, and the conversations beyond the slots recompute their own turns. With a slot
per conversation, a turn computes only what is new. Measured on Qwen3.5-0.8B (16
conversations of 10 turns over 5.8k shared tokens, 4 at a time, `--max-seqs 8
--max-running 4`, after 8 unrelated requests, `scripts/cache_replay.py --warm 8`):

| | cached share of prompt tokens | time to first token p50 / p99 | wall |
|---|---|---|---|
| before these changes | 38 % | 829 / 1931 ms | 53.0 s |
| now | 87 % | 225 / 653 ms | 15.8 s |
| now, `--cache-slots 24` | 96 % | 109 / 591 ms | 7.2 s |

Limits: a near-duplicate request (same history, another last turn) is computed in full the
first two times and forks a snapshot from the third on; a prefix shorter than 1024 tokens is
never snapshotted; with `AQ_COEXEC` or `AQ_BG_PREFILL` (study modes, off by default) tokens
are picked on the device, so only greedy, unconstrained requests are served, and a request
without `temperature` is greedy there.

## Build

```sh
docker build -t aqwenator .                                         # image
docker build --target binaries --output type=local,dest=bin .       # binaries only
```

Without Docker: CUDA 13 toolkit, clang 21, Rust 1.99.1 and `cargo-oxide` (see
`scripts/setup.sh`), then

```sh
source scripts/env.sh
cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins
```

Binaries: `aq-serve` (server), `aq-check` (correctness against reference data from
`scripts/golden.py`), `aq-bench` (synthetic load), `aq-ceiling` (hardware limits: memory
bandwidth, GEMM rate, and what they allow per step).

## Release

Pushing a tag `vX.Y.Z` builds amd64 and arm64 images, publishes
`ghcr.io/<owner>/aqwenator:X.Y.Z` and `:latest`, and attaches the binaries to a GitHub release.

## License

Apache-2.0, see [LICENSE](LICENSE).
