"""Replay real agent trajectories against any OpenAI-compatible /v1/completions server.

Unlike agentic_bench.py (random token ids, history = previous prompt + the server's own
output), this replays recorded OpenHands sessions from the Hub
(nebius/SWE-rebench-openhands-trajectories: system prompt + tools, issue, then assistant /
tool turns). Each agent walks one trajectory: for every assistant turn it sends the chat-
template rendering of everything before it (as token ids) and asks for as many tokens as the
recorded assistant turn had. So the benchmark has

  - real text: tool definitions, code, diffs and logs, which is what drives speculative
    acceptance and attention behaviour;
  - real turn shapes: prompt growth per turn and output lengths come from the trace;
  - the real cache pattern: the next prompt extends the previous *prompt* with the recorded
    assistant message as the template renders it, not with the tokens the server generated.
    A server that can only resume from "prompt + my own output" misses on every turn.

Step 1, once (downloads and tokenizes; needs `transformers`):
    python scripts/trace_bench.py prepare --model Qwen/Qwen3.5-9B --trajectories 40 --max-ctx 32768 --out traces-9b.json
Step 2 (add `--device <name> --record docs/results.jsonl` to keep the run):
    python scripts/trace_bench.py run --traces traces-9b.json --url http://localhost:8000 --model Qwen/Qwen3.5-9B \\
        --agents 16 --think-ms 500:3000 --warmup 60 --duration 300 --out result.json
Step 3, the published tables:
    python scripts/trace_bench.py report docs/results.jsonl > docs/results.md
"""

import argparse
import asyncio
import datetime
import json
import random
import time
import urllib.parse
import urllib.request

DATASET = "nebius/SWE-rebench-openhands-trajectories"


def fetch_rows(offset, length):
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": length})
    req = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}", headers={"User-Agent": "trace-bench"})
    for attempt in range(5):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(req, timeout=120))["rows"]]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)


def clean(msgs):
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if v is not None}
        for tc in m.get("tool_calls", []):
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    pass
        out.append(m)
    return out


def prepare(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    def render(msgs, tools, gen):
        ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=gen)
        return ids if isinstance(ids, list) else ids["input_ids"]

    sessions, offset, skipped = [], args.offset, 0
    while len(sessions) < args.trajectories:
        rows = fetch_rows(offset, 10)
        if not rows:
            break
        offset += len(rows)
        for r in rows:
            msgs = clean(r["trajectory"])
            turns = []
            try:
                for i, m in enumerate(msgs):
                    if m["role"] != "assistant":
                        continue
                    prompt = render(msgs[:i], r["tools"], True)
                    # Output length of the recorded turn: what rendering it adds to the prompt.
                    n_out = max(1, len(render(msgs[: i + 1], r["tools"], False)) - len(prompt))
                    n_out = min(n_out, args.max_out)
                    if len(prompt) + n_out > args.max_ctx:
                        break
                    turns.append({"prompt": prompt, "max_tokens": n_out})
            except Exception as e:  # a trace the chat template rejects
                skipped += 1
                print(f"skip {r['trajectory_id']}: {type(e).__name__}: {str(e)[:100]}")
                continue
            if len(turns) >= args.min_turns:
                sessions.append({"id": r["trajectory_id"], "turns": turns})
                print(f"{len(sessions):>3} {r['instance_id']:<40} {len(turns):>3} turns, ctx {len(turns[0]['prompt'])} -> {len(turns[-1]['prompt'])}", flush=True)
            if len(sessions) >= args.trajectories:
                break

    # Shape of the workload, and how much of each prompt a prefix cache can reuse.
    new, outs, ext, ext_gen = [], [], 0, 0
    n = 0
    for s in sessions:
        prev = None
        for t in s["turns"]:
            p = t["prompt"]
            outs.append(t["max_tokens"])
            if prev is not None:
                lcp = next((i for i, (a, b) in enumerate(zip(prev, p)) if a != b), min(len(prev), len(p)))
                new.append(len(p) - lcp)
                ext += lcp == len(prev)
                n += 1
            prev = p
    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, round(q * (len(xs) - 1)))]
    stats = {
        "sessions": len(sessions),
        "turns": sum(len(s["turns"]) for s in sessions),
        "skipped": skipped,
        "new_prompt_tokens_per_turn": {"p50": pct(new, 0.5), "p90": pct(new, 0.9), "p99": pct(new, 0.99), "mean": sum(new) / len(new)},
        "output_tokens_per_turn": {"p50": pct(outs, 0.5), "p90": pct(outs, 0.9), "p99": pct(outs, 0.99), "mean": sum(outs) / len(outs)},
        "turns_extending_previous_prompt": ext / max(1, n),
        "system_prefix_tokens_shared_by_all": next(
            (i for i, xs in enumerate(zip(*[s["turns"][0]["prompt"] for s in sessions])) if len(set(xs)) > 1),
            min(len(s["turns"][0]["prompt"]) for s in sessions),
        ),
    }
    print(json.dumps(stats, indent=1))
    json.dump({"model": args.model, "dataset": DATASET, "max_ctx": args.max_ctx, "stats": stats, "sessions": sessions}, open(args.out, "w"))


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q / 100 * (len(xs) - 1)))]


def summary(xs):
    return {k: pct(xs, q) for k, q in (("p50", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9))} | {
        "mean": sum(xs) / len(xs) if xs else float("nan"),
        "n": len(xs),
    }


async def one_turn(http, args, turn, new_tokens, first, turns_out, errors, t0_measure):
    body = {
        "model": args.model,
        "prompt": turn["prompt"],
        "max_tokens": turn["max_tokens"],
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    t_send = time.perf_counter()
    times = []
    cached = None
    async with http.post(f"{args.url}/v1/completions", json=body) as resp:
        if resp.status != 200:
            errors.append(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            return
        buf = b""
        async for chunk in resp.content.iter_any():
            now = time.perf_counter()
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    msg = json.loads(line[5:])
                    if msg.get("usage"):
                        cached = (msg["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                    for ch in msg.get("choices", []):
                        # Token ids where the server streams them (vLLM, aq-serve: `token_ids`;
                        # llama.cpp: `tokens`), else one token per chunk of text.
                        ids = ch.get("token_ids") or ch.get("tokens") or []
                        times.extend([now] * (len(ids) or (1 if ch.get("text") else 0)))
    if not times:
        errors.append("no tokens")
        return
    if t_send >= t0_measure:
        turns_out.append(
            {
                "t": t_send,
                "ctx": len(turn["prompt"]),
                "cached": cached,
                "new_prompt": new_tokens,
                "first": first,
                "ttft": times[0] - t_send,
                # Tokens that arrive together (speculation) count as one gap shared between them.
                "itl": [b - a for a, b in zip(times, times[1:])],
                "gen": len(times),
                "e2e": times[-1] - t_send,
            }
        )


async def agent(i, args, sessions, turns_out, errors, t0_measure, t_end):
    import aiohttp

    rng = random.Random(args.seed * 1000 + i)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900)) as http:
        await asyncio.sleep(rng.uniform(0, args.stagger_s))
        k = i
        while time.perf_counter() < t_end:
            s = sessions[k % len(sessions)]
            k += args.agents  # each agent walks its own stride of sessions
            prev = 0
            for turn in s["turns"]:
                if time.perf_counter() >= t_end:
                    break
                await one_turn(http, args, turn, len(turn["prompt"]) - prev, prev == 0, turns_out, errors, t0_measure)
                prev = len(turn["prompt"])
                await asyncio.sleep(rng.uniform(*args.think_ms) / 1000)


async def run(args):
    data = json.load(open(args.traces))
    sessions = data["sessions"]
    turns, errors = [], []
    t0_measure = time.perf_counter() + args.warmup
    total = args.duration * args.repeats
    t_end = t0_measure + total
    await asyncio.gather(*[agent(i, args, sessions, turns, errors, t0_measure, t_end) for i in range(args.agents)])
    turns = [t for t in turns if t["t"] < t_end]
    res = {
        "engine": args.engine,
        "model": args.model,
        "workload": f"{data['dataset']} ({len(sessions)} sessions, max ctx {data['max_ctx']})",
        "agents": args.agents,
        "turns": len(turns),
        "turns_per_min": len(turns) / total * 60,
        # One value per measurement window (--repeats), for the run-to-run spread.
        "turns_per_min_runs": [
            sum(1 for t in turns if int((t["t"] - t0_measure) // args.duration) == k) / args.duration * 60 for k in range(args.repeats)
        ],
        "gen_tok_s": sum(t["gen"] for t in turns) / total,
        "new_prompt_tok_s": sum(t["new_prompt"] for t in turns) / total,
        "mean_ctx": sum(t["ctx"] for t in turns) / max(1, len(turns)),
        # Share of prompt tokens the server reports as served from its cache (None if it does not).
        "cached_share": (
            sum(t["cached"] for t in turns if t["cached"] is not None) / max(1, sum(t["ctx"] for t in turns if t["cached"] is not None))
            if any(t["cached"] is not None for t in turns)
            else None
        ),
        "ttft_ms": summary([t["ttft"] * 1e3 for t in turns]),
        "itl_ms": summary([g * 1e3 for t in turns for g in t["itl"]]),
        "tpot_ms": summary([(t["e2e"] - t["ttft"]) / max(1, t["gen"] - 1) * 1e3 for t in turns if t["gen"] > 1]),
        "e2e_ms": summary([t["e2e"] * 1e3 for t in turns]),
        # One row per turn: [context, new prompt tokens, first turn of its session, TTFT ms,
        # output tokens, end-to-end ms], for comparisons at equal turn size.
        "per_turn": [[t["ctx"], t["new_prompt"], int(t["first"]), round(t["ttft"] * 1e3, 1), t["gen"], round(t["e2e"] * 1e3, 1)] for t in turns],
        "errors": len(errors),
        "error_samples": errors[:3],
    }
    print(
        f"{args.engine} agents={args.agents} turns/min={res['turns_per_min']:.1f} gen_tok/s={res['gen_tok_s']:.0f} mean_ctx={res['mean_ctx']:.0f} "
        f"ttft p50/p99/p99.9={res['ttft_ms']['p50']:.0f}/{res['ttft_ms']['p99']:.0f}/{res['ttft_ms']['p99.9']:.0f}ms "
        f"tpot p50/p99={res['tpot_ms']['p50']:.1f}/{res['tpot_ms']['p99']:.1f}ms errors={res['errors']}"
    )
    res["date"] = datetime.date.today().isoformat()
    res["device"] = args.device
    res["minutes"] = total / 60
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    if args.record:
        record(res, args.record)


def record(res, path):
    """Append a run, without its per-turn rows, to a results file (one JSON object per line)."""
    slim = {k: v for k, v in res.items() if k not in ("per_turn", "error_samples")}
    with open(path, "a") as f:
        f.write(json.dumps(slim) + "\n")


def report(args):
    """Markdown tables from a results file: the latest run of each (model, device, engine, agents)."""
    latest = {}
    for line in open(args.results):
        r = json.loads(line)
        latest[(r["model"], r["device"], r["engine"], r["agents"])] = r
    out = [
        "# Results",
        "",
        "Generated by `scripts/trace_bench.py report` from `docs/results.jsonl`: do not edit.",
        "Every row is one run and carries its date; method and caveats are in",
        "[benchmarks](benchmarks.md). TTFT, TPOT, inter-token and turn latencies are in ms; a range",
        "after turns per minute is the lowest and highest of the run's measurement windows. Cached",
        "is the share of prompt tokens the server reports as served from its cache.",
    ]
    groups = {}
    for (model, device, _, _), r in latest.items():
        groups.setdefault((model, device), []).append(r)
    for (model, device), rows in sorted(groups.items()):
        out += ["", f"## {model} on {device}", ""]
        out.append("| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 | TPOT p50 | Inter-token p99 | Turn p50 / p99 | Cached | Min | Date |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["agents"], -r["turns_per_min"])):
            f = lambda k, q: f"{r[k][q]:.0f}"
            runs = r.get("turns_per_min_runs", [])
            spread = f" ({min(runs):.1f}–{max(runs):.1f})" if len(runs) > 1 else ""
            cached = "" if r.get("cached_share") is None else f"{100 * r['cached_share']:.0f}%"
            out.append(
                f"| {r['agents']} | {r['engine']} | {r['turns_per_min']:.1f}{spread} | {r['gen_tok_s']:.0f} | {f('ttft_ms', 'p50')} / {f('ttft_ms', 'p99')} "
                f"| {r['tpot_ms']['p50']:.1f} | {f('itl_ms', 'p99')} | {f('e2e_ms', 'p50')} / {f('e2e_ms', 'p99')} | {cached} | {r['minutes']:g} | {r['date']} |"
            )
        # Capacity: the largest recorded load at which an engine meets the latency target.
        ok = lambda r: r["ttft_ms"]["p99"] <= args.ttft_p99 and r["tpot_ms"]["p99"] <= args.tpot_p99
        cap = {}
        for r in rows:
            best, top = cap.get(r["engine"], (0, 0))
            cap[r["engine"]] = (max(best, r["agents"]) if ok(r) else best, max(top, r["agents"]))
        out += ["", f"Largest recorded load with TTFT p99 ≤ {args.ttft_p99:g} ms and TPOT p99 ≤ {args.tpot_p99:g} ms:", ""]
        for engine, (best, top) in sorted(cap.items(), key=lambda kv: -kv[1][0]):
            out.append(f"- {engine}: " + ("none" if best == 0 else f"{best} agent{'s' * (best > 1)}" + (" (the largest recorded)" if best == top else "")))
    print("\n".join(out))


def span(s):
    a, _, b = s.partition(":")
    return (int(a), int(b or a))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--trajectories", type=int, default=40)
    a.add_argument("--offset", type=int, default=0)
    a.add_argument("--max-ctx", type=int, default=32768)
    a.add_argument("--max-out", type=int, default=1024)
    a.add_argument("--min-turns", type=int, default=8)
    b = sub.add_parser("run")
    b.add_argument("--traces", required=True)
    b.add_argument("--url", default="http://localhost:8000")
    b.add_argument("--model", required=True)
    b.add_argument("--engine", default="unknown")
    b.add_argument("--agents", type=int, default=16)
    b.add_argument("--think-ms", type=span, default=(500, 3000))
    b.add_argument("--stagger-s", type=float, default=10.0)
    b.add_argument("--warmup", type=float, default=60.0)
    b.add_argument("--duration", type=float, default=300.0)
    b.add_argument("--repeats", type=int, default=1, help="measurement windows of --duration each, run back to back")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out")
    b.add_argument("--device", default="unknown", help="label of the machine, for the results file")
    b.add_argument("--record", help="append the run to this results file (docs/results.jsonl)")
    c = sub.add_parser("report")
    c.add_argument("results")
    c.add_argument("--ttft-p99", type=float, default=1000.0, help="latency target for the capacity lines, ms")
    c.add_argument("--tpot-p99", type=float, default=50.0)
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "report":
        report(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
