"""Multi-turn agentic load generator for any OpenAI-compatible /v1/completions server.

N agents run concurrently. Each agent's session starts with a shared system prompt (identical
token ids for every agent: tool definitions, instructions) plus its own task message, then
loops: request a completion of the whole history (exact token ids), stream back G generated
tokens, append them plus a tool result of random length, wait a random "tool latency", repeat.
When the history would exceed --max-ctx the agent starts a fresh session (same system
prompt, new task). Prompts are random token ids; content does not matter for speed.

Every request carries the full history as token ids, so a server is free to reuse whatever
it cached from the previous turn. Metrics are taken after --warmup seconds, over --duration
seconds:
  per-turn TTFT (p50/p90/p99/p99.9), split by context-length bucket
  inter-token latency and time per output token (p50/p90/p99/p99.9)
  turns/min, generated tokens/s, new prompt tokens/s, context length reached

    python tools/agentic_bench.py --url http://localhost:8000 --model Qwen/Qwen3.5-9B \\
        --agents 16 --system-len 4096 --task-len 1024 --tool-len 512:4096 --gen-len 256 \\
        --think-ms 500:3000 --max-ctx 32768 --warmup 60 --duration 300 --out results.json
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp


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


def rand_ids(rng, n, lo=1000, hi=150000):
    return [rng.randrange(lo, hi) for _ in range(n)]


def span(s):
    a, _, b = s.partition(":")
    return (int(a), int(b or a))


class Stats:
    def __init__(self):
        self.turns = []  # dicts: t_start, ctx, ttft, itl list, tpot, gen, new_prompt
        self.errors = 0


async def one_turn(session, args, history, new_tokens, stats, t0_measure):
    body = {
        "model": args.model,
        "prompt": history,
        "max_tokens": args.gen_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    t_send = time.perf_counter()
    out_ids, times = [], []
    async with session.post(f"{args.url}/v1/completions", json=body) as resp:
        if resp.status != 200:
            stats.errors += 1
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        buf = b""
        async for chunk in resp.content.iter_any():
            now = time.perf_counter()
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    msg = json.loads(data)
                    for ch in msg.get("choices", []):
                        ids = ch.get("token_ids") or []
                        if ids:
                            out_ids.extend(ids)
                            times.extend([now] * len(ids))
    if len(out_ids) != args.gen_len:
        stats.errors += 1
        raise RuntimeError(f"expected {args.gen_len} tokens, got {len(out_ids)}")
    if t_send >= t0_measure:
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        stats.turns.append(
            {
                "t": t_send,
                "ctx": len(history),
                "new_prompt": new_tokens,
                "ttft": times[0] - t_send,
                "itl": gaps,
                "tpot": (times[-1] - times[0]) / max(1, len(times) - 1),
                "gen": len(out_ids),
                "e2e": times[-1] - t_send,
            }
        )
    return out_ids


async def agent(i, args, system_ids, stats, t0_measure, t_end):
    rng = random.Random(args.seed * 1000 + i)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Stagger starts so agents do not move in lockstep.
        await asyncio.sleep(rng.uniform(0, args.stagger_s))
        while time.perf_counter() < t_end:
            history = system_ids + rand_ids(rng, args.task_len)
            new = len(history)
            while time.perf_counter() < t_end:
                out = await one_turn(session, args, history, new, stats, t0_measure)
                tool = rand_ids(rng, rng.randint(*args.tool_len))
                if len(history) + len(out) + len(tool) + args.gen_len > args.max_ctx:
                    break  # session full: start a new one
                history = history + out + tool
                new = len(out) + len(tool)
                await asyncio.sleep(rng.uniform(*args.think_ms) / 1000)


async def main_async(args):
    rng = random.Random(args.seed)
    system_ids = rand_ids(rng, args.system_len)
    stats = Stats()
    t_start = time.perf_counter()
    t0_measure = t_start + args.warmup
    t_end = t0_measure + args.duration
    tasks = [asyncio.create_task(agent(i, args, system_ids, stats, t0_measure, t_end)) for i in range(args.agents)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errs = [r for r in results if isinstance(r, Exception)]
    turns = [t for t in stats.turns if t["t"] < t_end]
    wall = args.duration
    buckets = {}
    for t in turns:
        b = f"{(t['ctx'] // 8192) * 8}k-{(t['ctx'] // 8192 + 1) * 8}k"
        buckets.setdefault(b, []).append(t["ttft"] * 1e3)
    res = {
        "engine": args.engine,
        "model": args.model,
        "agents": args.agents,
        "config": {k: v for k, v in vars(args).items() if k not in ("url",)},
        "turns": len(turns),
        "turns_per_min": len(turns) / wall * 60,
        "gen_tok_s": sum(t["gen"] for t in turns) / wall,
        "new_prompt_tok_s": sum(t["new_prompt"] for t in turns) / wall,
        "mean_ctx": sum(t["ctx"] for t in turns) / max(1, len(turns)),
        "max_ctx_seen": max((t["ctx"] for t in turns), default=0),
        "ttft_ms": summary([t["ttft"] * 1e3 for t in turns]),
        "ttft_ms_by_ctx": {b: summary(v) for b, v in sorted(buckets.items())},
        "itl_ms": summary([g * 1e3 for t in turns for g in t["itl"]]),
        "tpot_ms": summary([t["tpot"] * 1e3 for t in turns]),
        "e2e_ms": summary([t["e2e"] * 1e3 for t in turns]),
        "errors": stats.errors + len(errs),
        "error_samples": [repr(e)[:300] for e in errs[:3]],
    }
    print(
        f"{args.engine} agents={args.agents} turns/min={res['turns_per_min']:.1f} gen_tok/s={res['gen_tok_s']:.0f} "
        f"mean_ctx={res['mean_ctx']:.0f} ttft p50/p99/p99.9={res['ttft_ms']['p50']:.0f}/{res['ttft_ms']['p99']:.0f}/{res['ttft_ms']['p99.9']:.0f}ms "
        f"itl p50/p99/p99.9={res['itl_ms']['p50']:.1f}/{res['itl_ms']['p99']:.1f}/{res['itl_ms']['p99.9']:.1f}ms errors={res['errors']}"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default="unknown", help="label stored in the results")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--system-len", type=int, default=4096)
    p.add_argument("--task-len", type=int, default=1024)
    p.add_argument("--tool-len", type=span, default=(512, 4096))
    p.add_argument("--gen-len", type=int, default=256)
    p.add_argument("--think-ms", type=span, default=(500, 3000))
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--stagger-s", type=float, default=10.0)
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
