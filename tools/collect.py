"""Collect benchmark JSONs from ~/aq/results into flat files for the progress page.

    python -I tools/collect.py ~/aq/results > bench.json
    python -I tools/collect.py ~/aq/results agentic > agentic.json

Rows: engine, model, input_len, output_len, concurrency, output_tok_s, ttft_p50_ms,
ttft_p99_ms, tpot_p50_ms, itl_p99_ms, run (file stem), gpu, date (file mtime, UTC).
"""

import datetime
import glob
import json
import os
import re
import sys


def num(x):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), 3)


def main():
    root = os.path.expanduser(sys.argv[1])
    rows = []
    # vLLM: one JSON per (shape, concurrency) under baseline/<tag>/
    for f in glob.glob(f"{root}/baseline/*/in*_out*_c*.json"):
        r = json.load(open(f))
        tag = os.path.basename(os.path.dirname(f))
        m = re.match(r"in(\d+)_out(\d+)_c(\d+)\.json", os.path.basename(f))
        meta = open(os.path.join(os.path.dirname(f), "meta.txt")).read()
        ver = re.search(r"vllm (\S+)", meta)
        rows.append(
            {
                "engine": f"vLLM {ver.group(1) if ver else ''}".strip(),
                "model": r.get("model_id") or re.search(r"model=(\S+)", meta).group(1),
                "input_len": int(m.group(1)),
                "output_len": int(m.group(2)),
                "concurrency": int(m.group(3)),
                "output_tok_s": num(r["output_throughput"]),
                "ttft_p50_ms": num(r["median_ttft_ms"]),
                "ttft_p99_ms": num(r["p99_ttft_ms"]),
                "tpot_p50_ms": num(r["median_tpot_ms"]),
                "itl_p99_ms": num(r["p99_itl_ms"]),
                "ttft_p999_ms": num(r.get("p99.9_ttft_ms")),
                "itl_p999_ms": num(r.get("p99.9_itl_ms")),
                "tpot_p99_ms": num(r.get("p99_tpot_ms")),
                "run": tag,
                "gpu": "B200",
                "date": datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.UTC).strftime("%Y-%m-%dT%H:%MZ"),
            }
        )
    # aqwenator: one JSON list per sweep under aq/
    for f in glob.glob(f"{root}/aq/*.json"):
        for r in json.load(open(f)):
            rows.append(
                {
                    "engine": "aqwenator " + os.path.basename(f).split("-")[0],
                    "model": r["model"],
                    "input_len": r["input_len"],
                    "output_len": r["output_len"],
                    "concurrency": r["concurrency"],
                    "output_tok_s": num(r["output_throughput"]),
                    "ttft_p50_ms": num(r["median_ttft_ms"]),
                    "ttft_p99_ms": num(r["p99_ttft_ms"]),
                    "tpot_p50_ms": num(r["median_tpot_ms"]),
                    "itl_p99_ms": num(r["p99_itl_ms"]),
                    "ttft_p999_ms": num(r.get("p99.9_ttft_ms")),
                    "itl_p999_ms": num(r.get("p99.9_itl_ms")),
                    "tpot_p99_ms": num(r.get("p99_tpot_ms")),
                    "cold_start_loaded_s": num(r.get("cold_start_loaded_s")),
                    "cold_start_first_token_s": num(r.get("cold_start_first_token_s")),
                    "run": os.path.basename(f)[:-5],
                    "gpu": "B200",
                    "date": datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.UTC).strftime("%Y-%m-%dT%H:%MZ"),
                }
            )
    rows.sort(key=lambda r: (r["model"], r["input_len"], r["engine"], r["concurrency"]))
    # vLLM tail-latency reruns supersede the first vLLM runs of the same shape.
    tail = {(r["model"], r["input_len"], r["concurrency"]) for r in rows if r["run"].startswith("vllm-tail")}
    rows = [r for r in rows if not (r["engine"].startswith("vLLM") and not r["run"].startswith("vllm-tail")
                                    and (r["model"], r["input_len"], r["concurrency"]) in tail)]
    which = sys.argv[2] if len(sys.argv) > 2 else "bench"
    if which == "agentic":
        rows = []
        for f in glob.glob(f"{root}/agentic/*/agents*.json"):
            r = json.load(open(f))
            rows.append({
                "engine": r["engine"], "model": r["model"], "agents": r["agents"],
                "turns_per_min": num(r["turns_per_min"]), "gen_tok_s": num(r["gen_tok_s"]),
                "new_prompt_tok_s": num(r["new_prompt_tok_s"]), "mean_ctx": num(r["mean_ctx"]),
                "ttft_p50_ms": num(r["ttft_ms"]["p50"]), "ttft_p99_ms": num(r["ttft_ms"]["p99"]),
                "ttft_p999_ms": num(r["ttft_ms"]["p99.9"]), "itl_p50_ms": num(r["itl_ms"]["p50"]),
                "itl_p99_ms": num(r["itl_ms"]["p99"]), "itl_p999_ms": num(r["itl_ms"]["p99.9"]),
                "turns": r["turns"], "errors": r["errors"], "run": os.path.basename(os.path.dirname(f)),
            })
        rows.sort(key=lambda r: (r["model"], r["engine"], r["agents"]))
    json.dump(rows, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
