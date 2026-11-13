"""Collect benchmark JSONs from ~/aq/results into one flat file for the progress page.

    python -I tools/collect.py ~/aq/results > bench.json

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
                    "cold_start_loaded_s": num(r.get("cold_start_loaded_s")),
                    "cold_start_first_token_s": num(r.get("cold_start_first_token_s")),
                    "run": os.path.basename(f)[:-5],
                    "gpu": "B200",
                    "date": datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.UTC).strftime("%Y-%m-%dT%H:%MZ"),
                }
            )
    rows.sort(key=lambda r: (r["model"], r["input_len"], r["engine"], r["concurrency"]))
    json.dump(rows, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
