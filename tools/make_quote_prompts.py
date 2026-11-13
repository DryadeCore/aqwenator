"""Build realistic "quote back with an edit" prompts (agents do this with tool outputs).

Each prompt is a chat-templated user turn (thinking disabled) that shows a source file and asks
for it back with a small edit, so most of the answer repeats the context. Writes a JSON list of
token-id lists for `aq-bench --prompts-file`.

    python -I tools/make_quote_prompts.py --model Qwen/Qwen3.5-9B --out ~/aq/prompts/quote.json \\
        --files crates/aq-engine/src/engine.rs crates/aq-engine/src/gpu.rs --max-lines 80
"""

import argparse
import json
import os

from transformers import AutoTokenizer

EDITS = [
    "Return the code exactly as given, but rename the variable `{a}` to `{a}_new` everywhere.",
    "Return the code exactly as given, but add a one-line comment above every function.",
    "Return the code exactly as given, with every `Result<()>` written as `anyhow::Result<()>`.",
    "Repeat the code exactly as given, then add a short summary line at the end.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--max-lines", type=int, default=80)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for f in args.files:
        lines = open(f).read().splitlines()
        for start in range(0, len(lines), args.max_lines):
            chunk = "\n".join(lines[start : start + args.max_lines])
            if len(chunk) < 400:
                continue
            words = [w for w in chunk.replace("(", " ").replace(")", " ").split() if w.isidentifier() and len(w) > 3]
            a = words[len(words) // 2] if words else "x"
            for e in EDITS:
                msg = f"```rust\n{chunk}\n```\n\n" + e.format(a=a)
                text = tok.apply_chat_template(
                    [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompts.append(tok(text, add_special_tokens=False).input_ids)
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    json.dump(prompts, open(os.path.expanduser(args.out), "w"))
    lens = sorted(len(x) for x in prompts)
    print(f"{len(prompts)} prompts, tokens min/median/max {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")


if __name__ == "__main__":
    main()
