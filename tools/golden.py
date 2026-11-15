"""Dump reference activations and logits from transformers for a Qwen3.5-family checkpoint.

Writes one safetensors file holding:
  input_ids          prompt token ids
  hidden.{i}         residual stream after layer i (fp32), hidden.0 = embeddings  [unless --no-hidden]
  logits_last        logits at the last prompt position (fp32)
  greedy_ids         greedy continuation (--greedy tokens)
  tf_logits          teacher-forced logits at every generated position: row k is the
                     distribution for greedy_ids[k] given prompt + greedy_ids[:k]  (fp32)

    python -I tools/golden.py --model Qwen/Qwen3.5-0.8B --out ~/aq/golden/qwen35-0.8b.safetensors
    python -I tools/golden.py --model Qwen/Qwen3.5-9B --len 16384 --no-hidden --attn sdpa --out ...
"""

import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"

# ~1.8k tokens of varied text, so the chunked/tiled prefill paths cross many chunk boundaries.
RECORDS = [
    f"Record {i}: the {a} {b} crossed {c} kilometres of {d} terrain in {i % 7 + 2} days, "
    f"carrying {i * 13 % 97} crates of {e}."
    for i, (a, b, c, d, e) in enumerate(
        [(a, b, c, d, e) for a in ("swift", "old", "patient", "noisy") for b in ("caravan", "courier", "barge", "convoy")
         for c in (12, 340, 77) for d in ("desert", "mountain") for e in ("salt", "tea", "copper")]
    )
]
LONG_PROMPT = " ".join(RECORDS[:60]) + " Summarize the records above in one sentence:"


def synthetic(tok, n_tokens):
    """A prompt of about n_tokens: the record list repeated with distinct indices."""
    parts, i = [], 0
    while True:
        r = RECORDS[i % len(RECORDS)].replace("Record ", f"Record {i // len(RECORDS)}.", 1)
        parts.append(r)
        i += 1
        if i % 50 == 0 and len(tok(" ".join(parts)).input_ids) >= n_tokens:
            break
    return " ".join(parts) + " Which record mentions the most crates of copper? Answer:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--greedy", type=int, default=32, help="greedy tokens to generate")
    p.add_argument("--long", action="store_true", help="use the ~1.8k-token prompt")
    p.add_argument("--len", type=int, default=0, help="synthetic prompt of about this many tokens")
    p.add_argument("--ids-file", help="JSON list of prompt token ids (e.g. one turn of a recorded trace)")
    p.add_argument("--no-hidden", action="store_true", help="skip per-layer hidden states (long prompts)")
    p.add_argument("--attn", default="eager", help="attention implementation (eager or sdpa)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation=args.attn
    ).eval()
    if args.ids_file:
        import json

        ids = torch.tensor([json.load(open(os.path.expanduser(args.ids_file)))]).cuda()
    else:
        prompt = synthetic(tok, args.len) if args.len else LONG_PROMPT if args.long else PROMPT
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    tensors = {"input_ids": ids[0].to(torch.int32).cpu()}
    with torch.no_grad():
        out = model(ids, output_hidden_states=not args.no_hidden, use_cache=False)
        if not args.no_hidden:
            for i, h in enumerate(out.hidden_states):
                tensors[f"hidden.{i}"] = h[0].float().cpu().contiguous()
        tensors["logits_last"] = out.logits[0, -1].float().cpu().contiguous()
        del out

        gen = model.generate(ids, max_new_tokens=args.greedy, do_sample=False)
        greedy = gen[0, ids.shape[1]:]
        tensors["greedy_ids"] = greedy.to(torch.int32).cpu()

        # Teacher forcing: one forward over prompt + greedy[:-1]; keep only the last rows.
        full = torch.cat([ids[0], greedy[:-1]])[None]
        logits = model(full, use_cache=False, logits_to_keep=len(greedy)).logits[0]
        tensors["tf_logits"] = logits.float().cpu().contiguous()

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    save_file(tensors, os.path.expanduser(args.out), metadata={"model": args.model, "prompt_tokens": str(ids.shape[1])})
    print("saved", len(tensors), "tensors;", ids.shape[1], "prompt tokens; greedy:", repr(tok.decode(greedy)[:120]))


if __name__ == "__main__":
    main()
