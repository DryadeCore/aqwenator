"""Dump reference activations from transformers for a Qwen3.5-family checkpoint.

Writes one safetensors file holding the prompt ids, the residual stream after
every decoder layer, the final-norm output, and the last-position logits.
Everything is in fp32, from a bf16 forward pass in eager mode.

    python -I tools/golden.py --model Qwen/Qwen3.5-0.8B --out ~/aq/golden/qwen35-0.8b.safetensors
"""

import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--greedy", type=int, default=32, help="greedy tokens to generate for an e2e check")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    ).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    tensors = {"input_ids": ids[0].to(torch.int32).cpu()}
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
        # hidden_states[0] = embeddings, [i+1] = after layer i (the last one is post final norm in HF)
        hs = out.hidden_states
        for i, h in enumerate(hs):
            tensors[f"hidden.{i}"] = h[0].float().cpu().contiguous()
        tensors["logits_last"] = out.logits[0, -1].float().cpu().contiguous()

        gen = model.generate(ids, max_new_tokens=args.greedy, do_sample=False)
        tensors["greedy_ids"] = gen[0, ids.shape[1]:].to(torch.int32).cpu()

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    save_file(tensors, os.path.expanduser(args.out), metadata={"model": args.model, "prompt": PROMPT})
    print("saved", len(tensors), "tensors; greedy:", repr(tok.decode(tensors["greedy_ids"])))


if __name__ == "__main__":
    main()
