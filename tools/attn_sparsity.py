"""How sparse is decode attention at long context, on a real agent trace?

Decides whether filter-and-refine decode attention is worth building: read a small sketch of
every key, bound each key's score, and read full K/V only for keys that can matter.

For the last --rows query positions of a real agent trajectory (OpenHands, rendered with the
model's chat template and tools), per full-attention layer and query head:

  within tau   fraction of keys whose score is within tau nats of the row's max score
  mass eps     fraction of keys needed to cover all but eps of the softmax mass
  sketch N-bit a linear N-bit sketch of each key row (per-row scale): the score error it
               causes, the rigorous bound |q| * |k - sketch(k)|, and the fraction of keys that
               survive the filter "upper bound >= (lower bound of the max) - tau_exact", where
               tau_exact = ln(n_keys) + 24 ln 2 makes the dropped mass < 2^-24 of the total
               (below fp32 accumulation error), and tau_bf16 = ln(n_keys) + 12 ln 2 (below
               bf16 output precision; not exact).

    python tools/attn_sparsity.py --model Qwen/Qwen3.5-0.8B --trace rows.json --len 16384 --out out.json

`rows.json` is a page of nebius/SWE-rebench-openhands-trajectories from the datasets server:
    https://datasets-server.huggingface.co/rows?dataset=nebius/SWE-rebench-openhands-trajectories&config=default&split=train&offset=0&length=3
"""

import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TAUS = (4, 8, 12, 16, 20, 24, 28)
EPS = (2.0**-8, 2.0**-12, 2.0**-16, 2.0**-24)
SKETCH_BITS = (8, 4)


def prompt_ids(tok, trace_path, row, n_tokens):
    rows = json.load(open(trace_path))["rows"]
    r = rows[row]["row"]
    msgs = []
    for m in r["trajectory"]:
        m = {k: v for k, v in m.items() if v is not None}
        for tc in m.get("tool_calls", []):
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    pass
        msgs.append(m)
    try:
        ids = tok.apply_chat_template(msgs, tools=r["tools"], tokenize=True, add_generation_prompt=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        how = "chat template + tools"
    except Exception as e:  # template rejected the trace: fall back to the raw message text
        text = "\n\n".join(f"[{m['role']}]\n{m.get('content') or json.dumps(m.get('tool_calls'))}" for m in msgs)
        ids = tok(text).input_ids
        how = f"plain text (template failed: {type(e).__name__}: {str(e)[:80]})"
    return ids[:n_tokens], len(ids), how


class Recorder:
    def __init__(self, rows):
        self.rows = rows
        self.on = False
        self.layers = []

    @torch.no_grad()
    def record(self, query, key, scaling):
        # query [B, H, Tq, D] (post q_norm + RoPE), key [B, Hkv, Tk, D]
        h, tq = query.shape[1], query.shape[2]
        hkv, tk = key.shape[1], key.shape[2]
        r = min(self.rows, tq)
        q = query[0, :, -r:, :].float()
        k = key[0].float()
        ki = torch.arange(h, device=k.device) // (h // hkv)  # kv head of each query head
        # Row i of the last r queries sits at position tk - r + i and sees keys 0..=that.
        pos = tk - r + torch.arange(r, device=k.device)
        valid = torch.arange(tk, device=k.device)[None, :] <= pos[:, None]  # [r, tk]
        n_valid = valid.sum(-1).double()  # [r]

        def scores(kmat):
            return (torch.einsum("hrd,htd->hrt", q, kmat[ki]) * scaling).masked_fill(~valid[None], float("-inf"))

        s = scores(k)
        m = s.max(-1, keepdim=True).values
        d = m - s
        out = {"n_keys": int(tk), "heads": h}
        frac = lambda cnt: (cnt.double() / n_valid[None])  # [h, r]
        agg = lambda x: {"mean": float(x.mean()), "worst_head": float(x.mean(-1).max()), "max": float(x.max())}
        out["within"] = {str(t): agg(frac((d <= t).sum(-1))) for t in TAUS}

        w = torch.softmax(s.double(), -1)
        asc = torch.sort(w, -1).values
        tail = torch.cumsum(asc, -1)
        masked = (~valid).sum(-1).double()[None]  # masked keys have weight 0 and sort first
        out["mass"] = {
            f"2^{round(math.log2(e))}": agg(1.0 - ((tail <= e).sum(-1).double() - masked).clamp(min=0) / n_valid[None]) for e in EPS
        }
        del w, asc, tail

        qn = q.norm(dim=-1)  # [h, r]
        tau_exact = torch.log(n_valid) + 24 * math.log(2)
        tau_bf16 = torch.log(n_valid) + 12 * math.log(2)
        out["tau_exact"] = float(tau_exact.mean())
        out["sketch"] = {}
        for bits in SKETCH_BITS:
            lv = 2 ** (bits - 1) - 1
            scale = k.abs().amax(-1, keepdim=True).clamp(min=1e-20) / lv
            kh = torch.round(k / scale) * scale
            rn = (k - kh).norm(dim=-1)  # [hkv, tk]
            sh = scores(kh)
            bound = qn[:, :, None] * rn[ki][:, None, :] * scaling  # rigorous: |s - sh| <= bound
            err = (s - sh).abs().masked_fill(~valid[None], 0)
            m_lo = (sh - bound).max(-1, keepdim=True).values
            res = {
                "err_max": float(err.max()),
                "err_p999": float(torch.quantile(err.flatten()[:: max(1, err.numel() // 4_000_000)], 0.999)),
                "bound_mean": float(bound.masked_fill(~valid[None], 0).sum() / (h * n_valid.sum())),
                "bound_max": float(bound.masked_fill(~valid[None], 0).max()),
            }
            for name, tau in (("survive_exact", tau_exact), ("survive_bf16", tau_bf16)):
                keep = (sh + bound >= m_lo - tau[None, :, None].float()) & valid[None]
                res[name] = agg(frac(keep.sum(-1)))
            out["sketch"][f"{bits}bit"] = res
        self.layers.append(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--len", type=int, default=16384)
    p.add_argument("--rows", type=int, default=64, help="query positions sampled (the last ones)")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, help="bfloat16 | float32 (default: bf16 on cuda, fp32 on cpu)")
    p.add_argument("--out")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype or ("bfloat16" if args.device == "cuda" else "float32"))
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, full, how = prompt_ids(tok, args.trace, args.row, args.len)
    print(f"prompt: {len(ids)} tokens (trace has {full}), {how}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    rec = Recorder(args.rows)
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def wrap(fn):
        def inner(module, query, key, value, *a, **kw):
            if rec.on:
                rec.record(query, key, kw.get("scaling", module.scaling))
            return fn(module, query, key, value, *a, **kw)

        return inner

    mod.eager_attention_forward = wrap(mod.eager_attention_forward)
    for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
        ALL_ATTENTION_FUNCTIONS[name] = wrap(ALL_ATTENTION_FUNCTIONS[name])

    x = torch.tensor([ids], device=args.device)
    pkv = None
    t0 = time.time()
    with torch.no_grad():
        for o in range(0, len(ids), args.chunk):
            rec.on = o + args.chunk >= len(ids)
            out = model(input_ids=x[:, o : o + args.chunk], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            print(f"  {min(o + args.chunk, len(ids))}/{len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)

    L = rec.layers
    print(f"\n{args.model}, {len(ids)} tokens, last {args.rows} queries, {len(L)} attention layers; tau_exact = {L[0]['tau_exact']:.1f} nats")
    print("fraction of keys (mean over heads and rows | worst head):")
    hdr = "layer  " + "  ".join(f"<={t:>2}nat" for t in (8, 16, 24)) + "   mass 2^-12  mass 2^-24   8bit: bound  surv.exact  surv.bf16   4bit: bound  surv.exact"
    print(hdr)
    for i, l in enumerate(L):
        w, ms, s8, s4 = l["within"], l["mass"], l["sketch"]["8bit"], l["sketch"]["4bit"]
        f = lambda a: f"{100 * a['mean']:5.1f}|{100 * a['worst_head']:5.1f}"
        print(
            f"{i:>5}  " + "  ".join(f"{100 * w[str(t)]['mean']:6.1f}%" for t in (8, 16, 24))
            + f"  {f(ms['2^-12'])}  {f(ms['2^-24'])}   {s8['bound_mean']:6.2f}nat  {f(s8['survive_exact'])}  {f(s8['survive_bf16'])}"
            + f"   {s4['bound_mean']:6.2f}nat  {f(s4['survive_exact'])}"
        )
    mean = lambda path: sum(path(l) for l in L) / len(L)
    print(
        f"\nall layers: survivors with an 8-bit sketch, exact threshold: {100 * mean(lambda l: l['sketch']['8bit']['survive_exact']['mean']):.1f}%"
        f" (bf16 threshold: {100 * mean(lambda l: l['sketch']['8bit']['survive_bf16']['mean']):.1f}%);"
        f" keys needed for all but 2^-24 of the mass: {100 * mean(lambda l: l['mass']['2^-24']['mean']):.1f}%"
    )
    if args.out:
        json.dump({"model": args.model, "tokens": len(ids), "prompt": how, "rows": args.rows, "layers": L}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
