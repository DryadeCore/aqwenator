"""Lossless bf16 compressibility study (weights and KV) for the 12-bit "PFOR" family.

Format (per group of G values, bf16 = sign:1 | exponent:8 | mantissa:7):
  plane A  8 bits/value : sign + 7-bit mantissa (verbatim)
  plane B  b bits/value : delta = base - exponent, base = max exponent in the group;
                          delta == 2^b - 1 is an escape (true exponent in a patch list)
  header   8 bits/group : base exponent
  patches  per escape   : 16-bit in-group index + 8-bit exponent (24 bits, upper bound)
Bits/value = 8 + b + 8/G + 24 * escape_rate. Roundtrip is checked bit-exactly.

    python -I tools/compress_study.py weights <model dir>           # all bf16 tensors
    python -I tools/compress_study.py kv <model id> <golden.safetensors> [layers]
"""

import glob
import json
import sys

import numpy as np


def exp_entropy(e):
    c = np.bincount(e.ravel(), minlength=256).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def encode_decode(bits16, G, b):
    """bits16: uint16 array with last dim = groups of G. Returns (roundtrip_ok, escapes)."""
    x = bits16.reshape(-1, G)
    sign_mant = ((x >> 8) & 0x80) | (x & 0x7F)  # 8-bit plane
    e = ((x >> 7) & 0xFF).astype(np.int32)
    base = e.max(axis=1, keepdims=True)
    delta = base - e
    esc_code = (1 << b) - 1
    esc = delta >= esc_code
    code = np.where(esc, esc_code, delta)
    # decode
    e_dec = np.where(esc, e, base - code)  # escapes come from the patch list
    y = (((sign_mant & 0x80).astype(np.uint16) << 8) | (e_dec.astype(np.uint16) << 7) | (sign_mant & 0x7F)).astype(np.uint16)
    return bool(np.array_equal(y, x)), int(esc.sum())


def study(bits16, G):
    e = ((bits16 >> 7) & 0xFF).astype(np.uint8)
    out = {"n": int(bits16.size), "exp_entropy_bits": exp_entropy(e)}
    for b in (3, 4):
        ok, n_esc = encode_decode(bits16, G, b)
        rate = n_esc / bits16.size
        out[f"b{b}"] = {"roundtrip": ok, "escape_rate": rate, "bits_per_value": 8 + b + 8 / G + 24 * rate}
    return out


def weights(model_dir):
    from safetensors import safe_open

    tot = {"n": 0, "esc3": 0, "esc4": 0, "ok": True, "hist": np.zeros(256, np.float64)}
    per = []
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, "np") as s:
            for k in s.keys():
                if "visual" in k:
                    continue
                sl = s.get_slice(k)
                if sl.get_dtype() != "BF16":
                    continue
                shape = sl.get_shape()
                n = int(np.prod(shape))
                if n % 128 != 0 or n < 4096:
                    continue
                # read raw bytes as uint16 (numpy has no bf16)
                import torch  # noqa: F401  (safetensors 'pt' path avoids bf16 numpy limits)

                with safe_open(f, "pt") as st:
                    t = st.get_tensor(k)
                bits = t.view(dtype=__import__("torch").int16).numpy().view(np.uint16).ravel()
                r = study(bits, 128)
                tot["n"] += n
                tot["esc3"] += r["b3"]["escape_rate"] * n
                tot["esc4"] += r["b4"]["escape_rate"] * n
                tot["ok"] &= r["b3"]["roundtrip"] and r["b4"]["roundtrip"]
                tot["hist"] += np.bincount(((bits >> 7) & 0xFF).astype(np.uint8), minlength=256)
                per.append((k, n, r["b4"]["escape_rate"], r["b3"]["escape_rate"]))
    p = tot["hist"][tot["hist"] > 0] / tot["hist"].sum()
    summary = {
        "tensors": len(per),
        "weights": tot["n"],
        "all_roundtrip_exact": tot["ok"],
        "exp_entropy_bits": float(-(p * np.log2(p)).sum()),
        "b4_escape_rate": tot["esc4"] / tot["n"],
        "b4_bits_per_weight": 12 + 8 / 128 + 24 * tot["esc4"] / tot["n"],
        "b3_escape_rate": tot["esc3"] / tot["n"],
        "b3_bits_per_weight": 11 + 8 / 128 + 24 * tot["esc3"] / tot["n"],
        "worst_b4_tensors": sorted(per, key=lambda x: -x[2])[:5],
    }
    print(json.dumps(summary, indent=1))


def kv(model, golden, layers):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    ids = load_file(golden)["input_ids"].long()[None].cuda()
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    with torch.no_grad():
        out = m(ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    res = {}
    full = [i for i, t in enumerate(m.config.get_text_config().layer_types) if t == "full_attention"]
    for li in full[: layers]:
        lay = cache.layers[li]
        for name, t in (("K", lay.keys), ("V", lay.values)):
            # [1, Hkv, T, D] -> rows of (token, head) with D=256 values
            x = t[0].transpose(0, 1).contiguous()  # [T, Hkv, D]
            bits = x.view(torch.int16).cpu().numpy().view(np.uint16)
            T, H, D = bits.shape
            r_th = study(bits.reshape(-1), 256)  # group = one (token, head) row
            r_th128 = study(bits.reshape(-1), 128)  # half rows
            # channel-major groups: each (head, channel) over 64 consecutive tokens (a KV page)
            Tp = (T // 64) * 64
            cm = bits[:Tp].reshape(Tp // 64, 64, H, D).transpose(0, 2, 3, 1).reshape(-1)
            r_ch = study(cm, 64)
            res[f"L{li}.{name}"] = {
                "tokens": T,
                "exp_entropy": r_th["exp_entropy_bits"],
                "row256_b4": r_th["b4"],
                "row128_b4": r_th128["b4"],
                "row128_b3": r_th128["b3"],
                "chan64_b4": r_ch["b4"],
                "chan64_b3": r_ch["b3"],
            }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        weights(sys.argv[2])
    else:
        kv(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
