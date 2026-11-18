"""Convert a 4-bit AutoRound / GPTQ checkpoint (packing `auto_round:auto_gptq`, symmetric,
group size 128) into the layout aqwenator loads: for every quantized linear `<name>.weight`,
`<name>.weight.q4` (uint8 [out, in / 2], two 4-bit codes per byte, low nibble first, value =
code - 8) and `<name>.weight.q4s` (float32 [out, in / 128], one scale per 128 inputs). Tensors
the quantized checkpoint lacks (the MTP head) are taken from the original checkpoint.

    python scripts/convert_autoround.py <quantized dir> <original dir> <out dir>

e.g. Intel/Qwen3.5-9B-int4-AutoRound and Qwen/Qwen3.5-9B (their snapshot directories).
"""
import glob, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

qdir, odir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(qdir, "config.json")))
qc = cfg.pop("quantization_config")
assert qc["bits"] == 4 and qc["group_size"] == 128 and qc["sym"], f"unsupported quantization: {qc}"
assert "auto_gptq" in qc.get("packing_format", ""), f"unsupported packing: {qc.get('packing_format')}"
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)


def index(d):
    names = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            names[k] = h
    return names


def unpack(p, axis):  # 8 4-bit codes per int32 along `axis`
    parts = [(p >> (4 * i)) & 15 for i in range(8)]
    shape = list(p.shape)
    shape[axis] *= 8
    return torch.stack(parts, dim=axis + 1).reshape(shape)


quant, orig = index(qdir), index(odir)
tensors, layers = {}, 0
for k, h in quant.items():
    if k.endswith(".qweight"):
        base = k[: -len(".qweight")]
        codes = unpack(h.get_tensor(k).to(torch.int32), 0).t().contiguous().to(torch.uint8)  # [out, in]
        zeros = unpack(quant[base + ".qzeros"].get_tensor(base + ".qzeros").to(torch.int32), 1)
        # This packing stores the zero point minus one; symmetric 4-bit has it at 8.
        assert bool((zeros == 7).all()), f"{base}: zero point is not 8"
        scales = quant[base + ".scales"].get_tensor(base + ".scales").float().t().contiguous()  # [out, groups]
        n_out, n_in = codes.shape
        assert n_in % 128 == 0 and scales.shape == (n_out, n_in // 128), f"{base}: shape {codes.shape}, scales {scales.shape}"
        tensors[base + ".weight.q4"] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        tensors[base + ".weight.q4s"] = scales
        layers += 1
    elif not k.endswith((".qzeros", ".scales", ".g_idx")):
        tensors[k] = h.get_tensor(k)
missing = [k for k in orig if k not in tensors and k + ".q4" not in tensors]
for k in missing:
    tensors[k] = orig[k].get_tensor(k)
save_file(tensors, os.path.join(out, "model.safetensors"))
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    if os.path.exists(os.path.join(odir, f)):
        shutil.copy(os.path.join(odir, f), out)
size = os.path.getsize(os.path.join(out, "model.safetensors")) / 2**30
print(f"{layers} quantized matrices, {len(missing)} tensors from the original checkpoint, {size:.2f} GiB -> {out}")
