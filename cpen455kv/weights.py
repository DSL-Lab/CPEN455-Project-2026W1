"""Load nanoGPT checkpoints without `transformers` (it is not a course dependency).

Two formats are supported:
  * nanoGPT `ckpt.pt` files ({'model_args': ..., 'model': state_dict}), as written by
    nanoGPT's train.py and by scripts/download_gpt2.py;
  * Hugging Face GPT-2 `model.safetensors` files, read with a 20-line parser.
"""
import hashlib
import json
import os
import struct

import torch

from nanogpt.model import GPT, GPTConfig

GPT2_CONFIGS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
}
# OpenAI checkpoints use Conv1D, so these weights are stored transposed
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        t = torch.frombuffer(bytearray(data[a:b]), dtype=_ST_DTYPES[meta["dtype"]])
        out[name] = t.reshape(meta["shape"])
    return out


def hf_gpt2_to_nanogpt(sd_hf, model_type):
    """Map a Hugging Face GPT-2 state dict onto nanoGPT parameter names."""
    cfg = dict(GPT2_CONFIGS[model_type], vocab_size=50257, block_size=1024, bias=True, dropout=0.0)
    sd = {}
    for k, v in sd_hf.items():
        k = k.removeprefix("transformer.")
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue  # causal-mask buffers, not parameters
        if any(k.endswith(w) for w in _TRANSPOSED):
            v = v.t()
        sd["transformer." + k if not k.startswith("lm_head") else k] = v.contiguous()
    sd["lm_head.weight"] = sd["transformer.wte.weight"]  # weight tying
    return cfg, sd


def load_model(path, device="cpu", dtype=torch.float32):
    """Return an eval-mode nanoGPT `GPT` from a ckpt.pt or model.safetensors file."""
    if path.endswith(".safetensors"):
        model_type = os.environ.get("CPEN455_MODEL_TYPE", "gpt2-medium")
        cfg, sd = hf_gpt2_to_nanogpt(read_safetensors(path), model_type)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg, sd = dict(ckpt["model_args"]), ckpt["model"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}  # torch.compile prefix
    cfg["dropout"] = 0.0
    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(sd, strict=False)
    tied = ("lm_head.weight",)  # shares its tensor with transformer.wte.weight
    missing = [k for k in model.state_dict() if k not in sd and not k.endswith(".attn.bias") and k not in tied]
    assert not missing, f"checkpoint is missing {missing[:5]}"
    return model.to(device=device, dtype=dtype).eval()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
