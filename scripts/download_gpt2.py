"""Download OpenAI GPT-2 weights and save them as a nanoGPT checkpoint (fp16).

Needs only torch (no `transformers`). Run during setup with network access:

    python scripts/download_gpt2.py gpt2          # 124M development model -> checkpoints/gpt2.pt
    python scripts/download_gpt2.py gpt2-medium   # 355M graded model      -> checkpoints/gpt2-medium.pt

The graded model is the base gpt2-medium checkpoint, unchanged (no fine-tuning).
"""
import os
import sys
import tempfile
import urllib.request

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpen455kv.weights import GPT2_CONFIGS, hf_gpt2_to_nanogpt, read_safetensors  # noqa: E402

URL = "https://huggingface.co/openai-community/{}/resolve/main/model.safetensors"


def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    assert model_type in GPT2_CONFIGS, f"choose one of {list(GPT2_CONFIGS)}"
    out_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{model_type}.pt")

    with tempfile.TemporaryDirectory() as tmp:
        st = os.path.join(tmp, "model.safetensors")
        print(f"downloading {URL.format(model_type)}")
        urllib.request.urlretrieve(URL.format(model_type), st)
        cfg, sd = hf_gpt2_to_nanogpt(read_safetensors(st), model_type)

    sd = {k: v.half() for k, v in sd.items() if k != "lm_head.weight"}  # tied to wte
    torch.save({"model_args": cfg, "model": sd, "source": f"openai-community/{model_type}"}, out)
    print(f"saved {out} ({os.path.getsize(out) / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
