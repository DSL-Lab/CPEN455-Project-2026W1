import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanogpt.model import GPT, GPTConfig  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model():
    """A 2-layer random GPT with GPT-2's vocabulary and context, fp32 on CPU."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=50257, block_size=1024, bias=True, dropout=0.0))
    return m.eval()


@pytest.fixture(scope="session")
def gpt2_model():
    """GPT-2 small in fp32 on CPU, if `python scripts/download_gpt2.py gpt2` has been run."""
    path = os.path.join(ROOT, "checkpoints", "gpt2.pt")
    if not os.path.exists(path):
        pytest.skip("checkpoints/gpt2.pt not found (python scripts/download_gpt2.py gpt2)")
    from cpen455kv.weights import load_model
    return load_model(path, "cpu", torch.float32)


@pytest.fixture(scope="session")
def story_items():
    from cpen455kv.data import load_split
    return load_split("cbt-val", limit=24)
