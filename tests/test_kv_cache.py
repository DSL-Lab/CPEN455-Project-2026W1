"""Part 2: the KV cache in student/kv_cache.py.

These public tests expose only some of the mutants in the provided code (README, "Mutant hunt").
The grader runs a larger hidden suite."""
import torch

from cpen455kv.part2_bench import vanilla_generate_greedy
from student.kv_cache import KVCache, forward_with_cache, generate_window, generate_with_cache, new_cache


def full_logits(model, idx):
    """Logits at every position from one plain nanoGPT forward."""
    logits, _ = model(idx, targets=torch.zeros_like(idx))
    return logits


def test_append_returns_all_cached_tokens():
    c = KVCache(n_layer=2, batch_size=3, n_head=4, head_dim=8, max_len=20, device="cpu", dtype=torch.float32)
    k1, v1 = torch.randn(3, 4, 5, 8), torch.randn(3, 4, 5, 8)
    K, V = c.append(0, k1, v1)
    assert K.shape == (3, 4, 5, 8) and torch.equal(K, k1) and torch.equal(V, v1)
    c.advance(5)
    k2, v2 = torch.randn(3, 4, 1, 8), torch.randn(3, 4, 1, 8)
    K, V = c.append(0, k2, v2)
    assert K.shape == (3, 4, 6, 8)
    assert torch.equal(K[:, :, :5], k1) and torch.equal(K[:, :, 5:], k2) and torch.equal(V[:, :, 5:], v2)


def test_single_token_step_matches_full_forward(tiny_model):
    """Compare one cached decoding step with a full forward pass."""
    torch.manual_seed(1)
    idx = torch.randint(0, 50257, (2, 40))
    ref = full_logits(tiny_model, idx)
    cache = new_cache(tiny_model, 2, 40, "cpu", torch.float32)
    with torch.no_grad():
        forward_with_cache(tiny_model, idx[:, :39], cache)
        out = forward_with_cache(tiny_model, idx[:, 39:], cache)
    torch.testing.assert_close(out[:, -1], ref[:, -1], atol=1e-4, rtol=1e-4)


def test_multi_token_chunk_after_prefill(tiny_model):
    """Compare a cached multi-token chunk with a full forward pass."""
    torch.manual_seed(2)
    idx = torch.randint(0, 50257, (2, 50))
    ref = full_logits(tiny_model, idx)
    cache = new_cache(tiny_model, 2, 50, "cpu", torch.float32)
    with torch.no_grad():
        forward_with_cache(tiny_model, idx[:, :30], cache)
        out = forward_with_cache(tiny_model, idx[:, 30:], cache)
    torch.testing.assert_close(out, ref[:, 30:], atol=1e-4, rtol=1e-4)


def test_positions_continue_after_prefill(tiny_model):
    cache = new_cache(tiny_model, 1, 16, "cpu", torch.float32)
    with torch.no_grad():
        forward_with_cache(tiny_model, torch.zeros(1, 10, dtype=torch.long), cache)
        forward_with_cache(tiny_model, torch.zeros(1, 1, dtype=torch.long), cache)
    assert cache.n == 11 and cache.pos == 11


def test_greedy_equivalence_tiny(tiny_model):
    torch.manual_seed(3)
    idx = torch.randint(0, 50257, (4, 32))
    ref = vanilla_generate_greedy(tiny_model, idx, 24)
    out = generate_with_cache(tiny_model, idx, 24)
    assert torch.equal(out, ref)


def test_greedy_equivalence_gpt2_stories(gpt2_model, story_items):
    """20 fixed prompts (the first 256 tokens of development stories), greedy, token-identical."""
    from cpen455kv.timing import fixed_sequences
    idx = fixed_sequences(story_items, 20, 256)
    ref = vanilla_generate_greedy(gpt2_model, idx, 16)
    out = generate_with_cache(gpt2_model, idx, 16)
    assert torch.equal(out, ref)


def test_window_larger_than_sequence_is_full_cache(tiny_model):
    torch.manual_seed(4)
    idx = torch.randint(0, 50257, (2, 20))
    assert torch.equal(generate_window(tiny_model, idx, 12, window=64), generate_with_cache(tiny_model, idx, 12))
