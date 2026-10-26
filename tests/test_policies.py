"""Part 3 (H2O) and the provided policy code in student/policy.py.

These public tests expose only some of the mutants in the provided code (README, "Mutant hunt").
The grader runs a larger hidden suite."""
import pytest
import torch

from cpen455kv.cache import BudgetedKVCache, forward_cached
from cpen455kv.data import collate
from student.policy import H2O, POLICIES, StreamingLLM, topk_keep


def shuffled_positions(B, H, L, n_pad=0, seed=0):
    """pos [B, H, L]: positions 0..L-n_pad-1 in a random slot order, plus n_pad padding slots."""
    g = torch.Generator().manual_seed(seed)
    base = torch.cat([torch.full((n_pad,), -1), torch.arange(L - n_pad)])
    return torch.stack([torch.stack([base[torch.randperm(L, generator=g)] for _ in range(H)]) for _ in range(B)])


# ---- topk_keep (provided) -------------------------------------------------------------------
def test_topk_keep_membership():
    B, H, L, budget, sink, recent = 2, 3, 50, 20, 4, 6
    pos = shuffled_positions(B, H, L, n_pad=5)
    scores = torch.rand(B, H, L)
    idx = topk_keep(scores, pos, budget, sink, recent)
    assert idx.shape == (B, H, budget)
    assert torch.equal(idx, idx.sort(-1).values), "indices must be sorted"
    for b in range(B):
        for h in range(H):
            kept = set(idx[b, h].tolist())
            p = pos[b, h]
            newest = p.max().item()
            must = {i for i in range(L) if 0 <= p[i] < sink or p[i] > newest - recent}
            assert must <= kept
            rest = [i for i in range(L) if i not in must and p[i] >= 0]
            best = sorted(rest, key=lambda i: -scores[b, h, i].item())[:budget - len(must)]
            assert kept == must | set(best)


def test_topk_keep_short_cache_and_padding():
    pos = torch.tensor([[[-1, -1, 0, 1, 2]]])
    idx = topk_keep(torch.zeros(1, 1, 5), pos, budget=8, sink=1, recent=1)
    assert idx.shape == (1, 1, 5)
    idx = topk_keep(torch.tensor([[[9.0, 9.0, 0.0, 0.0, 0.0]]]), pos, budget=3, sink=1, recent=1)
    assert set(idx[0, 0].tolist()) == {2, 3, 4}, "padding must never beat a real token"


# ---- budgets through the real harness -------------------------------------------------------
@pytest.mark.parametrize("name", ["StreamingLLM", "H2O", "MyPolicy"])
@pytest.mark.parametrize("budget", [64, 96])
def test_budget_never_exceeded(tiny_model, story_items, name, budget):
    batch = story_items[:3]
    ids, pos = collate(batch, "cpu")
    ids, pos = ids[:, :budget + 80], pos[:, :budget + 80]  # left-padded contexts, then 80 more tokens
    pol = POLICIES[name](budget, tiny_model.config)
    pol.reset(3, "cpu")
    cache = BudgetedKVCache(tiny_model.config, pol, 3, budget, ids.shape[1], "cpu", torch.float32, strict=True)
    caps = [pol.layer_capacity(l) for l in range(tiny_model.config.n_layer)]
    assert sum(caps) <= tiny_model.config.n_layer * pol.capacity
    with torch.no_grad():
        forward_cached(tiny_model, cache, ids[:, :budget], pos[:, :budget], 0, True)
        assert all(n <= c for n, c in zip(cache.n, caps))
        for s in range(budget, ids.shape[1]):
            forward_cached(tiny_model, cache, ids[:, s:s + 1], pos[:, s:s + 1], s - budget + 1, False)
            assert all(n <= c for n, c in zip(cache.n, caps)), f"step {s}: {cache.n} > {caps}"


def test_streaming_llm_keeps_sinks_and_recent(tiny_model):
    pol = StreamingLLM(16, tiny_model.config, sink=4)
    pos = shuffled_positions(1, 2, 17, seed=1)
    idx = pol.on_step(0, None, None, pos, 1, None)
    kept = pos.gather(-1, idx)
    for h in range(2):
        assert set(kept[0, h].tolist()) == set(range(4)) | set(range(5, 17))


def test_h2o_keeps_heavy_hitters(tiny_model):
    pol = H2O(8, tiny_model.config, sink=1, recent=1, window=4, interval=4)
    pol.reset(1, "cpu")
    pos = torch.tensor([[[4, 0, 7, 2, 8, 1, 5, 3, 6],
                         [1, 5, 0, 8, 4, 6, 2, 7, 3]]])
    # Fixed observations: the final query alone favours different slots from the
    # full window, and each head has its own expected retained slots.
    attn = torch.tensor([[
        [[5, 1, 4, 3, 2, 35, 20, 10, 20],
         [5, 1, 4, 3, 2, 35, 20, 10, 20],
         [5, 1, 4, 3, 2, 35, 20, 10, 20],
         [50, 1, 20, 3, 2, 1, 1, 2, 20]],
        [[20, 10, 1, 2, 5, 30, 3, 4, 25],
         [20, 10, 1, 2, 5, 30, 3, 4, 25],
         [20, 10, 1, 2, 5, 30, 3, 4, 25],
         [1, 2, 1, 2, 50, 1, 3, 20, 20]],
    ]], dtype=torch.float32) / 100
    expected_scores = torch.tensor([[[0.65, 0.04, 0.32, 0.12, 0.08, 1.06, 0.61, 0.32, 0.80],
                                     [0.61, 0.32, 0.04, 0.08, 0.65, 0.91, 0.12, 0.32, 0.95]]])
    torch.testing.assert_close(pol.scores(0, None, None, pos, lambda w: attn[:, :, -w:]), expected_scores)
    idx = pol.on_step(0, None, None, pos, 1, lambda w: attn[:, :, -w:])
    expected_indices = torch.tensor([[[0, 1, 4, 5, 8], [2, 3, 4, 5, 8]]])
    assert torch.equal(idx, expected_indices)


def test_h2o_as_tova_evicts_least_attended(tiny_model):
    """TOVA is H2O with window = interval = recent = 1 and no sinks."""
    pol = H2O(8, tiny_model.config, sink=0, recent=1, window=1, interval=1)
    pol.reset(1, "cpu")
    pos = torch.arange(9).view(1, 1, 9).expand(1, 3, 9).contiguous()
    # The newest slot is protected even though its observed attention is zero.
    attn = torch.tensor([[[[12, 12, 1, 15, 15, 15, 15, 15, 0]],
                          [[15, 15, 15, 15, 15, 1, 12, 12, 0]],
                          [[15, 15, 15, 15, 15, 12, 12, 1, 0]]]], dtype=torch.float32) / 100
    idx = pol.on_step(0, None, None, pos, 1, lambda w: attn[:, :, -w:])
    expected = torch.tensor([[[0, 1, 3, 4, 5, 6, 7, 8],
                              [0, 1, 2, 3, 4, 6, 7, 8],
                              [0, 1, 2, 3, 4, 5, 6, 8]]])
    assert torch.equal(idx, expected)


def test_policies_do_nothing_within_capacity(tiny_model):
    pos = torch.arange(16).view(1, 1, 16).expand(1, 2, 16).contiguous()
    for cls in (StreamingLLM, H2O):
        pol = cls(16, tiny_model.config)
        pol.reset(1, "cpu")
        assert pol.on_step(0, None, None, pos, 1, lambda w: torch.rand(1, 2, w, 16)) is None
