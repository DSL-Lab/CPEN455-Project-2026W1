"""Parts 3 and 4: cache policies.

Read cpen455kv/policy_base.py first: it defines the interface, and `Window` there is a
complete worked example. A policy may return indices only when a layer holds more than
`self.layer_capacity(layer)` tokens, and then must keep at most that many.

`topk_keep` and `StreamingLLM` are provided; you write `H2O` (Part 3) and `MyPolicy`
(Part 4). The provided code contains MUTANTS (see README, "Mutant hunt"): find them, fix
them, and log each one in student/MUTANTS.md.

Allowed imports: torch, numpy, math, the Python standard library, and the course packages.
"""
import torch

from cpen455kv.policy_base import CachePolicy, FullCache, Window, recent_indices  # noqa: F401


def topk_keep(scores, pos, budget, sink=4, recent=32):
    """Choose which cache slots to keep.

    scores: [B, H, L] importance of each slot (higher = more important)
    pos:    [B, H, L] absolute position of each slot, -1 for padding
    Returns [B, H, min(budget, L)] slot indices, sorted ascending, that always include
      * the slots holding the first `sink` tokens of the sequence (pos < sink),
      * the `recent` most recent tokens (the largest positions),
    and fill the rest of the budget with the highest-scoring other slots. Padding slots
    are chosen only if nothing else is left. Requires sink + recent <= budget.
    Slots are not in time order after an eviction: use `pos`, not the slot index.
    """
    assert sink + recent <= budget, "sinks and recent tokens must fit in the budget"
    m = min(budget, pos.shape[-1])
    s = scores.float().masked_fill(pos < 0, float("-inf"))
    newest = torch.full_like(pos[..., :1], pos.shape[-1] - 1)
    protected = ((pos >= 0) & (pos < sink)) | (pos > newest - recent)
    s = s.masked_fill(protected, float("inf"))
    return s.topk(m, dim=-1).indices.sort(dim=-1).values


# ---------------------------------------------------------------------------------------
# Provided baseline
# ---------------------------------------------------------------------------------------
class StreamingLLM(CachePolicy):
    """Attention sinks + recent window (Xiao et al., 2024): keep the first `sink` tokens
    and the most recent capacity − sink tokens. (H2O with constant scores.)"""

    name = "streaming_llm"

    def __init__(self, budget, model_cfg, sink=4, **kwargs):
        super().__init__(budget, model_cfg)
        self.sink = sink

    def on_step(self, layer, k, v, pos, step, probe):
        cap = self.layer_capacity(layer)
        if pos.shape[-1] <= cap:
            return None
        return topk_keep(torch.zeros_like(pos, dtype=torch.float), pos, cap, self.sink, cap - self.sink)


# ---------------------------------------------------------------------------------------
# Part 3: H2O
# ---------------------------------------------------------------------------------------
class H2O(CachePolicy):
    """Heavy-hitter oracle (Zhang et al., 2023), windowed.

    A token's score is the attention mass it received from the last `window` queries.
    Sinks and the `recent` newest tokens are always kept; the rest of the budget goes to the
    top scorers. To amortise the probe, eviction happens every `interval` steps: when the
    cache overflows, it is cut to capacity − interval + 1 tokens, so it can then grow for
    interval − 1 steps without another probe.

    Special cases: constant scores give StreamingLLM; window = interval = recent = 1 and
    sink = 0 give TOVA (evict the token the current query attends to least)."""

    name = "h2o"

    def __init__(self, budget, model_cfg, sink=4, recent=32, window=8, interval=8, **kwargs):
        super().__init__(budget, model_cfg)
        self.sink, self.recent, self.window, self.interval = sink, recent, window, interval
        self.probe_window = window

    def scores(self, layer, k, v, pos, probe):
        """Importance of every cached slot, [B, H, L]: the attention mass it received from the
        last `window` queries. probe(w) returns those attention probabilities, [B, H, w, L]."""
        # TODO (Part 3)
        raise NotImplementedError

    def on_step(self, layer, k, v, pos, step, probe):
        """Return None while the cache is within capacity. When it overflows, keep the sinks,
        the `recent` newest tokens and the top scorers, cutting it to capacity − interval + 1."""
        # TODO (Part 3)
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# Part 4: your improved H2O
# ---------------------------------------------------------------------------------------
class MyPolicy(StreamingLLM):
    """TODO (Part 4): your improved H2O. Describe the method here.

    It starts as a copy of StreamingLLM for local evaluation and validation.json.
    Replace it with your H2O and improve it."""

    name = "my_policy"


POLICIES = {"FullCache": FullCache, "Window": Window, "StreamingLLM": StreamingLLM, "H2O": H2O, "MyPolicy": MyPolicy}
