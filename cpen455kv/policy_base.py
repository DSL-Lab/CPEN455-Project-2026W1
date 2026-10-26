"""The cache-policy interface, plus the two reference policies the harness needs.

A policy decides which cached tokens to keep. The harness owns the model, the data, the
cache tensors, the gather, the timers and the FLOP counter. A policy only ever sees, per
layer, the cached keys and values, the absolute position of every cache slot, the step
number, and an attention probe whose cost is counted. It never sees token ids or text.

Shapes (gpt2-medium: H = 16 heads, D = 64):
    k, v   [B, H, L, D]   cached keys / values, L = slots currently in use
    pos    [B, H, L]      absolute position of each slot, -1 for padding (never attended)
    return [B, H, M]      slot indices to keep, M <= self.capacity, or None

Slots are not stored in time order after an eviction: use `pos` to know which token is which.
After the harness gathers your indices, slot j holds what was slot idx[..., j] before.
"""
import math

import torch


def capacity_for(budget, kv_bits=16, residual=0, key_group=32, head_dim=64):
    """Number of tokens that fit in `budget` fp16 tokens' worth of bytes.

    A quantised token costs `kv_bits` per element plus its scale and zero point (two fp16
    numbers): per token for values (one pair per head_dim elements) and per channel for
    keys (one pair per `key_group` tokens). The newest `residual` tokens stay in fp16."""
    if budget is None or kv_bits >= 16:
        return budget
    residual = min(residual, budget)
    overhead = (32 / key_group + 32 / head_dim) / 2  # bits per element, averaged over K and V
    return residual + math.floor((budget - residual) * 16 / (kv_bits + overhead))


class CachePolicy:
    """Base class. Subclasses override `on_prefill` and/or `on_step`."""

    name = "base"
    kv_bits = 16          # < 16: the harness charges quantised bytes and calls `compress`
    residual = 0          # fp16 recent window of a quantised policy
    key_group = 32        # tokens per key-quantisation group
    probe_window = 0      # how many recent queries `probe` may return; 0 = no probing

    def __init__(self, budget, model_cfg, **kwargs):
        self.budget = budget
        self.model_cfg = model_cfg
        self.n_layer = model_cfg.n_layer
        self.capacity = capacity_for(budget, self.kv_bits, self.residual, self.key_group,
                                     model_cfg.n_embd // model_cfg.n_head)

    def layer_capacity(self, layer):
        """Token capacity of one layer. Override for layer-wise budgets: the harness only
        requires the sum over layers to stay within n_layer x capacity."""
        return self.capacity

    def reset(self, batch_size, device):
        """Called before every new batch of sequences. Clear per-sequence state here."""

    def on_prefill(self, layer, k, v, pos, probe):
        """After the problem statement has been prefilled with the full cache.

        `probe(w)` returns softmax(q kᵀ / √D) of the last w prompt queries against the
        cache, shape [B, H, w, L] (0 at padding and at keys after the query). It costs
        FLOPs and time, both of which are charged to you. Must return indices if L > capacity."""
        return self.on_step(layer, k, v, pos, 0, probe)

    def on_step(self, layer, k, v, pos, step, probe):
        """After every decoding step, once the new token's K and V are in the cache and
        attention for that token has been computed. Must return indices if L > capacity."""
        raise NotImplementedError

    def compress(self, layer, k, v, pos):
        """Quantised policies only (kv_bits < 16): return (k, v) with the slots you store
        in low precision replaced by their quantise→dequantise round trip, or None."""
        return None


def recent_indices(pos, n):
    """Indices of the n most recent slots per [B, H], sorted by slot index."""
    n = min(n, pos.shape[-1])
    return pos.topk(n, dim=-1).indices.sort(dim=-1).values


class FullCache(CachePolicy):
    """Keeps everything. The quality reference; not ranked."""

    name = "full"

    def __init__(self, budget, model_cfg, **kwargs):
        super().__init__(None, model_cfg)

    def on_step(self, layer, k, v, pos, step, probe):
        return None


class Window(CachePolicy):
    """Keeps the `capacity` most recent tokens; the local R factor uses this baseline
    to compare policy throughput."""

    name = "window"

    def on_step(self, layer, k, v, pos, step, probe):
        cap = self.layer_capacity(layer)
        if pos.shape[-1] <= cap:
            return None
        return recent_indices(pos, cap)
