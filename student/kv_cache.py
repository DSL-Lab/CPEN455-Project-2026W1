"""Part 2: build the KV cache that nanoGPT does not have.

nanoGPT's `GPT.generate` re-runs the whole prefix through the model for every new token.
Here you add a cache so that each step only processes the new token, without changing
nanoGPT's model.py: the functions below reuse its modules (wte, wpe, ln_1, c_attn, ...).

Fill in every TODO. The code that is already written contains MUTANTS: small deliberate
bugs (see README, "Mutant hunt"). Find them, fix them, and log each one in
student/MUTANTS.md. The public tests expose only some of them.

Run the tests with:  python -m pytest tests/test_kv_cache.py -q
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right  # noqa: F401


class KVCache:
    """Per-layer keys and values for a batch, preallocated to `max_len` tokens.

    k[layer], v[layer]: [B, H, max_len, D]. The first `self.n` slots are in use.
    `self.pos` is the absolute position of the next token. It equals `self.n` until tokens
    are evicted (generate_window), after which it keeps growing while `self.n` does not."""

    def __init__(self, n_layer, batch_size, n_head, head_dim, max_len, device, dtype):
        shape = (batch_size, n_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.n = 0
        self.pos = 0

    def append(self, layer, k, v):
        """Store k, v ([B, H, T, D]) for T new tokens after the `self.n` cached ones and return
        (K, V) holding all tokens of this layer, [B, H, n + T, D]. Do not change self.n here:
        `advance` does that once, after the last layer."""
        T = k.shape[2]
        self.k[layer][:, :, self.n:self.n + T] = k
        self.v[layer][:, :, self.n:self.n + T] = v
        return self.k[layer][:, :, :self.n + T - 1], self.v[layer][:, :, :self.n + T - 1]

    def advance(self, T):
        self.n += T
        self.pos += T

    def evict_to(self, window):
        """Keep only the `window` most recent tokens in every layer (used by generate_window)."""
        # TODO (Part 2.5)
        raise NotImplementedError


def cached_attention(attn, x, cache, layer):
    """nanoGPT's CausalSelfAttention.forward, but attending over the cache.

    attn: a nanoGPT CausalSelfAttention module; x: [B, T, C] for the T new tokens."""
    B, T, C = x.size()
    q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)
    k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
    K, V = cache.append(layer, k, v)

    # same attention call as nanoGPT's original forward
    y = F.scaled_dot_product_attention(q, K, V, is_causal=True)
    y = y.contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    return attn.c_proj(y)


def forward_with_cache(model, idx, cache):
    """Like GPT.forward for the T new tokens idx [B, T] only. Returns logits [B, T, vocab]."""
    tr = model.transformer
    T = idx.shape[1]
    assert cache.pos + T <= model.config.block_size, "GPT-2 has only 1,024 position embeddings"
    pos = torch.arange(cache.n, cache.n + T, dtype=torch.long, device=idx.device)
    x = tr.wte(idx) + tr.wpe(pos)
    for layer, block in enumerate(tr.h):
        x = x + cached_attention(block.attn, block.ln_1(x), cache, layer)
        x = x + block.mlp(block.ln_2(x))
    cache.advance(T)
    return model.lm_head(x)


def _next_token(logits, greedy, temperature, top_k):
    """Pick the next token from logits [B, vocab]: argmax, or a sample at `temperature`
    from the `top_k` most likely tokens (all tokens if top_k is None)."""
    if greedy:
        return logits.argmax(-1, keepdim=True)
    logits = logits * temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [0]]] = -float("Inf")
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def new_cache(model, batch_size, max_len, device, dtype):
    cfg = model.config
    return KVCache(cfg.n_layer, batch_size, cfg.n_head, cfg.n_embd // cfg.n_head, max_len, device, dtype)


@torch.no_grad()
def generate_with_cache(model, idx, max_new_tokens, greedy=True, temperature=1.0, top_k=None):
    """Prefill the prompt idx [B, T0] once, then feed one token per step.
    Returns [B, T0 + max_new_tokens], like GPT.generate. Use _next_token to pick tokens."""
    # TODO (Part 2.3)
    raise NotImplementedError


@torch.no_grad()
def generate_window(model, idx, max_new_tokens, window=256, greedy=True):
    """generate_with_cache, but the cache keeps only the `window` most recent tokens.
    New tokens keep their true absolute positions (cache.pos), not the cache length."""
    # TODO (Part 2.5)
    raise NotImplementedError
