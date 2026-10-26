"""FLOP counting: torch's FlopCounterMode, cross-checked against closed-form formulas.

FlopCounterMode counts scaled_dot_product_attention only for some backends (it reports 0 for
the masked CPU and MPS kernels), so the harness counts attention itself, exactly, from the
shapes it runs (BudgetedKVCache.attn_flops) and takes everything else from the counter.

    N                        = 12 · n_layer · d²            (non-embedding parameters)
    non-attention FLOPs/tok  = 2 · N + 2 · d · V            (projections, MLP, LM head)
    attention FLOPs/tok      = 4 · n_layer · L · d          (q·Kᵀ and softmax·V at L cached tokens)
"""
from torch.utils.flop_counter import FlopCounterMode


def counter():
    return FlopCounterMode(display=False)


def non_sdpa_flops(fc):
    """FLOPs recorded by a FlopCounterMode, excluding scaled_dot_product_attention."""
    counts = fc.get_flop_counts().get("Global", {})
    return sum(v for op, v in counts.items() if "scaled_dot_product" not in str(op))


def sdpa_flops(B, H, T, L, D):
    """q·Kᵀ and softmax·V for T queries against L keys: 2 matmuls of 2·T·L·D per head."""
    return 4 * B * H * T * L * D


def nonattn_flops_per_token(cfg):
    d = cfg.n_embd
    return 2 * 12 * cfg.n_layer * d * d + 2 * d * cfg.vocab_size


def attn_flops_per_token(cfg, L):
    return 4 * cfg.n_layer * L * cfg.n_embd


def cached_step_flops(cfg, L):
    """One decoding step with L cached tokens (including the new one), per sequence."""
    return nonattn_flops_per_token(cfg) + attn_flops_per_token(cfg, L)


def vanilla_step_flops(cfg, T):
    """One step of nanoGPT's generate: a full forward over T tokens, LM head on the last only.
    SDPA is counted on the full T × T score matrix, as FlopCounterMode does."""
    d = cfg.n_embd
    return T * 2 * 12 * cfg.n_layer * d * d + 4 * cfg.n_layer * T * T * d + 2 * d * cfg.vocab_size


def vanilla_generation_flops(cfg, prompt, new):
    return sum(vanilla_step_flops(cfg, prompt + i) for i in range(new))


def cached_generation_flops(cfg, prompt, new):
    prefill = vanilla_step_flops(cfg, prompt)
    return prefill + sum(cached_step_flops(cfg, prompt + i + 1) for i in range(new - 1))
