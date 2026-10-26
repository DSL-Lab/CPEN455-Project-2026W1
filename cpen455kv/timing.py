"""Throughput, latency, GFLOPs and peak memory of a policy, measured by the harness.

Protocol: a fixed batch of story sequences, prefill `prompt_len` tokens, `warmup` untimed
decoding steps, then `steps` timed teacher-forced decoding steps with the device synchronised
around the timed region. Median of `repeats` runs. The policy runs inside the timed region,
so slow scoring code costs throughput. GFLOPs come from a separate instrumented replay of
the complete timed interval, so periodic policy work is averaged over the same steps.
"""
import statistics
import time

import torch

from . import flops
from .cache import BudgetedKVCache, forward_cached
from .data import EOT
from .env import peak_memory_gib, reset_peak_memory, synchronize


def fixed_sequences(items, batch, length):
    """[batch, length] token ids: row b is the context of item b, followed by the contexts of
    the next items (separated by EOT) until `length` tokens, cycling through items."""
    out = torch.empty((batch, length), dtype=torch.long)
    for b in range(batch):
        seq, j = [], b
        while len(seq) < length:
            seq += items[j % len(items)]["ctx"] + [EOT]
            j += 1
        out[b] = torch.tensor(seq[:length])
    return out


@torch.no_grad()
def decode_benchmark(model, items, make_policy, batch, device, dtype,
                     prompt_len=128, warmup=32, steps=512, repeats=3):
    total = prompt_len + warmup + steps
    assert steps > 0 and repeats > 0, "benchmark steps and repeats must be positive"
    assert total <= model.config.block_size, "benchmark sequence longer than GPT-2's context"
    seq = fixed_sequences(items, batch, total).to(device)
    pos = torch.arange(total, device=device)[None].expand(batch, -1)

    def run(timed_steps, count_flops=False):
        policy = make_policy()
        policy.reset(batch, device)
        cache = BudgetedKVCache(model.config, policy, batch, prompt_len, total, device, dtype, strict=False)
        forward_cached(model, cache, seq[:, :prompt_len], pos[:, :prompt_len], 0, prefill=True)
        t = prompt_len
        for i in range(warmup):
            forward_cached(model, cache, seq[:, t:t + 1], pos[:, t:t + 1], i + 1, prefill=False)
            t += 1
        if not count_flops:
            synchronize(device)
            t0 = time.perf_counter()
            for i in range(timed_steps):
                forward_cached(model, cache, seq[:, t:t + 1], pos[:, t:t + 1], warmup + i + 1, prefill=False)
                t += 1
            synchronize(device)
            dt = time.perf_counter() - t0
            return dt, None
        # Replay exactly the measured positions from a fresh policy/cache. Counting only
        # a short tail can miss or double periodic work (for example interval-32 eviction).
        # Keep this pass separate so instrumentation cannot affect reported throughput.
        outer = flops.counter()
        cache.count_policy_flops, cache.policy_flops, cache.attn_flops = True, 0, 0
        lens = []
        with outer:
            for i in range(timed_steps):
                lens.append(cache.n[0] + 1)  # keys attended by layer 0 in this step
                forward_cached(model, cache, seq[:, t:t + 1], pos[:, t:t + 1], warmup + i + 1, prefill=False)
                t += 1
        per_tok = batch * timed_steps * 1e9
        total_fl = flops.non_sdpa_flops(outer) + cache.attn_flops
        analytic = sum(flops.cached_step_flops(model.config, L) for L in lens) / timed_steps / 1e9
        kv_gib = sum(t.numel() * t.element_size() for t in cache.k + cache.v) / 2**30
        return None, (total_fl / per_tok, cache.policy_flops / per_tok, analytic, max(cache.n), kv_gib)

    times = []
    reset_peak_memory(device)
    for _ in range(repeats):
        dt, _ = run(steps)
        times.append(dt)
    _, fl = run(steps, count_flops=True)
    gflops, policy_gflops, analytic, cache_len, kv_gib = fl
    weights_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    med = statistics.median(times)
    return dict(
        tokens_per_s=batch * steps / med, ms_per_step=1e3 * med / steps,
        gflops_per_token=gflops, policy_gflops_per_token=policy_gflops,
        model_gflops_per_token=gflops - policy_gflops, cache_len=cache_len,
        analytic_gflops_per_token=analytic,
        peak_mem_gib=peak_memory_gib(device), kv_buffer_gib=kv_gib, weights_gib=weights_gib, times_s=times,
    )


@torch.no_grad()
def vanilla_benchmark(model, items, batch, device, prompt_len=128, warmup=32, steps=512, repeats=3, sample_steps=4):
    """nanoGPT's own generate step (full forward over the prefix, no cache).

    It is far too slow to run all `steps`, so `sample_steps` steps are timed at the middle
    context length of the cached benchmark and FLOPs are counted at its final context length."""
    mid = prompt_len + warmup + steps // 2
    end = prompt_len + warmup + steps
    seq = fixed_sequences(items, batch, end).to(device)
    model(seq[:, :mid])  # warm-up
    times = []
    reset_peak_memory(device)
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(sample_steps):
            model(seq[:, :mid])
        synchronize(device)
        times.append(time.perf_counter() - t0)
    with flops.counter() as fc:
        model(seq[:, :end])
    cfg = model.config
    attn = cfg.n_layer * flops.sdpa_flops(batch, cfg.n_head, end, end, cfg.n_embd // cfg.n_head)
    med = statistics.median(times)
    return dict(
        tokens_per_s=batch * sample_steps / med, ms_per_step=1e3 * med / sample_steps,
        gflops_per_token=(flops.non_sdpa_flops(fc) + attn) / batch / 1e9, policy_gflops_per_token=0.0,
        analytic_gflops_per_token=flops.vanilla_step_flops(model.config, end) / 1e9,
        peak_mem_gib=peak_memory_gib(device), context=mid, times_s=times,
    )
