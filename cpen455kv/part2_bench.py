"""Part 2.5: what does your KV cache buy and cost compared with vanilla nanoGPT?

    python -m cpen455kv.part2_bench --batches 1 16 --new 512 --out student/part2_results.json

Measures (a) vanilla `GPT.generate` (no cache), (b) your `generate_with_cache` and (c) your
`generate_window` with a 256-token window: peak memory, tokens/s and GFLOPs per new token.
Use the locked course environment and report your actual device/dtype.
"""
import argparse
import json
import time

import torch

from . import env, flops
from .data import load_split
from .timing import fixed_sequences
from .weights import load_model


@torch.no_grad()
def vanilla_generate_greedy(model, idx, max_new_tokens):
    """nanoGPT's generate with greedy decoding: a full forward over the prefix per token."""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        idx = torch.cat((idx, logits[:, -1, :].argmax(-1, keepdim=True)), dim=1)
    return idx


def measure(fn, device, *args):
    env.synchronize(device)
    env.reset_peak_memory(device)
    t0 = time.perf_counter()
    out = fn(*args)
    env.synchronize(device)
    return out, time.perf_counter() - t0, env.peak_memory_gib(device)


def count_gflops(fn, *args):
    with flops.counter() as fc:
        fn(*args)
    return fc.get_total_flops() / 1e9


def main():
    from student.kv_cache import generate_window, generate_with_cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--new", type=int, default=512)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--flops-new", type=int, default=32, help="new tokens in the FLOP-counted run")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--strict-env", action="store_true", default=None)
    ap.add_argument("--out", default="student/part2_results.json")
    args = ap.parse_args()

    env.check_env(strict=args.strict_env)
    device = env.get_device()
    dtype = env.get_dtype(device)
    ckpt = args.ckpt or env.default_ckpt()
    model = load_model(ckpt, device, dtype)
    items = load_split("cbt-val", limit=64)
    methods = {
        "vanilla": vanilla_generate_greedy,
        "full_cache": generate_with_cache,
        f"window_{args.window}": lambda m, x, n: generate_window(m, x, n, window=args.window),
    }
    weights_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    results = dict(device=device, dtype=str(dtype), prompt=args.prompt, new=args.new,
                   weights_gib=weights_gib, environment=env.runtime_metadata(ckpt), rows=[])
    for B in args.batches:
        idx = fixed_sequences(items, B, args.prompt).to(device)
        methods["full_cache"](model, idx, 4)  # warm-up
        for name, fn in methods.items():
            out, dt, peak = measure(fn, device, model, idx, args.new)
            gf = count_gflops(fn, model, idx, args.flops_new) / (B * args.flops_new)
            row = dict(batch=B, method=name, seconds=dt, tokens_per_s=B * args.new / dt,
                       peak_mem_gib=peak, mem_above_weights_gib=None if peak is None else peak - weights_gib,
                       gflops_per_token=gf, tokens=out[:, args.prompt:args.prompt + 16].tolist()[0])
            results["rows"].append(row)
            print(f"batch {B:3d} {name:12s} {dt:8.1f}s {row['tokens_per_s']:9.1f} tok/s  "
                  f"{gf:9.2f} GFLOPs/token (first {args.flops_new})  peak "
                  f"{'n/a' if peak is None else f'{peak:.2f} GiB'}")
    cfg = model.config
    results["analytic_gflops_whole_generation"] = dict(
        vanilla=flops.vanilla_generation_flops(cfg, args.prompt, args.new) / 1e9,
        full_cache=flops.cached_generation_flops(cfg, args.prompt, args.new) / 1e9)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
