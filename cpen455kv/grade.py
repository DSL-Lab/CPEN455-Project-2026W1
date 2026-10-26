"""Measure a policy locally and write a result JSON.

    python -m cpen455kv.grade --strict-env --split cbt-test --out validation.json   # submitted validation run
    python -m cpen455kv.grade --split cbt-val --limit 64 --skip-timing              # quick quality check

The policy is `CPEN455_POLICY` (default MyPolicy) from student/policy.py, configured by the
matching entry of `CPEN455_POLICY_CONFIG` (default student/config.yaml). Comparison rows
(vanilla nanoGPT, full cache, window, StreamingLLM, H2O) are computed in the same run.
StreamingLLM and H2O use the student implementations by default.

Every row uses the same frozen base model. `--ckpt` selects that model for local development;
use the course checkpoint for submitted validation. The reported Q, R, Score, loss differences
and FLOPs are local measurements, not course marks. The leaderboard benchmark is TBA.
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import time
import traceback

import torch
import yaml

from . import env, metrics, timing
from .cache import BudgetViolation
from .data import load_split
from .harness import evaluate_quality
from .policy_base import FullCache, Window
from .weights import load_model, sha256


def load_policy_classes(module_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, "POLICIES", {})


def load_config(path):
    if path and os.path.exists(path):
        return yaml.safe_load(open(path)) or {}
    return {}


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=env.REPO_ROOT, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="cbt-test", help="cbt-val | cbt-test | lambada | path to a jsonl file")
    ap.add_argument("--limit", type=int, default=0, help="number of items (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed before --limit")
    ap.add_argument("--seed-file", default=None, help="read the shuffle seed from a file")
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--policy", default=os.environ.get("CPEN455_POLICY", "MyPolicy"))
    ap.add_argument("--config", default=os.environ.get("CPEN455_POLICY_CONFIG",
                                                       os.path.join(env.REPO_ROOT, "student", "config.yaml")))
    ap.add_argument("--policy-module", default="student.policy")
    ap.add_argument("--reference-module", default=os.environ.get("CPEN455_REFERENCE_MODULE", "student.policy"),
                    help="module containing the StreamingLLM and H2O comparison policies")
    ap.add_argument("--reference-config", default=None,
                    help="settings of the comparison rows (default: `references:` in --config)")
    ap.add_argument("--ckpt", default=None, help="base model (default: $CPEN455_CKPT)")
    ap.add_argument("--quality-batch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64, help="batch size for throughput")
    ap.add_argument("--steps", type=int, default=512, help="timed decoding steps")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-timing", action="store_true", help="quality only (R is then reported as 1)")
    ap.add_argument("--strict-env", action="store_true", default=None,
                    help="reject package/environment drift (also CPEN455_STRICT_ENV=1)")
    ap.add_argument("--out", default="validation.json")
    args = ap.parse_args()

    env.check_env(strict=args.strict_env)
    torch.manual_seed(0)
    device = env.get_device()
    dtype = env.get_dtype(device)
    ckpt = args.ckpt or env.default_ckpt()
    model = load_model(ckpt, device, dtype)
    cfg = model.config
    ckpt_hash = sha256(ckpt)
    seed = int(open(args.seed_file).read().strip()) if args.seed_file else args.seed
    items = load_split(args.split, limit=args.limit or None, seed=seed)
    n_q = sum(len(it["questions"]) for it in items)
    print(f"device={device} dtype={dtype} ckpt={os.path.basename(ckpt)} split={args.split} "
          f"items={len(items)} questions={n_q} budgets={args.budgets}")

    conf = load_config(args.config)
    student_cls = load_policy_classes(args.policy_module)[args.policy]
    refs = load_policy_classes(args.reference_module)

    # rows: name -> (class, kwargs). The student's row is last.
    ref_conf = (load_config(args.reference_config) if args.reference_config else conf).get("references") or {}
    rows = {"window": (Window, {})}
    for ref in ("StreamingLLM", "H2O"):
        if ref in refs:
            rows[ref] = (refs[ref], ref_conf.get(ref, {}))
    rows["student"] = (student_cls, (conf.get("policies") or {}).get(args.policy, {}))

    result = dict(
        meta=dict(policy=args.policy, policy_config=rows["student"][1], split=args.split, n_items=len(items),
                  reference_config={n: kw for n, (_, kw) in rows.items() if n in ("StreamingLLM", "H2O")},
                  n_questions=n_q,
                  seed=seed, budgets=args.budgets, device=device, dtype=str(dtype),
                  gpu=torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                  torch=torch.__version__, ckpt=os.path.basename(ckpt), ckpt_sha256=ckpt_hash,
                  harness_git=git_hash(), environment=env.runtime_metadata(ckpt, ckpt_hash),
                  time=time.strftime("%Y-%m-%d %H:%M:%S")),
        rows={},
    )

    def run_quality(name, make):
        t0 = time.time()
        try:
            st = evaluate_quality(model, items, make, args.quality_batch, device, dtype, strict=True, verbose=True)
            return st, None
        except NotImplementedError:
            return None, "not implemented"
        except BudgetViolation as e:
            return None, f"budget violation: {e}"
        except Exception:
            return None, traceback.format_exc(limit=3)
        finally:
            print(f"  quality {name:14s} {time.time() - t0:6.0f}s")

    def run_timing(make, batch, steps):
        return timing.decode_benchmark(model, items, make, batch, device, dtype,
                                       steps=steps, repeats=args.repeats)

    print("full cache (quality reference)")
    full, err = run_quality("full", lambda: FullCache(None, cfg))
    assert err is None, err
    result["full_cache_accuracy"] = sum(q["correct"] for r in full for q in r["questions"]) / n_q

    for name, (cls, kw) in rows.items():
        result["rows"][name] = {"budgets": {}, "error": None}
        for B in args.budgets:
            make = (lambda cls=cls, kw=kw, B=B: cls(B, cfg, **kw))
            st, err = run_quality(f"{name}@{B}", make)
            if err:
                result["rows"][name]["error"] = err
                print(f"  {name}: {err.splitlines()[0]}")
                break
            q = metrics.quality(full, st, metrics.active_from(make()), metrics.scored_from(make()))
            result["rows"][name]["budgets"][B] = q

    if not args.skip_timing:
        print("timing (throughput at batch %d, latency at batch 1)" % args.batch)
        run_timing(lambda: Window(args.budgets[0], cfg), args.batch, min(64, args.steps))  # GPU warm-up, discarded
        t_full = run_timing(lambda: FullCache(None, cfg), args.batch, args.steps)
        result["full_cache"] = dict(throughput=t_full, latency=run_timing(lambda: FullCache(None, cfg), 1, args.steps))
        result["vanilla"] = timing.vanilla_benchmark(model, items, args.batch, device, steps=args.steps,
                                                     repeats=args.repeats)
        for name, (cls, kw) in rows.items():
            if result["rows"][name]["error"]:
                continue
            for B in args.budgets:
                make = (lambda cls=cls, kw=kw, B=B: cls(B, cfg, **kw))
                try:
                    thr, lat = run_timing(make, args.batch, args.steps), run_timing(make, 1, args.steps)
                except Exception as e:
                    result["rows"][name]["error"] = f"timing failed: {e!r}"
                    break
                result["rows"][name]["budgets"][B].update(throughput=thr, latency=lat)
                print(f"  timing {name}@{B}: {thr['tokens_per_s']:.0f} tok/s, {lat['ms_per_step']:.2f} ms/token")

    # ---- Local R and Score ------------------------------------------------------------------
    for name, row in result["rows"].items():
        row["score"] = None
        if row["error"]:
            continue
        for B, b in row["budgets"].items():
            if args.skip_timing:
                b["R"] = 1.0
                continue
            window = result["rows"]["window"]
            if window["error"]:
                row["error"] = "R and Score unavailable: window comparison failed"
                break
            win = window["budgets"][B]["throughput"]["tokens_per_s"]
            b["R"] = metrics.overhead_factor(b["throughput"]["tokens_per_s"], win)
        if not row["error"]:
            row["score"] = metrics.score(row["budgets"])

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print_table(result, args)
    print(f"wrote {args.out}")


def print_table(result, args):
    print(f"\nfull-cache accuracy {result['full_cache_accuracy']:.3f}. Acc = accuracy on the questions whose context "
          "has at least B + 2 tokens; Q = Acc / full-cache Acc on the same questions (1 = lossless)")
    print(f"{'row':14s} " + " ".join(f"Acc@{B:<5d}" for B in args.budgets) + " " +
          " ".join(f"Q@{B:<5d}" for B in args.budgets) + "  " +
          ("" if args.skip_timing else " ".join(f"R@{B:<5d}" for B in args.budgets)) + "  Local Score")
    for name, row in result["rows"].items():
        if row["error"]:
            print(f"{name:14s} error: {row['error'].splitlines()[0]}")
            continue
        if len(row["budgets"]) < len(args.budgets):
            continue
        bs = [row["budgets"][B] for B in args.budgets]
        accs = " ".join(f"{b['Acc']:.3f}   " if b["Acc"] is not None else "  -      " for b in bs)
        qs = " ".join(f"{b['Q']:.3f} " for b in bs)
        rs = "" if args.skip_timing else " ".join(f"{b.get('R', float('nan')):.3f} " for b in bs)
        print(f"{name:14s} {accs} {qs}  {rs}  {row['score']:.3f}")

if __name__ == "__main__":
    main()
