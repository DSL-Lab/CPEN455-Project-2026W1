"""Course code: the grader's cache, the reading-task protocol, budget enforcement, FLOPs and metrics."""
import json
import os
import sys

import pytest
import torch

from cpen455kv import flops, metrics
from cpen455kv.cache import BudgetViolation
from cpen455kv.data import answer_distance, encoder, load_item, load_split
from cpen455kv.env import data_dir
from cpen455kv.harness import PREFILL, evaluate_batch, evaluate_quality
from cpen455kv.policy_base import CachePolicy, FullCache, Window
from cpen455kv.timing import decode_benchmark


def test_full_cache_matches_plain_forward_with_padding(tiny_model, story_items):
    """Streaming + answering with the full cache = one plain forward over context + question."""
    batch = story_items[:4]  # different context lengths -> left padding
    out = evaluate_batch(tiny_model, batch, FullCache(None, tiny_model.config), "cpu", torch.float32)
    for r, it in zip(out, batch):
        assert r["ctx_len"] == len(it["ctx"])
        for q, rq in zip(it["questions"], r["questions"]):
            for o, ll in zip(q["options"], rq["scores"]):
                seq = torch.tensor([it["ctx"] + o])
                with torch.no_grad():
                    logits, _ = tiny_model(seq[:, :-1], targets=seq[:, 1:])
                n = len(it["ctx"])
                ref = torch.log_softmax(logits[0], -1)[n - 1:].gather(1, seq[0, n:, None]).sum()
                assert ll == pytest.approx(ref.item(), abs=1e-3)
            assert rq["pred"] == max(range(len(rq["scores"])), key=rq["scores"].__getitem__)
        # the streamed context tokens get the plain forward's loss too
        seq = torch.tensor([it["ctx"]])
        with torch.no_grad():
            logits, _ = tiny_model(seq[:, :-1], targets=seq[:, 1:])
        ref = -torch.log_softmax(logits[0], -1).gather(1, seq[0, 1:, None])[:, 0]
        torch.testing.assert_close(r["token_nll"], ref[r["token_pos"] - 1], atol=1e-4, rtol=1e-4)
        assert r["token_pos"].min() >= 1 and r["token_pos"].max() == len(it["ctx"]) - 1


def test_open_question_is_exact_match(tiny_model, story_items):
    """options = None: correct only if every answer token is the top-1 prediction."""
    it = dict(story_items[0])
    with torch.no_grad():
        logits, _ = tiny_model(torch.tensor([it["ctx"]]))
    top = logits[0, -1].argmax().item()
    word = encoder().decode([top])
    q = dict(options=[[top]], answer=0, open=True, kind="final word", distance=None)
    it["questions"] = [q]
    r = evaluate_batch(tiny_model, [it], FullCache(None, tiny_model.config), "cpu", torch.float32)[0]
    assert r["questions"][0]["correct"], word
    q["options"] = [[(top + 1) % 50257]]
    r = evaluate_batch(tiny_model, [it], FullCache(None, tiny_model.config), "cpu", torch.float32)[0]
    assert not r["questions"][0]["correct"]


class TooMany(CachePolicy):
    def on_step(self, layer, k, v, pos, step, probe):
        return torch.arange(pos.shape[-1]).expand(*pos.shape) if pos.shape[-1] > self.capacity else None


class Lazy(CachePolicy):
    def on_step(self, layer, k, v, pos, step, probe):
        return None


class Duplicates(CachePolicy):
    def on_step(self, layer, k, v, pos, step, probe):
        if pos.shape[-1] <= self.capacity:
            return None
        return torch.zeros(*pos.shape[:2], self.capacity, dtype=torch.long)


@pytest.mark.parametrize("cls", [TooMany, Lazy, Duplicates])
def test_budget_violations_are_caught(tiny_model, story_items, cls):
    with pytest.raises(BudgetViolation):
        evaluate_batch(tiny_model, story_items[:2], cls(32, tiny_model.config), "cpu", torch.float32)


class Eager(CachePolicy):
    """Evicts before the cache is full, which the harness forbids."""

    def on_step(self, layer, k, v, pos, step, probe):
        return torch.arange(pos.shape[-1] - 1).expand(*pos.shape[:2], -1) if pos.shape[-1] > 4 else None


def test_eviction_within_capacity_is_a_violation(tiny_model, story_items):
    with pytest.raises(BudgetViolation):
        evaluate_batch(tiny_model, story_items[:2], Eager(64, tiny_model.config), "cpu", torch.float32)


def test_questions_are_read_without_eviction(tiny_model, story_items):
    """The policy only acts while the context streams in; answering never calls it."""
    calls = []

    class Spy(Window):
        def on_step(self, layer, k, v, pos, step, probe):
            calls.append(pos.shape[-1])
            return super().on_step(layer, k, v, pos, step, probe)

    batch = story_items[:2]
    evaluate_batch(tiny_model, batch, Spy(48, tiny_model.config), "cpu", torch.float32)
    P = max(len(it["ctx"]) for it in batch)
    assert len(calls) == tiny_model.config.n_layer * (P - PREFILL + 1) and max(calls) == 49


def test_quality_of_full_is_one(tiny_model, story_items):
    st = evaluate_quality(tiny_model, story_items[:4], lambda: FullCache(None, tiny_model.config), 4, "cpu", torch.float32)
    q = metrics.quality(st, st, 0)
    assert q["Q"] == pytest.approx(1.0) and q["inactive_mean_dnll"] == 0 and q["dNLL"] == 0


def test_quality_counts_only_active_questions(tiny_model, story_items):
    """Before the first eviction a policy must reproduce the full cache exactly; Q only looks after."""
    cfg, items = tiny_model.config, story_items[:8]
    budget = sorted(len(it["ctx"]) for it in items)[3]  # about half the contexts fit
    full = evaluate_quality(tiny_model, items, lambda: FullCache(None, cfg), 4, "cpu", torch.float32)
    win = evaluate_quality(tiny_model, items, lambda: Window(budget, cfg), 4, "cpu", torch.float32)
    start = metrics.active_from(Window(budget, cfg))
    assert start == budget + 2
    q = metrics.quality(full, win, start)
    assert q["inactive_mean_dnll"] < 1e-4 and q["inactive_max_dnll"] < 1e-4
    n_active = sum(len(it["questions"]) for it in items if len(it["ctx"]) >= start)
    assert q["n_active"] == n_active > 0
    for r, f, it in zip(win, full, items):  # questions answered from the full context are unchanged
        if len(it["ctx"]) < start:
            assert [x["pred"] for x in r["questions"]] == [x["pred"] for x in f["questions"]]
    assert q["Q"] <= 1.0 and q["Q_ci95"][0] <= q["Q"] <= q["Q_ci95"][1]


class Uneven(Window):
    """A window with layer-wise capacities: 16 slots fewer in even layers, 16 more in odd ones."""

    def layer_capacity(self, layer):
        return self.capacity + (16 if layer % 2 else -16)


def test_scored_questions_do_not_depend_on_layer_capacities(tiny_model, story_items):
    """Smaller layer capacities must not add shorter (easier) contexts to a policy's scored set."""
    cfg, items = tiny_model.config, story_items[:8]
    budget = sorted(len(it["ctx"]) for it in items)[3] + 1  # one context sits just below budget + 2
    pol = Uneven(budget, cfg)
    assert metrics.active_from(pol) == budget - 14 and metrics.scored_from(pol) == budget + 2
    full = evaluate_quality(tiny_model, items, lambda: FullCache(None, cfg), 4, "cpu", torch.float32)
    st = evaluate_quality(tiny_model, items, lambda: Uneven(budget, cfg), 4, "cpu", torch.float32)
    q = metrics.quality(full, st, metrics.active_from(pol), metrics.scored_from(pol))
    assert q["n_active"] == sum(len(it["questions"]) for it in items if len(it["ctx"]) >= budget + 2)
    assert q["n_active"] < sum(len(it["questions"]) for it in items if len(it["ctx"]) >= budget - 14)


def test_window_loses_accuracy(gpt2_model, story_items):
    items = story_items[:6]
    full = evaluate_quality(gpt2_model, items, lambda: FullCache(None, gpt2_model.config), 6, "cpu", torch.float32)
    win = evaluate_quality(gpt2_model, items, lambda: Window(32, gpt2_model.config), 6, "cpu", torch.float32)
    q = metrics.quality(full, win, 34)
    assert q["Acc_full"] >= 0.7 and q["Q"] < 0.6


def test_cbt_answer_distance_points_at_last_mention():
    enc = encoder()
    for r in [json.loads(line) for line in open(os.path.join(data_dir(), "cbt_ne_val.jsonl"))][:50]:
        it = load_item(r)
        for q, raw in zip(it["questions"], r["questions"]):
            d = q["distance"]
            if d is None:
                assert raw["answer"] not in r["context"]
                continue
            assert 0 <= d < len(it["ctx"])
    assert len(load_split("cbt-test", limit=3)) == 3


@pytest.mark.parametrize("text,answer,remaining", [
    ("hello Alice", "Alice", ""),
    ("Alice went home", "Alice", " went home"),
    ("Marie Curie met Marie Curie.", "Marie Curie", "."),
    ("Zoë spoke to Zoë.", "Zoë", "."),
    ("Nag waited. Nagaina left.", "Nag", " waited. Nagaina left."),
    ("Draken came. Drakenflesh!", "Draken", " came. Drakenflesh!"),
    ("Annabel walked home", "Ann", None),
    ("Billy met Billings", "Bill", None),
])
def test_answer_distance_counts_tokens_after_complete_last_mention(text, answer, remaining):
    enc = encoder()
    expected = None if remaining is None else len(enc.encode(remaining))
    assert answer_distance(enc.encode(text), answer) == expected


def test_grade_rejects_student_checkpoint(monkeypatch, capsys):
    from cpen455kv import grade

    monkeypatch.setattr(sys, "argv", ["grade", "--student-ckpt", "weights.pt"])
    with pytest.raises(SystemExit) as exc:
        grade.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --student-ckpt" in capsys.readouterr().err


def test_capacity_for():
    from cpen455kv.policy_base import capacity_for
    assert capacity_for(128) == 128
    assert capacity_for(128, kv_bits=4, residual=32, key_group=32) == 32 + int(96 * 16 / 4.75)


class PeriodicWorkWindow(Window):
    """Exercise policy FLOP counting without implementing an eviction strategy."""

    interval = 4

    def reset(self, batch_size, device):
        self.work_left = torch.ones(batch_size, 7, 11, device=device)
        self.work_right = torch.ones(batch_size, 11, 5, device=device)

    def on_step(self, layer, k, v, pos, step, probe):
        if step > 0 and (step - 1) % self.interval == 0:
            torch.bmm(self.work_left, self.work_right)
        return super().on_step(layer, k, v, pos, step, probe)


def test_flop_counter_matches_formula_and_splits_policy(tiny_model, story_items):
    cfg = tiny_model.config
    r = decode_benchmark(tiny_model, story_items, lambda: FullCache(None, cfg), 2, "cpu", torch.float32,
                         prompt_len=16, warmup=2, steps=4, repeats=1)
    assert r["policy_gflops_per_token"] == 0
    assert r["gflops_per_token"] == pytest.approx(r["analytic_gflops_per_token"], rel=1e-3)
    v = decode_benchmark(tiny_model, story_items, lambda: Window(12, cfg), 2, "cpu", torch.float32,
                         prompt_len=16, warmup=2, steps=4, repeats=1)
    assert v["gflops_per_token"] == pytest.approx(flops.cached_step_flops(cfg, 13) / 1e9, rel=1e-3)
    assert v["gflops_per_token"] < r["gflops_per_token"]
    h = decode_benchmark(tiny_model, story_items, lambda: PeriodicWorkWindow(12, cfg),
                         2, "cpu", torch.float32, prompt_len=16, warmup=2, steps=4, repeats=1)
    # One synthetic [7, 11] @ [11, 5] multiply per layer in four timed steps.
    work_per_token = 2 * cfg.n_layer * 7 * 11 * 5 / 4
    assert h["policy_gflops_per_token"] == pytest.approx(work_per_token / 1e9, rel=1e-6)


def test_flop_counter_covers_periodic_work_in_the_timed_interval():
    """An interval-32 policy must not be charged twice for a 16-step FLOP sample."""
    from nanogpt.model import GPT, GPTConfig

    class WorkEvery32(PeriodicWorkWindow):
        interval = 32

    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=16, vocab_size=64, block_size=1024, bias=True, dropout=0.0)
    model = GPT(cfg).eval()
    # Start timing after the cache reaches steady state; 64 timed steps contain
    # exactly two workloads. A 16-step sample containing one would double its cost.
    result = decode_benchmark(model, [{"ctx": [1] * 1024}], lambda: WorkEvery32(256, cfg),
                              1, "cpu", torch.float32, prompt_len=128, warmup=160, steps=64, repeats=1)
    expected = 2 * cfg.n_layer * 7 * 11 * 5 / 32
    assert result["policy_gflops_per_token"] == pytest.approx(expected / 1e9, rel=1e-6)
    assert result["model_gflops_per_token"] == pytest.approx(result["analytic_gflops_per_token"], rel=1e-6)
    assert len(result["times_s"]) == 1
