"""Reject incomplete validation evidence and accept the harness's emitted result schema."""
import copy
import json
import sys

import pytest
import torch

from cpen455kv import check, env, grade
from cpen455kv.policy_base import Window
from nanogpt.model import GPT, GPTConfig


@pytest.fixture
def validation_file(tmp_path):
    (tmp_path / "student").mkdir()
    (tmp_path / "student/config.yaml").write_text("policies: {MyPolicy: {}}\n")
    with open(env.data_dir() + "/cbt_ne_test.jsonl") as f:
        n_items = sum(1 for _ in f)
    timing = dict(tokens_per_s=10.0, ms_per_step=100.0, gflops_per_token=0.1, times_s=[1.0])
    row = dict(Q=0.8, R=1.0, Acc=0.4, Acc_full=0.5, n_active=100,
               inactive_mean_dnll=0.0, inactive_max_dnll=0.0,
               throughput=timing, latency=copy.deepcopy(timing))
    result = dict(meta=dict(split="cbt-test", n_items=n_items, seed=None, policy="MyPolicy",
                            policy_config={}, budgets=[64, 128, 256]),
                  rows=dict(student=dict(error=None, budgets={str(b): copy.deepcopy(row) for b in check.BUDGETS})))
    return tmp_path, result


def validation_errors(root, result):
    (root / "validation.json").write_text(json.dumps(result))
    errors = []
    check.check_validation(str(root), errors)
    return errors


@pytest.mark.parametrize("result", [None, [], True, {}, {"meta": None, "rows": {}}])
def test_validation_rejects_malformed_top_level(validation_file, result):
    root, _ = validation_file
    assert validation_errors(root, result)


@pytest.mark.parametrize("budget", ["64", "128", "256"])
def test_validation_requires_result_object_at_each_budget(validation_file, budget):
    root, result = validation_file
    result["rows"]["student"]["budgets"][budget] = None
    assert any("not a result object" in e for e in validation_errors(root, result))


@pytest.mark.parametrize("metric", [
    "Q", "R", "Acc", "Acc_full", "n_active", "inactive_mean_dnll", "inactive_max_dnll",
])
def test_validation_requires_quality_evidence(validation_file, metric):
    root, result = validation_file
    del result["rows"]["student"]["budgets"]["128"][metric]
    assert validation_errors(root, result)


@pytest.mark.parametrize("metric,value", [
    ("Acc", None), ("Acc_full", float("nan")), ("Acc_full", 1.1),
    ("n_active", 0), ("n_active", True), ("n_active", 1.5),
    ("inactive_mean_dnll", float("inf")), ("inactive_max_dnll", -1),
])
def test_validation_rejects_invalid_quality_evidence(validation_file, metric, value):
    root, result = validation_file
    result["rows"]["student"]["budgets"]["256"][metric] = value
    assert validation_errors(root, result)


@pytest.mark.parametrize("target", ["Q", "gflops_per_token", "times_s"])
def test_validation_reports_huge_json_numbers_without_crashing(validation_file, target):
    root, result = validation_file
    row = result["rows"]["student"]["budgets"]["64"]
    huge = 10 ** 1000
    if target == "Q":
        row[target] = huge
    elif target == "times_s":
        row["throughput"][target] = [huge]
    else:
        row["throughput"][target] = huge
    assert validation_errors(root, result)


@pytest.mark.parametrize("diagnostic_shift", [0.0, 2.0])
def test_validation_accepts_actual_grade_output(tmp_path, monkeypatch, capsys, diagnostic_shift):
    """Run the real quality, metrics, timing, and JSON writer on a small CPU model."""
    (tmp_path / "student").mkdir()
    config = tmp_path / "student/config.yaml"
    config.write_text("policies: {MyPolicy: {}}\n")
    # The checker and harness see the same one-item test split. Context length makes
    # the question active at every required budget, without needing GPT-2 weights.
    (tmp_path / "cbt_ne_test.jsonl").write_text("{}\n")
    item = dict(id="test-story", ctx=[1] * 260,
                questions=[dict(options=[[1], [2]], answer=0, kind="cbt", distance=10, open=False)])
    model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8, vocab_size=32,
                          block_size=1024, bias=True, dropout=0.0)).eval()
    monkeypatch.setattr(check, "data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(grade, "load_split", lambda *args, **kwargs: [item])
    monkeypatch.setattr(grade, "load_model", lambda *args: model)
    class StudentWindow(Window):
        pass

    monkeypatch.setattr(grade, "load_policy_classes", lambda name: {"MyPolicy": StudentWindow})
    monkeypatch.setattr(grade, "sha256", lambda path: "test-model")
    monkeypatch.setattr(grade, "git_hash", lambda: "test-revision")
    monkeypatch.setattr(env, "check_env", lambda **kwargs: [])
    monkeypatch.setattr(env, "get_device", lambda: "cpu")
    monkeypatch.setattr(env, "runtime_metadata", lambda *args: {})
    evaluate_quality = grade.evaluate_quality
    decode_benchmark = grade.timing.decode_benchmark

    def measured_quality(model, items, make, *args, **kwargs):
        measurements = evaluate_quality(model, items, make, *args, **kwargs)
        if isinstance(make(), StudentWindow):
            for item in measurements:
                item["token_nll"] = item["token_nll"] + diagnostic_shift
        return measurements

    def measured_timing(model, items, make, *args, **kwargs):
        measurements = decode_benchmark(model, items, make, *args, **kwargs)
        if isinstance(make(), StudentWindow):
            measurements["gflops_per_token"] += diagnostic_shift
        return measurements

    monkeypatch.setattr(grade, "evaluate_quality", measured_quality)
    monkeypatch.setattr(grade.timing, "decode_benchmark", measured_timing)
    monkeypatch.setattr(sys, "argv", ["grade", "--ckpt", "test-model.pt", "--config", str(config),
                                      "--batch", "1", "--steps", "1", "--repeats", "1",
                                      "--out", str(tmp_path / "validation.json")])
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        grade.main()
    finally:
        torch.set_num_threads(previous_threads)
    errors = []
    check.check_validation(str(tmp_path), errors)
    assert errors == []
    result = json.loads((tmp_path / "validation.json").read_text())
    row = result["rows"]["student"]
    assert row["error"] is None and 0 <= row["score"] <= 1
    assert "valid" not in row
    for measurement in row["budgets"].values():
        assert {"inactive_ok", "flops_gate"}.isdisjoint(measurement)
        assert measurement["inactive_mean_dnll"] == pytest.approx(diagnostic_shift, abs=1e-4)
        if diagnostic_shift:
            assert measurement["throughput"]["gflops_per_token"] > result["full_cache"]["throughput"]["gflops_per_token"]
    assert "Local Score" in capsys.readouterr().out
