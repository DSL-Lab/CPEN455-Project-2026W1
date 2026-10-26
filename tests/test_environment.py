"""Environment drift and source-only submission checks; no model weights required."""
import hashlib
import json
import sys
import zipfile

import pytest
import torch

from cpen455kv import check, env


@pytest.fixture
def matching_environment(monkeypatch):
    for name, value in env.PINNED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(env, "get_device", lambda: "cpu")
    monkeypatch.setattr(env.sys, "version_info", (3, 11, 14))
    versions = env.expected_packages()
    versions["torch"] += "+cpu"
    monkeypatch.setattr(env, "installed_packages", lambda: versions)
    return versions


def test_strict_check_accepts_matching_environment(matching_environment):
    assert env.check_env(strict=True) == []


def test_strict_check_rejects_runtime_drift_without_container(matching_environment, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        env.check_env(strict=True)


def test_strict_check_rejects_dependency_drift(matching_environment):
    matching_environment["numpy"] = "0.0.0"
    with pytest.raises(RuntimeError, match="numpy"):
        env.check_env(strict=True)


def test_development_check_warns_on_drift(matching_environment, monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.warns(RuntimeWarning, match="PYTHONHASHSEED"):
        assert env.check_env(strict=False)


def test_strict_environment_variable_is_respected(matching_environment, monkeypatch):
    monkeypatch.setenv("CPEN455_STRICT_ENV", "1")
    monkeypatch.delenv("CPEN455_DTYPE")
    with pytest.raises(RuntimeError, match="CPEN455_DTYPE"):
        env.check_env()


def test_cuda_build_drift_is_rejected(matching_environment, monkeypatch):
    monkeypatch.setattr(env, "get_device", lambda: "cuda")
    monkeypatch.setattr(env.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(env.torch.version, "cuda", "13.0")
    with pytest.raises(RuntimeError, match="CUDA build"):
        env.check_env(strict=True)


def test_metadata_identifies_lock_and_checkpoint(matching_environment, monkeypatch, tmp_path):
    monkeypatch.setattr(env, "REPO_ROOT", str(tmp_path))
    (tmp_path / "uv.lock").write_bytes(b"lock content")
    ckpt = tmp_path / "base.pt"
    torch.save({"model_args": {"n_embd": 2}, "model": {"weight": torch.tensor([1., 2.])}}, ckpt)
    result = env.runtime_metadata(ckpt)
    assert result["uv_lock_sha256"] == hashlib.sha256(b"lock content").hexdigest()
    assert result["ckpt_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert result["ckpt_state_sha256"] == env.checkpoint_state_sha256(ckpt)
    assert result["ckpt"] == "base.pt"
    assert result["dtype"] == "torch.float32"
    assert result["packages"]["torch"].endswith("+cpu")
    assert result["environment"]["PYTHONHASHSEED"] == "0"


def test_checkpoint_fingerprint_ignores_serialization_and_key_order(tmp_path):
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    weight = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    torch.save({"model_args": {"n_embd": 3, "bias": True},
                "model": {"weight": weight, "bias": torch.zeros(3)}, "note": "first"}, first)
    torch.save({"note": "resaved", "model": {"bias": torch.zeros(3), "weight": weight.clone()},
                "model_args": {"bias": True, "n_embd": 3}}, second,
               _use_new_zipfile_serialization=False)
    assert env.file_sha256(first) != env.file_sha256(second)
    assert env.checkpoint_state_sha256(first) == env.checkpoint_state_sha256(second)


@pytest.mark.parametrize("change", ["value", "shape", "dtype", "config"])
def test_checkpoint_fingerprint_detects_model_changes(tmp_path, change):
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    config = {"n_embd": 2}
    weight = torch.tensor([1., 2.])
    torch.save({"model_args": config, "model": {"weight": weight}}, first)
    if change == "value":
        weight[0] = 3
    elif change == "shape":
        weight = weight.reshape(1, 2)
    elif change == "dtype":
        weight = weight.half()
    else:
        config["n_embd"] = 3
    torch.save({"model_args": config, "model": {"weight": weight}}, second)
    assert env.checkpoint_state_sha256(first) != env.checkpoint_state_sha256(second)


def validation_run(**meta):
    """The result JSON of a complete, timed cbt-test run of MyPolicy, as cpen455kv.grade writes it."""
    with open(env.data_dir() + "/cbt_ne_test.jsonl") as f:
        n_items = sum(1 for _ in f)
    base = dict(policy="MyPolicy", policy_config={"k": 2}, split="cbt-test", n_items=n_items, seed=None,
                budgets=[64, 128, 256])
    base.update(meta)
    timing = dict(tokens_per_s=1.0, ms_per_step=1000.0, gflops_per_token=0.1,
                  times_s=[1.0, 1.1, 0.9])
    budget = dict(Q=0.9, R=1.0, Acc=0.45, Acc_full=0.5, n_active=100,
                  inactive_mean_dnll=0.0, inactive_max_dnll=0.0,
                  throughput=timing, latency=timing)
    return json.dumps(dict(meta=base, rows={"student": dict(error=None, budgets={str(b): budget for b in (64, 128, 256)})}))


@pytest.fixture
def complete_submission(tmp_path):
    (tmp_path / "student/my_tests").mkdir(parents=True)
    files = {
        "student/kv_cache.py": "# submitted code\n",
        "student/policy.py": "# submitted code\n",
        "student/config.yaml": "policies:\n  MyPolicy: {k: 2}\n",
        "student/part1_answers.yaml": "answers: {}\n",
        "student/part2_results.json": '{"rows": [{"method": "vanilla"}]}\n',
        "student/MUTANTS.md": "Bug log\n",
        "student/my_tests/test_policy.py": "def test_example():\n    assert True\n",
        "validation.json": validation_run(),
        "report.pdf": "%PDF-test",
        "AI_USE.md": "none\n",
    }
    for filename, content in files.items():
        (tmp_path / filename).write_text(content)
    return tmp_path


def run_check(target, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check", str(target)])
    with pytest.raises(SystemExit) as exc:
        check.main()
    return exc.value.code, capsys.readouterr().out


def make_zip(root, path, names, prefix=""):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.write(root / name, prefix + name)
    return path


SUBMISSION = ["student/kv_cache.py", "student/policy.py", "student/config.yaml", "student/part1_answers.yaml",
              "student/part2_results.json", "student/MUTANTS.md", "student/my_tests/test_policy.py",
              "validation.json", "report.pdf", "AI_USE.md"]


def test_source_submission_needs_no_image_files(complete_submission, monkeypatch, capsys):
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 0, out
    assert "check passed" in out


def test_missing_required_file_fails_submission_check(complete_submission, monkeypatch, capsys):
    (complete_submission / "report.pdf").unlink()
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1
    assert "report.pdf missing" in out


def test_submission_zip_passes(complete_submission, tmp_path_factory, monkeypatch, capsys):
    path = make_zip(complete_submission, tmp_path_factory.mktemp("zip") / "submission.zip", SUBMISSION)
    code, out = run_check(path, monkeypatch, capsys)
    assert code == 0, out
    assert "ready to upload" in out


@pytest.mark.parametrize("extra,message", [
    ("notes.txt", "unexpected top-level entries in ZIP: notes.txt"),
    ("__MACOSX/report.pdf", "unexpected top-level entries in ZIP: __MACOSX"),
    ("student/__pycache__/policy.cpython-311.pyc", "remove from ZIP: student/__pycache__"),
    ("student/.DS_Store", "remove from ZIP: student/.DS_Store"),
])
def test_submission_zip_rejects_extra_files(complete_submission, tmp_path_factory, monkeypatch, capsys, extra, message):
    (complete_submission / extra).parent.mkdir(parents=True, exist_ok=True)
    (complete_submission / extra).write_text("x")
    path = make_zip(complete_submission, tmp_path_factory.mktemp("zip") / "submission.zip", SUBMISSION + [extra])
    code, out = run_check(path, monkeypatch, capsys)
    assert code == 1 and message in out


def test_submission_zip_rejects_enclosing_folder(complete_submission, tmp_path_factory, monkeypatch, capsys):
    path = make_zip(complete_submission, tmp_path_factory.mktemp("zip") / "submission.zip", SUBMISSION, "project/")
    code, out = run_check(path, monkeypatch, capsys)
    assert code == 1 and "no enclosing folder" in out


@pytest.mark.parametrize("meta,message", [
    (dict(split="cbt-val"), "not 'cbt-test'"),
    (dict(n_items=64), "not the complete split"),
    (dict(policy="H2O"), "not MyPolicy"),
    (dict(policy_config={"k": 3}), "but student/config.yaml has"),
])
def test_validation_must_be_the_submitted_full_run(complete_submission, monkeypatch, capsys, meta, message):
    (complete_submission / "validation.json").write_text(validation_run(**meta))
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and message in out


def test_validation_needs_timing_and_ai_use_needs_content(complete_submission, monkeypatch, capsys):
    v = json.loads(validation_run())
    for b in v["rows"]["student"]["budgets"].values():
        del b["throughput"]
    (complete_submission / "validation.json").write_text(json.dumps(v))
    (complete_submission / "AI_USE.md").write_text(open(env.REPO_ROOT + "/AI_USE.md").read())
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and "no usable throughput timing results" in out and "AI_USE.md is not filled in" in out


@pytest.mark.parametrize("budgets", [None, [], {}, {"64": {"throughput": None}}])
def test_validation_requires_every_result_budget(complete_submission, monkeypatch, capsys, budgets):
    v = json.loads(validation_run())
    v["rows"]["student"]["budgets"] = budgets
    (complete_submission / "validation.json").write_text(json.dumps(v))
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and "results for exactly budgets" in out


@pytest.mark.parametrize("metric,value", [
    ("Q", None), ("Q", float("nan")), ("Q", float("inf")), ("Q", True), ("Q", -0.1),
    ("R", 1.1), ("R", "1.0"),
])
def test_validation_rejects_invalid_quality_metrics(complete_submission, monkeypatch, capsys, metric, value):
    v = json.loads(validation_run())
    v["rows"]["student"]["budgets"]["128"][metric] = value
    (complete_submission / "validation.json").write_text(json.dumps(v))
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and f"{metric} must be finite" in out


@pytest.mark.parametrize("name,key,value", [
    ("throughput", None, None), ("latency", None, []),
    ("throughput", "tokens_per_s", 0), ("latency", "ms_per_step", float("nan")),
    ("throughput", "gflops_per_token", -1), ("latency", "times_s", []),
    ("latency", "times_s", [1, float("inf")]),
])
def test_validation_requires_usable_timing(complete_submission, monkeypatch, capsys, name, key, value):
    v = json.loads(validation_run())
    row = v["rows"]["student"]["budgets"]["256"]
    if key is None:
        row[name] = value
    else:
        row[name][key] = value
    (complete_submission / "validation.json").write_text(json.dumps(v))
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and f"no usable {name} timing results" in out


@pytest.mark.parametrize("student", [None, [], "incomplete"])
def test_validation_rejects_invalid_student_row(complete_submission, monkeypatch, capsys, student):
    v = json.loads(validation_run())
    v["rows"]["student"] = student
    (complete_submission / "validation.json").write_text(json.dumps(v))
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and "no student result object" in out


@pytest.mark.parametrize("config", ["[1, 2]", "policies: [MyPolicy]", "policies: {MyPolicy: [1]}"])
def test_validation_rejects_invalid_config_shapes(complete_submission, monkeypatch, capsys, config):
    (complete_submission / "student/config.yaml").write_text(config)
    code, out = run_check(complete_submission, monkeypatch, capsys)
    assert code == 1 and "must contain mappings" in out
