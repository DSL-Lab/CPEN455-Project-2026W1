"""Check a student submission before uploading it.

    python -m cpen455kv.check submission.zip   # the exact ZIP you upload to Canvas (required)
    python -m cpen455kv.check .                # the submission files in your project root, while working

Checks the ZIP layout, source/text file types, size, imports, required files, and that
validation.json is a complete timed `cbt-test` run of your submitted MyPolicy configuration.
Environment versions are checked separately by python -m cpen455kv.env --strict.
"""
import ast
import json
import math
import os
import sys
import tempfile
import zipfile

import yaml

from .env import data_dir

ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
ALLOWED_IMPORTS = {"torch", "numpy", "math", "cpen455kv", "nanogpt", "student", "pytest"}
FORBIDDEN_IMPORTS = {"transformers", "accelerate", "optimum", "onnxruntime", "vllm", "sglang", "tensorrt",
                     "tensorrt_llm", "lmdeploy", "flash_attn", "flashinfer", "xformers", "kvpress",
                     "deepspeed", "llama_cpp", "mlc_llm", "exllamav2", "ctranslate2", "triton"}
FLAGGED = {"ctypes", "subprocess", "importlib", "socket", "urllib", "requests", "multiprocessing"}
MAX_STUDENT_BYTES = 1 << 20
REQUIRED = ("validation.json", "report.pdf", "AI_USE.md", "student/kv_cache.py",
            "student/policy.py", "student/config.yaml", "student/part1_answers.yaml",
            "student/part2_results.json", "student/MUTANTS.md")
ZIP_TOP_LEVEL = {"student", "validation.json", "report.pdf", "AI_USE.md", "bonus"}
VALIDATION_SPLIT = "cbt-test"
BUDGETS = [64, 128, 256]
ZIP_COMMAND = ("zip -r submission.zip student validation.json report.pdf AI_USE.md "
               "-x '*__pycache__*' '*.DS_Store'   (add bonus/ if you did a bonus study)")


def check_student(root, errors, warnings):
    sdir = os.path.join(root, "student")
    if not os.path.isdir(sdir):
        errors.append("student/ missing")
        return
    total = 0
    stdlib = set(sys.stdlib_module_names)
    for dirpath, _, files in os.walk(sdir):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            total += os.path.getsize(p)
            if os.path.splitext(f)[1] not in ALLOWED_SUFFIXES:
                errors.append(f"file type not allowed in student/: {os.path.relpath(p, root)}")
                continue
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(p).read(), p)
            except SyntaxError as e:
                errors.append(f"syntax error in {os.path.relpath(p, root)}: {e}")
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for n in names:
                    top = n.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        errors.append(f"{os.path.relpath(p, root)}: import of excluded package '{top}'")
                    elif top in FLAGGED:
                        warnings.append(f"{os.path.relpath(p, root)}: '{top}' is flagged for TA review")
                    elif top not in ALLOWED_IMPORTS and top not in stdlib:
                        errors.append(f"{os.path.relpath(p, root)}: import '{top}' is not allowed")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"system", "popen", "exec", "spawn"} or (
                            isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr.startswith("exec")):
                        warnings.append(f"{os.path.relpath(p, root)}:{node.lineno}: call to {node.func.attr} flagged for TA review")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
                    warnings.append(f"{os.path.relpath(p, root)}:{node.lineno}: {node.func.id}() flagged for TA review")
    if total > MAX_STUDENT_BYTES:
        errors.append(f"student/ is {total / 2**20:.2f} MB > 1 MB")


def load_json(root, name, errors):
    try:
        return json.load(open(os.path.join(root, name)))
    except (OSError, ValueError) as e:
        errors.append(f"{name} is not valid JSON: {e}")
        return None


def check_validation(root, errors):
    """validation.json must be a complete, timed cbt-test run of the submitted MyPolicy config."""
    v = load_json(root, "validation.json", errors)
    rerun = f"rerun: python -m cpen455kv.grade --strict-env --split {VALIDATION_SPLIT} --out validation.json"
    if not (isinstance(v, dict) and isinstance(v.get("meta"), dict) and isinstance(v.get("rows"), dict)):
        errors.append(f"validation.json is not a result file written by cpen455kv.grade ({rerun})")
        return
    meta, student = v["meta"], v["rows"].get("student")
    if not isinstance(student, dict):
        errors.append(f"validation.json has no student result object ({rerun})")
        return
    with open(os.path.join(data_dir(), "cbt_ne_test.jsonl")) as f:
        n_items = sum(1 for _ in f)
    if meta.get("split") != VALIDATION_SPLIT:
        errors.append(f"validation.json uses split {meta.get('split')!r}, not {VALIDATION_SPLIT!r} ({rerun})")
    elif meta.get("n_items") != n_items or meta.get("seed") is not None:
        errors.append(f"validation.json covers {meta.get('n_items')} items, not the complete split of {n_items} ({rerun})")
    if meta.get("policy") != "MyPolicy":
        errors.append(f"validation.json evaluates {meta.get('policy')!r}, not MyPolicy ({rerun})")
    if meta.get("budgets") != BUDGETS:
        errors.append(f"validation.json budgets are {meta.get('budgets')}, not {BUDGETS} ({rerun})")
    if student.get("error"):
        errors.append(f"validation.json: MyPolicy failed: {str(student['error']).splitlines()[0]}")
    budgets = student.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != {str(b) for b in BUDGETS}:
        errors.append(f"validation.json must contain results for exactly budgets {BUDGETS} ({rerun})")
    else:
        for budget, row in budgets.items():
            if not isinstance(row, dict):
                errors.append(f"validation.json budget {budget} is not a result object ({rerun})")
                continue
            for metric in ("Q", "R", "Acc", "Acc_full"):
                value = row.get(metric)
                if not finite_number(value) or not 0 <= value <= 1:
                    errors.append(f"validation.json budget {budget}: {metric} must be finite and between 0 and 1 ({rerun})")
            n_active = row.get("n_active")
            if not isinstance(n_active, int) or isinstance(n_active, bool) or n_active <= 0:
                errors.append(f"validation.json budget {budget}: n_active must be a positive question count ({rerun})")
            for metric in ("inactive_mean_dnll", "inactive_max_dnll"):
                value = row.get(metric)
                if not finite_number(value) or value < 0:
                    errors.append(f"validation.json budget {budget}: {metric} must be finite and nonnegative ({rerun})")
            for name in ("throughput", "latency"):
                timing = row.get(name)
                if not valid_timing(timing):
                    errors.append(f"validation.json budget {budget} has no usable {name} timing results; "
                                  f"do not use --skip-timing ({rerun})")
    try:
        conf = yaml.safe_load(open(os.path.join(root, "student", "config.yaml"))) or {}
    except (OSError, yaml.YAMLError) as e:
        errors.append(f"student/config.yaml cannot be read: {e}")
        return
    policies = conf.get("policies", {}) if isinstance(conf, dict) else None
    submitted = policies.get("MyPolicy", {}) if isinstance(policies, dict) else None
    if not isinstance(submitted, dict):
        errors.append("student/config.yaml must contain mappings for policies and MyPolicy settings")
        return
    if meta.get("policy_config", {}) != submitted:
        errors.append(f"validation.json was run with MyPolicy settings {meta.get('policy_config')}, but "
                      f"student/config.yaml has {submitted} ({rerun})")


def finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_timing(timing):
    if not isinstance(timing, dict):
        return False
    for key in ("tokens_per_s", "ms_per_step"):
        value = timing.get(key)
        if not finite_number(value) or value <= 0:
            return False
    value = timing.get("gflops_per_token")
    if not finite_number(value) or value < 0:
        return False
    times = timing.get("times_s")
    return isinstance(times, list) and bool(times) and all(finite_number(t) and t > 0 for t in times)


def check_files(root, errors):
    """Content checks for the required files that are present (missing ones are reported separately)."""
    exists = lambda name: os.path.isfile(os.path.join(root, name))
    if exists("validation.json") and exists("student/config.yaml"):
        check_validation(root, errors)
    if exists("student/part2_results.json"):
        p2 = load_json(root, "student/part2_results.json", errors)
        if p2 is not None and not (isinstance(p2, dict) and p2.get("rows")):
            errors.append("student/part2_results.json has no measurements "
                          "(python -m cpen455kv.part2_bench --strict-env --out student/part2_results.json)")
    if exists("report.pdf"):
        with open(os.path.join(root, "report.pdf"), "rb") as f:
            if f.read(5) != b"%PDF-":
                errors.append("report.pdf is not a PDF file")
    if exists("AI_USE.md"):
        lines = [l.strip() for l in open(os.path.join(root, "AI_USE.md"))]
        rows = [l for l in lines if l.startswith("|") and "---" not in l][1:]  # table rows below the header
        if not (any(c.strip() for r in rows for c in r.strip("|").split("|")) or "none" in (l.lower() for l in lines)):
            errors.append("AI_USE.md is not filled in (list your AI use, or write `none` on its own line)")
    if os.path.isdir(os.path.join(root, "bonus")) and not os.path.isfile(os.path.join(root, "bonus", "README.md")):
        errors.append("bonus/README.md missing (exact commands for your bonus study)")


def check_zip(path, errors):
    """Check the layout of the ZIP and return its namelist, or None if it cannot be used."""
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        errors.append(f"{path} is not a readable ZIP file: {e}")
        return None
    names = zf.namelist()
    bad = sorted({n for n in names if n.startswith("/") or ".." in n.split("/")})
    extra = sorted({n.split("/")[0] for n in names} - ZIP_TOP_LEVEL - {""})
    junk = sorted({n for n in names if "__pycache__" in n or n.endswith(".pyc") or
                   any(part.startswith(".") for part in n.split("/") if part)})
    for n in bad:
        errors.append(f"unsafe path in ZIP: {n}")
    if extra:
        errors.append(f"unexpected top-level entries in ZIP: {', '.join(extra)}. The ZIP must contain only "
                      f"{', '.join(sorted(ZIP_TOP_LEVEL))} at its root (no enclosing folder). Build it with: {ZIP_COMMAND}")
    for n in junk:
        errors.append(f"remove from ZIP: {n}")
    return None if bad else zf


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    errors, warnings = [], []
    is_zip = target.endswith(".zip")
    with tempfile.TemporaryDirectory() as tmp:
        root = target
        if is_zip:
            zf = check_zip(target, errors)
            root = tmp
            if zf is not None:
                zf.extractall(tmp)
        check_student(root, errors, warnings)
        for f in REQUIRED:
            if not os.path.isfile(os.path.join(root, f)):
                errors.append(f"{f} missing (required in the final submission)")
        tests_dir = os.path.join(root, "student", "my_tests")
        if not os.path.isdir(tests_dir) or not any(
                name.startswith("test_") and name.endswith(".py")
                for _, _, files in os.walk(tests_dir) for name in files):
            errors.append("student/my_tests/ must contain at least one test_*.py file")
        check_files(root, errors)
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:  ", e)
    if errors:
        print(f"check FAILED ({len(errors)} errors)")
    elif is_zip:
        print(f"check passed: {target} is ready to upload")
    else:
        print(f"check passed. Now build the ZIP and check it before uploading:\n  {ZIP_COMMAND}\n"
              "  python -m cpen455kv.check submission.zip")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
