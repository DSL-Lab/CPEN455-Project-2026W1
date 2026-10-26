"""Device, dtype and reproducibility checks for uv and Conda environments.

    python -m cpen455kv.env
    python -m cpen455kv.env --strict --json environment.json

Source scripts/env.sh before starting Python. Strict checks work without a container.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tomllib
import warnings

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED_ENV = {
    "CPEN455_DTYPE": "float16",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
}


def expected_packages():
    """Read direct dependency pins from the single authoritative specification."""
    spec = tomllib.loads((Path(REPO_ROOT) / "pyproject.toml").read_text())["project"]
    requirements = spec["dependencies"] + spec["optional-dependencies"]["cpu"]
    return dict(requirement.split("==", 1) for requirement in requirements)


def installed_packages():
    packages = {}
    for name in expected_packages():
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def environment_issues():
    issues = []
    if sys.version_info[:2] != (3, 11):
        issues.append(f"Python {platform.python_version()} is not the course Python 3.11")
    installed = installed_packages()
    for name, expected in expected_packages().items():
        actual = installed[name]
        # torch encodes the CPU/CUDA build as a local version suffix.
        version = actual.split("+", 1)[0] if actual else None
        if version != expected:
            issues.append(f"{name}={actual!r}; expected {expected}")
    for name, expected in PINNED_ENV.items():
        if os.environ.get(name) != expected:
            issues.append(f"{name}={os.environ.get(name)!r}; expected {expected!r}")
    if get_device() == "cuda":
        if not torch.cuda.is_available():
            issues.append("CUDA selected but no CUDA device is available")
        elif torch.version.cuda != "12.6":
            issues.append(f"PyTorch CUDA build {torch.version.cuda!r}; expected '12.6'")
    return issues


def check_env(strict=None):
    """Warn about drift, or reject it with --strict-env / CPEN455_STRICT_ENV=1."""
    strict = os.environ.get("CPEN455_STRICT_ENV") == "1" if strict is None else strict
    issues = environment_issues()
    if issues:
        message = "Environment differs from the course setup:\n  " + "\n  ".join(issues)
        message += "\nUse the locked dependencies and source scripts/env.sh before starting Python."
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return issues


def get_device():
    if dev := os.environ.get("CPEN455_DEVICE"):
        if dev not in {"cpu", "cuda", "mps"}:
            raise ValueError("CPEN455_DEVICE must be cpu, cuda or mps")
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device):
    """CPU uses fp32; the course GPU configuration uses fp16."""
    if device == "cpu":
        return torch.float32
    name = os.environ.get("CPEN455_DTYPE", "float16")
    if name not in {"float16", "bfloat16", "float32"}:
        raise ValueError("CPEN455_DTYPE must be float16, bfloat16 or float32")
    return getattr(torch, name)


def data_dir():
    return os.environ.get("CPEN455_DATA", os.path.join(REPO_ROOT, "data"))


def default_ckpt():
    """Prefer the graded base GPT-2-medium, otherwise GPT-2 small for development."""
    if path := os.environ.get("CPEN455_CKPT"):
        return path
    for name in ("gpt2-medium.pt", "gpt2.pt"):
        path = os.path.join(REPO_ROOT, "checkpoints", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("no checkpoint found: run `python scripts/download_gpt2.py gpt2-medium`")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint_state_sha256(path):
    """Hash stored model configuration and tensors, independent of archive metadata."""
    if str(path).endswith(".safetensors"):
        from .weights import hf_gpt2_to_nanogpt, read_safetensors
        config, state = hf_gpt2_to_nanogpt(
            read_safetensors(path), os.environ.get("CPEN455_MODEL_TYPE", "gpt2-medium"))
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config, state = checkpoint["model_args"], checkpoint["model"]
    digest = hashlib.sha256(b"cpen455-checkpoint-state-v1\n")
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode() + b"\n")
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = [name, str(tensor.dtype), list(tensor.shape), tensor.numel() * tensor.element_size()]
        digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\n")
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def runtime_metadata(ckpt=None, ckpt_sha256=None):
    device = get_device()
    lock = Path(REPO_ROOT) / "uv.lock"
    result = dict(
        python=platform.python_version(), platform=platform.platform(),
        packages=installed_packages(), torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=device, dtype=str(get_dtype(device)),
        gpu=torch.cuda.get_device_name() if device == "cuda" and torch.cuda.is_available() else None,
        environment={name: os.environ.get(name) for name in PINNED_ENV},
        uv_lock_sha256=file_sha256(lock) if lock.exists() else None,
    )
    if ckpt:
        result.update(ckpt=os.path.basename(ckpt), ckpt_sha256=ckpt_sha256 or file_sha256(ckpt),
                      ckpt_state_sha256=checkpoint_state_sha256(ckpt))
    return result


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib(device):
    return torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None


def report(strict=False, json_path=None):
    issues = check_env(strict=strict)
    try:
        ckpt = default_ckpt()
    except FileNotFoundError:
        ckpt = None
    result = runtime_metadata(ckpt)
    print(f"Python {result['python']}, torch {result['torch']}, CUDA {result['cuda']}, "
          f"device {result['device']}, dtype {result['dtype']}")
    print(f"GPU: {result['gpu'] or 'not an NVIDIA GPU'}")
    print(f"Checkpoint: {result.get('ckpt', 'not downloaded')}")
    if "ckpt_sha256" in result:
        print(f"Checkpoint SHA256: {result['ckpt_sha256']}")
    if not issues:
        print("Course package versions and environment variables match")
    print("Record your actual device and dtype when reporting measurements")
    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {json_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on package or environment drift")
    parser.add_argument("--json", metavar="PATH", help="save environment and checkpoint metadata")
    args = parser.parse_args()
    report(strict=args.strict, json_path=args.json)


if __name__ == "__main__":
    main()
