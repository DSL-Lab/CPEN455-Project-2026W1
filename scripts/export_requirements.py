"""Export the shared uv lock for Conda/pip users; do not edit the exports by hand.

    python scripts/export_requirements.py
    python scripts/export_requirements.py --check
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = {"cpu": "requirements-cpu.txt", "cu126": "requirements.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if an export differs from uv.lock")
    args = parser.parse_args()
    stale = []
    for extra, filename in EXPORTS.items():
        result = subprocess.run(
            ["uv", "export", "--locked", "--extra", extra, "--no-emit-project", "--no-header"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        content = ("# Generated from uv.lock by scripts/export_requirements.py; do not edit.\n"
                   f"--extra-index-url https://download.pytorch.org/whl/{extra}\n"
                   + result.stdout)
        path = ROOT / filename
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(filename)
        else:
            path.write_text(content)
            print(f"wrote {filename}")
    if stale:
        raise SystemExit("Outdated exports: " + ", ".join(stale))
    if args.check:
        print("Both requirements exports match uv.lock")


if __name__ == "__main__":
    main()
