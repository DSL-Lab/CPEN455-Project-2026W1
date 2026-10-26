#!/usr/bin/env bash
# Create the locked course environment and download base checkpoints.
#   bash scripts/setup_t4.sh                 # Linux/Windows (WSL) NVIDIA GPU
#   bash scripts/setup_t4.sh --cpu           # CPU or Apple silicon (MPS)
#   bash scripts/setup_t4.sh --no-download   # skip model downloads
set -euo pipefail
cd "$(dirname "$0")/.."

backend=cu126
download=1
if [ "$(uname)" = Darwin ]; then backend=cpu; fi
for arg in "$@"; do
    case "$arg" in
        --cpu) backend=cpu ;;
        --no-download) download=0 ;;
        -h|--help)
            echo 'Usage: bash scripts/setup_t4.sh [--cpu] [--no-download]'
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
if ! command -v uv >/dev/null; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
uv sync --locked --extra "$backend"

if [ "$download" = 1 ]; then
    [ -f checkpoints/gpt2.pt ] || .venv/bin/python scripts/download_gpt2.py gpt2
    [ -f checkpoints/gpt2-medium.pt ] || .venv/bin/python scripts/download_gpt2.py gpt2-medium
fi
source scripts/env.sh
.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
.venv/bin/python -m cpen455kv.env --strict
