# Source from the project root before starting Python (uv or Conda):
#   source scripts/env.sh
export CPEN455_DTYPE=float16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR="$(pwd)/.cache/tiktoken"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
