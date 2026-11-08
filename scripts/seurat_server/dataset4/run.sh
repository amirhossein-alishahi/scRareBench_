#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source "$HOME/scrarebench-d3/.venv/bin/activate"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u run_seurat_dataset4.py 2>&1 | tee run.log
