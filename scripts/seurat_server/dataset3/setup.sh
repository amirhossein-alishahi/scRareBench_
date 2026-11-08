#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Reuse already-installed system R + Seurat. Keep Dataset 0's venv untouched.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps \
  "git+https://github.com/amirhossein-alishahi/scRareBench_.git@12c4e287f0367f2dc67945b45e5e79e707977f73"
export JAX_PLATFORMS=cpu
python - <<'PYSETUP'
from scrarebench.runtime import setup_runtime
setup_runtime(quiet=False)
print("Dataset 3 Python environment is ready")
PYSETUP
Rscript --vanilla -e 'library(Seurat); cat("Seurat: ", as.character(packageVersion("Seurat")), "\n")'
python -c 'import scrarebench, scib_metrics; print("scRareBench:", scrarebench.__version__)'
echo 'Setup complete for Dataset 3.'
