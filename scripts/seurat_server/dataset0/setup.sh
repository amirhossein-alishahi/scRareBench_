#!/usr/bin/env bash
set -Eeuo pipefail

# Ubuntu Server 24.04 — R / Seurat v5 and Python scRareBench v0.10.5.
# Run as normal SSH user from ~/scrarebench-d0. No GPU is required.
if [[ "${EUID}" -eq 0 ]]; then
  APT=(apt-get)
else
  APT=(sudo apt-get)
fi

export DEBIAN_FRONTEND=noninteractive
"${APT[@]}" update
"${APT[@]}" install -y \
  ca-certificates curl wget git unzip tmux \
  python3 python3-dev python3-pip python3-venv \
  r-base r-base-dev build-essential gfortran \
  libcurl4-openssl-dev libssl-dev libxml2-dev libhdf5-dev \
  libfontconfig1-dev libfreetype-dev libpng-dev libtiff-dev libjpeg-dev \
  libharfbuzz-dev libfribidi-dev libgit2-dev libglpk-dev \
  libblas-dev liblapack-dev libopenblas-dev \
  libuv1-dev libudunits2-dev libgdal-dev \
  libbz2-dev liblzma-dev libzstd-dev libreadline-dev libpcre2-dev

# Seurat is owned by R, not scRareBench.
# The personal R library is auto-discovered by future Rscript sessions.
Rscript --vanilla -e '
  user_lib <- Sys.getenv("R_LIBS_USER")
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_lib, .libPaths()))
  options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 1200)
  if (!requireNamespace("Seurat", quietly = TRUE) || packageVersion("Seurat") < "5.0.0") {
    install.packages("Seurat", lib = user_lib, dependencies = NA, Ncpus = 2)
  }
  if (!requireNamespace("Seurat", quietly = TRUE)) stop("Seurat installation failed")
  if (packageVersion("Seurat") < "5.0.0") stop("Seurat v5+ required")
  cat("Installed Seurat:", as.character(packageVersion("Seurat")), "\n")
'

# Isolated Python 3.12 environment for the benchmark.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.5"
python - <<'PYINSTALL'
from scrarebench.runtime import setup_runtime
setup_runtime(quiet=False)
print('scRareBench runtime is ready')
PYINSTALL

Rscript --vanilla -e 'library(Seurat); cat("Seurat: ",as.character(packageVersion("Seurat")),"\n")'
python -c 'import scrarebench; import scib_metrics; print("Python benchmark dependencies ready")'
echo 'Setup complete.'
