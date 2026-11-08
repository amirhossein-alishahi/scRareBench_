#!/usr/bin/env python3
"""Seurat v5 RPCA + CCA, Dataset 0, three method seeds.

Adapted from the user-provided, successful scRareBench v0.10.5 Colab notebook.
No Google Colab dependencies. Run from ~/scrarebench-d0.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from scrarebench import load_dataset

adata = load_dataset(0)
print(adata, flush=True)

# Reproducibility and Seurat controls.
METHOD_SEEDS = [42, 123, 2026]
BENCHMARK_SEED = 42

METHOD_CONFIG = {
    "nfeatures": 3000,
    "npcs": 50,
    "integration_dims": 30,
}


from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy import io, sparse

from scrarebench import MethodOutput, dataset_info

SEURAT_CACHE = {}
SEURAT_ROOT = Path.cwd() / "seurat_runtime"

R_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
workdir <- args[[1]]
seed <- as.integer(args[[2]])
nfeatures <- as.integer(args[[3]])
npcs <- as.integer(args[[4]])
ndims <- as.integer(args[[5]])

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

if (packageVersion("Seurat") < package_version("5.0.0")) {
  stop("Seurat >= 5.0.0 is required.")
}
if (ndims > npcs) stop("integration_dims must be <= npcs")

set.seed(seed)
options(future.globals.maxSize = 20 * 1024^3)

counts <- Matrix::readMM(file.path(workdir, "counts.mtx"))
genes <- read.csv(file.path(workdir, "genes.csv"), stringsAsFactors = FALSE)
barcodes <- read.csv(file.path(workdir, "barcodes.csv"), stringsAsFactors = FALSE)
meta <- read.csv(file.path(workdir, "metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)

stopifnot(nrow(counts) == nrow(genes))
stopifnot(ncol(counts) == nrow(barcodes))
stopifnot(nrow(meta) == nrow(barcodes))
stopifnot("BATCH" %in% colnames(meta))

rownames(counts) <- make.unique(as.character(genes$gene))
colnames(counts) <- as.character(barcodes$barcode)
rownames(meta) <- as.character(meta$barcode)
meta <- meta[colnames(counts), , drop = FALSE]
stopifnot(all(rownames(meta) == colnames(counts)))
meta$BATCH <- factor(meta$BATCH)

obj <- CreateSeuratObject(
  counts = counts,
  meta.data = meta,
  assay = "RNA",
  project = "scRareBench_Dataset0_Seurat"
)
rm(counts)
gc()

obj[["RNA"]] <- split(obj[["RNA"]], f = obj$BATCH)
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = nfeatures, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = npcs, verbose = FALSE)

save_latent <- function(object, reduction_name, output_name) {
  emb <- Embeddings(object[[reduction_name]])
  out <- data.frame(barcode = rownames(emb), emb, check.names = FALSE)
  write.csv(out, file.path(workdir, output_name), row.names = FALSE, quote = FALSE)
}

obj <- IntegrateLayers(
  object = obj,
  method = RPCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.rpca",
  dims = seq_len(ndims),
  verbose = FALSE
)
save_latent(obj, "integrated.rpca", "seurat_rpca_latent.csv")
gc()

obj <- IntegrateLayers(
  object = obj,
  method = CCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
  dims = seq_len(ndims),
  verbose = FALSE
)
save_latent(obj, "integrated.cca", "seurat_cca_latent.csv")
gc()

sink(file.path(workdir, "seurat_session_info.txt"))
print(sessionInfo())
sink()
'''


def _gex_mask(method_adata):
    for key in ("feature_types", "feature_type", "modality"):
        if key not in method_adata.var.columns:
            continue
        values = method_adata.var[key].astype(str).str.lower()
        mask = values.str.contains("gene expression|gex|rna", regex=True).to_numpy()
        if mask.any():
            return mask
    return np.ones(method_adata.n_vars, dtype=bool)


def _write_seurat_inputs(method_adata, workdir):
    meta = dataset_info(method_adata)
    batch_key = str(meta.get("batch_key") or "")
    count_layer = str(meta.get("count_layer") or "counts")

    if not batch_key or batch_key not in method_adata.obs.columns:
        raise KeyError("Dataset batch metadata is missing.")
    if count_layer not in method_adata.layers:
        raise KeyError(f"Raw count layer {count_layer!r} is required for Seurat.")

    mask = _gex_mask(method_adata)
    genes = method_adata.var_names[mask].astype(str)
    counts = method_adata.layers[count_layer][:, mask]
    counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)

    if counts.shape != (method_adata.n_obs, int(mask.sum())):
        raise RuntimeError("Unexpected count-matrix shape after GEX filtering.")
    if counts.nnz and np.min(counts.data) < 0:
        raise ValueError("Seurat counts must be non-negative.")
    sample = counts.data[: min(counts.nnz, 100_000)]
    if sample.size and not np.allclose(sample, np.rint(sample), rtol=0.0, atol=1e-6):
        raise ValueError("Seurat requires raw/count-like input; non-integer values were detected.")

    # AnnData is cells x genes; Seurat/Matrix Market input is genes x cells.
    io.mmwrite(workdir / "counts.mtx", counts.T.tocoo())
    pd.DataFrame({"gene": genes}).to_csv(workdir / "genes.csv", index=False)
    pd.DataFrame({"barcode": method_adata.obs_names.astype(str)}).to_csv(
        workdir / "barcodes.csv", index=False
    )
    pd.DataFrame({
        "barcode": method_adata.obs_names.astype(str),
        "BATCH": method_adata.obs[batch_key].astype(str).to_numpy(),
    }).to_csv(workdir / "metadata.csv", index=False)

    print(f"GEX features: {int(mask.sum()):,} / {method_adata.n_vars:,}")
    print(f"Cells: {method_adata.n_obs:,}; batches: {method_adata.obs[batch_key].nunique()}")


def _read_latent(path, expected_barcodes):
    frame = pd.read_csv(path)
    if "barcode" not in frame.columns:
        raise RuntimeError(f"Missing barcode column in {path}")
    barcodes = frame.pop("barcode").astype(str).to_numpy()
    expected = np.asarray(expected_barcodes, dtype=str)
    if not np.array_equal(barcodes, expected):
        raise RuntimeError("Seurat changed cell order; refusing to benchmark a misaligned latent.")
    latent = frame.to_numpy(dtype=np.float32)
    if latent.ndim != 2 or latent.shape[0] != expected.size or latent.shape[1] < 2:
        raise RuntimeError(f"Invalid latent shape: {latent.shape}")
    if not np.isfinite(latent).all():
        raise ValueError("Seurat latent contains NaN or infinite values.")
    return latent, barcodes


def _run_seurat_pair(method_adata, seed, config):
    seed = int(seed)
    cache_key = (seed, int(config["nfeatures"]), int(config["npcs"]), int(config["integration_dims"]))
    if cache_key in SEURAT_CACHE:
        return SEURAT_CACHE[cache_key]

    workdir = SEURAT_ROOT / f"seed_{seed}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    _write_seurat_inputs(method_adata, workdir)
    script_path = workdir / "run_seurat.R"
    script_path.write_text(R_SCRIPT, encoding="utf-8")

    command = [
        "Rscript", str(script_path), str(workdir), str(seed),
        str(int(config["nfeatures"])), str(int(config["npcs"])),
        str(int(config["integration_dims"])),
    ]
    print(f"Running Seurat RPCA + CCA for seed={seed} ...", flush=True)
    # Keep the R error visible and saved, instead of hiding it in CalledProcessError.
    log_path = workdir / "seurat_execution.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Rscript failed (exit={exit_code}). Full R log: {log_path}")

    expected = method_adata.obs_names.astype(str).to_numpy()
    rpca = _read_latent(workdir / "seurat_rpca_latent.csv", expected)
    cca = _read_latent(workdir / "seurat_cca_latent.csv", expected)

    provenance = {
        "seurat_runner": script_path,
        "seurat_session_info": workdir / "seurat_session_info.txt",
    }
    SEURAT_CACHE[cache_key] = {
        "rpca": (*rpca, provenance),
        "cca": (*cca, provenance),
    }
    return SEURAT_CACHE[cache_key]


def run_seurat_rpca(method_adata, seed, config):
    latent, barcodes, provenance = _run_seurat_pair(method_adata, seed, config)["rpca"]
    return MethodOutput(
        latent=latent,
        barcodes=barcodes,
        representation_key="X_seurat_rpca",
        provenance_files=provenance,
    )


def run_seurat_cca(method_adata, seed, config):
    latent, barcodes, provenance = _run_seurat_pair(method_adata, seed, config)["cca"]
    return MethodOutput(
        latent=latent,
        barcodes=barcodes,
        representation_key="X_seurat_cca",
        provenance_files=provenance,
    )


from scrarebench import MethodSpec, benchmark_method

methods = [
    MethodSpec(name="Seurat-RPCA", runner=run_seurat_rpca, config=METHOD_CONFIG),
    MethodSpec(name="Seurat-CCA", runner=run_seurat_cca, config=METHOD_CONFIG),
]

results = {}
for method in methods:
    print(f"\n===== Benchmarking {method.name} =====")
    results[method.name] = benchmark_method(
        adata,
        method,
        seeds=METHOD_SEEDS,
        benchmark_config={"random_state": BENCHMARK_SEED},
        install_dependencies=False,
        finalize=True,
        # Dataset 0 is large. The method runner never mutates adata, so reuse it
        # instead of creating multiple full AnnData copies in Colab.
        copy_input_per_seed=False,
        overwrite=True,
    )


# Store the six seed/method summaries next to the results.
summary = pd.concat(
    [result.summary().assign(method=name) for name, result in results.items()],
    ignore_index=True,
)
summary = summary[["method", "method_seed", "section", "metric", "value"]]
summary.to_csv("Seurat_Dataset0_3seeds_summary.csv", index=False)
print("\n--- Seurat benchmark summary ---", flush=True)
print(summary.round(5).to_string(index=False), flush=True)

for name, result in results.items():
    print(f"\n{name}: seeds={result.seeds}")
    print("  Report:", result.report_path)
    print("  Archive:", result.archive_path)
    print("  Summary JSON:", result.summary_path)

from zipfile import ZipFile, ZIP_DEFLATED
combined = Path.cwd() / "scRareBench_Seurat_Dataset0_3seeds_CCA_RPCA.zip"
with ZipFile(combined, "w", compression=ZIP_DEFLATED) as zf:
    for method_name, result in results.items():
        zf.write(result.archive_path, arcname=f"{method_name}/{Path(result.archive_path).name}")
    zf.write("Seurat_Dataset0_3seeds_summary.csv")
print(f"\nALL DONE. Combined ZIP: {combined}", flush=True)
