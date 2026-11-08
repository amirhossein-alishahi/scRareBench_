#!/usr/bin/env python3
"""scRareBench Dataset 5: Seurat RPCA and CCA, 3 seeds, Ubuntu server.

Reuses the established Seurat 5.6.0 R install and scRareBench 0.10.6 Python
runtime; dataset preparation, label-aware evaluation, scIB, and reports
remain owned by scRareBench. No method training uses biological cell labels.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io, sparse

from scrarebench import MethodOutput, MethodSpec, benchmark_method, dataset_info, load_dataset

METHOD_SEEDS = [42, 123, 2026]
BENCHMARK_SEED = 42
METHOD_CONFIG = {
    "n_hvg": 4000,
    "hvg_span": 0.3,
    "hvg_batch_key": "orig.ident",
    "npcs": 50,
    "integration_dims": 30,
}
BASE = Path(__file__).resolve().parent
SEURAT_ROOT = BASE / "seurat_runtime"
RESULTS_ROOT = BASE / "scrarebench_results"

print("Loading scRareBench Dataset 5 ...", flush=True)
adata = load_dataset(5)
print(adata, flush=True)
info = dataset_info(adata)
print(f"Integration batch: {info['batch_key']}; raw counts: adata.X", flush=True)

# The dataset-5 contract exposes raw RNA counts in X after its package-level QC.
# HVG selection is batch-aware and completely independent of celltype labels.
hvg = sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=METHOD_CONFIG["n_hvg"],
    batch_key=METHOD_CONFIG["hvg_batch_key"],
    span=METHOD_CONFIG["hvg_span"],
    inplace=False,
)["highly_variable"].to_numpy(dtype=bool)
hvg_indices = np.flatnonzero(hvg)
selected_names = adata.var_names[hvg_indices].astype(str)
counts = adata.X[:, hvg_indices]
counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
print(f"Method HVGs: {len(hvg_indices):,} (Seurat-v3, batch-aware); "
      f"SHA256: {hashlib.sha256(chr(10).join(selected_names).encode()).hexdigest()}", flush=True)

R_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
workdir <- args[[1]]
seed <- as.integer(args[[2]])
npcs <- as.integer(args[[3]])
ndims <- as.integer(args[[4]])

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

set.seed(seed)
options(future.globals.maxSize = 30 * 1024^3)

message("[Seurat] Reading selected raw-count genes and metadata")
counts <- as(Matrix::readMM(file.path(workdir, "counts.mtx")), "dgCMatrix")
genes <- read.csv(file.path(workdir, "genes.csv"), stringsAsFactors = FALSE)
meta <- read.csv(file.path(workdir, "metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
rownames(counts) <- make.unique(as.character(genes$gene))
colnames(counts) <- as.character(meta$barcode)
rownames(meta) <- as.character(meta$barcode)
meta$BATCH <- factor(meta$BATCH)

message("[Seurat] CreateSeuratObject")
obj <- CreateSeuratObject(
  counts = counts,
  meta.data = meta,
  assay = "RNA",
  project = "scRareBench_Dataset5_Seurat"
)
rm(counts)
gc()

message("[Seurat] Split RNA layers by batch")
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$BATCH)
VariableFeatures(obj) <- rownames(obj)
message("[Seurat] NormalizeData")
obj <- NormalizeData(obj, verbose = FALSE)
message("[Seurat] ScaleData")
obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
message("[Seurat] RunPCA")
obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = npcs, verbose = FALSE)

save_latent <- function(object, reduction_name, filename) {
  # Integration may return more than ndims: retain exactly the evaluation dims.
  emb <- Embeddings(object[[reduction_name]])[, seq_len(ndims), drop = FALSE]
  write.csv(
    data.frame(barcode = rownames(emb), emb, check.names = FALSE),
    file.path(workdir, filename), row.names = FALSE, quote = FALSE
  )
}

message("[Seurat] RPCAIntegration")
obj <- IntegrateLayers(
  object = obj,
  method = RPCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.rpca",
  features = VariableFeatures(obj),
  dims = seq_len(ndims),
  dims.to.integrate = seq_len(ndims),
  verbose = FALSE
)
save_latent(obj, "integrated.rpca", "seurat_rpca_latent.csv")
message("[Seurat] RPCA embedding written")
gc()

message("[Seurat] CCAIntegration")
obj <- IntegrateLayers(
  object = obj,
  method = CCAIntegration,
  svd.method = "rspectra",
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
  features = VariableFeatures(obj),
  dims = seq_len(ndims),
  dims.to.integrate = seq_len(ndims),
  verbose = FALSE
)
save_latent(obj, "integrated.cca", "seurat_cca_latent.csv")
message("[Seurat] CCA embedding written")
gc()

writeLines(VariableFeatures(obj), file.path(workdir, "seurat_selected_genes.txt"))
sink(file.path(workdir, "seurat_session_info.txt"))
print(sessionInfo())
sink()
message("[Seurat] Both integration modes completed")
'''

_PAIR_CACHE = {}


def _read_latent(path, expected_barcodes):
    frame = pd.read_csv(path).set_index("barcode")
    # Seurat can reorder cells internally; restore input order using barcode IDs.
    latent = frame.loc[expected_barcodes].to_numpy(dtype=np.float32)
    return latent, np.asarray(expected_barcodes, dtype=str)


def _run_pair(method_adata, seed, config):
    seed = int(seed)
    if seed in _PAIR_CACHE:
        return _PAIR_CACHE[seed]

    workdir = SEURAT_ROOT / f"seed_{seed}"
    workdir.mkdir(parents=True, exist_ok=True)
    completed = workdir / "seurat_pair_completed.flag"
    script = workdir / "run_seurat.R"
    if not completed.exists():
        print(f"Running Seurat RPCA + CCA for seed={seed} ...", flush=True)
        # Transfer only the 4000 batch-aware HVGs to R (not the entire RNA matrix).
        io.mmwrite(workdir / "counts.mtx", counts.T.tocoo())
        pd.DataFrame({"gene": selected_names}).to_csv(workdir / "genes.csv", index=False)
        pd.DataFrame({
            "barcode": adata.obs_names.astype(str),
            "BATCH": adata.obs[info["batch_key"]].astype(str).to_numpy(),
        }).to_csv(workdir / "metadata.csv", index=False)
        script.write_text(R_SCRIPT, encoding="utf-8")
        (workdir / "seurat_method_policy.json").write_text(json.dumps({
            "dataset": "nygc_seurat_v4_pbmc",
            "batch_key": info["batch_key"],
            "count_source": "adata.X (raw RNA counts)",
            "hvg_policy": "batch_aware_seurat_v3_raw_counts",
            "hvg_batch_key": config["hvg_batch_key"],
            "hvg_span": config["hvg_span"],
            "n_hvg": config["n_hvg"],
            "hvg_sha256": hashlib.sha256("\n".join(selected_names).encode()).hexdigest(),
            "npcs": config["npcs"], "integration_dims": config["integration_dims"],
            "method_seed": seed, "benchmark_labels_exported": False,
        }, indent=2), encoding="utf-8")

        cmd = ["Rscript", str(script), str(workdir), str(seed),
               str(config["npcs"]), str(config["integration_dims"])]
        log_path = workdir / "seurat_execution.log"
        with log_path.open("w", encoding="utf-8") as logfile:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
            ret = proc.wait()
        if ret:
            raise RuntimeError(f"Seurat failed for seed {seed} (exit {ret}); see {log_path}")
        completed.write_text("completed\n", encoding="utf-8")
    else:
        print(f"Reusing completed Seurat embeddings for seed={seed}", flush=True)

    expected = adata.obs_names.astype(str)
    provenance = {
        "seurat_runner": script,
        "seurat_session_info": workdir / "seurat_session_info.txt",
        "seurat_selected_genes": workdir / "seurat_selected_genes.txt",
        "seurat_method_policy": workdir / "seurat_method_policy.json",
    }
    rpca = _read_latent(workdir / "seurat_rpca_latent.csv", expected)
    cca = _read_latent(workdir / "seurat_cca_latent.csv", expected)
    _PAIR_CACHE[seed] = {"rpca": (*rpca, provenance), "cca": (*cca, provenance)}
    return _PAIR_CACHE[seed]


def run_rpca(method_adata, seed, config):
    latent, barcodes, provenance = _run_pair(method_adata, seed, config)["rpca"]
    return MethodOutput(latent=latent, barcodes=barcodes,
                        representation_key="X_seurat_rpca", provenance_files=provenance)


def run_cca(method_adata, seed, config):
    latent, barcodes, provenance = _run_pair(method_adata, seed, config)["cca"]
    return MethodOutput(latent=latent, barcodes=barcodes,
                        representation_key="X_seurat_cca", provenance_files=provenance)


methods = [
    MethodSpec(name="Seurat-RPCA", runner=run_rpca, config=METHOD_CONFIG),
    MethodSpec(name="Seurat-CCA", runner=run_cca, config=METHOD_CONFIG),
]
benchmark_config = {
    "random_state": BENCHMARK_SEED,
    "run_scib": True,
    "scib_n_hvg": 4000,
    "scib_reference_n_pcs": 50,
    "scib_n_jobs": 1,
    "scib_require_success": True,
    "write_pdf_report": False,
    "include_latent_in_bundle": False,
    "include_cell_ids": False,
}

results = {}
for method in methods:
    print(f"\n===== Benchmarking {method.name} =====", flush=True)
    results[method.name] = benchmark_method(
        adata, method, seeds=METHOD_SEEDS,
        output_dir=RESULTS_ROOT / method.name.lower().replace("-", "_"),
        benchmark_config=benchmark_config,
        install_dependencies=False, finalize=True, copy_input_per_seed=False,
        overwrite=True,
    )

summary = pd.concat([
    result.summary().assign(method=name) for name, result in results.items()
], ignore_index=True)
summary = summary[["method", "method_seed", "section", "metric", "value"]]
summary_path = BASE / "Seurat_Dataset5_3seeds_summary.csv"
summary.to_csv(summary_path, index=False)

combined = BASE / "scRareBench_Seurat_Dataset5_3seeds_CCA_RPCA.zip"
with ZipFile(combined, "w", compression=ZIP_DEFLATED) as zf:
    for method_name, result in results.items():
        zf.write(result.archive_path, arcname=f"{method_name}/{Path(result.archive_path).name}")
    zf.write(summary_path, arcname=summary_path.name)
for name, result in results.items():
    print(f"{name}: {result.seeds}; report={result.report_path}; ZIP={result.archive_path}", flush=True)
print(f"\nALL DONE. Combined ZIP: {combined}", flush=True)
