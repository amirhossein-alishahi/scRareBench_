#!/usr/bin/env python3
"""Dataset 4 — COVID-19 autoimmunity PBMC, Seurat v5 RPCA/CCA, 3 method seeds.

Standalone Ubuntu runner; installs nothing and reuses system R/Seurat.
Dataset preparation, scenario metrics, scIB and reports are delegated to scRareBench.
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

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io, sparse

from scrarebench import MethodOutput, MethodSpec, benchmark_method, dataset_info, load_dataset

METHOD_SEEDS = [42, 123, 2026]
BENCHMARK_SEED = 42
METHOD_CONFIG = {"n_hvg": 4000, "hvg_span": 0.3, "hvg_batch_key": "donor_id", "npcs": 50, "integration_dims": 30}
BASE = Path(__file__).resolve().parent
SEURAT_ROOT = BASE / "seurat_runtime"
RESULTS_ROOT = BASE / "scrarebench_results"

print("Loading scRareBench Dataset 4 ...", flush=True)
adata = load_dataset(4)
print(adata, flush=True)
info = dataset_info(adata)
print(f"Seurat batch: {info['batch_key']}; count source: {info['count_layer']}", flush=True)

# Donor-aware Seurat-v3 HVGs on package-provided canonical raw counts.
# Ground-truth cell_type and rare annotations are not used by this method.
counts = adata.layers[info["count_layer"]]
counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
hvg_data = ad.AnnData(
    X=counts,
    obs=pd.DataFrame(
        {"BATCH": adata.obs[info["batch_key"]].astype(str).to_numpy()},
        index=adata.obs_names.copy(),
    ),
    var=pd.DataFrame(index=adata.var_names.copy()),
)
sc.pp.highly_variable_genes(
    hvg_data,
    flavor="seurat_v3",
    n_top_genes=METHOD_CONFIG["n_hvg"],
    batch_key="BATCH",
    span=METHOD_CONFIG["hvg_span"],
    subset=False,
    check_values=True,
)
hvg_indices = np.flatnonzero(hvg_data.var["highly_variable"].to_numpy())
del hvg_data
hvg_names = adata.var_names[hvg_indices].astype(str)
print(f"Donor-aware method HVGs: {len(hvg_indices):,}; SHA256: "
      f"{hashlib.sha256(chr(10).join(hvg_names).encode()).hexdigest()}", flush=True)

R_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)

workdir <- args[[1]]
seed <- as.integer(args[[2]])
npcs <- as.integer(args[[3]])
ndims <- as.integer(args[[4]])
expected_hvg <- as.integer(args[[5]])

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

if (packageVersion("Seurat") < package_version("5.2.0")) {
  stop("Seurat >= 5.2.0 is required.")
}
if (ndims > npcs) {
  stop("integration_dims must be <= npcs.")
}

set.seed(seed)
options(future.globals.maxSize = 24 * 1024^3)

counts <- Matrix::readMM(file.path(workdir, "counts.mtx"))
genes <- read.csv(file.path(workdir, "genes.csv"), stringsAsFactors = FALSE)
barcodes <- read.csv(file.path(workdir, "barcodes.csv"), stringsAsFactors = FALSE)
meta <- read.csv(
  file.path(workdir, "metadata.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
hvg_indices <- read.csv(
  file.path(workdir, "hvg_indices.csv"),
  stringsAsFactors = FALSE
)

stopifnot(nrow(counts) == nrow(genes))
stopifnot(ncol(counts) == nrow(barcodes))
stopifnot(nrow(meta) == nrow(barcodes))
stopifnot("BATCH" %in% colnames(meta))
stopifnot("index_1based" %in% colnames(hvg_indices))

rownames(counts) <- make.unique(as.character(genes$gene))
colnames(counts) <- as.character(barcodes$barcode)

if (
  any(hvg_indices$index_1based < 1) ||
  any(hvg_indices$index_1based > nrow(counts))
) {
  stop("HVG indices are outside the feature range.")
}

selected_features <- rownames(counts)[hvg_indices$index_1based]

if (length(selected_features) != expected_hvg) {
  stop(
    paste0(
      "Expected ", expected_hvg,
      " fixed Dataset 4 HVGs, got ",
      length(selected_features), "."
    )
  )
}

if (anyDuplicated(selected_features)) {
  stop("Selected features are not unique after gene-name mapping.")
}

rownames(meta) <- as.character(meta$barcode)
meta <- meta[colnames(counts), , drop = FALSE]

if (!all(rownames(meta) == colnames(counts))) {
  stop("Metadata/cell order mismatch.")
}

meta$BATCH <- factor(meta$BATCH)

obj <- CreateSeuratObject(
  counts = counts,
  meta.data = meta,
  assay = "RNA",
  project = "scRareBench_Dataset4_Seurat"
)

rm(counts)
gc()

obj[["RNA"]] <- split(obj[["RNA"]], f = obj$BATCH)
VariableFeatures(obj) <- selected_features

obj <- NormalizeData(
  obj,
  verbose = FALSE
)

obj <- ScaleData(
  obj,
  features = selected_features,
  verbose = FALSE
)

obj <- RunPCA(
  obj,
  features = selected_features,
  npcs = npcs,
  verbose = FALSE
)

save_latent <- function(object, reduction_name, output_name) {
  emb <- Embeddings(object[[reduction_name]])

  if (ncol(emb) != ndims) {
    stop(
      paste0(
        "Integrated reduction ", reduction_name,
        " has ", ncol(emb),
        " dimensions; expected ", ndims, "."
      )
    )
  }

  out <- data.frame(
    barcode = rownames(emb),
    emb,
    check.names = FALSE
  )

  write.csv(
    out,
    file.path(workdir, output_name),
    row.names = FALSE,
    quote = FALSE
  )
}

rpca_file <- file.path(workdir, "seurat_rpca_latent.csv")
if (file.exists(rpca_file)) {
  message("[Seurat] Reusing existing RPCA embedding")
} else {
  obj <- IntegrateLayers(
    object = obj,
    method = RPCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated.rpca",
    features = selected_features,
    dims = seq_len(ndims),
    dims.to.integrate = seq_len(ndims),
    verbose = TRUE
  )
  save_latent(obj, "integrated.rpca", "seurat_rpca_latent.csv")
}
gc()

obj <- IntegrateLayers(
  object = obj,
  method = CCAIntegration,
  svd.method = "rspectra",
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
  features = selected_features,
  dims = seq_len(ndims),
  dims.to.integrate = seq_len(ndims),
  verbose = FALSE
)

save_latent(
  obj,
  "integrated.cca",
  "seurat_cca_latent.csv"
)
gc()

writeLines(
  selected_features,
  file.path(workdir, "seurat_selected_genes.txt")
)

sink(file.path(workdir, "seurat_session_info.txt"))
print(sessionInfo())
sink()
'''

# Both modes share the same R process and the same per-seed preprocessing.
PAIR_CACHE = {}


def _read_latent(path, expected_barcodes):
    data = pd.read_csv(path)
    barcodes = data.pop("barcode").astype(str).to_numpy()
    expected = np.asarray(expected_barcodes, dtype=str)
    # Barcode identity/order is part of the method-output contract, not dataset QC.
    if not np.array_equal(barcodes, expected):
        raise RuntimeError("Seurat output barcodes are not aligned to the input cells.")
    return data.to_numpy(dtype=np.float32), barcodes


def _run_pair(method_adata, seed, config):
    seed = int(seed)
    if seed in PAIR_CACHE:
        return PAIR_CACHE[seed]
    workdir = SEURAT_ROOT / f"seed_{seed}"
    workdir.mkdir(parents=True, exist_ok=True)
    done_flag = workdir / "seurat_pair_completed.flag"

    script = workdir / "run_seurat.R"
    if not done_flag.exists():
        print(f"Seurat RPCA + CCA for seed={seed} ...", flush=True)
        # Dataset 4 uses the complete canonical raw count matrix; only global HVGs
        # are scaled and used for PCA. Ground-truth labels are never sent to R.
        io.mmwrite(workdir / "counts.mtx", counts.T.tocoo())
        pd.DataFrame({"gene": adata.var_names.astype(str)}).to_csv(
            workdir / "genes.csv", index=False)
        pd.DataFrame({"barcode": adata.obs_names.astype(str)}).to_csv(
            workdir / "barcodes.csv", index=False)
        pd.DataFrame({
            "barcode": adata.obs_names.astype(str),
            "BATCH": adata.obs[info["batch_key"]].astype(str).to_numpy(),
        }).to_csv(workdir / "metadata.csv", index=False)
        pd.DataFrame({"index_1based": hvg_indices + 1}).to_csv(
            workdir / "hvg_indices.csv", index=False)
        script.write_text(R_SCRIPT, encoding="utf-8")
        policy = {
            "dataset": "covid19_autoimmunity_pbmc",
            "batch_key": info["batch_key"], "count_layer": info["count_layer"],
            "hvg_policy": "donor_aware_seurat_v3_raw_counts",
            "hvg_batch_key": config["hvg_batch_key"],
            "hvg_span": config["hvg_span"], "n_hvg": config["n_hvg"],
            "hvg_sha256": hashlib.sha256("\n".join(hvg_names).encode()).hexdigest(),
            "npcs": config["npcs"], "integration_dims": config["integration_dims"],
            "method_seed": seed, "benchmark_labels_exported": False,
            "cca_svd_backend": "rspectra",
            "rpca_integration_mode": "all_pairs",
        }
        (workdir / "seurat_method_policy.json").write_text(
            json.dumps(policy, indent=2), encoding="utf-8")
        command = [
            "Rscript", str(script), str(workdir), str(seed),
            str(config["npcs"]), str(config["integration_dims"]), str(config["n_hvg"]),
        ]
        r_log = workdir / "seurat_execution.log"
        with r_log.open("w", encoding="utf-8") as logfile:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Rscript failed for seed={seed} (exit {ret}). See {r_log}")
        # Only mark a full RPCA+CCA run complete after R returned successfully.
        done_flag.write_text("completed\n", encoding="utf-8")
    else:
        print(f"Reusing completed Seurat embeddings for seed={seed}", flush=True)

    rpca = _read_latent(workdir / "seurat_rpca_latent.csv", method_adata.obs_names)
    cca = _read_latent(workdir / "seurat_cca_latent.csv", method_adata.obs_names)
    provenance = {
        "seurat_runner": script,
        "seurat_session_info": workdir / "seurat_session_info.txt",
        "seurat_selected_genes": workdir / "seurat_selected_genes.txt",
        "seurat_method_policy": workdir / "seurat_method_policy.json",
    }
    PAIR_CACHE[seed] = {"rpca": (*rpca, provenance), "cca": (*cca, provenance)}
    return PAIR_CACHE[seed]


def run_rpca(method_adata, seed, config):
    latent, barcodes, evidence = _run_pair(method_adata, seed, config)["rpca"]
    return MethodOutput(latent=latent, barcodes=barcodes,
                        representation_key="X_seurat_rpca", provenance_files=evidence)


def run_cca(method_adata, seed, config):
    latent, barcodes, evidence = _run_pair(method_adata, seed, config)["cca"]
    return MethodOutput(latent=latent, barcodes=barcodes,
                        representation_key="X_seurat_cca", provenance_files=evidence)


methods = [
    MethodSpec(name="Seurat-RPCA", runner=run_rpca, config=METHOD_CONFIG),
    MethodSpec(name="Seurat-CCA", runner=run_cca, config=METHOD_CONFIG),
]
benchmark_config = {
    "random_state": BENCHMARK_SEED,
    "run_scib": True, "scib_n_hvg": 4000, "scib_reference_n_pcs": 50,
    "scib_n_jobs": 1, "scib_require_success": True,
    "write_pdf_report": False, "include_latent_in_bundle": False,
    "include_cell_ids": False,
}

results = {}
for method in methods:
    print(f"\n===== Benchmarking {method.name} =====", flush=True)
    results[method.name] = benchmark_method(
        adata, method, seeds=METHOD_SEEDS,
        output_dir=RESULTS_ROOT / method.name.lower().replace("-", "_"),
        benchmark_config=benchmark_config,
        install_dependencies=False,
        finalize=True, copy_input_per_seed=False, overwrite=True,
    )

summary = pd.concat([
    result.summary().assign(method=name) for name, result in results.items()
], ignore_index=True)
summary = summary[["method", "method_seed", "section", "metric", "value"]]
summary_path = BASE / "Seurat_Dataset4_3seeds_summary.csv"
summary.to_csv(summary_path, index=False)

combined = BASE / "scRareBench_Seurat_Dataset4_3seeds_CCA_RPCA.zip"
with ZipFile(combined, "w", compression=ZIP_DEFLATED) as zf:
    for method_name, result in results.items():
        zf.write(result.archive_path, arcname=f"{method_name}/{Path(result.archive_path).name}")
    zf.write(summary_path, arcname=summary_path.name)
print(f"\nALL DONE. Combined ZIP: {combined}", flush=True)
