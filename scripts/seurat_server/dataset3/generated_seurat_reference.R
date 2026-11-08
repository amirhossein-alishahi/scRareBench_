

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

if (packageVersion("Seurat") < package_version("5.0.0")) {
  stop("Seurat >= 5.0.0 is required.")
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
      " fixed Dataset 3 HVGs, got ",
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
  project = "scRareBench_Dataset3_Seurat"
)

rm(counts)
gc()

# Split layers by donor batch. HVGs were already fixed globally in Python.
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$BATCH)
VariableFeatures(obj) <- selected_features

# Normalize on the full raw-count feature universe.
obj <- NormalizeData(obj, verbose = FALSE)

# Scale/PCA only the fixed global HVGs.
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
  emb <- Embeddings(object[[reduction_name]])[, seq_len(ndims), drop = FALSE]
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
  message("[Seurat] Reusing previous RPCA embedding")
} else {
  obj <- IntegrateLayers(
    object = obj,
    method = RPCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated.rpca",
    dims = seq_len(ndims),
    verbose = FALSE
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
  dims = seq_len(ndims),
  verbose = FALSE
)
save_latent(obj, "integrated.cca", "seurat_cca_latent.csv")
gc()

writeLines(
  selected_features,
  file.path(workdir, "seurat_selected_genes.txt")
)

sink(file.path(workdir, "seurat_session_info.txt"))
print(sessionInfo())
sink()
