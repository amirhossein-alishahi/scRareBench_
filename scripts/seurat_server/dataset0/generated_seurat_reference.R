
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
