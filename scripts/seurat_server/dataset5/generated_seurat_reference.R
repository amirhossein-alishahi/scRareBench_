
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
