# scRareBench v0.9.1 — dataset-level HVG policy support

This release keeps the benchmark core method-independent and adds one generic scIB reference-preprocessing option: `ScibEvaluationConfig.hvg_batch_mode`.

- `evaluation_batch` (default): historical behavior; select scIB reference HVGs within the evaluation batch.
- `global`: select scIB reference HVGs globally while keeping the original evaluation batch labels for downstream scIB batch metrics.

The option was added because the executed all-dataset audit showed that mBDRC (dataset 2) has stable global Seurat-v3 HVG selection but numerically singular within-batch LOESS for the donor/assay batch definitions tested. Dataset 0 retains its batch-aware policy.

This is a dataset/preprocessing capability, not a method-specific patch. scVI, Harmony, MrVI, and future methods can share the same dataset policy.
