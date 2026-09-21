# Dataset evaluation policy

scRareBench keeps **method preprocessing** separate from **benchmark evaluation preprocessing**.

## Method preprocessing

The method developer owns all preprocessing needed to produce the submitted latent representation. scRareBench does not normalize, select HVGs, scale, compute PCA, or otherwise alter the method pipeline.

## scIB reference preprocessing

The standard scIB-compatible layer creates its own benchmark-only reference from the configured count source. `ScibEvaluationConfig.hvg_batch_mode` controls the reference HVG policy:

- `evaluation_batch` (default): select reference HVGs using the evaluation batch labels.
- `global`: select reference HVGs globally while preserving the evaluation batch labels for downstream batch metrics.

The built-in dataset profile chooses a validated policy where one is known. In particular, the mBDRC renal-cortex profile uses global reference HVG selection because batch-aware Seurat-v3 LOESS was numerically unstable for the registered donor/assay evaluation-batch definition during validation.

This is a **dataset policy**, not a method-specific exception. Every submitted method on the same dataset is evaluated against the same benchmark reference policy.

## Built-in dataset-level preparation

Dataset-level preparation is allowed only when it is part of a fixed, audited source contract rather than an integration-method choice. The downloaded H5AD is never overwritten.

- Dataset 0 and Dataset 1 retain their historical GSE194122 flows unchanged.
- Dataset 2 retains its historical `donor_id × assay` evaluation batch and global scIB-reference HVG policy unchanged.
- Dataset 3 uses `celltype_subset` as the evaluation label and `donor_id` as the evaluation batch. Canonical counts are aligned from `adata.raw.X` into `layers["counts"]`. Its scIB reference uses global Seurat-v3 HVGs because the audited donor-aware span-0.3 LOESS fit was numerically singular; this does not change the downstream evaluation batch.
- Dataset 4 uses `cell_type` and `donor_id`, with canonical counts aligned from `adata.raw.X`. Its audited donor-aware scIB-reference HVG selection succeeds and therefore keeps the default `evaluation_batch` policy.
- Dataset 5 first applies the fixed official scvi-tools PBMC source-cell QC filter in memory, reproducing 152,145 retained cells from the pinned 161,764-cell source. It uses `celltype.l2` and `orig.ident`; raw RNA counts remain in `adata.X`, so the built-in count-layer metadata is intentionally empty/None and canonical scIB reads `X` directly.

The package validates source shape, cell/feature identifiers, required label/batch fields, count integrity and registered scenario labels before benchmarking. Dataset 5 additionally validates the exact retained ordered-cell SHA256. If these invariants drift, loading fails closed and the dataset must be re-audited.

These dataset contracts do not normalize, log-transform, select method HVGs, scale, compute PCA, train an integration model, or otherwise alter user-owned method preprocessing.

