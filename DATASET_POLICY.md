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
