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

## Audited built-in external-dataset contracts

The registered loader standardizes dataset-specific source preparation while
keeping integration-method preprocessing user-owned.

| Selector | Dataset | Evaluation batch | Canonical counts | scIB HVG policy | Method-HVG guidance |
|---|---|---|---|---|---|
| 2 | mBDRC renal cortex | `donor_id × assay` | existing validated contract | global | existing validated global policy |
| 3 | Wu breast-cancer atlas | `donor_id` | aligned `adata.raw.X` in `layers["counts"]` | global | Seurat-v3 4000 global, span 0.3 |
| 4 | COVID-19 autoimmunity PBMC | `donor_id` | aligned `adata.raw.X` in `layers["counts"]` | evaluation batch | Seurat-v3 4000 donor-aware, span 0.3 |
| 5 | NYGC / Seurat v4 CITE-seq PBMC | `orig.ident` | `adata.X` after fixed official PBMC QC | evaluation batch | Seurat-v3 4000 batch-aware, span 0.3 |

Dataset 3 uses global reference HVGs because donor-aware Seurat-v3 at span 0.3
was numerically singular in the audited source, while global selection passed.
Datasets 4 and 5 passed evaluation-batch-aware selection at span 0.3.

For Dataset 5, `load_dataset(5)` applies the audited source-cell filter before
benchmark metadata are attached: protein log-library in `(7.6, 10.3)`, more
than 150 detected proteins, removal of `celltype.l2 == "Doublet"`, and
mitochondrial percentage below 12. The registered source must retain exactly
152,145 of 161,764 cells, with the audited retained-cell identity/order hash.
The downloaded H5AD remains unchanged.

The `method_hvg` entry exposed by `dataset_info()` for datasets 3--5 is
guidance/provenance only. The loader never executes HVG selection,
normalization, PCA, neighbors, clustering, or an integration method.

