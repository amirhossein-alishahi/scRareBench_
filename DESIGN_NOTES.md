# scRareBench 0.10.6 benchmark design

This document summarizes the public benchmark contract implemented by scRareBench.

## Method boundary

scRareBench evaluates user-generated integration or batch-correction latent spaces. It does not implement or train the integration method itself.

The submission boundary is:

- one latent row per benchmark cell;
- exact cell identity/alignment;
- optional method provenance files.

The submitted latent is evaluated as provided and is not re-integrated or rewritten by scRareBench.

## Seed contract

Method and benchmark randomness are separate:

- `method_seed` identifies a training/integration replicate and may vary across runs;
- the benchmark/evaluation seed is fixed across comparable method replicates;
- the canonical benchmark seed is `42`.

Multi-seed summaries aggregate scalar/tabular metrics only. Latents, UMAP coordinates, cluster assignments, Sankey topology, and cell-level states remain seed-specific.

Numeric multi-seed summaries use the arithmetic mean and sample standard deviation (`ddof=1`). A single contributing run has no fabricated zero standard deviation.

## Cell identity and missing data

Latent alignment is validated against benchmark cells. Silent cell deletion, duplication, or cross-dataset substitution is rejected.

Unavailable metrics and absent benchmark combinations remain missing rather than being converted to zero.

## Benchmark-controlled reference

Method preprocessing is owned by the method developer.

For standard scIB-compatible metrics, scRareBench constructs an independent benchmark-only reference from the configured counts. The default reference uses:

- the configured GEX count layer;
- 4,000 HVGs unless the dataset policy overrides it;
- total-count normalization to 10,000;
- log1p;
- PCA up to 50 components.

Canonical reference preprocessing fails closed when the required configured implementation is unavailable.

## Reference clustering

The default benchmark clustering contract uses:

- kNN = 15;
- Euclidean distance;
- Leiden reference resolution = 1.0;
- benchmark seed = 42.

Resolution sweeps are stored as sensitivity information and do not replace the explicit reference resolution.

## Rare-cell evaluation

Rare-cell evaluation is an additional layer beside the standard benchmark metrics.

A registered dataset may provide the six scenario slots:

```text
GR-DL  GR-RM  LE-DL  LE-RM  SR-DL  SR-RM
```

Custom datasets may instead provide only `rare_types`. In that case, generic rare-recovery metrics are available without requiring the six-state taxonomy.

A scenario table is applied only to the dataset for which it was provided or registered; scRareBench does not silently reuse another dataset's taxonomy.

## Rare-recovery metrics

The rare-cell layer includes complementary diagnostics such as:

- full-space selected-cell ASW;
- historical cluster-label-transfer precision/recall/F1;
- best-cluster precision/recall/F1;
- support-adjusted kNN Local Recovery;
- observed, expected, and maximum-achievable same-label kNN fractions;
- inverse-purity / dominant-cluster capture;
- within-type batch-dependence diagnostics;
- resolution-sensitivity tables;
- legacy and resolution-aware failure-archetype fields.

Metric direction is explicit metadata. Context-only quantities are kept separate from metrics intended for maximization/minimization.

Failure-archetype outputs are diagnostic interpretations and should be read together with their underlying metrics and sensitivity views.

## Result and provenance contract

`results.json` is the primary machine-readable benchmark result.

Result artifacts record benchmark configuration and relevant method/dataset identity. Bundles may include cell-order and latent hashes, artifact SHA256 information, method provenance files, reports, CSV tables, and optional latent/barcode arrays.

Barcode arrays are stored as Unicode NumPy arrays without pickle-based object serialization.

## Evaluation layers

Standard scIB-compatible evaluation and rare-cell evaluation remain separate result layers. scRareBench does not create an unvalidated combined score that merges them into a single benchmark objective.
