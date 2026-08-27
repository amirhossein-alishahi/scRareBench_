# scRareBench design decisions

## Core contract

- scRareBench benchmarks a **user-generated latent representation**; it does not implement integration methods.
- The method boundary is `latent + cell identity`.
- Dataset construction never performs method-specific normalization, HVG selection, scaling, PCA, graph construction, or dimensionality reduction.
- Cell alignment is strict when barcodes are supplied; DataFrame-indexed latent representations are preferred.
- Package-controlled clustering uses kNN=15, Euclidean distance, Leiden resolution=1.0, and seed=0 by default.
- The standard benchmark backend is pinned to `scib-metrics==0.5.9` for reproducibility.
- scIB aggregate outputs remain separate from rare-cell summaries; no unvalidated composite score is created.
- Metrics that cannot be inferred from the available latent/count contract are explicitly reported as not applicable.

## High-level API

- `load_dataset(selector)` loads a registered AnnData and attaches evaluation metadata in memory.
- `register_dataset()` records the evaluation contract for a user AnnData without changing the method pipeline.
- `benchmark_latent()` is the normal method-developer entry point: latent in, standardized benchmark and reports out.
- `attach_latent()` / `EvaluationConfig` / `evaluate_latent()` remain available for low-level control.
- There is no integration-method registry and no `methods/` package.

## Benchmark-only reference

- The user's preprocessing is never altered by scRareBench.
- The scIB-compatible layer creates an independent reference from the configured count source.
- The default reference workflow uses GEX features, 4,000 HVGs, library-size normalization to 10,000, `log1p`, and PCA with 50 components.
- Dataset profiles may select either evaluation-batch-aware or global reference HVG selection.

## Rare-cell policy

- Rare metrics are computed only from rare/scenario metadata actually associated with the current dataset or explicitly supplied by the user.
- A custom dataset never inherits the GSE194122 rare taxonomy implicitly.
- DL/RM topology is not inferred automatically from abundance alone.
- Failure-archetype thresholds remain diagnostic/provisional unless separately validated.

## Runtime policy

- The runtime helper is method-agnostic.
- User method dependencies are passed explicitly through `extra_requirements` / `extra_imports`.
- ABI-sensitive packages already present in notebook environments are constrained during installation.
- Any **new** `pip check` conflict introduced by runtime setup is fatal; pre-existing environment conflicts are reported separately.

## Deferred

- built-in integration-method adapters;
- corrected-expression or graph-only submissions;
- a formal leaderboard;
- automatic biological DL/RM inference for arbitrary custom datasets;
- a composite score combining standard scIB aggregates with rare-cell preservation.
