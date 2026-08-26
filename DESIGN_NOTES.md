# scRareBench 0.9.1 design decisions

## Locked

- The package name remains `scRareBench` / `scrarebench`.
- It is described as scIB-compatible, not as an official new scIB version.
- The package evaluates a user-generated integration latent and does not reimplement integration methods.
- The official GSE194122 benchmark applies cell subsetting only and preserves source-relative order.
- `load_gse194122_benchmark()` is the preferred high-level dataset API; download and preparation remain separately callable for advanced users.
- Dataset construction never performs method-specific normalization, HVG selection, scaling, PCA, graph construction, or dimensionality reduction.
- A latent must have one row per benchmark cell; barcode verification is strict when supplied.
- Official clustering is package-controlled: kNN=15, Euclidean, Leiden resolution=1.0, seed=0.
- Standard evaluation uses pinned `scib-metrics==0.5.9` with every metric exposed by its Benchmarker, with both Leiden and KMeans NMI/ARI enabled.
- Classic batch silhouette is added separately.
- scIB aggregate scores remain separate from rare-cell summaries; no unvalidated extended total score is created.
- Standard, paper-style, and rare-cell outputs are stored in separate namespaces and combined in one self-contained HTML report.
- Metrics that cannot be inferred from a latent alone are listed explicitly as not applicable.

## Benchmark-only reference

- The user’s preprocessing is never altered.
- scRareBench independently creates a canonical reference from counts for standard metrics.
- Default reference: GEX only, 4,000 batch-aware HVGs, normalize-total 10,000, log1p, PCA 50.
- PCR comparison uses this canonical pre-integrated PCA and the submitted latent.

## Provisional

- Exact six-scenario biological mapping.
- Parent/sibling lineage metadata for RM populations.
- Failure-archetype thresholds.
- Resolution-sweep interpretation beyond the paper reference resolution.
- Future input contracts for corrected expression, cell-cycle scores, and pseudotime.

## Deferred

- Built-in adapters for integration methods.
- Corrected-expression and graph-only submissions.
- A formal leaderboard.
- Automatic DL/RM inference for custom datasets.
- A composite score combining standard scIB aggregates and rare-cell preservation.


## Method-independent runtime contract

The benchmark package never depends on scVI, Harmony, or another integration implementation. The method boundary is the latent representation plus cell alignment. Notebook dependency installation is external to the package and uses the shared stdlib-only runtime installer so adding a new method does not require changing pyproject.toml or benchmark code.
