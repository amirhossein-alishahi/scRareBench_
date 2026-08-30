# scRareBench 0.10.5 design decisions
## 0.10.4 provenance and resume contract

A method seed is the only stochastic training replicate dimension allowed to vary inside one aggregate family. `method_training_hash` identifies the exact method/training-input contract; `evaluation_contract_hash` identifies the requested benchmark contract and includes the fixed benchmark seed. A changed training contract invalidates the latent cache. A changed evaluation contract may reuse a valid latent but must regenerate benchmark outputs, reports, status and bundles.

Final delivery treats the per-seed HTML run as the scientific identity anchor and binds companion status, result bundle and latent artifacts by run ID, method/dataset identity, training/evaluation hashes, benchmark seed, dataset/reference contract, cell order and latent-array hash. Scientific table row identities are invariant across method seeds; unavailable metric values may remain missing, but whole biological/scenario rows may not disappear silently.

## 0.10.3 atomic delivery and presentation contract

Multi-seed handoff creation is centralized in `finalize_multiseed_delivery()`. A delivery is published only after the merged HTML payload, aggregate/run consistency, expected seed coverage, per-seed status files, result bundles, and requested latent arrays pass validation. The final ZIP is then reopened, CRC-checked, member-checked, and the exact embedded HTML is deep-validated again before atomic publication. Colab notebooks do not execute the full standalone JavaScript dashboard inline; they validate it in Python and download/open the standalone HTML instead.


## 0.10.1 multi-seed contract

A **run** is one method/dataset/configuration realization with one `method_seed`. The evaluator's `random_state`/`BENCHMARK_SEED` is intentionally independent and remains fixed across seed replicates. Compatible seed replicates must have the same method identity, cell universe/order fingerprint and seed-independent method configuration hash. Duplicate method seeds are rejected to prevent pseudo-replication.

Numeric aggregate summaries use the arithmetic mean and **sample standard deviation (`ddof=1`)** over currently included runs. When only one run contributes, SD is missing/not applicable rather than encoded as zero. Categorical failure interpretations use mode/consensus plus agreement and counts.

Latent vectors, UMAP coordinates, Leiden clusters, Sankey flows and sandbox cluster state are run-specific objects and are never averaged. Aggregate views summarize metrics only; detailed geometry is inspected by selecting a seed or using independent small-multiple panels.

Excluding a run from an aggregate is a display/analysis-state operation, not deletion. The complete run remains embedded and inspectable. Browser-imported compatible runs become portable only when the user saves an updated self-contained HTML; duplicate seeds and configuration/cell-universe mismatches are rejected.

Notebook execution is resumable and seed-scoped. A following seed is not started until the current seed has produced a validated latent, benchmark outputs, report and bundle. The final multi-seed report can be rebuilt from completed per-seed reports even if they were produced in separate Colab sessions.

## Locked benchmark contract

- scRareBench evaluates user-generated integration latent spaces; it does not implement integration methods.
- The method boundary is the latent matrix plus strict cell alignment.
- Missing data are never converted to zero.
- Canonical benchmark seed is `42`.
- Official clustering uses kNN=15, Euclidean distance, Leiden reference resolution=1.0 and the canonical benchmark seed.
- Standard scIB-compatible evaluation and rare-cell evaluation remain separate result layers.
- No unvalidated combined “extended total” score is created.
- The submitted latent is never re-integrated or rewritten by the benchmark.
- Canonical scIB reference preprocessing fails closed if the required HVG implementation is unavailable.
- Dataset-specific rare scenario metadata is required unless rare evaluation is explicitly disabled.
- The GSE194122 paper scenario table is never silently applied to an unknown dataset.
- Barcode caches/result bundles use Unicode NumPy arrays and `allow_pickle=False`.
- `results.json` is the stable machine-readable result interface; HTML is a presentation/backward-compatibility format.
- Result bundles include cell-order/latent hashes and a SHA256 artifact inventory.


## Rare-recovery additions in 0.9.4–0.9.7

- Historical majority-vote precision/recall/F1 are retained for backward compatibility and explicitly interpreted as resolution-dependent cluster-label-transfer metrics.
- Full-space rare-cell ASW is reported alongside the historical within-subset ASW; the full-space value keeps abundant competitors in the geometry.
- Leiden-resolution-independent kNN local recovery is a primary complementary diagnostic: observed same-label neighborhood fraction is normalized against the cell type's global abundance.
- Best-cluster precision/recall/F1 is reported per cell type without requiring that the cell type win majority ownership of a cluster.
- Reference clustering metadata records exact Leiden flavor and iteration count; there is no silent implementation fallback.
- Resolution sensitivity is stored as a first-class output rather than choosing a resolution from the ground-truth number of labels.
- Historical failure-archetype precedence is frozen for backward compatibility. A separate resolution-aware v2 precedence adds `resolution_limited`; both primary and matched-rule columns are persisted.
- Inferred GR/LE/SR classes require a global rarity gate.

## Benchmark-only reference

- The user’s integration-method preprocessing is not altered.
- scRareBench independently constructs the reference used by the standard metrics.
- Default reference: configured GEX count layer, 4,000 HVGs unless dataset policy overrides, normalize-total 10,000, log1p, PCA up to 50 components.
- Method-independent PCR/reference comparisons use this package-controlled reference and the submitted latent.

## Rare-cell layer

- Six scenario slots are retained: GR-DL, GR-RM, LE-DL, LE-RM, SR-DL, SR-RM.
- A dataset may legitimately have empty slots.
- Registered scenario tables are attached to loaded datasets in memory; published source files are not rewritten.
- Failure-archetype thresholds are diagnostic/provisional and must be sensitivity-tested before strong biological claims.

## Metric semantics

- Metric direction is explicit metadata, not a UI assumption.
- Most benchmark/recovery scores are maximized.
- Within-type batch NMI and dominant wrong fraction are minimized.
- Diagnostic non-rare/rare ratios are stored separately from ordinary subset metrics.
- Near-zero ratio denominators are marked `unstable_denominator` instead of producing extreme values.

## Reproducibility

- Method seed and benchmark seed are separate concepts even if both are numerically 42 in current examples.
- Optional backend failures remain machine-readable.
- Method-specific config, HVGs, training history and manifests may be added to bundles without making those method libraries package dependencies.
- Legacy examples can be migrated into the new artifact schema without claiming their metrics were recomputed.

## Provisional / research-facing

- Exact DL/RM biological mapping for external datasets.
- Parent/sibling metadata for RM populations.
- Failure-archetype thresholds.
- Interpretation of resolution sweeps beyond the fixed reference resolution.
- Formal manuscript-level inferential policy across seeds/datasets. Release 0.10.1 implements descriptive N-seed aggregation (mean, sample SD, n, range, consensus/agreement) but intentionally does not invent a new inferential or stability-adjusted composite score.

## Deferred

- Built-in integration-method adapters.
- Corrected-expression or graph-only submission contracts.
- Formal hosted leaderboard.
- Automatic DL/RM inference for arbitrary custom datasets.
- Composite score that combines scIB and rare-preservation layers.

## 0.9.6 support-adjusted local recovery

The v0.9.4 raw kNN local-recovery score is retained but is context-only because populations with support smaller than the realized graph degree cannot attain raw score 1 even under perfect isolation. v0.9.6 adds `knn_local_recovery_adjusted`, normalizing observed same-label neighborhoods against the abundance null and the support/realized-degree achievable ceiling. This adjusted score is the primary local-recovery metric.

`preserved_fraction` remains the historical provisional majority-vote rule outcome and is explicitly context-only; it is not promoted as a primary endpoint. Canonical scenario labels are strict by default. UMAP generation failure must be surfaced as a latent-dimension fallback rather than silently relabeled as UMAP.

## 0.9.6 resolution-aware interpretation contract

The historical `failure_archetype` columns are immutable compatibility outputs. The current dashboard uses additive `failure_archetype_v2` as its primary interpretation. `resolution_limited` can become the v2 primary label only when the historical lineage-assimilation rule already matches and both support-adjusted local recovery and dominant-cluster capture exceed the configured guard thresholds. The historical `lineage_assimilation` match is not deleted; it remains in legacy columns and in the v2 matched-rule list.

Scenario metadata are validated before filtering. Invalid scenario codes or curated labels missing from the loaded data fail closed when `strict_scenario_labels=True`; exploratory mode records the exact drift. kNN graph degree is defined everywhere as the number of non-self neighbors. Singleton adjusted local recovery is undefined/not assessable and remains missing rather than zero.


## 0.9.7 export-consistency contract

The primary per-population v2 taxonomy remains unchanged from 0.9.6. Release 0.9.7 makes its aggregate context first-class in machine-readable CSV outputs: `rare_metrics_summary.csv` includes `resolution_limited_fraction`, and `scenario_metrics.csv` includes v2 archetype counts/fractions for every canonical scenario slot. `resolution_limited_fraction` is context-only: it means assimilation cannot be confirmed at the canonical clustering resolution under the v2 guard; it does **not** mean preserved. Empty scenario slots report zero archetype counts and missing fractions rather than fabricated zeros.
