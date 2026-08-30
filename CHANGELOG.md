# Changelog

## 0.10.5 — high-level multi-seed developer release

- Added generic `MethodSpec`, `MethodOutput`, and `benchmark_method()` orchestration.
- Kept integration-method implementations and dependencies user-controlled.
- Integrated multi-seed aggregation, rare-aware metric registry, support-adjusted local recovery, and provenance hardening.
- Added generic high-level and low-level Colab templates; all shipped notebooks are pinned to v0.10.5.

## 0.10.4 — provenance, resume and handoff contract freeze

- Separated `method_training_hash` from `evaluation_contract_hash`; benchmark seed is now part of the evaluation invariant rather than volatile seed state.
- Colab resume logic reuses a latent only when the exact training-input contract matches, and re-runs evaluation/report/bundle when evaluation settings change. Harmony additionally binds its PCA preprocessing seed.
- Added stronger dataset/reference contract hashing over ordered cells, labels, batches, rare scenarios and feature IDs.
- Bound report, status, result bundle and latent artifacts by run/config/dataset/cell/latent identity; valid-but-wrong companion artifacts now fail closed.
- Added strict cross-seed scientific row identity checks and deep stored-vs-recomputed aggregate validation.
- Added current/legacy bundle canonicalization so validated 0.10.0–0.10.3 seed artifacts can still be recovered without weakening the 0.10.4 contract.
- Final delivery ZIPs now re-hash every member listed in `delivery_manifest.json` after archive creation.
- Non-finite configuration values are rejected; results/bundle schema bumped to `1.6`.
- No rare-recovery metric formula, failure threshold, scIB metric definition or ranking rule changed in this release.

## 0.10.3 — reproducibility and UI interpretation hardening

- Canonical seed sorting for order-independent multi-run aggregation/report identity.
- Strict-majority failure consensus with explicit no-consensus semantics.
- Deterministic set/NumPy configuration serialization and contextual hash errors.
- Six-scenario full-width pivot with raw-table export retained.
- Single-run presentation mode without redundant aggregate/seed-stability panels.
- Seed Stability dot plot with mean and ±SD band; Comparator v9 rank dot plot with 1 = best.
- Wide-table scroll cues, metric-aware score formatting, clearer latent-delivery errors, and synchronized six-notebook release contract.


## 0.10.2 — validated multi-seed delivery hardening

- Added `finalize_multiseed_delivery()` as one atomic report/summary/ZIP finalization path.
- Added deep dashboard payload validation for required benchmark tables, rare-cell tables, UMAP coordinate payload lengths, encoded point fields, reproducibility metadata, aggregate/run consistency, and expected seed coverage.
- Final delivery ZIPs are reopened, CRC-tested, member-checked, hash-checked, and the exact HTML stored inside the ZIP is re-validated before success is returned.
- Missing/empty per-seed bundles, status files, or requested latent files now fail loudly instead of producing a partial handoff.
- All six Colab notebooks now use the package finalizer, verify final artifacts before download, and avoid inline execution of the large standalone dashboard in Colab output cells.
- Added direct per-seed interactive HTML reports to the final handoff ZIP for easier inspection.
- Retained the 0.10.1 canonical configuration-hash fix and backward compatibility for 0.10.0 per-seed reports.

## 0.10.1 — multi-seed configuration identity hotfix

- Fixed false `Incompatible run: configuration_hash differs` failures when seed-specific embeddings produced different realized Leiden cluster counts.
- Excluded realized clustering/kNN outcomes from configuration identity while preserving requested scientific settings in the hash.
- Added backward-compatible canonicalization so already-generated v0.10.0 per-seed HTML reports can be merged without retraining.
- Added regression coverage for differing `reference_n_clusters` and realized kNN degree statistics.
- Updated all six Colab example notebooks with the corrected version contract and stronger multi-seed preflight checks.

## 0.10.0 — 2026-08-15

- Added first-class N-seed provenance (`method_seed`, `run_id`, seed-independent `configuration_hash`, dataset/cell-order fingerprint) and bumped results/bundle schema to `1.5`.
- Added multi-seed aggregation with mean, sample SD, n, min/max and categorical failure-archetype consensus/agreement; SD is missing/not-applicable for n=1.
- Interactive report now stores one or many complete seed runs, provides aggregate and individual views, small-multiple UMAPs, a Seed Stability tab, dynamic run import, duplicate/incompatibility guards, non-destructive exclude/restore, edit history and portable Save Updated Report persistence. UMAP/latent coordinates, cluster IDs and Sankey topology are never averaged.
- Comparator v8 imports single- or multi-seed HTML/ZIP/JSON artifacts, supports aggregate/individual/specific-seed views, merges compatible separately imported seed runs, displays seed coverage and error bars, and retains context-only ranking guards.
- All six Colab notebooks now accept scalar or sequence `METHOD_SEED`, use a fixed benchmark/preprocessing seed, perform method smoke tests, checkpoint seed-specific artifacts, resume completed seeds, gate each next seed on full end-to-end success, clean GPU memory, and produce one final multi-seed HTML/ZIP.
- Fixed scVI cached-model loading to use `device="auto"`, avoiding Lightning interpreting GPU device index `0` as `devices=0`.
- Notebook runtime installation now selects the wheel matching the source-tree version exactly instead of lexicographic wheel order, preventing a stale `0.9.x` wheel from being chosen over `0.10.0` when both are present.
- No rare-recovery metric definition, failure threshold or resolution-aware interpretation rule was changed in this release.

## 0.9.7 — 2026-08-14

### Final export consistency / release polish
- Added `resolution_limited_fraction` to `rare_metrics_summary.csv` so the machine-readable summary exposes the same v2 diagnostic context as the dashboard/PDF. It is context-only and must not be interpreted as preserved fraction or used for method ranking.
- Added per-scenario resolution-aware v2 archetype counts and fractions to `scenario_metrics.csv`; all six canonical scenario slots remain explicit, with zero counts and missing fractions for genuinely empty slots.
- Registered `resolution_limited_fraction` as a context-only metric and surfaced it consistently in the interactive report.
- Removed the remaining style-lint traps in reporting/scIB compatibility code without changing runtime semantics.
- Results/bundle schema bumped to `1.4`; Comparator v7 and all six Colab notebooks synchronized to 0.9.7.

### Validation
- Added regression tests for measured-zero vs missing v2 fractions and the complete per-scenario v2 export contract.
- Re-ran warnings-as-errors tests, source/wheel comparison, browser report/comparator smoke tests, PDF generation, notebook/API validation, manifest/hash verification, and side-by-side Colab project-ZIP discovery.

## 0.9.6 — 2026-08-14

### Final scientific hardening
- Added additive resolution-aware failure taxonomy v2 while retaining the historical majority-vote taxonomy unchanged.
- Added `resolution_limited`: a legacy assimilation match is guarded when support-adjusted local recovery and dominant-cluster capture remain high.
- Invalid rare-scenario codes are validated before filtering; strict/canonical runs now fail closed and exploratory runs record exact dropped rows/codes.
- kNN realized-degree provenance excludes diagonal/self-loop edges exactly as the metric implementation does.
- Singleton support-adjusted recovery remains explicitly not assessable (`NaN`) and is explained as such in the UI.

### Reports / comparison
- Dashboard uses resolution-aware v2 failure labels as the primary visual interpretation and exposes the legacy label side-by-side.
- Failure-threshold sensitivity lab implements the same v2 precedence/rules client-side.
- UMAP coloring and lasso export include both v2 and legacy failure taxonomies.
- PDF and static failure summaries prioritize v2 while retaining separate legacy summaries.
- Comparator v6 consumes schema 1.3 and prefers v2 failure labels while preserving legacy fields.

### Reproducibility / release
- Results and bundle schemas bumped to 1.3.
- Added adversarial regression tests for resolution-limited classification, invalid-code drift, self-loop degree consistency, and context-only diagnostic directions.
- Colab notebooks synchronized to 0.9.6 and validated against side-by-side project ZIP discovery.

## 0.9.5 — 2026-08-14

### Scientific correctness
- Added `knn_local_recovery_adjusted`, correcting the support-dependent ceiling of the v0.9.4 raw kNN score using the per-population support and realized graph degree.
- Retained `knn_local_recovery` unchanged for backward comparison, but registered it as context-only.
- Added `knn_max_achievable_fraction`, realized-degree provenance, and per-cell adjusted recovery in cluster/lasso exports.
- Made canonical scenario-label drift fail closed by default; exploratory continuation explicitly records absent curated labels.
- Kept the historical `preserved_fraction` and failure-rule contract but downgraded it to a legacy/provisional context-only outcome because its precision/recall inputs remain majority-vote dependent.

### Reports / UI / comparator
- Made support-adjusted kNN recovery primary in Overview, Rare Explorer, scenario plots, heatmaps, PDF summaries, results methodology, and Comparator v5.
- Raw support-limited kNN recovery remains visible beside the adjusted score.
- UMAP failures now surface an explicit latent-dimensions-1–2 fallback warning.
- Sandbox text now states that batch exclusion filters existing clusters; integration/clustering are not recomputed.
- Results and bundle schema bumped to 1.2.

### Validation
- Added adversarial support-ceiling tests across rare supports 4/8/12/16/200, absorption, singleton/not-assessable behavior, metric-direction safety, and strict annotation-drift behavior.
- Updated browser/comparator/release contracts for v0.9.5 while preserving all prior functional views and exports.

## 0.9.4 — 2026-08-14

### Rare-cell scientific correctness
- Retained all historical majority-vote metrics and added Leiden-resolution-independent kNN local recovery.
- Added best-cluster precision/recall/F1 to remove the structural cluster-majority ownership ceiling as a complementary diagnostic.
- Added full-space selected-cell ASW while retaining the historical within-subset ASW.
- Fixed measured-zero `preserved_fraction`, global rarity gating for inferred LE/SR populations, and silhouette seed propagation.
- Canonical scIB preprocessing now fails closed when the declared raw-count layer is missing.
- Leiden flavor and iteration count are explicit, persisted, and no longer silently switched.
- Failure-archetype precedence is explicit and all matched archetypes are retained.
- Added rare recovery across the configured resolution sweep without tuning resolution to ground-truth label count.

### Interactive report / handoff artifact
- Added live failure-threshold sensitivity sliders with canonical-result warning.
- Added stored-resolution what-if sandbox for batch exclusion and label merging; exploratory values never overwrite canonical benchmark files.
- Added optional marker-gene UMAP overlays with compact quantized payloads.
- Added lasso/box cell selection and CSV barcode export.
- Added plain-language metric glossary, cluster-count warning, resolution-sensitivity view, and local-recovery hover information.
- Report generation no longer mutates caller AnnData when UMAP must be computed.
- Removed duplicate standalone self-contained Sankey artifact; the interactive Sankey tab is retained.
- Dashboard CSS/JS/template moved to package assets and are still inlined into the final self-contained HTML.
- Compact categorical/coordinate payload encoding reduces report data size while preserving offline behavior.

### Comparator / artifacts
- Results and bundle schemas bumped to 1.1.
- Comparator v4 understands the new rare metrics and context-only metric directions.
- Rare Deep Dive prefers kNN local recovery when available and preserves legacy F1 for older reports.
- PDF summary prioritizes full-space ASW, local recovery, best-cluster recovery, and resolution sensitivity while preserving historical tables.

### Validation
- Added adversarial tests for absorption, majority-vote ceiling, best-cluster recovery, zero preservation, abundant batch-specific populations, failure-rule overlap, seed propagation, canonical count-layer enforcement, kNN graph edge cases, full-space ASW sampling, and ratio-status semantics.
- Final release validation: 69 tests pass with Python warnings promoted to errors; dashboard/comparator JavaScript syntax checks and browser runtime smoke tests pass without console/page errors.

## 0.9.3 — 2026-08-14

### Correctness / benchmark contract
- Canonical benchmark seed synchronized to 42.
- Method and benchmark seed separated in release notebooks.
- Rare-cell evaluation now requires dataset-specific scenario metadata by default.
- Legacy GSE194122 scenario fallback requires explicit opt-in.
- Canonical scIB HVG preprocessing fails closed; exploratory fallback requires explicit opt-in.
- Optional scIB backend failures now persist status, exception type and message.
- Non-rare/rare ratios moved out of the ordinary subset table; near-zero denominators are flagged unstable.
- Metric directions added for correct ranking/Pareto behavior.

### Reproducibility / artifacts
- Added strict `results.json` schema v1.0.
- Added stronger bundle manifest, latent hash, cell-order hash and artifact SHA256 inventory.
- Added dataset provenance in loaded AnnData/result bundles.
- Added method-specific provenance-file support.
- Barcode arrays are written as Unicode with pickle disabled.
- Added legacy bundle migration utility with explicit no-recomputation provenance.

### Reports / comparison
- PDF redesigned as a readable scientific summary rather than an over-wide raw-table dump.
- Comparator v3 supports JSON, result ZIP and legacy HTML/ZIP import.
- Comparator Pareto/“Best”/ranking logic is direction-aware.
- Added strict common-dataset ranking mode.
- PNG export retained for all major comparator plots.

### Quality assurance
- Added release-contract, ratio, scenario-policy, scIB-failure, barcode and end-to-end synthetic evaluation regression tests.
- Release notebooks synchronized to 0.9.3 and cleaned of stale outputs.

## 0.9.2
- Registered six-state scenario metadata for selected external benchmark datasets.
- Expanded interactive reporting/dashboard workflows.

## 0.9.1
- Added dataset-level scIB HVG policy support.

## Earlier releases
Historical runtime/UI audit documents are retained for provenance. They describe the release in which those decisions were introduced and are not current-version specifications.
