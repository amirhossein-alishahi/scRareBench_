# Interactive reports and multi-report comparison

scRareBench treats the HTML report as a first-class scientific handoff artifact. A benchmark result can be inspected without Python, a notebook, or a server: open the generated HTML file in a modern browser.

The interactive layer does **not** replace the machine-readable result. `results.json`, CSV exports, provenance hashes, and result bundles remain the reproducible data contract; the HTML provides a portable interface for exploring those results.

## 1. Single-run interactive report

A single-run report opens on the **Overview** page and exposes the main benchmark sections as tabs.

![Interactive report overview](docs/images/report_overview.jpg)

### Overview

Use the Overview as the first-pass health check. It summarizes:

- number of cells, cell types, batches, and latent dimensions;
- reference-cluster count;
- rare-population and rare-cell counts;
- scIB biological conservation / batch correction / total aggregates when available;
- selected rare-cell summary metrics;
- benchmark warnings that affect interpretation.

The top seed card identifies the active method seed and links to **Run & Provenance**.

### Metrics

The Metrics tab provides overall / rare / non-rare benchmark tables. Use it for the standard integration-performance view before drilling into individual rare populations.

### scIB

The scIB tab contains individual scIB-compatible metrics and aggregate scores together with applicability/status information. Metrics that cannot be computed remain unavailable rather than being converted to zero.

### Rare-cell Explorer

The Rare-cell Explorer is the main diagnostic view for low-frequency populations.

![Rare-cell Explorer](docs/images/report_rare_explorer.jpg)

It combines:

- historical cluster-majority precision / recall / F1;
- best-cluster precision / recall / F1;
- support-adjusted kNN Local Recovery;
- raw support-limited local recovery and neighborhood context;
- inverse-purity / dominant-cluster capture;
- within-type batch dependence;
- rare-scenario metadata where available;
- legacy and resolution-aware failure interpretations;
- stored resolution and threshold sensitivity views.

Failure labels are diagnostic annotations. Use the underlying metrics and sensitivity views when making biological claims.

### UMAP

When an embedding is available, UMAP views allow interactive coloring by true label, batch, scenario, cluster, prediction, and failure state. Depending on the report payload, marker-gene overlays and cell selection/export can also be available.

UMAP is a visualization layer. The canonical benchmark metrics are computed from the benchmark contract and are not replaced by manual visual interpretation.

### Sankey

The Sankey tab exposes label → cluster → prediction/failure flow. It is useful for seeing where a rare population is being split, absorbed, or relabeled.

### Run & Provenance

For a single run, this tab records the seed/run identity and reproducibility information stored with the report. The report and bundle contain stronger machine-readable provenance, including hashes where configured.

### Reproducibility and Figures

The Reproducibility tab shows benchmark configuration and environment/provenance information. Figures collects generated static plots and provides an easier handoff for presentations or manuscripts.

---

## 2. Multi-seed reports

A multi-seed report stores several stochastic method runs in one portable HTML workspace.

![Multi-seed seed stability](docs/images/report_seed_stability.jpg)

The top **Multi-seed workspace** card displays:

- stored runs;
- runs currently included in aggregate statistics;
- planned seeds;
- the currently selected detailed run.

### Aggregate versus detailed run

These are separate concepts:

- **Aggregate view** summarizes metric values over included seeds.
- **Detailed run** selects which seed-specific geometry and run-level evidence you are inspecting.

Changing the detailed run does not change which runs contribute to the aggregate.

### Seed Stability

The Seed Stability tab visualizes per-seed metric values. For multiple included runs, numeric summaries use the arithmetic mean and **sample SD** (`ddof=1`). A single available seed does not fabricate `SD = 0`.

Use this tab to distinguish a stable method result from a result that is strongly seed-sensitive.

### Runs & Seeds

The Runs & Seeds manager lets you inspect stored seed identities and control aggregate membership.

A seed can be excluded from aggregate statistics without deleting it. The complete run remains in the report so the exclusion is reversible and auditable.

### Save Updated Report

After changing include/exclude state or importing compatible runs into a report, **Save Updated Report** creates a new self-contained HTML containing the updated workspace state.

This is a presentation/analysis state operation. It does not retrain a method or alter canonical per-seed benchmark outputs.

### What is never averaged

Multi-seed aggregation applies to scalar/tabular metrics. The following remain run-specific:

- latent coordinates;
- UMAP coordinates;
- Leiden cluster assignments;
- Sankey topology;
- cell-level prediction/failure state.

This prevents a visually smooth but scientifically meaningless “average embedding.”

---

# 3. Multi-Report Comparator

Open:

```text
comparator/scRareBench_Multi_Report_Comparator_v10.html
```

No Python installation or web server is required. The comparator is a client-side HTML application; it loads pinned Plotly/JSZip browser libraries from public CDNs while keeping imported report data local to the browser.

![Multi-Report Comparator](docs/images/comparator_methods_v10.jpg)

## Importing reports

Use **Import reports** or drag files into the import area.

Supported inputs include:

- current scRareBench v0.10.5/schema-1.6 single-run interactive HTML;
- current multi-seed interactive HTML;
- scRareBench result/delivery ZIPs containing compatible interactive HTML or `results.json`;
- standalone `results.json`;
- compatible legacy self-contained scRareBench HTML.

Files are processed locally by the browser. The comparator does not require uploading results to an external service.

## Dataset and method identity

Current schema-1.6 reports carry method, dataset, seed, configuration, and provenance identity in the report wrapper. Comparator v10 uses that identity when grouping reports.

For current reports it prefers the stored `dataset_key` (for example `gse194122`) instead of inventing an anonymous `Dataset A` label.

When separate files contribute seeds to the same method × dataset group, the comparator checks compatible run identity before aggregating. It uses available evaluation/method configuration hashes, benchmark seed, dataset fingerprint/reference contract, and cell order.

- Exact duplicate runs are ignored rather than double-counted.
- A same-seed report with a different run identity is rejected as a conflict.
- Different dataset/cell-universe identities are not silently merged.

## Seed view selector

The top **Seed view** control changes the comparison level:

### Aggregate included seeds

Use this for the main method comparison. Each method × dataset group is represented by the aggregate across runs marked as included in its source report.

### Show individual seeds

Each seed becomes its own comparison row (for example `scVI · seed 42`, `scVI · seed 123`). This is useful for diagnosing stochastic variation directly.

### Specific seed

Select one seed number and compare only methods that contain that seed. Methods without the selected seed remain absent rather than being filled with zero.

---

## Comparator tabs

### Import & Data

Shows all loaded groups, method/dataset labels, stored/included/planned seed coverage, dimensions, cell counts, rare-type counts, and import messages.

If automatic labels are ambiguous for older reports, the registry allows label editing without modifying the original result files.

### Methods on Dataset

Choose one dataset and compare every imported method available for it.

The view provides:

- metric-family selection;
- metric heatmap;
- selected-metric bar chart;
- comparison table;
- PNG export for major plots.

### Datasets for Method

Select one method and inspect its performance across the datasets currently imported into the comparator.

Coverage gaps remain explicit.

### Metric Plotter

Build an X/Y comparison for one dataset. Metric direction is read from the report metric registry when available. This allows direction-aware “best” interpretation and Pareto visualization without assuming every metric is higher-is-better.

Context-only quantities should not be treated as ranking objectives unless the user explicitly chooses an interpretation.

### Rare-cell Deep Dive

Compare cell-type-level rare-recovery behavior across methods. This includes support-adjusted kNN recovery and best-cluster diagnostics alongside historical cluster-majority metrics.

Local recovery is independent of the Leiden resolution choice but remains dependent on the benchmark kNN graph definition; the report stores the relevant graph context.

### Coverage & Ranking

Coverage and performance are kept separate. A method is not penalized as score zero merely because a dataset has not been benchmarked for it.

Use strict common-dataset comparisons when you need paired method ranking over the same evaluated dataset set.

---

# 4. Recommended workflow

A practical workflow is:

1. Run each integration method and let it produce a scRareBench report/bundle.
2. For stochastic methods, use multiple method seeds while keeping the benchmark seed fixed.
3. Open each interactive report and inspect Overview, Rare-cell Explorer, and seed stability before ranking methods.
4. Open `comparator/scRareBench_Multi_Report_Comparator_v10.html`.
5. Import the final reports or result ZIPs for all methods.
6. Start in **Aggregate included seeds** mode.
7. Compare methods on the same dataset using global/scIB metrics and rare-cell metrics separately.
8. Switch to **Show individual seeds** if a method appears unstable.
9. Use **Rare-cell Deep Dive** to identify which populations drive a global difference.
10. Use **Coverage & Ranking** to verify that a ranking is not driven by different dataset coverage.
11. Export plots or keep the standalone HTML artifacts with the result bundles for handoff/review.

---

# 5. Output and privacy model

The interactive report and comparator are static browser applications. They can be opened from local disk and do not require a hosted backend for normal use.

This makes them useful for:

- collaborator handoff;
- review meetings;
- supplementary benchmark inspection;
- offline result exploration;
- comparing methods without rebuilding a Python dashboard environment.

For automated pipelines or long-term analysis, prefer `results.json`/CSV/bundles as the data interface and treat the HTML as the interactive evidence layer.
