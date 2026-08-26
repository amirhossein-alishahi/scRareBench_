# scRareBench 0.9.2

`scRareBench` evaluates scRNA-seq integration latent spaces with two deliberately separated layers:

1. **Standard scIB-compatible evaluation**, using the pinned `scib-metrics==0.5.9` backend.
2. **Rare-cell-specific evaluation**, reproducing and extending the metrics and six-scenario analysis from the rare-cell preservation study.

The package name remains **scRareBench**. It is not presented as an official scIB release or fork; the manuscript can state that the package provides the current scIB-compatible output suite in addition to rare-cell-specific outputs.

## Version-0.3 contract

1. `load_dataset(selector, data_dir)` is the recommended multi-dataset Python entry point. It accepts either a stable integer index `0..5` or a registered dataset name/alias. The original `load_gse194122_benchmark()`, `download_gse194122()`, and `prepare_gse194122_paper_main()` APIs remain public for backward compatibility and advanced use.
2. Benchmark construction modifies **cell membership only** to reproduce the executed paper dataset:
   - retain `pDC` only in `s3d7`, `s3d6`, `s2d1`;
   - retain `CD8+ T CD57+ CD45RA+` only in `s3d6`, `s4d8`, `s1d3`;
   - preserve the source-relative order of all remaining cells.
3. It performs no normalization, filtering, scaling, HVG selection, or dimensionality reduction on the file delivered to the user.
4. The user chooses preprocessing and runs any batch-effect-removal method.
5. The user returns a two-dimensional latent matrix with one row per benchmark cell, preferably with an exact barcode vector.
6. scRareBench performs package-controlled clustering, the complete current `scib-metrics` evaluation layer, paper-style metrics, rare-cell metrics, six-scenario summaries, and provisional failure-archetype analysis.

The executed dataset contains **89,199 cells** after removing 1,062 target-cell instances. The implementation follows the executed `scDML_modify` notebook and retains the named batches.

## Installation

Python 3.11 or newer is required because the pinned current `scib-metrics` release requires it.

Install the current package directly from GitHub:

```bash
python -m pip install "scrarebench @ git+https://github.com/amirhossein-alishahi/scRareBench_.git@main"
```

Method libraries remain optional rather than mandatory core dependencies. In a clean environment they can be requested as extras:

```bash
pip install '.[scvi]'     # scvi-tools==1.4.3
pip install '.[mrvi]'     # scvi-tools==1.4.3, including MRVI
pip install '.[harmony]'  # harmonypy==2.0.0
pip install '.[methods]'  # all currently registered method libraries
```

For Colab and other pre-populated scientific environments, use the notebooks instead of installing the extras directly. The notebooks bootstrap the package with `--no-deps` and then call `scrarebench.runtime.setup_notebook()`, which preserves already-installed ABI-sensitive scientific packages, installs the core + selected-method requirements in one constrained transaction, runs `pip check`, and performs a fresh-process smoke test.

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Dataset registry

scRareBench 0.9.2 exposes six stable dataset selectors and adds interactive HTML / PDF reporting utilities. Index `0` is the paper benchmark and is the **only** selector that applies scRareBench-specific cell editing. Index `1` exposes the original GSE194122 source without that edit. The four added datasets are downloaded as published. No selector performs normalization, log transform, HVG selection, scaling, PCA, neighbors, or integration-method preprocessing.

| Index | Canonical name | Dataset | scRareBench cell editing |
|---:|---|---|---|
| 0 | `gse194122` | GSE194122 paper benchmark | Yes: paper cell subsetting + six-scenario annotation |
| 1 | `gse194122_raw` | Original GSE194122 | No |
| 2 | `mbdrc_renal_cortex` | mBDRC renal cortex | No |
| 3 | `wu_breast_cancer_atlas` | Wu breast-cancer atlas | No |
| 4 | `covid19_autoimmunity_pbmc` | COVID-19 autoimmunity PBMC | No |
| 5 | `nygc_seurat_v4_pbmc` | NYGC / Seurat v4 CITE-seq PBMC | No |

```python
from scrarebench import list_datasets, load_dataset

print(list_datasets())

# Equivalent selectors for the edited paper benchmark
adata = load_dataset(0, "./data")
adata = load_dataset("gse194122", "./data")

# Original GSE194122 without benchmark editing
raw = load_dataset(1, "./data")

# Added public datasets
renal = load_dataset("mbdrc", "./data")
breast = load_dataset(3, "./data")
covid = load_dataset("covid_pbmc", "./data")
nygc = load_dataset(5, "./data")
```

To download without loading the H5AD into memory:

```python
from scrarebench import download_dataset

path = download_dataset("wu_breast_cancer_atlas", "./data")
print(path)
```

For CELLxGENE-backed datasets, the package resolves the current published H5AD from the official collection metadata and writes a sibling `*.source.json` manifest containing collection ID, dataset ID/version, resolved URL, and SHA256.

CLI equivalents:

```bash
scrarebench datasets
scrarebench download-dataset 2 --data-dir ./data
scrarebench download-dataset gse194122_raw --data-dir ./data
```

## Prepare the official dataset

### GSE194122-specific convenience API

The original convenience function remains available and performs both GSE194122 benchmark-construction stages:

```python
from scrarebench import load_gse194122_benchmark

adata = load_gse194122_benchmark("./data")
```

On the first call it downloads the original GSE194122 file, constructs the 89,199-cell paper benchmark, annotates the curated six rare-cell scenarios, saves the prepared benchmark, and returns it. Later calls reuse the prepared file.

This is **benchmark construction only**. It performs no normalization, gene filtering, log transform, HVG selection, scaling, PCA, neighbors, or dimensionality reduction. Method-specific preprocessing remains the responsibility of scVI, Scanpy, Harmony, Scanorama, or the selected integration workflow.

### Manual two-stage Python API

Advanced users can still run the two stages separately:

```python
from scrarebench.datasets import download_gse194122, prepare_gse194122_paper_main

source = download_gse194122("./data/cache")
adata, manifest = prepare_gse194122_paper_main(
    source,
    "./data/gse194122_paper_main.h5ad",
)
```

### CLI

```bash
scrarebench prepare-gse194122 \
  --cache-dir ./data/cache \
  --output ./data/gse194122_paper_main.h5ad
```

Outputs next to the H5AD file:

- `.manifest.json`: source/output hashes, counts, and modification rules;
- `.cell_order.npy`: exact cell order expected from every latent space;
- `.distribution.csv`: checks for the two controlled LE populations.

## User integration stage

The user may preprocess the returned AnnData according to the integration method’s official workflow. The cell set and order must be preserved.

```python
import numpy as np

from scrarebench import load_gse194122_benchmark

adata = load_gse194122_benchmark("data")
latent = run_my_integration(adata)
np.save("my_method_latent.npy", latent)
np.save("my_method_barcodes.npy", adata.obs_names.astype(str).to_numpy())
```

## Evaluate a latent space

```bash
scrarebench evaluate \
  --adata ./data/gse194122_paper_main.h5ad \
  --latent ./my_method_latent.npy \
  --latent-barcodes ./my_method_barcodes.npy \
  --method MyMethod \
  --output-dir ./results/MyMethod
```

The standard scIB-compatible layer runs by default. It can be disabled only for debugging with `--skip-scib`.

## Standard scIB-compatible layer

The current backend is pinned to `scib-metrics==0.5.9`. The package creates a **benchmark-only reference representation** from the canonical raw-count layer:

1. GEX feature selection;
2. batch-aware HVG selection, default 4,000 genes;
3. library-size normalization to 10,000;
4. `log1p`;
5. canonical unintegrated PCA, default 50 components.

This reference is used only to compute benchmark metrics. It does not alter or constrain the user’s preprocessing or integration model.

### Biological-conservation outputs

- Isolated labels;
- Leiden NMI;
- Leiden ARI;
- KMeans NMI;
- KMeans ARI;
- Silhouette label;
- cLISI.

### Batch-correction outputs

- BRAS;
- iLISI;
- kBET per label;
- Graph connectivity;
- PCR comparison;
- classic batch silhouette/ASW as a separately reported supplement.

### Aggregate outputs

- Bio conservation;
- Batch correction;
- Total, using the standard `0.6 × biological + 0.4 × batch` weighting returned by `scib-metrics`.

`scib-metrics` documents that its values can deviate from the legacy `scib` repository. Every run therefore records the backend and exact version.

### Legacy scIB metrics not identifiable from a latent alone

The output always contains `scib/scib_metric_status.csv`. Metrics that require information outside the latent-only contract are explicitly listed as `not_applicable`, with a reason:

- HVG overlap requires corrected gene expression;
- cell-cycle conservation requires expression and organism-specific cell-cycle annotation;
- trajectory conservation requires curated reference pseudotime;
- Moran’s I requires a gene-level representation.

They are never silently omitted. Future interfaces can compute them when the corresponding inputs are supplied.

## Package-controlled clustering

The official paper-style and rare-cell results use:

- kNN graph on the submitted latent;
- `n_neighbors=15`;
- Euclidean distance;
- Leiden resolution `1.0`;
- random seed `0`;
- a separate graph and cluster key per method.

Additional resolutions may be requested with `--resolution-sweep`.

## Rare-cell-specific outputs

Paper-style global/subset metrics:

- biological ASW on the submitted latent;
- ARI and AMI between standard Leiden clusters and reference cell types;
- cluster-majority Accuracy;
- macro F1, weighted F1, and multiclass G-Mean.

Per-cell-type outputs:

- precision;
- recall;
- F1;
- inverse purity/completeness;
- number of containing and assigned clusters;
- dominant wrong label and fraction;
- within-cell-type cluster–batch NMI.

Curated rare-cell outputs additionally contain:

- all six GR/LE/SR × DL/RM scenarios;
- scenario-level summaries;
- configurable provisional failure archetypes;
- self-contained Sankey and static figures.

## Output structure

```text
results/MyMethod/
├── report.html                         # self-contained combined report
├── subset_metrics.csv                  # backward-compatible alias
├── per_type_metrics.csv                # backward-compatible alias
├── rare_metrics.csv                    # backward-compatible alias
├── scib/
│   ├── scib_results_wide.csv
│   ├── scib_metrics.csv
│   ├── scib_aggregate_scores.csv
│   ├── scib_metric_status.csv
│   ├── scib_reference_config.yaml
│   └── scib_metric_scores.png
├── rare_cell/
│   ├── rare_metrics_per_type.csv
│   ├── rare_metrics_summary.csv
│   ├── scenario_metrics.csv
│   ├── all_cell_type_metrics.csv
│   ├── sankey_all.html
│   └── figures/
├── clustering/
│   ├── clusters.csv
│   ├── resolution_results.csv
│   └── cluster_majority_mapping.json
└── reproducibility/
    ├── run_config.yaml
    └── failure_rules.yaml
```

`report.html` embeds all static images as Base64 and can be opened independently. The Plotly Sankey file also contains Plotly inline.

## Python API

```python
from scrarebench import EvaluationConfig, ScibEvaluationConfig, evaluate_latent

result = evaluate_latent(
    adata,
    EvaluationConfig(
        method_name="MyMethod",
        representation_key="X_my_method",
        label_key="celltype",
        batch_key="BATCH",
        scib=ScibEvaluationConfig(
            enabled=True,
            count_layer="counts",
            n_hvg=4000,
            reference_n_pcs=50,
        ),
    ),
    "results/MyMethod",
)
```

## Google Colab notebooks

The six notebooks under `notebooks/` now read scRareBench directly from this GitHub repository; no source ZIP upload or local project extraction is required. Each notebook has only a small bootstrap layer:

```python
SCRAREBENCH_GITHUB = "git+https://github.com/amirhossein-alishahi/scRareBench_.git@main"
METHOD = "scvi"  # or "harmony" / "mrvi"

# Install only the lightweight package code first.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps", SCRAREBENCH_GITHUB
])

from scrarebench.runtime import setup_notebook
setup_notebook(METHOD, quiet=False)
```

The method profile owns the method-specific dependency version, smoke imports, and safe environment defaults. After setup, the notebook prepares the registered dataset, builds the method-specific latent representation, verifies cell order, runs the unchanged scIB-compatible + rare-cell evaluation, and generates the standalone HTML/PDF/bundle outputs.

## Scientific status

The current scIB-compatible output layer is implemented and version-pinned. The six-scenario biological table and failure-archetype thresholds remain editable scientific specifications and should undergo final expert curation and sensitivity analysis before manuscript submission.


## Reporting utilities

This release adds three user-facing reporting functions:

- `write_interactive_report(adata, result, output_html)`
- `write_pdf_report(adata, result, output_pdf)`
- `create_report_bundle(adata, result, output_zip, include_latent=False)`

The interactive HTML is standalone and offline-friendly. It contains a linked Sankey explorer and embedding explorer, supports threshold-based flow filtering, category highlighting, and PNG export of the current view.


### 0.5.0 reliability fixes

- Interactive report generation now safely handles pandas `Categorical` scenario columns with missing values.
- Colab notebooks no longer require a source ZIP; they bootstrap scRareBench directly from GitHub and delegate runtime dependencies to `scrarebench.runtime`.
- The final bundle avoids duplicating generated interactive/PDF reports inside `benchmark_results`.


## Complete interactive dashboard

`write_interactive_report` now produces one standalone benchmark dashboard with optional tabs for overview, all paper-style metrics, all scIB-compatible metrics/status tables, a dedicated rare-cell explorer, general UMAP, Sankey flow explorer, reproducibility metadata, and embedded static figures. All section flags default to `True`; disabling a section omits its payload to reduce file size.

```python
write_interactive_report(
    adata, result, "interactive_report.html",
    representation_key="X_scVI",
    include_overview=True,
    include_metrics=True,
    include_scib=True,
    include_rare=True,
    include_rare_umap=True,
    include_rare_heatmaps=True,
    include_rare_scenario_analysis=True,
    include_umap=True,
    include_sankey=True,
    include_reproducibility=True,
    include_static_figures=True,
    include_cell_ids=True,
)
```


### 0.6.0 interactive dashboard

The standalone HTML dashboard now includes lazy tab rendering, CSV export for every data table, original-image download and an in-report lightbox/zoom viewer for static figures, human-readable metric labels, filtered rare-cell heatmaps, and a six-scenario rare-cell breakdown. The Rare-cell Explorer compares GR-DL, GR-RM, LE-DL, LE-RM, SR-DL, and SR-RM side by side, reports mean precision/recall/F1/dominant-cluster capture for each scenario, summarizes failure-archetype counts, and lists the exact rare cell types responsible for each scenario-level failure profile.


### Runtime compatibility (introduced in 0.6.1; retained in 0.9.0)

The scIB-compatible backend now includes an automatic scoped compatibility bridge for upstream API changes. In particular, `scib-metrics==0.5.9` still calls the legacy top-level `pandas.value_counts` API in graph connectivity, while pandas 3 removes that API. scRareBench detects this condition at runtime and temporarily maps the legacy call to `Series.value_counts()` only for the duration of the scIB benchmark, then restores the pandas module. This avoids forcing users to downgrade pandas and applies uniformly to any submitted latent representation (scVI, Harmony, Scanorama, custom embeddings, and other methods). Runtime versions and any activated compatibility adjustments are written to `scib_reference_config.yaml`.



### 0.9.2 GitHub-first notebook runtime

The benchmark package remains identical across integration methods, while method libraries remain optional. The runtime installer now lives inside the package as `scrarebench.runtime` rather than in a notebook-side `tools/` helper. Registered profiles currently cover `scvi`, `mrvi`, and `harmony`; the corresponding optional extras are declared in `pyproject.toml` for ordinary clean-environment installation.

Colab notebooks first install only the package code from GitHub with `--no-deps`, then call `setup_notebook(method)`. This lets the stdlib-only runtime helper constrain already-installed NumPy/SciPy/scikit-learn/JAX/Torch and related ABI-sensitive anchors before installing any scientific dependency. The setup performs a single constrained dependency transaction, avoids an unconditional `--upgrade`, runs scoped `pip check`, and validates the core scRareBench/scIB modules plus the selected method in a fresh Python process. The public benchmark API remains unchanged: attach a cell-by-dimension representation with `attach_latent()` and call `evaluate_latent()`.

### 0.8.0 dashboard interaction audit

The interactive dashboard received a full UI/state audit. Rare-cell filters now use one shared state across the population table, rare UMAP, recovery heatmap, batch-dependence heatmap, population details, match chips, and filter counts. Cell-type search is restricted to the `cell_type` field, single matches auto-focus, and no-match states are explicit. Scenario comparison and Rare Explorer filtering are separated: scenario cards/tabs change the selected scenario analysis, while an explicit action applies that scenario to the Rare Explorer.

Additional reliability and usability changes include independent rare subfeature flags (`include_rare_umap`, `include_rare_heatmaps`, `include_rare_scenario_analysis`), fail-safe plot export, per-column table sort state, raw/full versus visible-view CSV export, Sankey empty states and dynamic threshold range, UMAP controls that adapt when Sankey is omitted, keyboard-accessible tabs and scenario cards, figure gallery filtering, and a scrollable high-resolution lightbox with previous/next navigation and original-image download. Within-type batch NMI is intentionally separated from recovery-quality heatmaps because its direction of interpretation differs from precision/recall/F1/capture.


### 0.8.0 dashboard interaction hardening

Version 0.8.0 completes the interaction audit of the standalone dashboard. The main UMAP now supports click-to-focus selection with a details panel and rare-population handoff; Rare→UMAP and Sankey→UMAP state transitions are synchronized; rare preservation outcome is separated from failure-mode classification; rare heatmaps are clickable; scenario outcome bars apply their selected scenario/failure filter; Sankey numeric/range thresholds remain synchronized; figure lightbox navigation respects gallery filters and provides explicit Fit vs 100% actual-size controls; and very small metrics use scientific notation instead of being displayed as zero.

### 0.9.1 dataset-level scIB HVG policy

`ScibEvaluationConfig.hvg_batch_mode` now accepts `evaluation_batch` (default, historical behavior) or `global`. This controls only the scIB reference HVG-selection step; the evaluation `batch_key` used by downstream batch metrics is unchanged. This is intended for datasets where within-batch Seurat-v3 LOESS is numerically unstable, while keeping one benchmark package across all integration methods.


## Registered six-state topology for external benchmark datasets (v0.9.2)

`load_dataset()` now attaches in-memory rare-scenario metadata for registry datasets 2, 3, and 4:

- `2 / mbdrc_renal_cortex`: 10 six-state rare populations, covering GR-DL, GR-RM, LE-DL, and LE-RM. The broad `lymphocyte` rare label is retained as topology-ambiguous in annotation provenance and excluded from six-state summaries.
- `3 / wu_breast_cancer_atlas`: 17 six-state rare populations, covering all six GR/LE/SR × DL/RM combinations.
- `4 / covid19_autoimmunity_pbmc`: 12 six-state rare populations, covering GR-DL, GR-RM, LE-DL, and SR-DL.

These DL/RM labels are **provisional annotation-driven benchmark metadata**, derived from the supplied rare-cell benchmark report. They are not presented as marker-validated biological truth. The downloaded H5AD files remain unchanged; only the returned in-memory AnnData receives `scrarebench_*` metadata.

When dataset-specific scenario metadata exists in `adata.uns["scrarebench_scenario_table"]`, `evaluate_latent()` now uses it automatically if `scenario_table` is not passed explicitly. The interactive dashboard always renders all six scenario slots; unsupported scenarios appear as `0 types` rather than disappearing.
