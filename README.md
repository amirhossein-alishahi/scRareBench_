# scRareBench

**scRareBench** is a benchmarking framework for evaluating **user-generated scRNA-seq integration latent spaces**, with standard scIB-compatible metrics and rare-cell-aware diagnostics.

The package has one deliberately strict boundary:

```text
AnnData -> your method -> latent representation (+ cell identity) -> scRareBench
```

scRareBench **does not implement, select, configure, or train integration methods**. Your method can be scVI, Harmony, MrVI, scRareP, an unpublished model, or any future method. As long as it produces one latent row per cell, the same benchmark API applies.

## The developer workflow

For most method developers, the complete workflow is three steps:

```python
from scrarebench import load_dataset, benchmark_latent

# 1) Get a benchmark dataset as AnnData.
adata = load_dataset("gse194122")

# 2) Run YOUR method. scRareBench is not involved here.
latent = my_integration_method(adata)

# 3) Benchmark the latent.
result = benchmark_latent(
    adata,
    latent,
    method="MyMethod",
)

print(result.summary())
print(result.interactive_report_path)
```

That is the intended high-level interface. The lower-level evaluation API remains available for users who need complete control.

---

## What scRareBench owns — and what you own

### scRareBench owns

- loading and identifying registered benchmark datasets;
- recording the evaluation contract for custom datasets;
- cell/barcode alignment checks;
- package-controlled kNN + Leiden clustering for paper-style/rare-cell evaluation;
- the validated `scib-metrics==0.5.9` evaluation layer;
- global, per-type, rare-cell, scenario and failure-analysis outputs;
- reproducibility metadata;
- static HTML, interactive HTML, PDF and ZIP bundle generation.

### You own

- gene filtering used by your method;
- normalization/transformation used by your method;
- HVG selection used by your method;
- model architecture and hyperparameters;
- training;
- the definition of your integration algorithm;
- production of the final latent representation.

scRareBench never modifies your method to make it conform to a reference implementation.

---

## Installation

scRareBench requires **Python >=3.11**. Its benchmark backend is intentionally pinned to `scib-metrics==0.5.9` for reproducibility.

### Release install from GitHub

```bash
python -m pip install "scrarebench @ git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.3"
```

### Development install

```bash
git clone https://github.com/amirhossein-alishahi/scRareBench_.git
cd scRareBench_
python -m pip install -e '.[dev]'
python -m pytest
```

There are **no method-specific package extras** and no method registry. Installing scRareBench does not install scVI, Harmony, MrVI, or any other integration method.

---

## Google Colab / pre-populated scientific environments

For normal virtual environments, ordinary `pip install` is preferred. The generic runtime helper exists primarily for environments such as Google Colab where replacing an already-loaded NumPy/JAX/Torch stack can create ABI or dependency conflicts.

The release notebooks were validated on **Google Colab runtime 2026.07** (Python 3.12.13, NumPy 2.0.2, PyTorch 2.11.0, JAX 0.7.2). That exact runtime is a reproducibility reference, **not a hard requirement** for scRareBench. Each notebook contains a separate, fully commented compatibility-check cell that users may optionally uncomment to compare their environment with the validated anchors. The default execution path does not stop merely because NumPy/PyTorch/JAX versions differ. The scVI/MrVI examples still depend on the requirements of their user-side method (`scvi-tools==1.4.3` requires Python 3.12+).

The official notebooks bootstrap scRareBench with `--no-deps`, then install the package requirements and the **user-declared method requirements** in one transaction while preserving ABI-sensitive scientific packages already present in the running environment:

```python
import subprocess
import sys

PACKAGE_URL = "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.3"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-deps", PACKAGE_URL
])

from scrarebench.runtime import setup_runtime

setup_runtime(
    # Optional: dependencies owned by YOUR method, not by scRareBench.
    extra_requirements=("my-method-package==1.0.0",),
    extra_imports=("my_method_package",),
    quiet=False,
)
```

For a local/unpublished method that needs no additional packages, omit `extra_requirements` and `extra_imports`.

`setup_runtime()` is method-agnostic. It does not receive a method name and does not infer dependencies from one. It:

1. records pre-existing `pip check` issues;
2. snapshots already-installed ABI-sensitive packages;
3. constrains those anchors during installation;
4. installs scRareBench dependencies plus any requirements explicitly supplied by the user;
5. reruns `pip check` and fails on **new** conflicts introduced by the installation;
6. verifies that scientific anchors were not unexpectedly replaced;
7. performs fresh-process import smoke tests.

The optional release constraint file under `constraints/` records documented/validated Colab anchors for users who want stricter reproducibility checks. The notebooks do not enforce it by default, and it is not presented as a complete lockfile.

---

# 1. Loading a registered dataset

```python
from scrarebench import load_dataset, dataset_info

adata = load_dataset(0)
print(adata)
print(dataset_info(adata))
```

By default, datasets are cached under:

```text
~/.cache/scrarebench/datasets
```

Override this with either:

```bash
export SCRAREBENCH_DATA_DIR=/path/to/data
```

or:

```python
adata = load_dataset(0, data_dir="/path/to/data")
```

`load_dataset()` returns an `AnnData` object and **does not** run normalization, log transformation, HVG selection, scaling, PCA, neighbors, or integration-method preprocessing.

## Registered datasets

| Index | Key | Dataset | Ready for `benchmark_latent()` immediately? | Evaluation notes |
|---:|---|---|---|---|
| 0 | `gse194122` | Curated GSE194122 paper benchmark | **Yes** | `label=celltype`, `batch=BATCH`, curated six-state rare scenarios |
| 1 | `gse194122_raw` | Original GSE194122 | **Yes for standard/scIB** | `label=celltype`, `batch=BATCH`; no automatic paper rare taxonomy |
| 2 | `mbdrc_renal_cortex` | mBDRC renal cortex | **Yes** | `label=cell_type`; evaluation batch is `donor_id × assay`; registered rare scenarios |
| 3 | `wu_breast_cancer_atlas` | Wu breast-cancer atlas | **Needs batch registration** | registered label/scenario metadata are available, but the biological batch must be chosen explicitly |
| 4 | `covid19_autoimmunity_pbmc` | COVID-19 autoimmunity PBMC | **Needs batch registration** | registered label/scenario metadata are available, but the biological batch must be chosen explicitly |
| 5 | `nygc_seurat_v4_pbmc` | NYGC / Seurat v4 CITE-seq PBMC | **Needs label + batch registration** | source H5AD is loaded without an imposed evaluation contract |

List the registry programmatically:

```python
from scrarebench import list_datasets

for row in list_datasets():
    print(row["index"], row["key"], row["display_name"])
```

To download without loading into memory:

```python
from scrarebench import download_dataset

path = download_dataset("mbdrc_renal_cortex")
print(path)
```

For CELLxGENE-backed sources, scRareBench stores source-resolution metadata and a SHA256 in a sibling `*.source.json` manifest. Downloaded source H5AD files are not edited in place.

---

# 2. Running your method

This section belongs to **your codebase**, not to scRareBench.

```python
# Example only: implement this however you want.
latent = my_method(adata)
```

The only required benchmark contract is:

- `latent` is finite and two-dimensional;
- rows correspond one-to-one with `adata.obs_names`;
- the number of latent rows equals `adata.n_obs`.

## Preferred latent format: DataFrame with cell IDs

The safest interface is a `pandas.DataFrame` whose index contains the cell barcodes:

```python
import pandas as pd

latent_df = pd.DataFrame(
    latent,
    index=adata.obs_names.astype(str),
)
```

This lets scRareBench verify exact cell order.

A plain NumPy array is also accepted, but cell order cannot be independently verified unless you provide barcodes separately.

---

# 3. Benchmarking the latent

```python
from scrarebench import benchmark_latent

result = benchmark_latent(
    adata,
    latent_df,
    method="MyMethod",
)
```

The method name is a **display/reproducibility label only**. It is never validated against a registry. This is valid:

```python
result = benchmark_latent(
    adata,
    latent_df,
    method="scRareP-v1",
)
```

## Supported latent inputs

### NumPy array

```python
result = benchmark_latent(adata, latent, method="MyMethod")
```

### DataFrame with barcode index — recommended

```python
result = benchmark_latent(adata, latent_df, method="MyMethod")
```

### Existing `adata.obsm` representation

```python
adata.obsm["X_my_method"] = latent
result = benchmark_latent(
    adata,
    "X_my_method",
    method="MyMethod",
    representation_key="X_my_method_benchmark",
)
```

### `.npy`, `.npz`, `.csv`, or `.tsv`

```python
result = benchmark_latent(
    adata,
    "my_method_latent.npz",
    method="MyMethod",
)
```

For `.npz`, the preferred keys are `latent` and optional `barcodes`.

## Explicit barcode vector

```python
result = benchmark_latent(
    adata,
    latent,
    barcodes=my_cell_barcodes,
    method="MyMethod",
)
```

By default, a barcode order mismatch raises an error. If the sets are identical but ordering differs:

```python
result = benchmark_latent(
    adata,
    latent,
    barcodes=my_cell_barcodes,
    method="MyMethod",
    allow_reorder=True,
)
```

scRareBench verifies the barcode sets before reordering.

---

# Custom datasets

A user dataset must be given an explicit evaluation contract once. Registration is metadata-only: it does not preprocess expression values or run a method.

```python
import scanpy as sc
from scrarebench import register_dataset, benchmark_latent

adata = sc.read_h5ad("my_atlas.h5ad")

register_dataset(
    adata,
    name="MyAtlas",
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
)

latent = my_method(adata)
result = benchmark_latent(adata, latent, method="MyMethod")
```

### Important requirements

- `adata.obs_names` must be unique;
- `label_key` and `batch_key` must exist in `adata.obs`;
- if `count_layer` is specified, it must exist in `adata.layers`;
- raw counts are strongly recommended for the scIB reference-preprocessing layer.

If the expression matrix in `adata.X` is intentionally the count source, use:

```python
register_dataset(
    adata,
    label_key="cell_type",
    batch_key="batch",
    count_layer=None,
)
```

## Custom rare populations without topology classes

```python
register_dataset(
    adata,
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
    rare_types=["pDC", "MAIT"],
)
```

This enables rare-vs-non-rare and per-type rare metrics without inventing a six-state biological scenario assignment.

## Custom scenario table

```python
scenario_table = ...  # pandas DataFrame

register_dataset(
    adata,
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
    scenario_table=scenario_table,
)
```

Required columns:

```text
cell_type
scenario
distribution
topology
```

Optional provenance columns include `parent_type` and `curation_source`.

Valid six-state scenario labels are:

```text
GR-DL  GR-RM  LE-DL  LE-RM  SR-DL  SR-RM
```

Distribution-only `GR`, `LE`, or `SR` rows may be supplied with unassigned topology when a defensible DL/RM annotation does not exist.

A custom dataset with no rare metadata receives standard/global and scIB-compatible metrics only. scRareBench does **not** silently reuse the GSE194122 rare taxonomy.

---

# Built-in dataset metadata and partial registration

Some registered source datasets deliberately do not impose a batch definition. You can load them first, inspect the source annotations, then register the missing contract:

```python
from scrarebench import load_dataset, register_dataset, dataset_info

adata = load_dataset("wu_breast_cancer_atlas")
print(dataset_info(adata))
print(adata.obs.columns.tolist())

# Example only: choose the biologically correct batch column for your analysis.
register_dataset(
    adata,
    name="Wu breast-cancer atlas",
    label_key="celltype_subset",
    batch_key="YOUR_BATCH_COLUMN",
    count_layer="counts",
)
```

Any scenario metadata already attached by the registered loader remains in the AnnData object unless you explicitly replace it.

---

# Benchmark configuration

Defaults reproduce the package's standard evaluation contract:

```python
from scrarebench import BenchmarkConfig, benchmark_latent

config = BenchmarkConfig(
    reference_resolution=1.0,
    resolution_sweep=(1.0,),
    n_neighbors=15,
    distance_metric="euclidean",
    random_state=0,
    run_scib=True,
    scib_n_hvg=4000,
    scib_reference_n_pcs=50,
)

result = benchmark_latent(
    adata,
    latent_df,
    method="MyMethod",
    config=config,
)
```

A mapping is accepted as a convenience:

```python
result = benchmark_latent(
    adata,
    latent_df,
    method="MyMethod",
    config={
        "n_neighbors": 30,
        "resolution_sweep": (0.5, 1.0, 1.5),
    },
)
```

Unknown configuration keys fail loudly instead of being silently ignored.

---

# Result object and outputs

`benchmark_latent()` returns a `BenchmarkResult`:

```python
result.summary()
result.metrics
result.per_type_metrics
result.rare_metrics
result.rare_summary
result.scenario_metrics
result.scib_metrics
result.scib_aggregates

result.report_path
result.interactive_report_path
result.pdf_path
result.bundle_path
```

Default output location:

```text
./scrarebench_results/<dataset>/<method>/
```

You can override it:

```python
result = benchmark_latent(
    adata,
    latent_df,
    method="MyMethod",
    output_dir="results/MyMethod",
)
```

The standard output includes reproducibility metadata and tabular benchmark results. By default the high-level API also creates:

- static self-contained HTML report;
- interactive HTML dashboard;
- PDF summary;
- ZIP result bundle.

Disable optional artifacts through `BenchmarkConfig` when running large parameter sweeps.

---

# Scientific evaluation layers

## Package-controlled clustering

Paper-style and rare-cell metrics use a graph/clustering step controlled by scRareBench rather than by the submitted method:

- kNN graph on the submitted latent;
- default `n_neighbors=15`;
- Euclidean distance;
- Leiden reference resolution `1.0`;
- random seed `0`.

This prevents each method from receiving a different downstream clustering pipeline.

## scIB-compatible layer

The validated backend is pinned to:

```text
scib-metrics==0.5.9
```

scRareBench constructs an independent benchmark-only reference representation from the configured count source. This reference preprocessing does not alter the user's integration workflow.

The package reports biological-conservation metrics, batch-correction metrics, aggregate scores, and a metric-status table. Metrics that require information outside the latent/count contract are reported as not applicable rather than silently omitted.

The `scib-metrics` aggregate score and rare-cell summaries remain separate scientific outputs; scRareBench does not create an unvalidated composite score between them.

## Rare-cell layer

When rare metadata exists, scRareBench reports per-population precision, recall, F1, inverse purity, within-type batch NMI, scenario summaries and provisional failure-archetype diagnostics.

Failure-archetype thresholds should be treated as diagnostic/provisional unless separately validated for the intended scientific claim.

---

# Low-level API

Advanced users can bypass the high-level orchestration:

```python
from scrarebench import (
    attach_latent,
    EvaluationConfig,
    ScibEvaluationConfig,
    evaluate_latent,
)

attach_latent(
    adata,
    latent,
    key="X_my_method",
    latent_barcodes=cell_ids,
)

config = EvaluationConfig(
    method_name="MyMethod",
    representation_key="X_my_method",
    label_key="cell_type",
    batch_key="batch",
    scib=ScibEvaluationConfig(enabled=True),
)

result = evaluate_latent(
    adata,
    config,
    "results/MyMethod",
)
```

Use the low-level API when you need explicit control over alignment, scenario tables, individual report-generation calls, or scIB configuration. The high-level API uses this same evaluation engine internally.

---

# Example notebooks

The repository includes two notebook styles:

### High-level developer examples

- `scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb`
- `scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb`

These are the recommended onboarding examples. They make the ownership boundary obvious:

```text
scRareBench: load data
        -> user code: run scVI and produce latent
        -> scRareBench: benchmark latent
```

### Detailed reference notebooks

- `scRareBench_scVI_Colab.ipynb`
- `scRareBench_Harmony_Colab.ipynb`
- `scRareBench_scVI_Dataset2_mBDRC_Colab.ipynb`
- `scRareBench_Harmony_Dataset2_mBDRC_Colab.ipynb`
- `scRareBench_MrVI_Dataset0_GSE194122_Colab.ipynb`
- `scRareBench_MrVI_Dataset2_mBDRC_Colab.ipynb`

These expose method-side preprocessing/training and report-generation details for reproducibility. They are **examples**, not method implementations shipped by scRareBench.

Release notebooks install a fixed Git tag (`v0.10.3`), never the moving `main` branch.

---

# Common errors and how to interpret them

### `Latent has N rows but dataset has M cells`

Your method changed cell membership or you passed the wrong latent. scRareBench requires one latent row per benchmark cell.

### `Latent barcode order differs ...`

The same cells are not in the same order. Prefer a DataFrame indexed by cell IDs, or pass `allow_reorder=True` only when the barcode sets are identical.

### `adata.obs['...'] is required`

The dataset evaluation contract is incomplete or refers to a missing label/batch column. For a custom or partially registered dataset, call `register_dataset()` with explicit keys.

### `count_layer='counts' is not present`

Register the actual raw-count layer, or set `count_layer=None` if `adata.X` is intentionally the benchmark count source.

### Re-running the same method on the same AnnData raises an existing-key/output error

Use a new method/representation name or explicitly pass `overwrite=True` when replacement is intentional.

### Colab dependency resolver/import failure

Use the official release notebook/runtime version and `setup_runtime()` rather than manually upgrading NumPy/JAX/Torch in the middle of a running kernel. `setup_runtime()` fails if the install introduces a new `pip check` conflict or changes protected scientific anchors.

---

# Reproducibility and release policy

- The package version, Git tag and official notebook ref must match.
- Official notebooks are pinned to the release tag.
- `scib-metrics` is pinned to the validated backend version.
- Colab constraint anchors are stored under `constraints/`.
- Every benchmark run records package/runtime configuration in its output directory.
- GitHub Actions validates source compilation, notebook code-cell syntax, tests, package build/data, and a VCS bootstrap smoke test on Python 3.11 and 3.12.

See `RELEASE_CHECKLIST.md` before publishing a release.

---

# Repository layout

```text
scRareBench_/
├── src/scrarebench/          # package code
│   ├── benchmark.py          # high-level latent benchmarking API
│   ├── datasets/             # registered dataset loaders + metadata
│   ├── evaluation.py         # low-level benchmark orchestration
│   ├── scib_backend.py       # validated scib-metrics layer
│   ├── scenarios.py          # rare/scenario metadata
│   ├── reporting.py          # static/PDF/bundle outputs
│   ├── dashboard.py          # interactive report
│   └── runtime.py            # generic Colab/runtime helper
├── notebooks/                # user-method examples
├── constraints/              # release environment anchors
├── tests/                    # regression and contract tests
├── scripts/                  # repository validators
└── .github/workflows/ci.yml  # CI
```

There is intentionally no `methods/` package and no integration-method registry.

---

# Design principles

1. **Method agnostic:** a new integration method does not require a scRareBench code change.
2. **Latent boundary:** benchmarking begins at the user-generated latent representation.
3. **Strict cell identity:** latent-to-cell alignment is validated whenever identity information is supplied.
4. **Dataset-aware evaluation:** registered datasets may carry label, batch, count and rare-scenario metadata.
5. **No silent scientific fallback:** missing rare metadata does not trigger reuse of another dataset's taxonomy.
6. **Reproducible downstream evaluation:** clustering and metric settings are controlled and recorded by the benchmark.
7. **Progressive disclosure:** the high-level API is simple; low-level APIs remain available for auditability.

---

# Citation

If you use scRareBench, cite the accompanying rare-cell integration benchmarking study, the validated `scib-metrics` backend, and the original scIB benchmarking study as appropriate. See `CITATION.cff` for software citation metadata.
