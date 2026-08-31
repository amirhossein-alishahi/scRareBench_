# scRareBench

**Rare-cell-aware, scIB-compatible benchmarking for single-cell RNA-seq integration latent spaces**

[![Release](https://img.shields.io/badge/release-0.10.6-blue)](https://github.com/amirhossein-alishahi/scRareBench_/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Benchmark seed](https://img.shields.io/badge/benchmark_seed-42-6d4aff)](#reproducibility)

**Current release: `0.10.6`**

scRareBench evaluates single-cell integration and batch-correction **latent representations** with conventional scIB-compatible metrics and an explicit rare-cell recovery layer.

The package is method-agnostic. It does not implement or train scVI, MrVI, Harmony, scRareP, or another integration method. Your code produces the latent representation; scRareBench validates cell alignment, evaluates the representation under one benchmark contract, and creates reproducible machine-readable **and interactive** result artifacts.

## Why scRareBench

Global integration scores can look strong while a low-frequency population is absorbed into an abundant neighboring lineage. scRareBench therefore keeps three views separate:

1. **Biological conservation** — is biological structure retained?
2. **Batch correction** — is unwanted technical/sample variation reduced?
3. **Rare-cell recovery** — do low-frequency or context-specific populations remain recoverable?

No unvalidated combined score collapses these layers into one number.

The method boundary is intentionally narrow:

```text
AnnData
  -> user-owned integration method
  -> cell x latent matrix
  -> scRareBench validation + evaluation
  -> metrics + interactive report + reproducible bundle
```

---

# Interactive results are a first-class output

A scRareBench run does not end with a collection of CSV files. It can produce a **self-contained interactive HTML workspace** that can be opened directly in a browser and shared with collaborators without a Python environment or a running server.

The report combines benchmark scores, rare-population diagnostics, UMAP/Sankey views, reproducibility metadata, sensitivity views, and seed management in one portable artifact. Canonical benchmark results remain immutable; exploratory controls are visually separated from the reported result.

> **Screenshot placeholder — add:** `docs/images/report_overview.webp`

The main report tabs include:

- **Overview** — cells, batches, cell types, scIB aggregates, rare-cell summaries, and benchmark warnings.
- **Metrics** — overall / rare / non-rare benchmark metrics.
- **scIB** — individual scIB-compatible metrics and aggregate scores.
- **Rare-cell Explorer** — per-population recovery, support-adjusted kNN recovery, best-cluster recovery, failure interpretations, and scenario drill-down.
- **UMAP** — interactive views of labels, batches, scenarios, clusters, predictions, and failure states when available.
- **Sankey** — true-label → cluster → prediction/failure flow inspection.
- **Run & Provenance / Runs & Seeds** — exact run identity, hashes, seed state, and multi-seed controls.
- **Reproducibility** — benchmark configuration and environment/provenance information.
- **Figures** — generated static figures with browser inspection/export controls.

### Rare-cell Explorer

The rare-cell layer is designed to make local failures inspectable rather than hiding them behind a single global score.

> **Screenshot placeholder — add:** `docs/images/report_rare_explorer.webp`

### Native multi-seed workspace

For stochastic integration methods, scRareBench can keep several method seeds in one report. The report stores each run separately and aggregates **metrics**, not latent geometry.

> **Screenshot placeholder — add:** `docs/images/report_seed_stability.webp`

A multi-seed report lets you:

- inspect aggregate mean ± sample SD across included runs;
- switch the detailed view to any stored seed;
- inspect seed-stability plots;
- exclude/restore a seed from aggregate statistics without deleting the run;
- keep planned/stored/included seed counts visible;
- save the updated report state as another self-contained HTML file.

Latent coordinates, UMAP coordinates, Leiden clusters, Sankey topology, and cell-level states are always seed-specific and are **never averaged**.

See **[INTERACTIVE_REPORTS.md](INTERACTIVE_REPORTS.md)** for the full report and multi-seed workflow.

---

# Compare several reports in the browser

The repository includes a standalone comparator:

```text
comparator/scRareBench_Multi_Report_Comparator_v10.html
```

Open the file in a browser and import scRareBench outputs by drag-and-drop or with **Import reports**. Nothing needs to be uploaded to a server; report data are processed locally in the browser. The comparator loads its pinned Plotly/JSZip browser libraries from public CDNs.

The comparator accepts current v0.10.6/schema-1.6 single-run and multi-seed interactive HTML reports, result ZIPs, `results.json`, and compatible legacy HTML reports.

> **Screenshot placeholder — add:** `docs/images/comparator_methods_v10.webp`

It provides:

- **Methods on Dataset** — compare methods evaluated on the same dataset;
- **Datasets for Method** — inspect one method across datasets;
- **Metric Plotter** — direction-aware metric-vs-metric plots and Pareto views;
- **Rare-cell Deep Dive** — compare cell-type-level rare-recovery diagnostics;
- **Coverage & Ranking** — separate benchmark coverage from performance;
- **Seed view** — switch between aggregate included seeds, individual seeds, or one specific seed.

Current multi-seed reports are merged only when their method/dataset/evaluation identities are compatible. Exact duplicate runs are not double-counted, and a conflicting duplicate seed is rejected instead of silently replacing data.

Detailed comparator instructions are in **[INTERACTIVE_REPORTS.md](INTERACTIVE_REPORTS.md#multi-report-comparator)**.

---

# Installation

## Standard installation from the release tag

```bash
pip install "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.6"
```

Python `>=3.11` is required.

## Google Colab / controlled runtime bootstrap

The shipped Colab notebooks bootstrap the package first and then let scRareBench install its benchmark dependencies while preserving ABI-sensitive packages already present in the runtime:

```python
%pip install -q --no-deps "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.6"

from scrarebench.runtime import setup_runtime
runtime_report = setup_runtime()
```

Method-specific dependencies remain under your control:

```python
runtime_report = setup_runtime(
    extra_requirements=("my-method==1.2.0",),
    extra_imports=("my_method",),
)
```

You can also preinstall method dependencies with pip, conda, uv, Docker, or another environment manager.

---

# High-level API

For most method developers, use `MethodSpec` and `benchmark_method()`.

```python
from scrarebench import MethodOutput, MethodSpec, benchmark_method, load_dataset

adata = load_dataset(0)

def run_my_method(method_adata, seed, config):
    # Your preprocessing / training / integration code.
    latent = my_integration_method(
        method_adata,
        seed=seed,
        latent_dim=config["latent_dim"],
    )
    return MethodOutput(
        latent=latent,
        barcodes=method_adata.obs_names,
    )

method = MethodSpec(
    name="MyMethod",
    runner=run_my_method,
    config={"latent_dim": 30},
    dependencies=(),
)

result = benchmark_method(
    adata,
    method,
    seeds=[42, 123, 2026],
    benchmark_config={"random_state": 42},
    install_dependencies=False,
)

print(result.summary())
print(result.report_path)
print(result.archive_path)
```

`MethodSpec.runner` may return a latent array/DataFrame, an `adata.obsm` key, or `MethodOutput`. `MethodOutput` is useful when you want to provide explicit barcodes, a representation name, or method provenance files.

See [HIGH_LEVEL_API.md](HIGH_LEVEL_API.md) and [METHOD_INTEGRATION_GUIDE.md](METHOD_INTEGRATION_GUIDE.md) for the developer contract.

## Benchmark a precomputed latent

```python
from scrarebench import benchmark_latent

result = benchmark_latent(
    adata,
    latent,
    method="MyMethod",
    barcodes=adata.obs_names,
    output_dir="./results/MyMethod",
    config={"random_state": 42},
)
```

The latent must contain exactly one row per benchmark cell. When barcodes are supplied, cell identity/order is verified rather than inferred from shape.

---

# Output artifacts

Depending on configuration, a benchmark run can create:

```text
results/
├── results.json                 # stable machine-readable result
├── subset_metrics.csv
├── subset_metric_ratios.csv
├── per_type_metrics.csv
├── rare_cell/
├── scib/
├── reproducibility/
├── interactive_report.html     # standalone browser workspace
├── summary_report.pdf
└── scrarebench_bundle.zip      # reproducible handoff bundle
```

For multi-seed execution, each seed remains an auditable run and the finalizer can additionally create a combined multi-seed HTML/report bundle. `results.json` and CSV files are suitable for downstream automation; the HTML report is designed for interactive inspection and scientific handoff.

---

# Custom datasets

Register evaluation metadata once and use the same high-level or low-level APIs:

```python
from scrarebench import register_dataset

register_dataset(
    adata,
    name="MyDataset",
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
    rare_types=["RareTypeA", "RareTypeB"],  # optional
)
```

A custom dataset does **not** need the six-state paper taxonomy. With only `rare_types`, generic rare-recovery metrics are computed and scenario-specific fields remain unassigned.

For dataset evaluation policy, see [DATASET_POLICY.md](DATASET_POLICY.md). For the registered rare-scenario taxonomy, see [SCENARIO_TAXONOMY.md](SCENARIO_TAXONOMY.md).

---

# Reproducibility

Method randomness and benchmark randomness are separate:

- **method seed** controls the stochastic integration method;
- **benchmark seed** controls deterministic/sampled benchmark operations.

A multi-seed benchmark therefore evaluates several method seeds under one fixed evaluation contract.

The package records run identity and provenance fields such as method/dataset identity, method seed, benchmark seed, configuration/evaluation hashes, dataset/cell-order identity, latent hash, and environment metadata where available.

For a multi-seed group, scalar metrics are summarized over included runs. Seed-specific geometry and cell-level structures are retained per run rather than averaged.

---

# Included notebooks

The repository ships Colab-oriented examples for:

- custom methods through the high-level API;
- scVI high-level examples;
- low-level / multi-seed templates;
- explicit scVI, Harmony, and MrVI example workflows on included datasets.

Method code remains notebook/user code. scRareBench itself stays method-agnostic.

---

# Project documentation

- [Interactive reports and comparator](INTERACTIVE_REPORTS.md)
- [High-level API](HIGH_LEVEL_API.md)
- [Method integration guide](METHOD_INTEGRATION_GUIDE.md)
- [Dataset policy](DATASET_POLICY.md)
- [Scenario taxonomy](SCENARIO_TAXONOMY.md)
- [Benchmark contract notes](DESIGN_NOTES.md)
- [Changelog](CHANGELOG.md)

---

# Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). If you use scRareBench in research, cite the associated benchmark work/package release as appropriate.

# License

MIT — see [LICENSE](LICENSE).
