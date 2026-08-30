# scRareBench

**Rare-cell-aware, scIB-compatible benchmarking for single-cell RNA-seq integration latent spaces**

[![Release](https://img.shields.io/badge/release-0.10.5-blue)](https://github.com/amirhossein-alishahi/scRareBench_/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Benchmark seed](https://img.shields.io/badge/benchmark_seed-42-6d4aff)](#reproducibility)

**Current release: `0.10.5`**

scRareBench evaluates single-cell integration and batch-correction **latent representations** with both conventional scIB-compatible metrics and an explicit rare-cell recovery layer.

The package is method-agnostic. It does not implement or train scVI, MrVI, Harmony, scRareP, or any other integration method. Your code produces the latent representation; scRareBench validates cell alignment, runs the benchmark, and creates reproducible result artifacts.

## Core design

scRareBench keeps three evaluation views separate:

1. **Biological conservation** — whether biological structure is retained.
2. **Batch correction** — whether unwanted technical/sample variation is reduced.
3. **Rare-cell recovery** — whether low-frequency or context-specific populations remain recoverable.

No unvalidated combined score is used to collapse these layers into a single number.

The method boundary is intentionally narrow:

```text
AnnData
  -> user-owned integration method
  -> cell x latent matrix
  -> scRareBench validation and evaluation
  -> metrics + reports + reproducible bundle
```

## Installation

### Standard installation from the release tag

```bash
pip install "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.5"
```

Python `>=3.11` is required.

### Google Colab / controlled runtime bootstrap

The shipped Colab notebooks bootstrap the package first and then let scRareBench install its benchmark dependencies while preserving already-installed ABI-sensitive packages:

```python
%pip install -q --no-deps "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.5"

from scrarebench.runtime import setup_runtime

runtime_report = setup_runtime()
```

Method-specific dependencies remain under your control. For example:

```python
runtime_report = setup_runtime(
    extra_requirements=("my-method==1.2.0",),
    extra_imports=("my_method",),
)
```

You can also preinstall method dependencies yourself with pip, conda, uv, Docker, or another environment manager.

## High-level API

For most method developers, use `MethodSpec` and `benchmark_method()`.

```python
from scrarebench import MethodOutput, MethodSpec, benchmark_method, load_dataset

adata = load_dataset(0)

def run_my_method(method_adata, seed, config):
    # Your preprocessing/training/integration code.
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

`MethodSpec.runner` may return a latent array/DataFrame, an `adata.obsm` key, or `MethodOutput`. `MethodOutput` is useful when you want to provide explicit barcodes, a representation name, or provenance files.

See [HIGH_LEVEL_API.md](HIGH_LEVEL_API.md) and [METHOD_INTEGRATION_GUIDE.md](METHOD_INTEGRATION_GUIDE.md) for the full developer contract.

## Benchmark a precomputed latent

If your method has already produced a latent representation, use `benchmark_latent()` directly:

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

print(result.summary())
```

The latent must contain exactly one row per benchmark cell. When barcodes are supplied, scRareBench verifies cell identity/order rather than trusting shape alone.

## Custom datasets

Register evaluation metadata once and then use the same high-level or low-level APIs:

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

For dataset preprocessing policy, see [DATASET_POLICY.md](DATASET_POLICY.md). For the registered rare-scenario taxonomy, see [SCENARIO_TAXONOMY.md](SCENARIO_TAXONOMY.md).

## Reproducibility

Method randomness and benchmark randomness are separate:

```text
method_seed     = training/integration replicate
benchmark_seed  = fixed evaluation seed
```

The canonical benchmark seed is `42`.

For multi-seed benchmarking, vary the method seed while keeping the benchmark seed fixed:

```python
result = benchmark_method(
    adata,
    method,
    seeds=[42, 123, 2026],
    benchmark_config={"random_state": 42},
)
```

Multi-seed aggregation summarizes scalar/tabular metrics. Latent coordinates, UMAP coordinates, Leiden clusters, Sankey topology, and cell-level states remain seed-specific and are never averaged.

Reference clustering defaults include:

- kNN neighbors: `15`
- distance: Euclidean
- Leiden reference resolution: `1.0`
- benchmark seed: `42`

Missing values remain missing; scRareBench does not convert unavailable metrics or untested method-dataset combinations to zero.

## Metrics

scRareBench reports complementary metric families rather than relying on one score.

### Global and subset metrics

Where applicable, the benchmark reports metrics such as:

- `ASW_true_on_latent`
- `ARI_true_vs_cluster`
- `AMI_true_vs_cluster`
- `Accuracy`
- `F1_macro`
- `F1_weighted`
- `G_Mean`

Subset rows include `overall`, `rare`, and `non_rare`.

### Rare-cell diagnostics

The rare-aware layer includes:

- **full-space selected-cell ASW** — `ASW_selected_cells_in_full_latent`
- **best-cluster precision/recall/F1**
- **support-adjusted kNN Local Recovery** — `knn_local_recovery_adjusted`
- observed/expected/maximum-achievable same-label kNN fractions
- inverse-purity / dominant-cluster capture
- within-type batch dependence diagnostics
- stored Leiden-resolution sensitivity
- legacy and resolution-aware failure-archetype fields

Metric direction is explicit in `METRIC_REGISTRY`; context-only diagnostics are not automatically treated as higher-is-better ranking metrics.

Failure-archetype labels are diagnostic interpretations. For publication-level biological claims, use them together with the underlying metrics and the available threshold/resolution sensitivity views.

## scIB-compatible evaluation

scRareBench uses the pinned `scib-metrics==0.5.9` backend defined by the package dependencies. The submitted method latent is never re-integrated by scRareBench.

The benchmark-only reference is constructed independently from the configured counts. The default reference policy uses:

- the configured GEX count layer
- 4,000 HVGs unless the dataset policy overrides it
- total-count normalization to 10,000
- log1p
- PCA up to 50 components

Canonical mode fails closed instead of silently switching the configured HVG algorithm.

## Registered datasets

The package registry exposes stable selectors for the prepared benchmark datasets:

| Index | Key | Dataset / role |
|---:|---|---|
| 0 | `gse194122` | GSE194122 paper benchmark |
| 1 | `gse194122_raw` | original/unmodified GSE194122 source |
| 2 | `mbdrc_renal_cortex` | mBDRC renal cortex |
| 3 | `wu_breast_cancer_atlas` | Wu breast-cancer atlas |
| 4 | `covid19_autoimmunity_pbmc` | COVID-19 autoimmunity PBMC |
| 5 | `nygc_seurat_v4_pbmc` | NYGC / Seurat v4 CITE-seq PBMC |

Load a registered dataset:

```python
from scrarebench import load_dataset

adata = load_dataset(0, data_dir="./data")
```

Dataset provenance is attached to the loaded object and propagated into benchmark artifacts. Cite each dataset's original publication/database record in scientific work.

## Outputs

A benchmark run can create:

- `results.json` — machine-readable benchmark result
- CSV metric tables
- an interactive self-contained HTML report
- a scientific-summary PDF
- reproducibility/provenance metadata
- a hashed ZIP result bundle
- optional latent/barcode artifacts

For multi-seed runs, compatible per-seed artifacts can be finalized into a combined report and delivery archive while preserving each seed as an independent auditable run.

## Colab notebooks

The repository includes both runnable examples and generic templates:

- `notebooks/scRareBench_CustomMethod_HighLevel_Colab.ipynb` — generic high-level template for any method
- `notebooks/scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb` — high-level scVI example on Dataset 0
- `notebooks/scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb` — high-level scVI example on mBDRC
- `notebooks/scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb` — low-level multi-seed template
- additional Harmony, MrVI, and scVI notebooks — method-specific examples

Method-specific notebooks are examples only; those method implementations are not embedded in the scRareBench package.

## CLI

List registered datasets:

```bash
scrarebench datasets
```

Download/prepare a registered dataset:

```bash
scrarebench download-dataset 0 --data-dir ./data
```

The Python API is recommended when integrating scRareBench into a method-development workflow.

## Documentation

- [High-level API quick reference](HIGH_LEVEL_API.md)
- [Method developer guide](METHOD_INTEGRATION_GUIDE.md)
- [Dataset evaluation policy](DATASET_POLICY.md)
- [Rare-scenario taxonomy](SCENARIO_TAXONOMY.md)
- [Reproducibility constraints](constraints/README.md)
- [Release history](CHANGELOG.md)

## Citation

A synchronized [CITATION.cff](CITATION.cff) is included. When using the standard benchmark layer, cite scIB/scib-metrics as appropriate. When using registered biological datasets, cite the original dataset publications/database records.

## License

MIT License — see [LICENSE](LICENSE).
