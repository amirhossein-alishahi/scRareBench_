# Method developer guide

scRareBench is method-agnostic: a new integration or batch-effect-removal method can be benchmarked **without adding that method to the scRareBench package**.

## Two supported developer paths

### High-level API — recommended for most method developers

Use `MethodSpec` plus `benchmark_method`. You own the method runner, dependency list, and method configuration; scRareBench owns seed orchestration, latent alignment, evaluation, reports, and optional multi-seed finalization.

```python
from scrarebench import MethodOutput, MethodSpec, benchmark_method

METHOD_DEPS = ("my-method==1.2.0",)
METHOD_CONFIG = {"latent_dim": 30, "epochs": 200}

def run_my_method(method_adata, seed, config):
    # Your preprocessing + training/integration code.
    latent = my_method(method_adata, seed=seed, **config)
    return MethodOutput(latent=latent, barcodes=method_adata.obs_names)

method = MethodSpec(
    name="MyMethod",
    runner=run_my_method,
    config=METHOD_CONFIG,
    dependencies=METHOD_DEPS,
)

result = benchmark_method(
    adata,
    method,
    seeds=[42, 123, 2026],        # or one integer for a single run
    benchmark_config={"random_state": 42},  # fixed evaluation seed
    install_dependencies=False,  # your environment remains under your control
)
```

`MethodSpec.runner` may return a NumPy array, a DataFrame, an `adata.obsm` key, or `MethodOutput`. `MethodOutput` is preferred when you want to supply explicit barcodes or provenance files.

### Low-level API — maximum control

Use `evaluate_latent`, `write_interactive_report`, `create_report_bundle`, `write_multiseed_interactive_report`, and `finalize_multiseed_delivery` directly. This is appropriate when your project owns custom caching/resume logic, artifact layout, or evaluation orchestration.

The repository includes `scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb` for this path.

## Dependency ownership

Method dependencies are never hardcoded into scRareBench. Choose one of three approaches:

```python
# 1. Preinstall dependencies yourself (conda/uv/Docker/pip).

# 2. In Colab, use the runtime helper.
from scrarebench.runtime import setup_runtime
setup_runtime(extra_requirements=("my-method==1.2.0",), extra_imports=("my_method",))

# 3. Explicit high-level opt-in.
result = benchmark_method(..., install_dependencies=True)
```

For custom environment policy, pass `MethodSpec(..., installer=my_installer)`. scRareBench calls that installer only when `install_dependencies=True`.

## Single-seed versus multi-seed

The **method seed** and **benchmark seed** are separate contracts:

- method seed varies across stochastic training/integration replicates;
- benchmark/evaluation seed stays fixed across a comparable family;
- multi-seed aggregation summarizes metric tables only;
- embeddings, UMAP coordinates, clusters, Sankey topology, and cell-level state remain seed-specific and are never averaged.

Multi-seed finalization validates compatible method/evaluation/dataset identity before producing the combined report and delivery archive.

## Cell identity contract

The latent must contain exactly one row per benchmark cell. Prefer returning barcodes with the latent. If rows are reordered, `benchmark_latent(..., allow_reorder=True)` can reorder only after verifying that cell sets match. Silent cell removal, duplication, or cross-dataset latents are rejected.

## Custom datasets and rare populations

```python
from scrarebench import register_dataset

register_dataset(
    adata,
    name="MyDataset",
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
    rare_types=["RareType"],  # optional
)
```

The six-state GR/LE/SR × DL/RM taxonomy is **not required** for a custom dataset. If only `rare_types` are supplied, scRareBench still computes generic rare-recovery metrics and marks scenario metadata as `UNASSIGNED`. Scenario-specific paper summaries remain separate.

## Additional rare-recovery diagnostics

The hardened metric layer includes the historical metrics plus:

- full-space selected-cell ASW (`ASW_selected_cells_in_full_latent`);
- best-cluster precision/recall/F1 without majority-vote ownership competition;
- kNN same-label fraction and abundance-null expectation;
- support-adjusted kNN local recovery for very small populations;
- separate, status-aware non-rare/rare ratio diagnostics;
- rare-resolution sensitivity tables;
- resolution-aware failure taxonomy while retaining legacy failure labels.

Metric direction and plain-language semantics are exposed through `METRIC_REGISTRY`, `metric_info()`, and `metric_direction()`.

## Notebook map

- `scRareBench_CustomMethod_HighLevel_Colab.ipynb`: generic high-level template for any method.
- `scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb`: runnable high-level scVI example with scalar/list method seeds.
- `scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb`: same high-level contract on mBDRC.
- `scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb`: explicit low-level multi-seed orchestration.
- Existing method-specific notebooks remain examples; they are not package-supported method implementations.
