# Method developer guide

scRareBench is designed so that a new integration method can be benchmarked **without modifying scRareBench**.

## The required boundary

```text
benchmark AnnData
    -> your preprocessing/training
    -> latent representation + cell identity
    -> scRareBench evaluation
```

Your method may be a Python package, notebook implementation, local module, command-line program, or unpublished research prototype. scRareBench does not require a method adapter or registration entry.

## Recommended workflow

```python
from scrarebench import load_dataset, benchmark_latent
import pandas as pd

adata = load_dataset(0)

# Completely external method code.
latent = run_my_method(adata)

# Recommended: attach cell IDs to the latent itself.
latent = pd.DataFrame(latent, index=adata.obs_names.astype(str))

result = benchmark_latent(
    adata,
    latent,
    method="MyMethod",
)
```

## Cell identity contract

The latent must contain exactly one row per benchmark cell. Prefer a DataFrame indexed by `adata.obs_names`. If a method changes order, provide barcodes and use `allow_reorder=True`; scRareBench validates that the cell sets match before reordering.

A method should not silently remove cells, duplicate cells, or return a latent for a different dataset instance.

## Custom dataset workflow

```python
from scrarebench import register_dataset, benchmark_latent

register_dataset(
    adata,
    name="MyDataset",
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
)

latent = run_my_method(adata)
result = benchmark_latent(adata, latent, method="MyMethod")
```

Rare population metadata is optional. If absent, scRareBench runs the global/standard benchmark without inventing a rare taxonomy.

## Dependencies in Colab

Method dependencies belong to the method developer. The generic runtime helper can install them safely in a pre-populated environment:

```python
from scrarebench.runtime import setup_runtime

setup_runtime(
    extra_requirements=("my-method-package==1.0.0",),
    extra_imports=("my_method_package",),
)
```

No scRareBench code change is needed for a new method name or dependency set.

## Release notebooks

Official release notebooks install a fixed scRareBench Git tag (`v0.10.3` for this release), not the moving `main` branch. Method dependencies are declared explicitly inside the example notebook. The package itself remains method-agnostic.

## What belongs in a method example notebook

A method-specific notebook may show:

1. method dependency setup;
2. loading a scRareBench dataset;
3. method-specific preprocessing;
4. training/inference;
5. latent extraction;
6. one `benchmark_latent()` call or the equivalent low-level API;
7. optional method-specific plots.

It should not move method implementation into the scRareBench package.
