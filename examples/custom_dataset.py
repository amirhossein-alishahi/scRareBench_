"""Minimal workflow for a user-provided AnnData."""

import scanpy as sc
import pandas as pd

from scrarebench import benchmark_latent, register_dataset

adata = sc.read_h5ad("my_atlas.h5ad")
register_dataset(
    adata,
    name="MyAtlas",
    label_key="cell_type",
    batch_key="batch",
    count_layer="counts",
)

# Replace this with your own integration method.
latent = my_integration_method(adata)  # noqa: F821
latent = pd.DataFrame(latent, index=adata.obs_names.astype(str))
result = benchmark_latent(adata, latent, method="MyMethod")
print(result.summary())
