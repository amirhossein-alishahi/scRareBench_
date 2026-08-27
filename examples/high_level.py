"""Minimal method-developer workflow using a registered scRareBench dataset."""

import pandas as pd

from scrarebench import benchmark_latent, load_dataset

adata = load_dataset(0)

# Replace this with your own integration method.
latent = my_integration_method(adata)  # noqa: F821
latent = pd.DataFrame(latent, index=adata.obs_names.astype(str))

result = benchmark_latent(adata, latent, method="MyMethod")
print(result.summary())
print(result.interactive_report_path)
