"""Minimal Python API example after the benchmark dataset and latent exist."""
from pathlib import Path

import numpy as np

from scrarebench import (
    EvaluationConfig,
    ScibEvaluationConfig,
    attach_latent,
    evaluate_latent,
    load_dataset,
    load_latent,
)

adata = load_dataset(0, "data")
latent, _ = load_latent("my_method_latent.npy")
barcodes = np.load("my_method_barcodes.npy", allow_pickle=False)
attach_latent(adata, latent, key="X_my_method", latent_barcodes=barcodes)

result = evaluate_latent(
    adata,
    EvaluationConfig(
        method_name="MyMethod",
        representation_key="X_my_method",
        scib=ScibEvaluationConfig(enabled=True, count_layer="counts"),
    ),
    Path("results/MyMethod"),
)
print(result.files["report"])
print(result.scib.aggregate_scores if result.scib is not None else "scIB layer disabled")
