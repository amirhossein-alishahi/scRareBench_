"""Low-level API example for users who need explicit benchmark configuration."""

from scrarebench import EvaluationConfig, ScibEvaluationConfig, attach_latent, evaluate_latent

attach_latent(
    adata,  # noqa: F821
    latent,  # noqa: F821
    key="X_my_method",
    latent_barcodes=cell_ids,  # noqa: F821
)

config = EvaluationConfig(
    method_name="MyMethod",
    representation_key="X_my_method",
    label_key="cell_type",
    batch_key="batch",
    scib=ScibEvaluationConfig(enabled=True, count_layer="counts"),
)

result = evaluate_latent(adata, config, "results/MyMethod")  # noqa: F821
print(result.files["report"])
