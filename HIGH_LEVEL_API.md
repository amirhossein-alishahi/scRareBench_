# High-level API quick reference

```python
from scrarebench import MethodOutput, MethodSpec, benchmark_method

method = MethodSpec(
    name="MyMethod",
    dependencies=("my-method==1.0",),
    config={"latent_dim": 30},
    runner=lambda adata, seed, config: MethodOutput(
        latent=run_my_method(adata, seed=seed, **config),
        barcodes=adata.obs_names,
    ),
)

result = benchmark_method(
    adata,
    method,
    seeds=[42, 123, 2026],
    benchmark_config={"random_state": 42},
)

print(result.summary())
print(result.report_path)
print(result.archive_path)
```

The method runner and dependencies are user-owned. For a one-off latent that was generated elsewhere, continue to use `benchmark_latent()` directly.
