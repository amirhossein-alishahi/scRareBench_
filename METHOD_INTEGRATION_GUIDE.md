# scRareBench method-independent integration contract

scRareBench does not implement or depend on an integration method. A method is responsible only for producing a finite 2-D cell-by-dimension representation in the exact benchmark cell order (or with explicit barcodes that scRareBench can align).

Stable benchmark contract:

```python
from scrarebench import attach_latent, EvaluationConfig, evaluate_latent

attach_latent(
    adata,
    method_latent,
    key="X_my_method",
    latent_barcodes=barcodes,   # strongly recommended
    allow_reorder=False,
    overwrite=True,
)

result = evaluate_latent(
    adata,
    EvaluationConfig(
        method_name="my_method",
        representation_key="X_my_method",
    ),
    output_dir,
)
```

The package and benchmark logic remain identical across scVI, Harmony, Scanorama, scANVI, custom neural models, or any other method. Only the method-specific notebook cells that generate `method_latent` change.

## Notebook installation rule

Method notebooks must use `tools/notebook_runtime.py`. It resolves the package and the method in a single pip transaction, preserves already-installed scientific ABI anchors, runs `pip check`, and performs a fresh-process import smoke test before model work starts. Do not add `pip install --upgrade <project>` to method notebooks.
