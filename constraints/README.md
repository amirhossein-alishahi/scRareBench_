# Reproducibility constraints

`colab-2026.07-anchors.txt` is an optional reference constraint file for the documented Google Colab environment used with the release notebooks.

It is not intended to be a complete environment lock. scRareBench's runtime helper detects ABI-sensitive packages already present in the running environment and constrains them during dependency installation.

Typical Colab bootstrap:

```python
%pip install -q --no-deps "git+https://github.com/amirhossein-alishahi/scRareBench_.git@v0.10.5"

from scrarebench.runtime import setup_runtime

report = setup_runtime()
```

To add your own constraint file:

```python
report = setup_runtime(
    constraint_files=("constraints/colab-2026.07-anchors.txt",),
)
```

Method-specific dependencies can be supplied independently through `extra_requirements=` and `extra_imports=`. The integration method remains user-owned and is not inferred from its name by scRareBench.
