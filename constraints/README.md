# Notebook reproducibility constraints

`colab-2026.07-anchors.txt` records only versions that are explicitly documented/validated for the release workflow: the published Colab 2026.07 core anchors used as a **validation reference** for the official notebooks and scRareBench's pinned benchmark backend. The notebooks do not enforce this file by default.

It is **not** a complete environment lock. Do not add exact Scanpy, AnnData, SciPy, pandas, Leiden, etc. versions merely to make the file look complete. Add them only after a successful end-to-end release notebook run records the actual environment.

`scrarebench.runtime.setup_runtime()` already snapshots and constrains ABI-sensitive packages present in the running environment. Users may additionally pass one or more files via `constraint_files=`.
