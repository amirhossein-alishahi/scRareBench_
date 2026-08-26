# scRareBench v0.9.0 runtime audit

## Root cause fixed

The v0.8 method notebooks installed the method package and then reinstalled the local scRareBench project with a standalone `--upgrade`. In a long-running notebook kernel that can replace ABI-sensitive packages such as NumPy while SciPy/scikit-learn or other compiled extensions remain from a different compatible set. The observed Harmony notebook failed during import inside NumPy/SciPy/sklearn before Harmony evaluation began.

## v0.9.0 architecture

- scRareBench remains one method-independent benchmark package.
- `harmonypy` and `scvi-tools` are not scRareBench package dependencies.
- A method contributes only a cell-by-dimension representation and method metadata.
- Both notebooks use the same stdlib-only `tools/notebook_runtime.py` installer.
- scRareBench + the method dependency are resolved in one pip transaction.
- The installer never uses a standalone `pip install --upgrade`.
- Already-installed ABI-sensitive anchors (NumPy, SciPy, pandas, scikit-learn, matplotlib, h5py, numba/llvmlite, JAX/JAXLIB, PyTorch when present) are constrained to their current versions for that transaction.
- Relevant `pip check` conflicts are fatal; unrelated preinstalled-environment warnings are reported but do not abort a valid benchmark environment.
- A fresh Python subprocess imports NumPy/SciPy/sklearn/pandas, executes a SciPy sparse + sklearn metric smoke test, and imports scRareBench, scIB, and the selected method before any expensive model computation.

## Validation performed

- Python package tests: 34/34 passed.
- `compileall`: package + runtime helper passed.
- Harmony notebook: 13/13 code cells parsed without Python syntax errors.
- scVI notebook: 13/13 code cells parsed without Python syntax errors.
- Both notebooks contain no standalone `--upgrade` installation flag.
- Both notebooks use the same shared runtime installer.
- Built wheel metadata: `scrarebench==0.9.0`.
- Wheel installed into a clean target directory and imported as version 0.9.0 using the available scientific stack.
- Source package contains one 0.9.0 wheel, both notebooks, the shared runtime installer, and the method-integration guide.

## Stable method contract

For any new method, do not edit scRareBench. Produce the latent representation, attach it with `attach_latent`, and evaluate it with the unchanged `EvaluationConfig` / `evaluate_latent` API. Method-specific dependencies belong in the notebook/runtime request only.
