# scRareBench release checklist

For v0.10.4:

1. Ensure the repository root contains `pyproject.toml`, `src/`, `notebooks/`, `tests/`, `constraints/`, `scripts/`, and `.github/`. Do not commit generated `build/`, `dist/`, `*.egg-info`, cache, or benchmark-result directories.
2. Run the local release checks from a clean source tree:
   ```bash
   python scripts/check_notebooks.py
   python -m compileall -q src tests scripts examples
   python -m pytest
   python -m build
   ```
3. Verify version coherence: `pyproject.toml`, `src/scrarebench/__init__.py`, `CITATION.cff`, official notebook Git refs, and this checklist must all refer to `0.10.4` / `v0.10.4`.
4. Push the validated release commit to GitHub and confirm GitHub Actions is green on that commit.
5. Create the immutable release tag from exactly that commit:
   ```bash
   git tag -a v0.10.4 -m "scRareBench v0.10.4"
   git push origin v0.10.4
   ```
6. Confirm that every official notebook resolves `...scRareBench_.git@v0.10.4`, never `@main`. The optional Colab compatibility-check cell should remain fully commented by default.
7. For final release validation, Google Colab runtime **2026.07** is recommended because it is the documented reference environment, but the notebooks must not hard-fail solely because NumPy/PyTorch/JAX anchors differ. Run both high-level scVI notebooks from a fresh runtime with **Run all**, then smoke-test at least one detailed Dataset 0 notebook and one detailed Dataset 2 notebook. Save the generated package-version/environment manifest from successful runs.
8. If any release-tagged notebook or CI check fails, **do not move or overwrite the tag**. Fix the issue, increment the patch version, and create a new release tag.
9. Only after tag CI and fresh-Colab validation pass should the release be announced or used as the manuscript/reproducibility reference.
