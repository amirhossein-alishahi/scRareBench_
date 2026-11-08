# High-Level scRareBench benchmark notebooks

These notebooks are the **complete user-supplied high-level execution collection** for dataset indices 0, 2, 3, 4 and 5. Each notebook uses `MethodSpec` + `benchmark_method()`; method-specific training/HVG selection stays inside the user-owned runner.

## Matrix

| Method | Dataset 0 · BMMC | Dataset 2 · Kidney | Dataset 3 · Breast cancer | Dataset 4 · COVID PBMC | Dataset 5 · NYGC PBMC |
| --- | --- | --- | --- | --- | --- |
| Harmony | [D0](scRareBench_HighLevel_Harmony_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_Harmony_Dataset2_Colab.ipynb) | [D3](scRareBench_HighLevel_Harmony_Dataset3_Colab.ipynb) | [D4](scRareBench_HighLevel_Harmony_Dataset4_Colab.ipynb) | [D5](scRareBench_HighLevel_Harmony_Dataset5_Colab.ipynb) |
| MrVI | [D0](scRareBench_HighLevel_MrVI_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_MrVI_Dataset2_Colab.ipynb) | [D3](scRareBench_HighLevel_MrVI_Dataset3_Colab.ipynb) | [D4](scRareBench_HighLevel_MrVI_Dataset4_Colab.ipynb) | [D5](scRareBench_HighLevel_MrVI_Dataset5_Colab.ipynb) |
| scDisInFact | [D0](scRareBench_HighLevel_scDisInFact_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_scDisInFact_Dataset2_Colab.ipynb) | [D3](scRareBench_HighLevel_scDisInFact_Dataset3_Colab.ipynb) | [D4](scRareBench_HighLevel_scDisInFact_Dataset4_Colab.ipynb) | [D5](scRareBench_HighLevel_scDisInFact_Dataset5_Colab.ipynb) |
| scDML | [D0](scRareBench_HighLevel_scDML_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_scDML_Dataset2_Colab.ipynb) | [D3](scRareBench_HighLevel_scDML_Dataset3_Colab.ipynb) | [D4](scRareBench_HighLevel_scDML_Dataset4_Colab.ipynb) | [D5](scRareBench_HighLevel_scDML_Dataset5_Colab.ipynb) |
| scVI | [D0](scRareBench_HighLevel_scVI_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_scVI_Dataset2_Colab.ipynb) | [D3](scRareBench_HighLevel_scVI_Dataset3_Colab.ipynb) | [D4](scRareBench_HighLevel_scVI_Dataset4_Colab.ipynb) | [D5](scRareBench_HighLevel_scVI_Dataset5_Colab.ipynb) |
| Seurat | [D0](scRareBench_HighLevel_Seurat_Dataset0_Colab.ipynb) | [D2](scRareBench_HighLevel_Seurat_Dataset2_Colab.ipynb) | — | — | — |

## Conventions

- Names use `scRareBench_HighLevel_<Method>_Dataset<N>_Colab.ipynb`; no `(1)`, `_TEST`, `_FINAL`, `_PINNED`, or `_CLEAN` suffixes.
- All existing execution outputs, transient widget state, and Colab-specific run IDs were removed; the scientific code and method-specific runtime settings are preserved.
- scRareBench Git dependency points to the reviewed taxonomy commit `cfdb720f5488b9ceef9a225b35fc7558bcb0251c` (D3 pDC-IRF7 GR-DL; D4 exhausted B and proliferating NK GR-RM). Review the benchmark paper before changing the pin.
- The default independent method seeds are intended to remain `[42, 123, 2026]`, with benchmark seed 42. The notebooks preserve their own method-specific and dataset-specific configurations.
- **Running the notebooks needs Google Colab, dataset downloads, external method dependencies (and for Seurat, R).** CI performs structural, code-identity and import/static checks; CI does not train large scRNA-seq models.
- For published result reproduction, retain the exact scRareBench revision, method seed, dataset fingerprint and the original HTML/report artifacts; new scenario classifications affect scenario-level summaries.

## Standalone Seurat server runs

The four companion archives were reviewed separately. Their latest Python entrypoint, setup/runner shell scripts, and one representative generated R script are kept in `../../scripts/seurat_server/`. Logs, backups and duplicated per-seed R scripts are excluded from version control. These are **historical Ubuntu execution records** whose virtual-environment paths are machine-specific; they are not directly runnable Colab notebooks.

## Audit

`NOTEBOOK_MANIFEST.json` records the source filename, canonical location and SHA256 of each normalized notebook. `NOTEBOOK_AUDIT.json` documents the nonfunctional edits and structural validations.
