from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scrarebench import benchmark_latent


def test_declared_missing_count_layer_fails_closed_for_scib(tmp_path):
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "B", "B"], "BATCH": ["b1", "b2", "b1", "b2"]},
        index=["c0", "c1", "c2", "c3"],
    )
    adata = ad.AnnData(X=np.ones((4, 3), dtype=float), obs=obs)
    adata.uns["scrarebench"] = {
        "name": "demo",
        "dataset_key": "demo",
        "label_key": "celltype",
        "batch_key": "BATCH",
        "count_layer": "counts",
        "scenario_key": "scrarebench_scenario",
        "scib_hvg_batch_mode": "global",
    }
    with pytest.raises(KeyError, match=r"count_layer='counts'.*not present"):
        benchmark_latent(
            adata,
            np.arange(8, dtype=float).reshape(4, 2),
            method="Demo",
            output_dir=tmp_path / "out",
            config={
                "run_scib": True,
                "write_interactive_report": False,
                "write_pdf_report": False,
                "create_bundle": False,
            },
        )


def test_dataset2_high_level_notebook_uses_validated_count_and_global_hvg_contract():
    path = Path("notebooks/scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
    code = "\n".join(code_cells)
    runner = next(cell for cell in code_cells if "def run_scvi(" in cell)

    assert "ensure_counts_layer(adata, count_layer)" in code
    assert '"hvg_policy": "global_seurat_v3_raw_counts"' in code
    assert '"hvg_batch_key": None' in code
    assert 'if config.get("hvg_batch_key") is not None' in runner
    assert "batch_key=batch_key" not in runner.split("sc.pp.highly_variable_genes", 1)[1].split("scvi.settings.seed", 1)[0]
