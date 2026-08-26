import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scrarebench.dashboard import _build_payload
from scrarebench.reporting import write_interactive_report


class DemoAdata:
    def __init__(self):
        self.obs = pd.DataFrame(
            {
                "celltype": ["ILC", "ILC", "ILC1", "ILC1", "pDC", "pDC", "X"],
                "BATCH": ["b1", "b2", "b1", "b2", "b1", "b2", "b1"],
                "scrarebench_scenario": ["GR-DL", "GR-DL", "GR-DL", "GR-DL", "LE-DL", "LE-DL", ""],
                "cluster": ["0", "0", "1", "1", "2", "2", "3"],
                "pred": ["NK", "NK", "T", "T", "pDC", "pDC", "X"],
            },
            index=[f"cell_{i}" for i in range(7)],
        )
        self.obsm = {"X_test": np.asarray([[i, i / 10, 0.0] for i in range(7)], dtype=float)}
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)


def result(tmp_path: Path):
    rare = pd.DataFrame(
        {
            "cell_type": ["ILC", "ILC1", "pDC"],
            "scenario": ["GR-DL", "GR-DL", "LE-DL"],
            "distribution": ["GR", "GR", "LE"],
            "topology": ["DL", "DL", "DL"],
            "support": [2, 2, 2],
            "precision": [0.0, 0.0, 1.0],
            "recall": [0.0, 0.0, 1.0],
            "f1": [0.0, 0.0, 1.0],
            "inverse_purity": [0.9, 0.8, 1.0],
            "within_type_batch_nmi": [0.05, 0.1, 0.0],
            "failure_archetype": ["lineage_assimilation", "lineage_assimilation", "preserved"],
            "failure_rationale": ["absorbed", "absorbed", "preserved"],
            "dominant_wrong_label": ["NK", "T", ""],
            "dominant_wrong_fraction": [0.9, 0.8, 0.0],
            "n_clusters_found_in": [2, 2, 1],
            "curation_source": ["paper_curated"] * 3,
            "parent_type": ["", "", ""],
        }
    )
    return SimpleNamespace(
        output_dir=tmp_path,
        subset_metrics=pd.DataFrame({"subset": ["overall"], "F1_macro": [0.5]}),
        per_type_metrics=rare[["cell_type", "precision", "recall", "f1"]].copy(),
        rare_metrics=rare,
        rare_summary=pd.DataFrame(
            {"metric": ["precision", "recall", "f1", "preserved_fraction"], "mean": [1 / 3, 1 / 3, 1 / 3, 1 / 3], "median": [0, 0, 0, np.nan], "n_valid": [3, 3, 3, 3]}
        ),
        scenario_metrics=pd.DataFrame({"scenario": ["GR-DL", "LE-DL"], "f1_mean": [0.0, 1.0]}),
        cluster_keys={1.0: "cluster"},
        prediction_key="pred",
        scib=None,
        files={},
    )


def test_ui_audit_tokens_and_js_syntax(tmp_path: Path):
    path = write_interactive_report(DemoAdata(), result(tmp_path), tmp_path / "dashboard.html", representation_key="X_test")
    text = path.read_text(encoding="utf-8")
    for token in [
        "One shared filter state controls the rare table, UMAP, heatmaps, and population details",
        "Filtered rare-cell metric heatmaps",
        "Recovery metrics",
        "Batch-dependence diagnostic",
        "Matching populations",
        "Cell type name, e.g. ILC or pDC",
        "scenario-chip",
        "Reset filters",
        "Download original",
        "Export view",
        "Export all",
        "No Sankey flows match the threshold",
        "role=\"tablist\"",
    ]:
        assert token in text

    # Search must be restricted to the cell-type field rather than JSON.stringify(row).
    assert "String(r.cell_type||'').toLowerCase().includes(q)" in text
    assert "JSON.stringify(r).toLowerCase().includes(q)" not in text

    scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.S)
    runtime = tmp_path / "runtime.js"
    runtime.write_text(scripts[-1], encoding="utf-8")
    subprocess.run(["node", "--check", str(runtime)], check=True)


def test_granular_rare_flags_omit_points_when_possible(tmp_path: Path):
    adata = DemoAdata()
    res = result(tmp_path)
    payload = _build_payload(
        adata,
        res,
        representation_key="X_test",
        label_key="celltype",
        batch_key="BATCH",
        scenario_key="scrarebench_scenario",
        umap_key=None,
        random_state=0,
        include_overview=False,
        include_metrics=False,
        include_scib=False,
        include_rare=True,
        include_rare_umap=False,
        include_rare_heatmaps=True,
        include_rare_scenario_analysis=False,
        include_umap=False,
        include_sankey=False,
        include_reproducibility=False,
        include_static_figures=False,
        include_cell_ids=False,
    )
    assert payload["features"] == {
        "rare_umap": False,
        "rare_heatmaps": True,
        "rare_scenario_analysis": False,
    }
    assert "points" not in payload
    assert payload["rare"]["category_breakdown"] == []
