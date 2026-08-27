from pathlib import Path
from types import SimpleNamespace
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scrarebench.reporting import write_interactive_report


class DemoAdata:
    def __init__(self):
        scenarios = ["GR-DL", "GR-RM", "LE-DL", "LE-RM", "SR-DL", "SR-RM"]
        celltypes = ["A", "B", "C", "D", "E", "F"]
        rows = []
        coords = []
        for j, (ct, scn) in enumerate(zip(celltypes, scenarios)):
            for i in range(5):
                rows.append((ct, "b1" if i < 3 else "b2", scn, str(j), ct))
                coords.append([j + 0.05 * i, (j % 3) + 0.04 * i, 0.1 * j])
        self.obs = pd.DataFrame(rows, columns=["celltype", "BATCH", "scrarebench_scenario", "cluster", "pred"])
        self.obs.index = [f"cell_{i}" for i in range(len(self.obs))]
        self.obsm = {"X_test": np.asarray(coords, dtype=float)}
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)


def make_result(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    fig_path = out / "rare_metric_heatmap.png"
    fig, ax = plt.subplots()
    ax.imshow(np.arange(9).reshape(3, 3))
    fig.savefig(fig_path)
    plt.close(fig)
    rare = pd.DataFrame(
        {
            "cell_type": list("ABCDEF"),
            "scenario": ["GR-DL", "GR-RM", "LE-DL", "LE-RM", "SR-DL", "SR-RM"],
            "distribution": ["GR", "GR", "LE", "LE", "SR", "SR"],
            "topology": ["DL", "RM", "DL", "RM", "DL", "RM"],
            "support": [5] * 6,
            "precision": [0.9, 0.3, 0.8, 0.2, 0.7, 0.4],
            "recall": [0.9, 0.2, 0.8, 0.3, 0.6, 0.4],
            "f1": [0.9, 0.24, 0.8, 0.24, 0.65, 0.4],
            "inverse_purity": [0.9, 0.7, 0.8, 0.5, 0.6, 0.4],
            "within_type_batch_nmi": [0.1, 0.2, 0.1, 0.6, 0.3, 0.7],
            "failure_archetype": ["preserved", "lineage_assimilation", "preserved", "batch_driven_fragmentation", "lineage_leakage", "mixed_or_uncertain"],
            "failure_rationale": ["ok", "absorbed", "ok", "fragmented", "false positives", "mixed"] * 1,
            "dominant_wrong_label": ["", "X", "", "Y", "Z", "Q"],
            "dominant_wrong_fraction": [0, .8, 0, .4, .3, .2],
            "n_clusters_found_in": [1, 3, 1, 4, 2, 5],
            "parent_type": ["", "A", "", "B", "", "C"],
            "curation_source": ["paper_curated"] * 6,
        }
    )
    scib = SimpleNamespace(
        backend="scib-metrics", backend_version="0.5.9",
        metrics_long=pd.DataFrame({"metric": ["iLISI"], "value": [0.7], "metric_type": ["Batch correction"]}),
        aggregate_scores=pd.DataFrame({"metric": ["Total"], "value": [0.7], "metric_type": ["Aggregate score"]}),
        metric_status=pd.DataFrame({"metric": ["HVG overlap"], "status": ["not_applicable"], "reason": ["latent only"]}),
        results_wide=pd.DataFrame({"Total": [0.7]}, index=["X_test"]),
        reference_config={"n_hvg": 4000}, files={},
    )
    return SimpleNamespace(
        output_dir=out,
        subset_metrics=pd.DataFrame({"subset": ["overall"], "F1_macro": [0.6]}),
        per_type_metrics=rare[["cell_type", "precision", "recall", "f1"]].copy(),
        rare_metrics=rare,
        rare_summary=pd.DataFrame({"metric": ["precision", "recall", "f1", "preserved_fraction"], "mean": [.55, .53, .54, 2/6], "median": [.55, .5, .52, np.nan], "n_valid": [6, 6, 6, 6]}),
        scenario_metrics=pd.DataFrame({"scenario": rare["scenario"], "f1_mean": rare["f1"]}),
        cluster_keys={1.0: "cluster"}, prediction_key="pred", scib=scib,
        files={"rare_metric_heatmap": fig_path},
    )


def test_dashboard_payload_features_and_js(tmp_path: Path):
    html_path = write_interactive_report(DemoAdata(), make_result(tmp_path), tmp_path / "dashboard.html", representation_key="X_test")
    text = html_path.read_text(encoding="utf-8")
    for token in [
        "Performance and outcome by rare scenario",
        "Cell-type performance within selected rare scenario",
        "Outcome and failure-mode profile across rare scenarios",
        "Export view",
        "Export all",
        "figModal",
        "Enlarge figure",
        "const rendered=new Set()",
        '"category_breakdown"',
        '"scenario":"GR-DL"',
        '"scenario":"SR-RM"',
    ]:
        assert token in text

    scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.S)
    assert len(scripts) >= 2
    js_path = tmp_path / "dashboard_runtime.js"
    js_path.write_text(scripts[-1], encoding="utf-8")
    subprocess.run(["node", "--check", str(js_path)], check=True)
