from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scrarebench.reporting import create_report_bundle, write_interactive_report, write_pdf_report


class MiniAdata:
    def __init__(self):
        self.obs = pd.DataFrame(
            {
                "celltype": ["A", "A", "B", "B", "C", "C"],
                "BATCH": ["x", "x", "y", "y", "x", "y"],
                "scrarebench_scenario": pd.Categorical(["GR-DL", "GR-DL", None, None, "LE-RM", "LE-RM"]),
                "cluster": ["0", "0", "1", "1", "2", "2"],
                "pred": ["A", "A", "B", "B", "C", "B"],
            },
            index=[f"cell_{i}" for i in range(6)],
        )
        self.obsm = {
            "X_test": np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.2, 0.1, 1.1],
                    [1.0, 0.9, 0.2],
                    [1.1, 0.8, 0.3],
                    [2.0, 1.8, 0.4],
                    [1.9, 1.7, 0.5],
                ],
                dtype=float,
            )
        }
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)



def build_result(tmp_path: Path):
    out = tmp_path / "result_dir"
    out.mkdir()
    static_report = out / "report.html"
    static_report.write_text("<html><body>stub</body></html>", encoding="utf-8")
    return SimpleNamespace(
        output_dir=out,
        subset_metrics=pd.DataFrame({"subset": ["overall"], "accuracy": [0.9]}),
        rare_metrics=pd.DataFrame(
            {
                "cell_type": ["A", "C"],
                "precision": [1.0, 0.5],
                "recall": [1.0, 0.5],
                "f1": [1.0, 0.5],
                "inverse_purity": [1.0, 0.5],
                "within_type_batch_nmi": [0.0, 0.4],
                "failure_archetype": ["preserved", "lineage_assimilation"],
            }
        ),
        rare_summary=pd.DataFrame({"metric": ["precision"], "mean": [0.75]}),
        scenario_metrics=pd.DataFrame({"scenario": ["GR-DL"], "precision_mean": [1.0]}),
        cluster_keys={1.0: "cluster"},
        prediction_key="pred",
        scib=None,
        files={"report": static_report},
    )



def test_interactive_pdf_and_bundle(tmp_path: Path):
    adata = MiniAdata()
    result = build_result(tmp_path)
    html_path = write_interactive_report(adata, result, tmp_path / "interactive.html", representation_key="X_test")
    pdf_path = write_pdf_report(adata, result, tmp_path / "summary.pdf", representation_key="X_test")
    bundle_path = create_report_bundle(adata, result, tmp_path / "bundle.zip", representation_key="X_test", include_latent=True)

    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Plotly.react" in html_text
    assert "non_rare" in html_text
    assert pdf_path.exists() and pdf_path.stat().st_size > 500
    assert bundle_path.exists() and bundle_path.stat().st_size > 1000
