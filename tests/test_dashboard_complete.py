from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scrarebench.reporting import write_interactive_report


class MiniAdata:
    def __init__(self):
        self.obs = pd.DataFrame(
            {
                "celltype": ["A", "A", "B", "B", "C", "C"],
                "BATCH": ["x", "x", "y", "y", "x", "y"],
                "scrarebench_scenario": pd.Categorical(["GR-DL", "GR-DL", None, None, "LE-RM", "LE-RM"]),
                "cluster": ["0", "0", "1", "1", "2", "2"],
                "pred": ["A", "A", "B", "B", "C", "B"],
            }, index=[f"cell_{i}" for i in range(6)]
        )
        self.obsm = {"X_test": np.array([[0,0,1],[.2,.1,1],[1,.9,.2],[1.1,.8,.3],[2,1.8,.4],[1.9,1.7,.5]], dtype=float)}
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)


def _result(tmp_path: Path):
    out = tmp_path / "out"; out.mkdir()
    fig_path = out / "scib_plot.png"
    fig, ax = plt.subplots(); ax.bar(["Total"], [.8]); fig.savefig(fig_path); plt.close(fig)
    for name, text in [("run.yaml", "seed: 42\n"),("rules.yaml", "preserved: true\n"),("versions.txt", "python=3.x\n")]:
        (out/name).write_text(text)
    scib = SimpleNamespace(
        backend="scib-metrics", backend_version="0.5.9",
        metrics_long=pd.DataFrame({"metric":["iLISI","Graph connectivity"],"value":[.7,.8],"metric_type":["Batch correction","Batch correction"]}),
        aggregate_scores=pd.DataFrame({"metric":["Bio conservation","Batch correction","Total"],"value":[.8,.7,.76],"metric_type":["Aggregate score"]*3}),
        metric_status=pd.DataFrame({"metric":["iLISI","HVG overlap"],"status":["computed","not_applicable"],"reason":["ok","latent only"]}),
        results_wide=pd.DataFrame({"iLISI":[.7],"Total":[.76]}, index=["X_test"]),
        reference_config={"n_hvg":4000}, files={"metric_plot":fig_path},
    )
    return SimpleNamespace(
        output_dir=out,
        subset_metrics=pd.DataFrame({"subset":["overall","rare"],"ARI_true_vs_cluster":[.8,.5],"F1_macro":[.75,.55]}),
        per_type_metrics=pd.DataFrame({"cell_type":["A","B","C"],"precision":[1,.9,.5],"recall":[1,.9,.5]}),
        rare_metrics=pd.DataFrame({"cell_type":["A","C"],"scenario":["GR-DL","LE-RM"],"distribution":["GR","LE"],"topology":["DL","RM"],"support":[2,2],"precision":[1,.5],"recall":[1,.5],"f1":[1,.5],"inverse_purity":[1,.5],"within_type_batch_nmi":[0,.4],"failure_archetype":["preserved","lineage_assimilation"]}),
        rare_summary=pd.DataFrame({"metric":["precision","recall","f1","preserved_fraction"],"mean":[.75,.75,.75,.5],"median":[.75,.75,.75,np.nan],"n_valid":[2,2,2,2]}),
        scenario_metrics=pd.DataFrame({"scenario":["GR-DL","LE-RM"],"f1_mean":[1,.5]}),
        cluster_keys={1.0:"cluster"}, prediction_key="pred", scib=scib,
        files={"run_config":out/"run.yaml","failure_rules":out/"rules.yaml","package_versions":out/"versions.txt","scib_metric_plot":fig_path},
    )


def test_complete_dashboard_and_flags(tmp_path: Path):
    adata=MiniAdata(); result=_result(tmp_path)
    full=write_interactive_report(adata,result,tmp_path/"full.html",representation_key="X_test")
    text=full.read_text()
    for token in ["Rare-cell Explorer","All scIB-compatible metrics","Rare-cell UMAP","Metric availability and applicability","Reproducibility","Export PNG"]:
        assert token in text
    assert "HVG overlap" in text
    assert "data:image/png;base64," in text

    light=write_interactive_report(
        adata,result,tmp_path/"light.html",representation_key="X_test",
        include_scib=False,include_rare=False,include_umap=False,include_sankey=False,
        include_reproducibility=False,include_static_figures=False,include_cell_ids=False,
    )
    light_text=light.read_text()
    assert '"scib":false' in light_text
    assert '"rare":false' in light_text
    assert '"figures":false' in light_text
    assert "scib_plot.png" not in light_text
