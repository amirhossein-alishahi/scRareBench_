from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scrarebench.reporting import write_html_report


def test_combined_report_contains_scib_and_rare_sections(tmp_path: Path):
    image = tmp_path / "score.png"
    fig, ax = plt.subplots()
    ax.bar(["Total"], [0.75])
    fig.savefig(image)
    plt.close(fig)

    report = write_html_report(
        tmp_path / "report.html",
        title="Combined",
        metadata={"method": "demo"},
        global_table=pd.DataFrame({"subset": ["overall"], "ARI": [0.5]}),
        rare_table=pd.DataFrame({"cell_type": ["pDC"], "f1": [0.4]}),
        figure_names=[image],
        scib_metrics=pd.DataFrame({"metric": ["iLISI"], "value": [0.8]}),
        scib_aggregates=pd.DataFrame({"metric": ["Total"], "value": [0.75]}),
        scib_status=pd.DataFrame({"metric": ["HVG overlap"], "status": ["not_applicable"]}),
        rare_summary=pd.DataFrame({"metric": ["f1"], "mean": [0.4]}),
        scenario_table=pd.DataFrame({"scenario": ["LE-DL"], "f1_mean": [0.4]}),
    )
    text = report.read_text(encoding="utf-8")
    assert "Standard scIB-compatible evaluation" in text
    assert "Rare-cell-specific evaluation" in text
    assert "HVG overlap" in text
    assert 'src="data:image/png;base64,' in text
