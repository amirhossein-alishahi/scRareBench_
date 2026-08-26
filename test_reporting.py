from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scrarebench.reporting import write_html_report


def test_html_report_embeds_static_figures(tmp_path: Path):
    figures = tmp_path / "figures"
    figures.mkdir()
    image_path = figures / "example.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(image_path)
    plt.close(fig)

    report = write_html_report(
        tmp_path / "report.html",
        title="Test",
        metadata={"method": "demo"},
        global_table=pd.DataFrame({"metric": ["ARI"], "value": [0.5]}),
        rare_table=pd.DataFrame({"cell_type": ["pDC"], "recall": [0.8]}),
        figure_names=[image_path.name],
    )
    text = report.read_text(encoding="utf-8")
    assert 'src="data:image/png;base64,' in text
    assert 'src="figures/example.png"' not in text
