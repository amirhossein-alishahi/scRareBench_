import re
import subprocess
from pathlib import Path

from scrarebench.reporting import write_interactive_report
from test_dashboard_payload import DemoAdata, make_result


def test_dashboard_interaction_contract(tmp_path: Path):
    path = write_interactive_report(DemoAdata(), make_result(tmp_path), tmp_path / "dashboard.html", representation_key="X_test")
    text = path.read_text(encoding="utf-8")

    required = [
        'id="umapSelectionDetail"',
        "el.on?.('plotly_click'",
        "umapState.selected=new Set([String(row.cell_type)])",
        "renderUmapSelectionDetail();renderUmap()",
        'label for="rareOutcome"',
        'Failure mode <span class="note">(applies to not-preserved populations)</span>',
        "rareOutcome(r)===rareState.outcome",
        "failure==='preserved'",
        "Click to inspect this population",
        "e.target.value=String(v);range.value=String(v)",
        "modalNavIndices",
        'id="zoomFit"',
        'id="zoomActual"',
        "a<1e-4)return v.toExponential(3)",
        "Failure modes among not-preserved populations",
        "Most frequent failure mode among not-preserved populations",
        "DL — Distinct lineage",
        "RM — Related manifold / state",
    ]
    for token in required:
        assert token in text, token

    # Removed/renamed misleading semantics.
    assert "Rare failure snapshot:" not in text
    assert "<b>Main observed failure:</b>" not in text
    assert 'label for="rareFailure">Failure archetype' not in text

    scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.S)
    runtime = tmp_path / "runtime.js"
    runtime.write_text(scripts[-1], encoding="utf-8")
    subprocess.run(["node", "--check", str(runtime)], check=True)
