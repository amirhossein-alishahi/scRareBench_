from pathlib import Path
from scrarebench.metric_registry import METRIC_REGISTRY, metric_direction

ROOT = Path(__file__).parents[1]

def test_dashboard_assets_and_metric_semantics_are_packaged():
    assets = ROOT / "src/scrarebench/assets"
    for name in ("dashboard.css", "dashboard.js", "dashboard_template.html"):
        assert (assets / name).exists()
    assert metric_direction("knn_local_recovery_adjusted") == "maximize"
    assert metric_direction("within_type_batch_nmi") == "minimize"
    assert "ASW_selected_cells_in_full_latent" in METRIC_REGISTRY
