from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scrarebench
from scrarebench.constants import DEFAULT_BENCHMARK_SEED
from scrarebench.evaluation import EvaluationConfig, evaluate_latent
from scrarebench.metric_registry import metric_direction
from scrarebench.metrics import subset_metric_ratios
from scrarebench.reporting import create_report_bundle
from scrarebench.scib_backend import ScibEvaluationConfig, run_scib_evaluation


class MiniAdata:
    def __init__(self, *, label_key: str = "celltype", batch_key: str = "BATCH"):
        self.obs = pd.DataFrame(
            {
                label_key: ["A", "A", "B", "B"],
                batch_key: ["x", "y", "x", "y"],
                "cluster": ["0", "0", "1", "1"],
                "pred": ["A", "A", "B", "B"],
            },
            index=["c0", "c1", "c2", "c3"],
        )
        self.obs_names = self.obs.index
        self.n_obs = 4
        self.obsm = {"X_test": np.arange(12, dtype=np.float32).reshape(4, 3)}
        self.uns = {"scrarebench_dataset": {"dataset_key": "synthetic", "dataset_index": 99}}


def _bundle_result(tmp_path: Path):
    out = tmp_path / "result"
    out.mkdir()
    (out / "report.html").write_text("<html><body>stub</body></html>", encoding="utf-8")
    return SimpleNamespace(
        output_dir=out,
        subset_metrics=pd.DataFrame({"subset": ["overall"], "F1_macro": [0.9]}),
        subset_metric_ratios=pd.DataFrame(),
        per_type_metrics=pd.DataFrame({"cell_type": ["A", "B"], "f1": [1.0, 1.0]}),
        rare_metrics=pd.DataFrame(),
        rare_summary=pd.DataFrame(),
        scenario_metrics=pd.DataFrame(),
        cluster_keys={1.0: "cluster"},
        prediction_key="pred",
        scib=None,
        rare_evaluation_status={"status": "disabled"},
        scib_status={"status": "disabled"},
        files={"report": out / "report.html"},
    )


def test_version_and_seed_contract():
    assert scrarebench.__version__ == "0.10.6"
    assert DEFAULT_BENCHMARK_SEED == 42
    assert EvaluationConfig(method_name="m", representation_key="X").random_state == 42


def test_missing_scenario_metadata_fails_closed(monkeypatch, tmp_path: Path):
    import scrarebench.evaluation as ev

    adata = MiniAdata()
    monkeypatch.setattr(ev, "scenario_table_from_adata", lambda _: None)
    with pytest.raises(ValueError, match="dataset-specific scenario metadata"):
        ev._resolve_rare_metadata(adata, EvaluationConfig(method_name="m", representation_key="X_test"), None)


def test_rare_evaluation_can_be_explicitly_disabled():
    import scrarebench.evaluation as ev

    metadata, status = ev._resolve_rare_metadata(
        MiniAdata(),
        EvaluationConfig(method_name="m", representation_key="X_test", rare_evaluation=False),
        None,
    )
    assert metadata is None
    assert status["status"] == "disabled"


def test_subset_ratio_near_zero_is_not_extreme_number():
    frame = pd.DataFrame(
        {
            "subset": ["rare", "non_rare"],
            "F1_macro": [0.1, 0.5],
            "G_Mean": [1e-12, 0.5],
            "n_cells": [10, 90],
            "n_cell_types": [2, 5],
        }
    )
    ratios = subset_metric_ratios(frame)
    f1 = ratios.set_index("metric").loc["F1_macro"]
    gm = ratios.set_index("metric").loc["G_Mean"]
    assert f1["ratio"] == pytest.approx(5.0)
    assert gm["status"] == "unstable_denominator"
    assert pd.isna(gm["ratio"])


def test_metric_direction_registry_distinguishes_minimize_metrics():
    assert metric_direction("f1") == "maximize"
    assert metric_direction("within_type_batch_nmi") == "minimize"
    assert metric_direction("dominant_wrong_fraction") == "minimize"


def test_bundle_supports_nonstandard_keys_and_unicode_pickle_free_barcodes(tmp_path: Path, monkeypatch):
    import scrarebench.reporting as reporting

    adata = MiniAdata(label_key="labels", batch_key="samples")
    result = _bundle_result(tmp_path)
    # Keep the test focused on the bundle contract, not browser/PDF rendering.
    monkeypatch.setattr(reporting, "write_interactive_report", lambda *a, **k: Path(a[2]).write_text("html") or Path(a[2]))
    monkeypatch.setattr(reporting, "write_pdf_report", lambda *a, **k: Path(a[2]).write_bytes(b"pdf") or Path(a[2]))

    bundle = create_report_bundle(
        adata,
        result,
        tmp_path / "bundle.zip",
        representation_key="X_test",
        include_latent=True,
        label_key="labels",
        batch_key="samples",
        write_interactive=False,
        write_pdf=False,
    )
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        manifest = json.loads(z.read("bundle_manifest.json"))
        assert manifest["scrarebench_version"] == "0.10.6"
        assert manifest["cell_order_sha256"]
        results = json.loads(z.read("benchmark_results/results.json"))
        assert results["benchmark"]["label_key"] == "labels"
        assert results["benchmark"]["batch_key"] == "samples"
        barcode_name = next(n for n in names if n.endswith("_barcodes.npy"))
        extracted = tmp_path / "barcodes.npy"
        extracted.write_bytes(z.read(barcode_name))
        barcodes = np.load(extracted, allow_pickle=False)
        assert barcodes.dtype.kind == "U"
        assert barcodes.tolist() == list(adata.obs_names)


def test_scib_failure_reason_is_persisted_when_backend_is_optional(tmp_path: Path, monkeypatch):
    import scrarebench.scib_backend as backend

    def fail_dependency():
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(backend, "_require_scib_metrics", fail_dependency)
    result = run_scib_evaluation(
        MiniAdata(),
        representation_key="X_test",
        label_key="celltype",
        batch_key="BATCH",
        output_dir=tmp_path,
        config=ScibEvaluationConfig(enabled=True, require_backend=False),
    )
    assert result is None
    status = json.loads((tmp_path / "scib_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "synthetic backend failure" in status["error_message"]


def test_evaluate_latent_end_to_end_with_explicit_scenarios(tmp_path: Path, monkeypatch):
    import scrarebench.evaluation as ev

    labels = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    batches = np.array(["x", "y", "x", "y"] * 3)
    latent = np.vstack([
        np.random.default_rng(1).normal(0, 0.05, (4, 3)),
        np.random.default_rng(2).normal(2, 0.05, (4, 3)),
        np.random.default_rng(3).normal(4, 0.05, (4, 3)),
    ]).astype(np.float32)
    adata = SimpleNamespace()
    adata.obs = pd.DataFrame({"celltype": labels, "BATCH": batches}, index=[f"c{i}" for i in range(12)])
    adata.obs_names = adata.obs.index
    adata.n_obs = 12
    adata.obsm = {"X_demo": latent}
    adata.uns = {}

    def fake_clustering(adata, **kwargs):
        key = "demo_cluster"
        adata.obs[key] = pd.Categorical(["0"] * 4 + ["1"] * 4 + ["2"] * 4)
        return SimpleNamespace(cluster_keys={1.0: key}, neighbors_key="demo_neighbors")

    monkeypatch.setattr(ev, "run_standard_clustering", fake_clustering)
    def fake_plot(df, p, **kwargs):
        path = Path(p); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fake-png"); return path
    monkeypatch.setattr(ev, "plot_rare_metric_heatmap", fake_plot)
    monkeypatch.setattr(ev, "plot_precision_recall", fake_plot)
    monkeypatch.setattr(ev, "plot_failure_counts", fake_plot)
    def fake_scib(*args, output_dir, **kwargs):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "scib_status.json").write_text(json.dumps({"attempted": False, "success": False, "status": "disabled"}), encoding="utf-8")
        return None
    monkeypatch.setattr(ev, "run_scib_evaluation", fake_scib)
    scenarios = pd.DataFrame({
        "cell_type": ["A", "C"], "scenario": ["GR-DL", "LE-RM"],
        "distribution": ["GR", "LE"], "topology": ["DL", "RM"],
        "parent_type": ["", "B"], "curation_source": ["test", "test"],
    })
    config = EvaluationConfig(method_name="Demo", representation_key="X_demo", random_state=42, scib=ScibEvaluationConfig(enabled=False))
    result = ev.evaluate_latent(adata, config, tmp_path / "out", scenario_table=scenarios)
    assert result.rare_evaluation_status["status"] == "available"
    assert set(result.rare_metrics["cell_type"]) == {"A", "C"}
    payload = json.loads((result.output_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["benchmark"]["n_cells"] == 12
    assert payload["benchmark"]["benchmark_seed"] == 42
    assert (result.output_dir / "subset_metric_ratios.csv").exists()
    assert (result.output_dir / "rare_cell" / "rare_resolution_sensitivity.csv").exists()
    assert "knn_local_recovery" in set(result.rare_metrics.columns)
    assert "knn_local_recovery_adjusted" in set(result.rare_metrics.columns)
    rare_summary = result.rare_summary.set_index("metric")
    assert "resolution_limited_fraction" in rare_summary.index
    scenario_export = result.scenario_metrics.set_index("scenario")
    assert "v2_resolution_limited_count" in scenario_export.columns
    assert "v2_resolution_limited_fraction" in scenario_export.columns
    assert "knn_max_achievable_fraction" in set(result.rare_metrics.columns)
    assert "best_cluster_f1" in set(result.rare_metrics.columns)
    assert "ASW_selected_cells_in_full_latent" in set(result.subset_metrics.columns)
    assert payload["schema_version"] == "1.6"
    assert payload["rare"]["methodology"]["primary_local_recovery_metric"] == "knn_local_recovery_adjusted"
    assert payload["metric_registry"]["knn_local_recovery_adjusted"]["direction"] == "maximize"
    assert payload["metric_registry"]["knn_local_recovery"]["direction"] == "context"
    assert payload["metric_registry"]["preserved_fraction"]["direction"] == "context"
    assert (result.output_dir / "reproducibility" / "metric_registry.json").exists()


def test_release_metadata_is_synchronized():
    root = Path(__file__).parents[1]
    assert "Current release: `0.10.6`" in (root / "README.md").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 0.10.6" in citation
    assert "# scRareBench 0.10.6" in (root / "DESIGN_NOTES.md").read_text(encoding="utf-8")
    for notebook in (root / "notebooks").glob("*.ipynb"):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        sources = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
        if "EXPECTED_SCRAREBENCH_VERSION" in sources:
            assert 'EXPECTED_SCRAREBENCH_VERSION = "0.10.6"' in sources
            assert 'EXPECTED_SCRAREBENCH_VERSION = "0.9.1"' not in sources
            assert 'EXPECTED_SCRAREBENCH_VERSION = "0.9.2"' not in sources
