from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scrarebench import (
    aggregate_dashboard_runs,
    configuration_hash,
    normalize_method_seeds,
    write_interactive_report,
    write_multiseed_interactive_report,
)
from scrarebench.dashboard import _dashboard_run_entry
from scrarebench.multiseed import (
    canonicalize_embedded_run,
    extract_embedded_report_payload,
    make_multirun_container,
    mean_sd,
    validate_compatible_run,
)


class MiniAdata:
    def __init__(self) -> None:
        self.obs = pd.DataFrame(
            {
                "celltype": ["A", "A", "B", "B", "C", "C"],
                "BATCH": ["x", "y", "x", "y", "x", "y"],
                "scrarebench_scenario": ["GR-DL", "GR-DL", "non_rare", "non_rare", "SR-RM", "SR-RM"],
                "cluster": pd.Categorical(["0", "0", "1", "1", "2", "2"]),
                "pred": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
            },
            index=[f"c{i}" for i in range(6)],
        )
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)
        self.var_names = pd.Index(["G1", "G2"])
        self.X = np.ones((6, 2), dtype=np.float32)
        self.layers = {"counts": self.X.copy()}
        self.obsm = {
            "X_demo": np.arange(18, dtype=np.float32).reshape(6, 3) / 10,
            "X_umap_demo": np.array([[0, 0], [0.1, 0], [1, 1], [1.1, 1], [2, 0], [2.1, 0]], dtype=float),
        }
        self.uns = {"scrarebench_dataset": {"dataset_key": "demo", "dataset_index": 0}}


class MiniResult:
    def __init__(self, root: Path) -> None:
        self.output_dir = root
        repro = root / "reproducibility"
        repro.mkdir(parents=True, exist_ok=True)
        (repro / "run_config.yaml").write_text(
            "method_name: Demo\nrepresentation_key: X_demo\nlabel_key: celltype\nbatch_key: BATCH\nbenchmark_seed: 42\nrandom_state: 42\n",
            encoding="utf-8",
        )
        self.files = {"run_config": repro / "run_config.yaml"}
        self.prediction_key = "pred"
        self.cluster_keys = {1.0: "cluster"}
        self.local_recovery_key = None
        self.local_recovery_adjusted_key = None
        self.subset_metrics = pd.DataFrame([
            {"subset": "overall", "n_cells": 6, "n_cell_types": 3, "F1_macro": 0.8},
            {"subset": "rare", "n_cells": 4, "n_cell_types": 2, "F1_macro": 0.6},
        ])
        self.subset_metric_ratios = pd.DataFrame()
        self.per_type_metrics = pd.DataFrame()
        self.rare_metrics = pd.DataFrame([
            {
                "cell_type": "A", "scenario": "GR-DL", "distribution": "GR", "topology": "DL",
                "support": 2, "knn_local_recovery_adjusted": 0.5, "knn_local_recovery": 0.3,
                "knn_max_achievable_fraction": 0.5, "best_cluster_f1": 0.8, "inverse_purity": 1.0,
                "precision": 0.8, "recall": 0.8, "f1": 0.8, "failure_archetype": "preserved",
                "failure_archetype_v2": "preserved",
            },
            {
                "cell_type": "C", "scenario": "SR-RM", "distribution": "SR", "topology": "RM",
                "support": 2, "knn_local_recovery_adjusted": 0.2, "knn_local_recovery": 0.1,
                "knn_max_achievable_fraction": 0.5, "best_cluster_f1": 0.4, "inverse_purity": 0.5,
                "precision": 0.3, "recall": 0.5, "f1": 0.375, "failure_archetype": "lineage_assimilation",
                "failure_archetype_v2": "resolution_limited",
            },
        ])
        self.rare_summary = pd.DataFrame([
            {"metric": "knn_local_recovery_adjusted", "mean": 0.35, "median": 0.35, "minimum": 0.2, "maximum": 0.5, "n_valid": 2},
            {"metric": "best_cluster_f1", "mean": 0.6, "median": 0.6, "minimum": 0.4, "maximum": 0.8, "n_valid": 2},
            {"metric": "resolution_limited_fraction", "mean": 0.5, "median": np.nan, "minimum": np.nan, "maximum": np.nan, "n_valid": 2},
        ])
        self.scenario_metrics = pd.DataFrame([
            {"scenario": "GR-DL", "distribution": "GR", "topology": "DL", "n_cell_types": 1, "knn_local_recovery_adjusted_mean": 0.5},
            {"scenario": "SR-RM", "distribution": "SR", "topology": "RM", "n_cell_types": 1, "knn_local_recovery_adjusted_mean": 0.2},
        ])
        self.resolution_rare_metrics = pd.DataFrame([
            {"resolution": 1.0, "cell_type": "A", "scenario": "GR-DL", "best_cluster_f1": 0.8},
            {"resolution": 1.0, "cell_type": "C", "scenario": "SR-RM", "best_cluster_f1": 0.4},
        ])
        self.rare_evaluation_status = {"status": "available"}
        self.scib_status = {"status": "disabled"}
        self.scib = None


def test_seed_normalization_and_duplicate_guard():
    assert normalize_method_seeds(42) == [42]
    assert normalize_method_seeds([42, 123, np.int64(2026)]) == [42, 123, 2026]
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_method_seeds([42, 42])
    with pytest.raises(TypeError):
        normalize_method_seeds("42,123")
    with pytest.raises(ValueError):
        normalize_method_seeds([-1])


def test_configuration_hash_is_seed_path_runtime_independent_but_parameter_sensitive():
    a = {"seed": 42, "model_dir": "/a", "training_seconds": 10, "n_latent": 30, "nested": {"method_seed": 42, "dropout": 0.1}}
    b = {"seed": 123, "model_dir": "/b", "training_seconds": 999, "n_latent": 30, "nested": {"method_seed": 123, "dropout": 0.1}}
    c = {**b, "n_latent": 50}
    assert configuration_hash(a) == configuration_hash(b)
    assert configuration_hash(a) != configuration_hash(c)


def test_mean_sd_single_seed_is_missing_not_zero():
    one = mean_sd([0.5])
    assert one == {"mean": 0.5, "sd": None, "min": 0.5, "max": 0.5, "n": 1, "values": [0.5]}
    two = mean_sd([0.5, 0.7])
    assert two["mean"] == pytest.approx(0.6)
    assert two["sd"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_aggregate_exclusion_keeps_data_but_changes_summary():
    runs = [
        {"run_id": "a", "method_seed": 42, "included": True, "payload": {"rare": {"summary": {"rows": [{"metric": "best_cluster_f1", "metric_type": "x", "mean": 0.4}]}}}},
        {"run_id": "b", "method_seed": 123, "included": False, "payload": {"rare": {"summary": {"rows": [{"metric": "best_cluster_f1", "metric_type": "x", "mean": 0.8}]}}}},
    ]
    agg = aggregate_dashboard_runs(runs)
    assert agg["n_stored"] == 2
    assert agg["n_included"] == 1
    assert agg["excluded_seeds"] == [123]
    assert agg["rare_summary"][0]["mean"] == pytest.approx(0.4)
    assert agg["rare_summary"][0]["mean__sd"] is None
    assert len(runs) == 2 and runs[1]["payload"]


def test_compatible_run_rejects_duplicate_seed_and_config_mismatch():
    base = {"run_id": "r1", "method_seed": 42, "method_name": "scVI", "dataset_fingerprint": "d", "configuration_hash": "c"}
    with pytest.raises(ValueError, match="Duplicate method seed"):
        validate_compatible_run([base], {**base, "run_id": "r2"})
    with pytest.raises(ValueError, match="configuration_hash"):
        validate_compatible_run([base], {**base, "run_id": "r2", "method_seed": 123, "configuration_hash": "other"})


def test_single_and_multiseed_report_roundtrip(tmp_path: Path):
    adata = MiniAdata()
    result = MiniResult(tmp_path / "r")
    reports = []
    for seed, delta in [(42, 0.0), (123, 0.02), (2026, -0.01)]:
        adata.obsm["X_demo"] = (np.arange(18, dtype=np.float32).reshape(6, 3) / 10) + delta
        path = tmp_path / f"seed_{seed}.html"
        write_interactive_report(
            adata, result, path, representation_key="X_demo", umap_key="X_umap_demo",
            method_seed=seed, method_config={"n_latent": 3, "dropout": 0.1, "seed": seed},
            expected_seeds=[42, 123, 2026], include_static_figures=False,
        )
        data = extract_embedded_report_payload(path)
        assert data["report_type"] == "scrarebench_multi_run"
        assert data["runs"][0]["method_seed"] == seed
        reports.append(path)
    merged = tmp_path / "multi.html"
    write_multiseed_interactive_report(reports, merged, expected_seeds=[42, 123, 2026])
    data = extract_embedded_report_payload(merged)
    assert [r["method_seed"] for r in data["runs"]] == [42, 123, 2026]
    assert data["aggregate"]["n_included"] == 3
    assert "Seed Stability" in merged.read_text(encoding="utf-8")
    assert "Runs & Seeds" in merged.read_text(encoding="utf-8")
    assert "Save Updated Report" in merged.read_text(encoding="utf-8")
    assert "Duplicate seeds are not allowed" in merged.read_text(encoding="utf-8")
    assert "UMAP coordinates are never averaged" in merged.read_text(encoding="utf-8")


def test_multiseed_merge_rejects_duplicate_seed(tmp_path: Path):
    adata = MiniAdata(); result = MiniResult(tmp_path / "r")
    p = tmp_path / "seed42.html"
    write_interactive_report(adata, result, p, representation_key="X_demo", umap_key="X_umap_demo", method_seed=42, method_config={"n_latent": 3})
    with pytest.raises(ValueError, match="Duplicate method seed"):
        write_multiseed_interactive_report([p, p], tmp_path / "bad.html")


def test_container_preserves_excluded_run_and_expected_seeds():
    runs = [
        {"run_id": "r1", "method_seed": 42, "included": True, "method_name": "M", "dataset_fingerprint": "D", "configuration_hash": "C", "payload": {}},
        {"run_id": "r2", "method_seed": 123, "included": False, "method_name": "M", "dataset_fingerprint": "D", "configuration_hash": "C", "payload": {"x": 1}},
    ]
    c = make_multirun_container(runs, expected_seeds=[42, 123, 2026])
    assert c["expected_seeds"] == [42, 123, 2026]
    assert c["aggregate"]["n_included"] == 1
    assert c["aggregate"]["excluded_seeds"] == [123]
    assert c["runs"][1]["payload"] == {"x": 1}


def test_dashboard_run_configuration_hash_includes_method_config(tmp_path: Path):
    adata = MiniAdata(); result = MiniResult(tmp_path / "r")
    payload = {"meta": {}}
    a = _dashboard_run_entry(adata, result, payload=payload.copy(), representation_key="X_demo", method_seed=42, method_config={"n_latent": 30})
    b = _dashboard_run_entry(adata, result, payload={"meta": {}}, representation_key="X_demo", method_seed=123, method_config={"n_latent": 30})
    c = _dashboard_run_entry(adata, result, payload={"meta": {}}, representation_key="X_demo", method_seed=123, method_config={"n_latent": 50})
    assert a["configuration_hash"] == b["configuration_hash"]
    assert a["configuration_hash"] != c["configuration_hash"]


def test_multiseed_dashboard_has_distinct_umap_hosts_and_portable_save_sanitizer():
    root = Path(__file__).parents[1]
    js = (root / "src" / "scrarebench" / "assets" / "dashboard.js").read_text(encoding="utf-8")
    assert 'id="seedUmapMultiples"' in js
    assert 'id="seedStabilityUmapMultiples"' in js
    assert "renderSeedUmapMultiples('seedStabilityUmapMultiples')" in js
    assert "function portableReport()" in js
    assert "delete pts.x" in js
    assert "delete pts.y" in js
    assert "delete pts._decoded" in js
    assert "recordEdit('add_run'" in js
    assert "recordEdit(c.checked?'include_run':'exclude_run'" in js


@pytest.mark.skip(reason="Standalone Comparator v9 is not shipped in the core developer repository")
def test_comparator_v8_preserves_first_class_seed_and_context_metric_guard():
    root = Path(__file__).parents[1]
    html = (root / "comparator" / "scRareBench_Multi_Report_Comparator_v9.html").read_text(encoding="utf-8")
    assert "Multi-Report Comparator v9" in html
    assert "method_seed:r.methodSeed??r.method_seed??null" in html
    assert "Show individual seeds" in html
    assert "Aggregate included seeds" in html
    assert "Not automatically rankable." in html
    assert '"resolution_limited_fraction":"context"' in html


def test_configuration_hash_ignores_realized_evaluation_outcomes_but_not_requested_settings():
    base = {
        "benchmark": {
            "method_name": "scVI",
            "n_neighbors": 15,
            "random_state": 42,
            "method_seed": 42,
            "method_config": {"seed": 42, "n_latent": 30},
            "knn_graph": {
                "source": "scanpy_neighbors_graph",
                "requested_n_neighbors": 15,
                "realized_degree_mean": 15.0,
                "realized_degree_min": 15,
                "realized_degree_max": 15,
            },
            "reference_n_clusters": 22,
            "n_reference_cell_types": 45,
            "cluster_count_warning": True,
        },
        "method": {"seed": 42, "n_latent": 30},
    }
    other = json.loads(json.dumps(base))
    other["benchmark"]["method_seed"] = 123
    other["benchmark"]["method_config"]["seed"] = 123
    other["method"]["seed"] = 123
    other["benchmark"]["reference_n_clusters"] = 27
    other["benchmark"]["cluster_count_warning"] = False
    other["benchmark"]["knn_graph"]["realized_degree_mean"] = 14.75
    other["benchmark"]["knn_graph"]["realized_degree_min"] = 14
    assert configuration_hash(base) == configuration_hash(other)

    requested_change = json.loads(json.dumps(other))
    requested_change["benchmark"]["n_neighbors"] = 30
    requested_change["benchmark"]["knn_graph"]["requested_n_neighbors"] = 30
    assert configuration_hash(base) != configuration_hash(requested_change)


def test_canonicalize_embedded_run_repairs_v0100_stale_hash_and_generated_run_id():
    run_config_22 = """method_name: scVI\nn_neighbors: 15\nrandom_state: 42\nmethod_seed: 42\nmethod_config:\n  seed: 42\n  n_latent: 30\nreference_n_clusters: 22\ncluster_count_warning: true\nknn_graph:\n  source: scanpy_neighbors_graph\n  requested_n_neighbors: 15\n  realized_degree_mean: 15.0\n  realized_degree_min: 15\n  realized_degree_max: 15\n"""
    method_cfg = {"seed": 42, "n_latent": 30}
    # Simulate the stale 0.10.0 identity by hashing without the 0.10.1 outcome stripping.
    stale_hash = "stale-v0100-hash"
    base = {
        "run_id": make_multirun_container.__name__,  # custom ID must be preserved
        "method_seed": 42,
        "method_name": "scVI",
        "dataset_fingerprint": "dataset-fp",
        "configuration_hash": stale_hash,
        "method_configuration": method_cfg,
        "latent_sha256": "latent-a",
        "payload": {"meta": {"configuration_hash": stale_hash, "method_configuration": method_cfg}, "reproducibility": {"run_config": run_config_22}},
    }
    repaired = canonicalize_embedded_run(base)
    assert repaired["configuration_hash"] != stale_hash
    assert repaired["payload"]["meta"]["configuration_hash"] == repaired["configuration_hash"]
    assert repaired["run_id"] == base["run_id"]  # explicit/custom ID is not rewritten


def test_multiseed_merge_repairs_stale_v0100_outcome_hashes(tmp_path: Path):
    adata = MiniAdata()
    result = MiniResult(tmp_path / "r")
    reports = []
    for seed, n_clusters in [(42, 22), (123, 27)]:
        cfg_path = result.files["run_config"]
        cfg_path.write_text(
            "method_name: Demo\nrepresentation_key: X_demo\nlabel_key: celltype\nbatch_key: BATCH\n"
            f"benchmark_seed: 42\nrandom_state: 42\nmethod_seed: {seed}\n"
            f"method_config:\n  seed: {seed}\n  n_latent: 3\n"
            f"reference_n_clusters: {n_clusters}\ncluster_count_warning: true\n"
            "knn_graph:\n  source: scanpy_neighbors_graph\n  requested_n_neighbors: 15\n"
            f"  realized_degree_mean: {15.0 if seed == 42 else 14.75}\n  realized_degree_min: 14\n  realized_degree_max: 15\n",
            encoding="utf-8",
        )
        path = tmp_path / f"seed_{seed}.html"
        write_interactive_report(
            adata, result, path, representation_key="X_demo", umap_key="X_umap_demo",
            method_seed=seed, method_config={"seed": seed, "n_latent": 3},
            expected_seeds=[42, 123], include_static_figures=False,
        )
        container = extract_embedded_report_payload(path)
        # Force the exact failure mode of 0.10.0: each embedded run carries a
        # different stale hash despite identical requested configuration.
        run = container["runs"][0]
        stale = f"stale-{n_clusters}"
        run["configuration_hash"] = stale
        run["payload"]["meta"]["configuration_hash"] = stale
        reports.append(container)

    merged = tmp_path / "merged.html"
    write_multiseed_interactive_report(reports, merged, expected_seeds=[42, 123])
    data = extract_embedded_report_payload(merged)
    assert len(data["runs"]) == 2
    assert data["runs"][0]["configuration_hash"] == data["runs"][1]["configuration_hash"]
    assert [r["method_seed"] for r in data["runs"]] == [42, 123]
