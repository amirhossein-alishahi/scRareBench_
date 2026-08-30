from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scrarebench.evaluation import EvaluationConfig, _resolve_rare_metadata
from scrarebench.metric_registry import metric_direction
from scrarebench.metrics import knn_local_recovery_from_graph


def _fixed_degree_graph(labels: np.ndarray, *, degree: int = 15, absorbed: bool = False) -> sparse.csr_matrix:
    """Build a directed graph with exactly ``degree`` neighbors per cell.

    Rare cells get every available same-label peer first when ``absorbed=False``;
    remaining slots are filled from other labels.  This makes a perfectly isolated
    finite population attain its *mathematical* same-label ceiling without changing
    the graph degree.  With ``absorbed=True`` rare cells receive no same-label peers.
    """
    labels = np.asarray(labels).astype(str)
    n = len(labels)
    if degree >= n:
        raise ValueError("degree must be smaller than n")
    rows: list[int] = []
    cols: list[int] = []
    for i, label in enumerate(labels):
        same = [j for j in range(n) if j != i and labels[j] == label]
        other = [j for j in range(n) if labels[j] != label]
        if absorbed and label.startswith("Rare"):
            chosen = other[:degree]
        else:
            chosen = (same + other)[:degree]
        # Common cells may not have enough in the first list; fill deterministically.
        if len(chosen) < degree:
            remaining = [j for j in range(n) if j != i and j not in chosen]
            chosen.extend(remaining[: degree - len(chosen)])
        assert len(chosen) == degree
        rows.extend([i] * degree)
        cols.extend(chosen)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


@pytest.mark.parametrize("support", [4, 8, 12, 16, 200])
def test_support_adjusted_knn_reaches_one_for_perfectly_recovered_populations(support: int):
    # Keep a large background so the abundance-null expectation remains small and
    # the same test covers both support < k and support > k.
    n_common = max(240, support + 40)
    labels = np.array(["RareA"] * support + ["Common"] * n_common)
    graph = _fixed_degree_graph(labels, degree=15, absorbed=False)
    frame, _ = knn_local_recovery_from_graph(labels, graph)
    row = frame.set_index("cell_type").loc["RareA"]

    expected_ceiling = min(support - 1, 15) / 15
    assert row["knn_same_label_fraction"] == pytest.approx(expected_ceiling)
    assert row["knn_max_achievable_fraction"] == pytest.approx(expected_ceiling)
    assert row["knn_local_recovery_adjusted"] == pytest.approx(1.0, abs=1e-12)
    if support < 16:
        assert row["knn_local_recovery"] < 1.0
    else:
        assert row["knn_local_recovery"] == pytest.approx(1.0)


def test_support_adjusted_knn_is_near_null_for_complete_absorption():
    labels = np.array(["RareA"] * 8 + ["Common"] * 300)
    graph = _fixed_degree_graph(labels, degree=15, absorbed=True)
    frame, _ = knn_local_recovery_from_graph(labels, graph)
    row = frame.set_index("cell_type").loc["RareA"]
    assert row["knn_same_label_fraction"] == 0.0
    # Below-null is allowed to be slightly negative; importantly it is nowhere near
    # the perfect-recovery endpoint.
    assert -0.1 < row["knn_local_recovery_adjusted"] <= 0.0


def test_support_adjusted_knn_singleton_is_explicitly_not_assessable():
    labels = np.array(["RareSingleton"] + ["Common"] * 40)
    graph = _fixed_degree_graph(labels, degree=15, absorbed=False)
    frame, _ = knn_local_recovery_from_graph(labels, graph)
    row = frame.set_index("cell_type").loc["RareSingleton"]
    assert row["knn_same_label_fraction"] == 0.0
    assert row["knn_max_achievable_fraction"] == 0.0
    assert np.isnan(row["knn_local_recovery_adjusted"])


def test_metric_registry_prevents_context_diagnostics_from_being_ranked():
    assert metric_direction("knn_local_recovery_adjusted") == "maximize"
    for metric in (
        "knn_local_recovery",
        "knn_max_achievable_fraction",
        "knn_mean_neighbors",
        "knn_valid_cells",
        "failure_match_count",
        "preserved_fraction",
    ):
        assert metric_direction(metric) == "context"


def test_strict_scenario_label_drift_fails_closed_and_exploratory_mode_records_it():
    adata = SimpleNamespace(
        obs=pd.DataFrame({"celltype": ["A", "A", "B"], "BATCH": ["x", "y", "x"]}),
        uns={},
    )
    scenarios = pd.DataFrame(
        {
            "cell_type": ["A", "MissingRare"],
            "scenario": ["GR-DL", "SR-RM"],
            "distribution": ["GR", "SR"],
            "topology": ["DL", "RM"],
        }
    )
    strict = EvaluationConfig(method_name="Demo", representation_key="X", strict_scenario_labels=True)
    with pytest.raises(ValueError, match="annotation/source drift"):
        _resolve_rare_metadata(adata, strict, scenarios)

    exploratory = EvaluationConfig(method_name="Demo", representation_key="X", strict_scenario_labels=False)
    metadata, status = _resolve_rare_metadata(adata, exploratory, scenarios)
    assert metadata is not None
    assert status["annotation_drift_warning"] is True
    assert status["scenario_cell_types_absent_from_data"] == ["MissingRare"]
    assert status["strict_scenario_labels"] is False


def test_support_adjusted_knn_handles_variable_realized_degree_without_support_bias():
    labels = np.array(["RareA"] * 8 + ["Common"] * 80)
    rows: list[int] = []
    cols: list[int] = []
    # Rare cells alternate degree 7 and 15.  In both cases they include every
    # available same-label peer, so each cell reaches its own achievable maximum.
    for i in range(8):
        same = [j for j in range(8) if j != i]
        other = list(range(8, len(labels)))
        degree = 7 if i % 2 == 0 else 15
        chosen = (same + other)[:degree]
        rows.extend([i] * len(chosen)); cols.extend(chosen)
    # Common rows only need valid graph neighborhoods.
    for i in range(8, len(labels)):
        peers = [j for j in range(8, len(labels)) if j != i][:15]
        rows.extend([i] * len(peers)); cols.extend(peers)
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(labels), len(labels)))
    frame, _ = knn_local_recovery_from_graph(labels, graph)
    rare = frame.set_index("cell_type").loc["RareA"]
    assert rare["knn_mean_neighbors"] == pytest.approx(11.0)
    assert rare["knn_local_recovery_adjusted"] == pytest.approx(1.0, abs=1e-12)


def test_dashboard_umap_failure_is_visibly_labeled_as_latent_projection(monkeypatch):
    import sys
    from types import SimpleNamespace
    import scrarebench.dashboard as dashboard

    adata = SimpleNamespace(obsm={"X_demo": np.arange(18, dtype=float).reshape(6, 3)})
    # Force the UMAP construction path to fail before any caller mutation.
    monkeypatch.setitem(sys.modules, "scanpy", None)
    coords, key, kind = dashboard._coordinates(adata, "X_demo", umap_key=None, random_state=42)
    assert coords.shape == (6, 2)
    assert key == "X_demo"
    assert "UMAP unavailable" in kind
