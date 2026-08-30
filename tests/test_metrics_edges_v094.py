from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scrarebench.metrics import (
    full_space_silhouette_group_means,
    knn_local_recovery_from_graph,
    safe_silhouette,
    selected_cells_silhouette_in_full_space,
    subset_metric_ratios,
)


def test_selected_cells_full_space_validates_alignment_and_empty_selection():
    X = np.zeros((4, 2), dtype=float)
    labels = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError):
        selected_cells_silhouette_in_full_space(X[:3], labels, [True] * 4)
    assert np.isnan(selected_cells_silhouette_in_full_space(X, labels, [False] * 4))


def test_selected_cells_full_space_forces_small_group_into_bounded_sample():
    rng = np.random.default_rng(12)
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["Rare"] * 3)
    X = np.vstack([
        rng.normal([0, 0], 0.2, size=(40, 2)),
        rng.normal([4, 4], 0.2, size=(40, 2)),
        rng.normal([8, 0], 0.1, size=(3, 2)),
    ])
    mask = labels == "Rare"
    value = selected_cells_silhouette_in_full_space(
        X, labels, mask, max_cells=20, random_state=5, min_selected_sample=3
    )
    assert np.isfinite(value)
    assert value > 0.5


def test_full_space_group_means_handles_empty_and_bounded_sampling():
    rng = np.random.default_rng(1)
    labels = np.array(["A"] * 30 + ["B"] * 30 + ["R"] * 4)
    X = np.vstack([
        rng.normal([0, 0], 0.2, size=(30, 2)),
        rng.normal([5, 5], 0.2, size=(30, 2)),
        rng.normal([9, 0], 0.1, size=(4, 2)),
    ])
    masks = {
        "rare": labels == "R",
        "non_rare": labels != "R",
        "empty": np.zeros(len(labels), dtype=bool),
    }
    out = full_space_silhouette_group_means(
        X, labels, masks, max_cells=24, random_state=9, min_group_sample=4
    )
    assert out["rare"] > 0.5
    assert np.isfinite(out["non_rare"])
    assert np.isnan(out["empty"])


def test_full_space_group_means_rejects_bad_shape_and_degenerate_labels():
    with pytest.raises(ValueError):
        full_space_silhouette_group_means(
            np.zeros((3, 2)), ["A"] * 4, {"all": np.ones(4, dtype=bool)}
        )
    out = full_space_silhouette_group_means(
        np.zeros((4, 2)), ["A"] * 4, {"all": np.ones(4, dtype=bool)}
    )
    assert np.isnan(out["all"])


def test_safe_silhouette_returns_nan_for_invalid_label_geometry():
    X = np.zeros((3, 2))
    assert np.isnan(safe_silhouette(X, ["A", "A", "A"]))
    assert np.isnan(safe_silhouette(X, ["A", "B", "C"]))


def test_knn_local_recovery_validates_graph_and_handles_isolated_cells():
    labels = np.array(["A", "A", "B"])
    with pytest.raises(ValueError):
        knn_local_recovery_from_graph(labels, sparse.eye(2, format="csr"))
    with pytest.raises(TypeError):
        knn_local_recovery_from_graph(labels, np.eye(3))

    graph = sparse.csr_matrix(
        np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=float)
    )
    per_type, per_cell = knn_local_recovery_from_graph(labels, graph)
    row_a = per_type.set_index("cell_type").loc["A"]
    row_b = per_type.set_index("cell_type").loc["B"]
    assert row_a["knn_same_label_fraction"] == pytest.approx(1.0)
    assert row_a["knn_valid_cells"] == 2
    assert np.isnan(row_b["knn_same_label_fraction"])
    assert row_b["knn_valid_cells"] == 0
    assert np.isnan(per_cell[2])


def test_subset_ratio_statuses_distinguish_missing_zero_and_nonfinite():
    empty = subset_metric_ratios(pd.DataFrame())
    assert list(empty.columns) == [
        "metric", "numerator_subset", "denominator_subset", "numerator",
        "denominator", "ratio", "status"
    ]

    frame = pd.DataFrame([
        {"subset": "rare", "stable": 0.5, "near_zero": 0.0, "bad": np.nan},
        {"subset": "non_rare", "stable": 0.75, "near_zero": 0.6, "bad": 0.8},
    ])
    rows = subset_metric_ratios(frame).set_index("metric")
    assert rows.loc["stable", "status"] == "computed"
    assert rows.loc["stable", "ratio"] == pytest.approx(1.5)
    assert rows.loc["near_zero", "status"] == "unstable_denominator"
    assert np.isnan(rows.loc["near_zero", "ratio"])
    assert rows.loc["bad", "status"] == "non_finite_input"


def test_subset_ratio_requires_both_named_subsets():
    frame = pd.DataFrame([{"subset": "overall", "metric": 0.7}])
    assert subset_metric_ratios(frame).empty
