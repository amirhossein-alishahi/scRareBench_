from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scrarebench.evaluation import _rare_summary
from scrarebench.failures import classify_failure_archetypes
from scrarebench.metrics import (
    global_metrics,
    knn_local_recovery_from_graph,
    majority_vote_predictions,
    per_type_metrics,
    subset_metrics,
)
from scrarebench.scenarios import infer_distribution_classes


def test_full_space_rare_asw_detects_absorption_while_historical_metric_is_retained():
    rng = np.random.default_rng(7)
    # Two rare types are far from one another, but each is embedded in a distinct
    # abundant population. Rare-only ASW therefore looks excellent while the new
    # full-space rare ASW correctly sees the abundant competitors.
    common_c = rng.normal([0.0, 0.0], 0.10, size=(120, 2))
    rare_a = rng.normal([0.0, 0.0], 0.10, size=(20, 2))
    common_d = rng.normal([5.0, 5.0], 0.10, size=(120, 2))
    rare_b = rng.normal([5.0, 5.0], 0.10, size=(20, 2))
    X = np.vstack([common_c, rare_a, common_d, rare_b])
    true = np.array(["C"] * 120 + ["A"] * 20 + ["D"] * 120 + ["B"] * 20)
    clusters = np.array(["0"] * 140 + ["1"] * 140)
    pred, _ = majority_vote_predictions(true, clusters)
    frame = subset_metrics(X, true, clusters, pred, ["A", "B"], random_state=11).set_index("subset")
    assert frame.loc["rare", "ASW_true_on_latent"] > 0.8
    assert frame.loc["rare", "ASW_selected_cells_in_full_latent"] < 0.25


def test_knn_local_recovery_survives_majority_vote_ceiling():
    true = np.array(["Rare"] * 3 + ["Common"] * 9)
    clusters = np.array(["0"] * 12)
    pred, _ = majority_vote_predictions(true, clusters)
    legacy = per_type_metrics(true, clusters, pred).set_index("cell_type")
    assert legacy.loc["Rare", "f1"] == 0.0
    # The additive best-cluster diagnostic remains informative even when Rare can
    # never win majority ownership of the shared cluster.
    assert legacy.loc["Rare", "best_cluster_recall"] == pytest.approx(1.0)
    assert legacy.loc["Rare", "best_cluster_f1"] > 0.0

    rows, cols = [], []
    # Rare cells are locally perfectly coherent despite sharing the global cluster.
    for i in range(3):
        for j in range(3):
            if i != j:
                rows.append(i); cols.append(j)
    for i in range(3, 12):
        for j in range(3, 12):
            if i != j:
                rows.append(i); cols.append(j)
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(12, 12))
    local, _ = knn_local_recovery_from_graph(true, graph)
    rare = local.set_index("cell_type").loc["Rare"]
    assert rare["knn_same_label_fraction"] == pytest.approx(1.0)
    assert rare["knn_local_recovery"] == pytest.approx(1.0)


def test_preserved_fraction_is_measured_zero_not_missing():
    rare = pd.DataFrame([
        {"precision": 0.0, "recall": 0.0, "f1": 0.0, "inverse_purity": 0.2, "within_type_batch_nmi": 0.8, "failure_archetype": "lineage_assimilation"},
        {"precision": 0.1, "recall": 0.2, "f1": 0.13, "inverse_purity": 0.3, "within_type_batch_nmi": 0.7, "failure_archetype": "batch_driven_fragmentation"},
    ])
    row = _rare_summary(rare).set_index("metric").loc["preserved_fraction"]
    assert row["mean"] == 0.0
    assert row["n_valid"] == 2


def test_abundant_batch_specific_population_is_not_inferred_rare():
    rows = []
    # ~30% of all cells, but restricted to one batch.
    rows += [{"batch": "b1", "label": "BigBatchSpecific"}] * 300
    rows += [{"batch": "b1", "label": "Common"}] * 200
    for b in ["b2", "b3", "b4"]:
        rows += [{"batch": b, "label": "Common"}] * 167
    table = infer_distribution_classes(
        pd.DataFrame(rows), batch_key="batch", label_key="label",
        global_abundance_threshold=0.05,
        batch_fraction_threshold=0.25,
        local_abundance_threshold=0.05,
    ).set_index("cell_type")
    assert table.loc["BigBatchSpecific", "global_abundance"] > 0.05
    assert table.loc["BigBatchSpecific", "distribution"] == "COMMON_OR_UNASSIGNED"


def test_failure_taxonomy_exposes_overlapping_matches_with_precedence():
    frame = pd.DataFrame([{
        "cell_type": "overlap",
        "precision": 0.2,
        "recall": 0.4,
        "inverse_purity": 0.3,
        "within_type_batch_nmi": 0.8,
        "n_clusters_found_in": 3,
        "dominant_wrong_fraction": 0.8,
    }])
    row = classify_failure_archetypes(frame).iloc[0]
    assert row["failure_archetype"] == "batch_driven_fragmentation"
    assert row["failure_matched_archetypes"].split(";") == [
        "batch_driven_fragmentation", "lineage_assimilation"
    ]
    assert row["failure_match_count"] == 2


def test_global_metrics_propagates_requested_seed(monkeypatch):
    import scrarebench.metrics as metrics
    seen = {}
    def fake_asw(X, labels, *, max_cells=10_000, random_state=42):
        seen["seed"] = random_state
        return 0.123
    monkeypatch.setattr(metrics, "safe_silhouette", fake_asw)
    X = np.arange(18, dtype=float).reshape(6, 3)
    true = np.array(["A", "A", "A", "B", "B", "B"])
    clusters = np.array(["0", "0", "0", "1", "1", "1"])
    pred = true.copy()
    out = global_metrics(X, true, clusters, pred, random_state=917)
    assert out["ASW_true_on_latent"] == 0.123
    assert seen["seed"] == 917


def test_standard_clustering_records_exact_leiden_contract_without_fallback(monkeypatch):
    import scrarebench.clustering as clustering
    from types import SimpleNamespace

    calls = []
    class FakePP:
        @staticmethod
        def neighbors(adata, **kwargs):
            adata.uns[kwargs["key_added"]] = {"params": kwargs.copy()}
    class FakeTL:
        @staticmethod
        def leiden(adata, **kwargs):
            calls.append(kwargs.copy())
            adata.obs[kwargs["key_added"]] = pd.Categorical(["0", "0", "1", "1"])
    monkeypatch.setattr(clustering, "_require_scanpy", lambda: SimpleNamespace(pp=FakePP(), tl=FakeTL()))
    adata = SimpleNamespace(
        obsm={"X_demo": np.arange(12, dtype=float).reshape(4, 3)},
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        uns={},
    )
    result = clustering.run_standard_clustering(
        adata,
        representation_key="X_demo",
        method_name="Demo",
        resolutions=(0.5, 1.0),
        random_state=77,
        leiden_flavor="igraph",
        leiden_n_iterations=2,
    )
    assert result.leiden_flavor == "igraph"
    assert result.leiden_n_iterations == 2
    assert [c["flavor"] for c in calls] == ["igraph", "igraph"]
    assert [c["n_iterations"] for c in calls] == [2, 2]
    assert [c["random_state"] for c in calls] == [77, 77]


def test_dashboard_coordinate_generation_does_not_mutate_caller(monkeypatch):
    import copy
    import sys
    from types import SimpleNamespace
    import scrarebench.dashboard as dashboard

    class TinyAdata:
        def __init__(self):
            self.obsm = {"X_demo": np.arange(15, dtype=float).reshape(5, 3)}
            self.uns = {}
            self.obsp = {}
        def copy(self):
            out = TinyAdata()
            out.obsm = {k: np.array(v, copy=True) for k, v in self.obsm.items()}
            out.uns = copy.deepcopy(self.uns)
            out.obsp = copy.deepcopy(self.obsp)
            return out

    seen = {}
    def neighbors(adata, **kwargs):
        seen["neighbors_seed"] = kwargs.get("random_state")
        adata.uns[kwargs["key_added"]] = {"generated": True}
    def umap(adata, **kwargs):
        seen["umap_seed"] = kwargs.get("random_state")
        adata.obsm["X_umap"] = np.column_stack([np.arange(5), np.arange(5)[::-1]])

    fake_scanpy = SimpleNamespace(pp=SimpleNamespace(neighbors=neighbors), tl=SimpleNamespace(umap=umap))
    class FakeAnnData:
        def __init__(self, obs):
            self.obs = obs
            self.obs_names = obs.index
            self.obsm = {}
            self.uns = {}
            self.obsp = {}
    fake_anndata = SimpleNamespace(AnnData=FakeAnnData)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    monkeypatch.setitem(sys.modules, "anndata", fake_anndata)
    adata = TinyAdata()
    original_keys = set(adata.obsm)
    coords, key, kind = dashboard._coordinates(adata, "X_demo", umap_key=None, random_state=123)
    assert coords.shape == (5, 2)
    assert key == "X_umap_scrarebench_interactive"
    assert kind == "UMAP"
    assert set(adata.obsm) == original_keys
    assert adata.uns == {}
    assert adata.obsp == {}
    assert seen == {"neighbors_seed": 123, "umap_seed": 123}
