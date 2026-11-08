"""Regression guard for the November 2026 biological topology review.

These tests deliberately keep original cell-type labels and distribution classes
unchanged. Only the three registered DL/RM scenario assignments are reviewed.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scrarebench.evaluation import _scenario_summary
from scrarebench.scenarios import (
    SIX_SCENARIOS,
    annotate_registered_scenarios,
    load_registered_scenario_table,
    scenario_table_from_adata,
)


EXPECTED = {
    "wu_breast_cancer_atlas": {
        "B cells Naive": "LE-RM",
        "CAFs MSC iCAF-like s2": "GR-RM",
        "CAFs Transitioning s3": "GR-RM",
        "CAFs myCAF like s4": "GR-RM",
        "Cycling PVL": "SR-RM",
        "Cycling_Myeloid": "GR-RM",
        "Endothelial Lymphatic LYVE1": "GR-DL",
        "Myeloid_c0_DC_LAMP3": "SR-RM",
        "Myeloid_c11_cDC2_CD1C": "GR-DL",
        "Myeloid_c3_cDC1_CLEC9A": "SR-DL",
        "Myeloid_c4_DCs_pDC_IRF7": "GR-DL",
        "Myeloid_c5_Macrophage_3_SIGLEC1": "SR-RM",
        "Myeloid_c8_Monocyte_2_S100A9": "GR-RM",
        "Myeloid_c9_Macrophage_2_CXCL10": "GR-RM",
        "Myoepithelial": "LE-DL",
        "PVL_Immature s2": "GR-RM",
        "T_cells_c5_CD8+_GZMK": "LE-RM",
    },
    "covid19_autoimmunity_pbmc": {
        "B cell": "GR-RM",
        "conventional dendritic cell": "GR-DL",
        "gamma-delta T cell": "GR-DL",
        "hematopoietic stem cell": "GR-DL",
        "immature B cell": "GR-RM",
        "memory B cell": "GR-RM",
        "mucosal-associated invariant T cell": "GR-DL",
        "natural killer cell": "GR-RM",
        "neutrophil": "LE-DL",
        "plasma cell": "GR-RM",
        "plasmacytoid dendritic cell": "SR-DL",
        "regulatory T cell": "GR-RM",
    },
}

EXPECTED_COUNTS = {
    "wu_breast_cancer_atlas": {
        "GR-DL": 3, "GR-RM": 7, "LE-DL": 1,
        "LE-RM": 2, "SR-DL": 1, "SR-RM": 3,
    },
    "covid19_autoimmunity_pbmc": {
        "GR-DL": 4, "GR-RM": 6, "LE-DL": 1,
        "LE-RM": 0, "SR-DL": 1, "SR-RM": 0,
    },
}

REVIEWED = {
    "wu_breast_cancer_atlas": ("Myeloid_c4_DCs_pDC_IRF7",),
    "covid19_autoimmunity_pbmc": ("B cell", "natural killer cell"),
}


def test_registered_labels_and_all_scenario_assignments_are_frozen():
    for dataset_key, expected in EXPECTED.items():
        table = load_registered_scenario_table(dataset_key, include_unassigned=True)
        assert len(table) == len(expected)
        assert not table["cell_type"].duplicated().any()
        assert table.set_index("cell_type")["scenario"].to_dict() == expected
        assert table["include_in_six_state"].astype(bool).all()
        assert table["distribution"].astype(str).tolist() == [
            scenario.split("-")[0] for scenario in table["scenario"]
        ]
        assert table["topology"].astype(str).tolist() == [
            scenario.split("-")[1] for scenario in table["scenario"]
        ]
        assert {
            scenario: int((table["scenario"] == scenario).sum())
            for scenario in SIX_SCENARIOS
        } == EXPECTED_COUNTS[dataset_key]
        assert set(table["scenario"]).issubset(SIX_SCENARIOS)

        by_type = table.set_index("cell_type")
        for cell_type in REVIEWED[dataset_key]:
            assert by_type.loc[cell_type, "curation_source"] == (
                "author_annotation_marker_review_2026_11"
            )
            assert bool(by_type.loc[cell_type, "low_confidence"])
            assert bool(by_type.loc[cell_type, "threshold_sensitive"])


def test_registered_scenario_annotation_never_mutates_expression_or_truth():
    for dataset_key, label_key in (
        ("wu_breast_cancer_atlas", "celltype_subset"),
        ("covid19_autoimmunity_pbmc", "cell_type"),
    ):
        table = load_registered_scenario_table(dataset_key)
        labels = table["cell_type"].tolist() + ["common cell"]
        n = len(labels)
        obs = pd.DataFrame(
            {label_key: labels, "donor_id": ["d1", "d2"] * (n // 2) + (["d1"] if n % 2 else [])},
            index=[f"cell_{i}" for i in range(n)],
        )
        x = np.arange(n * 3, dtype=float).reshape(n, 3)
        adata = ad.AnnData(
            X=x.copy(),
            obs=obs,
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        adata.layers["counts"] = x.copy()
        adata.obsm["X_method"] = np.arange(n * 2, dtype=float).reshape(n, 2)
        before = {
            "X": adata.X.copy(),
            "counts": adata.layers["counts"].copy(),
            "method": adata.obsm["X_method"].copy(),
            "labels": adata.obs[label_key].tolist(),
            "batch": adata.obs["donor_id"].tolist(),
        }

        returned = annotate_registered_scenarios(
            adata, dataset_key=dataset_key, strict_labels=True
        )
        assert returned is adata
        np.testing.assert_array_equal(adata.X, before["X"])
        np.testing.assert_array_equal(adata.layers["counts"], before["counts"])
        np.testing.assert_array_equal(adata.obsm["X_method"], before["method"])
        assert adata.obs[label_key].tolist() == before["labels"]
        assert adata.obs["donor_id"].tolist() == before["batch"]

        expected_scenarios = [EXPECTED[dataset_key][label] for label in labels[:-1]]
        assert adata.obs["scrarebench_scenario"].tolist() == expected_scenarios + [""]
        assert adata.obs["scrarebench_topology"].tolist() == [
            s.split("-")[1] for s in expected_scenarios
        ] + [""]
        embedded = scenario_table_from_adata(adata)
        assert embedded is not None
        assert embedded.set_index("cell_type")["scenario"].to_dict() == EXPECTED[dataset_key]
        assert int(adata.obs["scrarebench_is_six_state"].sum()) == len(table)
        assert not bool(adata.obs["scrarebench_is_six_state"].iloc[-1])


def test_scenario_summary_reaggregates_under_reviewed_assignments():
    for dataset_key in EXPECTED:
        table = load_registered_scenario_table(dataset_key)
        rare = table[["cell_type", "scenario", "distribution", "topology"]].copy()
        rare["best_cluster_f1"] = np.arange(1, len(rare) + 1) / 100.0
        rare["knn_local_recovery_adjusted"] = (
            np.arange(1, len(rare) + 1) / 110.0
        )
        summary = _scenario_summary(rare).set_index("scenario")
        assert set(summary.index) == set(SIX_SCENARIOS)
        for scenario in SIX_SCENARIOS:
            group = rare[rare["scenario"] == scenario]
            assert int(summary.loc[scenario, "n_cell_types"]) == len(group)
            assert bool(summary.loc[scenario, "is_empty"]) == group.empty
            if not group.empty:
                assert np.isclose(
                    summary.loc[scenario, "best_cluster_f1_mean"],
                    group["best_cluster_f1"].mean(),
                )
                assert np.isclose(
                    summary.loc[scenario, "knn_local_recovery_adjusted_mean"],
                    group["knn_local_recovery_adjusted"].mean(),
                )
            else:
                assert np.isnan(summary.loc[scenario, "best_cluster_f1_mean"])
