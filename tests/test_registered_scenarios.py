from __future__ import annotations

import pandas as pd

from scrarebench.dashboard import _rare_category_breakdown
from scrarebench.scenarios import (
    SIX_SCENARIOS,
    annotate_registered_scenarios,
    load_registered_scenario_table,
    registered_scenario_info,
    scenario_table_from_adata,
)


class DummyAnnData:
    def __init__(self, obs: pd.DataFrame):
        self.obs = obs.copy()
        self.uns = {}

    def copy(self):
        clone = DummyAnnData(self.obs.copy())
        clone.uns = dict(self.uns)
        return clone


def test_registered_scenario_counts_and_coverage():
    expected = {
        "mbdrc_renal_cortex": (10, {"GR-DL", "GR-RM", "LE-DL", "LE-RM"}),
        "wu_breast_cancer_atlas": (17, set(SIX_SCENARIOS)),
        "covid19_autoimmunity_pbmc": (12, {"GR-DL", "GR-RM", "LE-DL", "SR-DL"}),
    }
    for key, (n_rows, coverage) in expected.items():
        table = load_registered_scenario_table(key)
        assert len(table) == n_rows
        assert set(table["scenario"]) == coverage
        assert set(table["scenario"]).issubset(SIX_SCENARIOS)


def test_mbdrc_tracks_ambiguous_lymphocyte_without_forcing_topology():
    table = load_registered_scenario_table("mbdrc_renal_cortex", include_unassigned=True)
    row = table.set_index("cell_type").loc["lymphocyte"]
    assert row["distribution"] == "GR"
    assert row["topology"] == "AMBIGUOUS"
    assert row["scenario"] == ""
    assert not bool(row["include_in_six_state"])
    info = registered_scenario_info("mbdrc_renal_cortex")
    assert info["unassigned_cell_types"] == ["lymphocyte"]


def test_annotation_embeds_dataset_specific_six_state_table():
    table = load_registered_scenario_table("covid19_autoimmunity_pbmc", include_unassigned=True)
    obs = pd.DataFrame(
        {
            "cell_type": table["cell_type"].tolist() + ["common population"],
        },
        index=[f"c{i}" for i in range(len(table) + 1)],
    )
    adata = DummyAnnData(obs)
    annotate_registered_scenarios(
        adata,
        dataset_key="covid19_autoimmunity_pbmc",
        strict_labels=True,
    )
    assert "scrarebench_scenario_table" in adata.uns
    embedded = scenario_table_from_adata(adata)
    assert embedded is not None
    assert len(embedded) == 12
    assert set(embedded["scenario"]) == {"GR-DL", "GR-RM", "LE-DL", "SR-DL"}
    assert int(adata.obs["scrarebench_is_six_state"].sum()) == 12
    assert int(adata.obs["scrarebench_is_rare"].sum()) == 12


def test_annotation_fails_loudly_on_source_label_revision():
    adata = DummyAnnData(pd.DataFrame({"cell_type": ["B cell"]}, index=["c0"]))
    try:
        annotate_registered_scenarios(
            adata,
            dataset_key="covid19_autoimmunity_pbmc",
            strict_labels=True,
        )
    except ValueError as exc:
        assert "source/annotation revision" in str(exc)
    else:
        raise AssertionError("Expected missing registered labels to fail loudly.")


def test_dashboard_keeps_all_six_scenario_slots_even_when_empty():
    rare = pd.DataFrame(
        [
            {
                "cell_type": "x",
                "scenario": "GR-DL",
                "distribution": "GR",
                "topology": "DL",
                "support": 12,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "inverse_purity": 0.9,
                "within_type_batch_nmi": 0.1,
                "failure_archetype": "preserved",
            }
        ]
    )
    rows = _rare_category_breakdown(rare)
    assert [row["scenario"] for row in rows] == list(SIX_SCENARIOS)
    assert len(rows) == 6
    empty = {row["scenario"]: row for row in rows}
    assert empty["GR-DL"]["n_cell_types"] == 1
    assert empty["SR-RM"]["n_cell_types"] == 0
    assert empty["SR-RM"]["is_empty"] is True
