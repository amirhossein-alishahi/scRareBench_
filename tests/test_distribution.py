import pandas as pd

from scrarebench.scenarios import infer_distribution_classes


def test_distribution_only_inference():
    rows = []
    for batch in ["b1", "b2", "b3", "b4"]:
        rows += [{"batch": batch, "label": "GR"}] * 1
        rows += [{"batch": batch, "label": "Common"}] * 99

    # LE and SR must both satisfy the global-rarity gate.  Six LE cells are
    # globally <2% while still reaching >5% abundance in the one batch where
    # they occur; one SR cell stays below both thresholds.
    rows += [{"batch": "b1", "label": "LE"}] * 6
    rows += [{"batch": "b1", "label": "SR"}] * 1

    table = infer_distribution_classes(
        pd.DataFrame(rows),
        batch_key="batch",
        label_key="label",
        global_abundance_threshold=0.02,
        batch_fraction_threshold=0.25,
        local_abundance_threshold=0.05,
    ).set_index("cell_type")
    assert table.loc["GR", "distribution"] == "GR"
    assert table.loc["LE", "distribution"] == "LE"
    assert table.loc["SR", "distribution"] == "SR"
    assert table.loc["Common", "distribution"] == "COMMON_OR_UNASSIGNED"


def test_batch_restricted_but_globally_common_population_is_not_called_rare():
    rows = []
    for batch in ["b1", "b2", "b3", "b4"]:
        rows += [{"batch": batch, "label": "Common"}] * 100
    rows += [{"batch": "b1", "label": "RestrictedButCommon"}] * 20

    table = infer_distribution_classes(
        pd.DataFrame(rows),
        batch_key="batch",
        label_key="label",
        global_abundance_threshold=0.02,
        batch_fraction_threshold=0.25,
        local_abundance_threshold=0.05,
    ).set_index("cell_type")
    assert table.loc["RestrictedButCommon", "global_abundance"] >= 0.02
    assert table.loc["RestrictedButCommon", "distribution"] == "COMMON_OR_UNASSIGNED"
