import pandas as pd

from scrarebench.scenarios import infer_distribution_classes


def test_distribution_only_inference():
    rows = []
    for batch in ["b1", "b2", "b3", "b4"]:
        rows += [{"batch": batch, "label": "GR"}] * 1
        rows += [{"batch": batch, "label": "Common"}] * 99
    rows += [{"batch": "b1", "label": "LE"}] * 20
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
