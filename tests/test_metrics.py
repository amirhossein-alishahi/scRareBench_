import numpy as np

from scrarebench.metrics import majority_vote_predictions, per_type_metrics


def test_majority_vote_and_per_type_metrics():
    true = np.array(["Rare", "Rare", "Rare", "Common", "Common", "Common"])
    clusters = np.array(["0", "0", "1", "0", "1", "1"])
    pred, mapping = majority_vote_predictions(true, clusters)
    assert mapping == {"0": "Rare", "1": "Common"}
    metrics = per_type_metrics(true, clusters, pred).set_index("cell_type")
    assert metrics.loc["Rare", "recall"] == 2 / 3
    assert metrics.loc["Rare", "precision"] == 2 / 3
    assert metrics.loc["Rare", "inverse_purity"] == 2 / 3
