import pandas as pd

from scrarebench.failures import classify_failure_archetypes


def test_provisional_failure_rules():
    frame = pd.DataFrame(
        [
            {"cell_type": "ok", "precision": 0.9, "recall": 0.9, "inverse_purity": 0.9, "within_type_batch_nmi": 0.1, "n_clusters_found_in": 1, "dominant_wrong_fraction": 0.0},
            {"cell_type": "leak", "precision": 0.2, "recall": 0.8, "inverse_purity": 0.8, "within_type_batch_nmi": 0.1, "n_clusters_found_in": 1, "dominant_wrong_fraction": 0.1},
            {"cell_type": "assim", "precision": 0.0, "recall": 0.1, "inverse_purity": 0.8, "within_type_batch_nmi": 0.1, "n_clusters_found_in": 1, "dominant_wrong_fraction": 0.8},
            {"cell_type": "frag", "precision": 0.8, "recall": 0.4, "inverse_purity": 0.3, "within_type_batch_nmi": 0.8, "n_clusters_found_in": 3, "dominant_wrong_fraction": 0.2},
        ]
    )
    result = classify_failure_archetypes(frame).set_index("cell_type")
    assert result.loc["ok", "failure_archetype"] == "preserved"
    assert result.loc["leak", "failure_archetype"] == "lineage_leakage"
    assert result.loc["assim", "failure_archetype"] == "lineage_assimilation"
    assert result.loc["frag", "failure_archetype"] == "batch_driven_fragmentation"
