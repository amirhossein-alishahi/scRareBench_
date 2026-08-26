import pandas as pd

from scrarebench.evaluation import _scenario_summary
from scrarebench.scenarios import SIX_SCENARIOS


def test_scenario_summary_emits_all_six_slots():
    rare = pd.DataFrame(
        {
            "scenario": ["GR-DL", "LE-RM"],
            "distribution": ["GR", "LE"],
            "topology": ["DL", "RM"],
            "precision": [0.8, 0.6],
            "recall": [0.7, 0.5],
            "f1": [0.75, 0.55],
            "inverse_purity": [0.9, 0.7],
            "within_type_batch_nmi": [0.1, 0.2],
        }
    )
    out = _scenario_summary(rare)
    assert out["scenario"].tolist() == list(SIX_SCENARIOS)
    assert len(out) == 6
    empty = out.set_index("scenario").loc["SR-RM"]
    assert bool(empty["is_empty"])
    assert int(empty["f1_count"]) == 0
    present = out.set_index("scenario").loc["GR-DL"]
    assert not bool(present["is_empty"])
    assert int(present["f1_count"]) == 1
