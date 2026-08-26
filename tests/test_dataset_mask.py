import pandas as pd

from scrarebench.datasets.gse194122 import build_paper_main_mask


def test_paper_mask_preserves_order_and_keeps_only_configured_batches():
    obs = pd.DataFrame(
        {
            "BATCH": ["s3d7", "other", "s3d6", "s4d8", "other", "s1d1"],
            "celltype": [
                "pDC",
                "pDC",
                "CD8+ T CD57+ CD45RA+",
                "CD8+ T CD57+ CD45RA+",
                "CD8+ T CD57+ CD45RA+",
                "Other",
            ],
        },
        index=["a", "b", "c", "d", "e", "f"],
    )
    mask = build_paper_main_mask(obs)
    assert obs.index[mask].tolist() == ["a", "c", "d", "f"]
