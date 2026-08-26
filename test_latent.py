import numpy as np
import pytest

from scrarebench.exceptions import LatentAlignmentError
from scrarebench.latent import validate_and_align_latent


def test_exact_alignment():
    latent = np.arange(12).reshape(4, 3)
    aligned, report = validate_and_align_latent(latent, ["a", "b", "c", "d"], latent_barcodes=["a", "b", "c", "d"])
    assert np.array_equal(aligned, latent)
    assert report["exact_order_match"] is True


def test_mismatched_order_fails_by_default():
    latent = np.arange(12).reshape(4, 3)
    with pytest.raises(LatentAlignmentError):
        validate_and_align_latent(latent, ["a", "b", "c", "d"], latent_barcodes=["b", "a", "c", "d"])


def test_optional_reorder():
    latent = np.array([[20], [10], [30]])
    aligned, report = validate_and_align_latent(
        latent,
        ["a", "b", "c"],
        latent_barcodes=["b", "a", "c"],
        allow_reorder=True,
    )
    assert aligned[:, 0].tolist() == [10, 20, 30]
    assert report["reordered"] is True
