import pandas as pd

from scrarebench.scib_backend import _parse_benchmarker_results, _status_catalog


def test_scib_result_parser_separates_metrics_and_aggregates():
    frame = pd.DataFrame(
        {
            "KMeans NMI": [0.7, "Bio conservation"],
            "iLISI": [0.8, "Batch correction"],
            "Bio conservation": [0.7, "Aggregate score"],
            "Batch correction": [0.8, "Aggregate score"],
            "Total": [0.74, "Aggregate score"],
        },
        index=["X_demo", "Metric Type"],
    )
    metrics, aggregates = _parse_benchmarker_results(
        frame,
        representation_key="X_demo",
        backend_version="0.5.9",
    )
    assert set(metrics["metric"]) == {"KMeans NMI", "iLISI"}
    assert set(aggregates["metric"]) == {"Bio conservation", "Batch correction", "Total"}
    assert float(aggregates.loc[aggregates["metric"] == "Total", "value"].iloc[0]) == 0.74


def test_metric_status_explicitly_lists_latent_incompatible_legacy_metrics():
    status = _status_catalog().set_index("metric")
    assert status.loc["HVG overlap", "status"] == "not_applicable"
    assert status.loc["Cell cycle conservation", "status"] == "not_applicable"
    assert status.loc["Trajectory conservation", "status"] == "not_applicable"
    assert status.loc["Silhouette batch", "status"] == "supported"


def test_scib_runtime_compatibility_bridges_pandas3_value_counts(monkeypatch):
    import numpy as np
    import scrarebench.scib_backend as backend

    # Simulate pandas 3, where the top-level pandas.value_counts API was removed.
    monkeypatch.delattr(backend.pd, "value_counts", raising=False)
    assert not hasattr(backend.pd, "value_counts")

    with backend._scib_runtime_compatibility() as adjustments:
        assert hasattr(backend.pd, "value_counts")
        counts = backend.pd.value_counts(np.array([2, 2, 1, 3, 2]))
        assert int(counts.loc[2]) == 3
        assert any("pandas>=3 compatibility" in item for item in adjustments)

    # The compatibility bridge must not permanently mutate the user's pandas module.
    assert not hasattr(backend.pd, "value_counts")


def test_scib_runtime_compatibility_is_noop_when_api_exists(monkeypatch):
    import scrarebench.scib_backend as backend

    marker = object()

    def existing(values, *args, **kwargs):
        return marker

    monkeypatch.setattr(backend.pd, "value_counts", existing, raising=False)
    with backend._scib_runtime_compatibility() as adjustments:
        assert backend.pd.value_counts([1, 1]) is marker
        assert adjustments == []
    assert backend.pd.value_counts is existing


def test_scib_hvg_batch_mode_defaults_to_evaluation_batch():
    from scrarebench.scib_backend import ScibEvaluationConfig, _resolve_hvg_batch_key

    config = ScibEvaluationConfig()
    assert config.hvg_batch_mode == "evaluation_batch"
    assert _resolve_hvg_batch_key(
        evaluation_batch_key="BATCH", mode=config.hvg_batch_mode
    ) == "BATCH"


def test_scib_hvg_batch_mode_can_be_global_without_changing_evaluation_batch():
    from scrarebench.scib_backend import _resolve_hvg_batch_key

    assert _resolve_hvg_batch_key(
        evaluation_batch_key="donor_assay", mode="global"
    ) is None


def test_scib_hvg_batch_mode_rejects_unknown_value():
    import pytest
    from scrarebench.scib_backend import _resolve_hvg_batch_key

    with pytest.raises(ValueError, match="hvg_batch_mode"):
        _resolve_hvg_batch_key(evaluation_batch_key="BATCH", mode="adaptive")
