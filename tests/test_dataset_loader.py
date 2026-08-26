from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import scrarebench.datasets.gse194122 as gse


def _fake_prepared(n_obs=24):
    obs = pd.DataFrame(
        {
            "BATCH": [f"b{i % gse.EXPECTED_BATCHES}" for i in range(n_obs)],
            "celltype": ["Other"] * n_obs,
            "scrarebench_scenario": [""] * n_obs,
            "scrarebench_distribution": [""] * n_obs,
            "scrarebench_topology": [""] * n_obs,
            "scrarebench_parent_type": [""] * n_obs,
            "scrarebench_curation_source": [""] * n_obs,
            "scrarebench_is_rare": [False] * n_obs,
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    return SimpleNamespace(
        obs=obs,
        obs_names=obs.index,
        n_obs=n_obs,
    )


def test_high_level_loader_reuses_existing_prepared_dataset(tmp_path, monkeypatch):
    output = tmp_path / gse.PAPER_MAIN_H5AD_NAME
    output.touch()
    prepared = _fake_prepared()

    monkeypatch.setattr(gse, "load_gse194122", lambda path: prepared)
    monkeypatch.setattr(
        gse,
        "download_gse194122",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not run")),
    )
    monkeypatch.setattr(
        gse,
        "prepare_gse194122_paper_main",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prepare should not run")),
    )

    result = gse.load_gse194122_benchmark(tmp_path, strict_expected_counts=False)
    assert result is prepared


def test_high_level_loader_downloads_and_prepares_when_missing(tmp_path, monkeypatch):
    prepared = _fake_prepared()
    calls = []

    def fake_download(cache_dir, *, force=False, url=gse.GSE194122_SOURCE_URL):
        calls.append(("download", Path(cache_dir), force, url))
        return tmp_path / "source.h5ad"

    def fake_prepare(source, output, **kwargs):
        calls.append(("prepare", Path(source), Path(output), kwargs))
        return prepared, {"output_n_obs": prepared.n_obs}

    monkeypatch.setattr(gse, "download_gse194122", fake_download)
    monkeypatch.setattr(gse, "prepare_gse194122_paper_main", fake_prepare)

    result = gse.load_gse194122_benchmark(tmp_path, strict_expected_counts=False)

    assert result is prepared
    assert calls[0][0] == "download"
    assert calls[1][0] == "prepare"
    assert calls[1][2] == tmp_path / gse.PAPER_MAIN_H5AD_NAME


def test_high_level_loader_does_not_perform_method_preprocessing(tmp_path, monkeypatch):
    prepared = _fake_prepared()
    monkeypatch.setattr(gse, "download_gse194122", lambda *args, **kwargs: tmp_path / "source.h5ad")
    monkeypatch.setattr(gse, "prepare_gse194122_paper_main", lambda *args, **kwargs: (prepared, {}))

    result = gse.load_gse194122_benchmark(tmp_path, strict_expected_counts=False)

    # The dataset API only constructs/annotates benchmark membership. Method-specific
    # preprocessing is intentionally absent from this layer.
    assert not hasattr(result, "obsm")
    assert set(gse.PAPER_SCENARIO_COLUMNS).issubset(result.obs.columns)
