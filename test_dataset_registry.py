from pathlib import Path

import pytest

import scrarebench.datasets.registry as registry


def test_registry_has_exactly_six_stable_indices():
    assert [spec.index for spec in registry.DATASET_REGISTRY] == list(range(6))
    assert registry.resolve_dataset(0).key == "gse194122"
    assert registry.resolve_dataset(1).key == "gse194122_raw"
    assert registry.resolve_dataset(5).key == "nygc_seurat_v4_pbmc"


def test_selectors_accept_names_aliases_and_numeric_strings():
    assert registry.resolve_dataset("gse194122").index == 0
    assert registry.resolve_dataset("GSE194122").index == 0
    assert registry.resolve_dataset("0").index == 0
    assert registry.resolve_dataset("gse194122_unmodified").index == 1
    assert registry.resolve_dataset("mBDRC renal cortex").index == 2
    assert registry.resolve_dataset("GSE176078").index == 3
    assert registry.resolve_dataset("covid_pbmc").index == 4
    assert registry.resolve_dataset("pbmc_seurat_v4").index == 5


def test_unknown_selector_has_clear_error():
    with pytest.raises(ValueError, match="Available datasets"):
        registry.resolve_dataset("not-a-dataset")
    with pytest.raises(ValueError, match="0..5"):
        registry.resolve_dataset(9)


def test_index_zero_downloads_source_and_applies_benchmark_edit(tmp_path, monkeypatch):
    calls = []

    def fake_download(cache_dir, *, force=False, **kwargs):
        calls.append(("download", Path(cache_dir), force))
        source = tmp_path / "source.h5ad"
        source.touch()
        return source

    def fake_prepare(source, output, **kwargs):
        calls.append(("prepare", Path(source), Path(output), kwargs))
        Path(output).touch()
        return object(), {}

    monkeypatch.setattr(registry, "download_gse194122", fake_download)
    monkeypatch.setattr(registry, "prepare_gse194122_paper_main", fake_prepare)

    result = registry.download_dataset(0, tmp_path, strict_expected_counts=False)

    assert result == tmp_path / "gse194122_paper_main.h5ad"
    assert calls[0][0] == "download"
    assert calls[1][0] == "prepare"


def test_index_one_exposes_original_without_prepare(tmp_path, monkeypatch):
    expected = tmp_path / "cache" / "source.h5ad"
    expected.parent.mkdir()
    expected.touch()

    monkeypatch.setattr(
        registry,
        "download_gse194122",
        lambda cache_dir, *, force=False, **kwargs: expected,
    )
    monkeypatch.setattr(
        registry,
        "prepare_gse194122_paper_main",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prepare must not run")),
    )

    assert registry.download_dataset(1, tmp_path) == expected


def test_cellxgene_dataset_selection_prefers_title_and_excludes_companion():
    spec = registry.resolve_dataset(2)
    datasets = [
        {"title": "multimodal benchmarking dataset for renal cortex characterization - multiome only (fragment files included)", "cell_count": 37717},
        {"title": "multimodal benchmarking dataset for renal cortex characterization", "cell_count": 97125},
    ]
    selected = registry._select_cellxgene_dataset(spec, datasets)
    assert selected["cell_count"] == 97125


def test_cellxgene_version_id_builds_permanent_h5ad_url(monkeypatch):
    spec = registry.resolve_dataset(3)
    monkeypatch.setattr(
        registry,
        "_fetch_json",
        lambda url: {
            "name": "Wu collection",
            "datasets": [
                {
                    "dataset_id": "dataset-id",
                    "dataset_version_id": "version-id",
                    "title": "A single-cell and spatially resolved atlas of human breast cancers",
                    "cell_count": 99876,
                    "dataset_assets": [],
                }
            ],
        },
    )
    info = registry._cellxgene_download_info(spec)
    assert info["download_url"] == "https://datasets.cellxgene.cziscience.com/version-id.h5ad"
    assert info["dataset_id"] == "dataset-id"


def test_external_download_is_unmodified_and_writes_manifest(tmp_path, monkeypatch):
    spec = registry.resolve_dataset(4)
    monkeypatch.setattr(
        registry,
        "_cellxgene_download_info",
        lambda spec: {
            "collection_id": spec.collection_id,
            "dataset_id": "dataset-id",
            "dataset_version_id": "version-id",
            "dataset_title": "COVID PBMC",
            "download_url": "https://example.org/file.h5ad",
        },
    )

    def fake_download(url, destination, *, force=False, timeout=120):
        destination.write_bytes(b"fake-h5ad")
        return destination

    monkeypatch.setattr(registry, "_download_url", fake_download)
    path = registry.download_dataset(4, tmp_path)
    manifest = path.with_suffix(path.suffix + ".source.json")

    assert path.name == "covid19_autoimmunity_pbmc.h5ad"
    assert manifest.exists()
    assert '"scrarebench_modification_applied": false' in manifest.read_text().lower()
