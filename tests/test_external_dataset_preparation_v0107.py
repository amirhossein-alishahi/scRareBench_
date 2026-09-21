from __future__ import annotations

import hashlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scrarebench.datasets.gse194122 as gse
import scrarebench.datasets.preparation as prep
import scrarebench.datasets.registry as registry
from scrarebench import dataset_info
from scrarebench.datasets.metadata import attach_builtin_dataset_metadata


def _small_adata(obs: pd.DataFrame, *, n_vars: int = 3) -> ad.AnnData:
    x = sparse.csr_matrix(np.ones((len(obs), n_vars), dtype=float))
    return ad.AnnData(
        X=x,
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


def test_dataset0_and_dataset2_metadata_contracts_are_unchanged():
    obs0 = pd.DataFrame(
        {"celltype": ["A", "B"], "BATCH": ["b1", "b2"]},
        index=["c0", "c1"],
    )
    a0 = _small_adata(obs0)
    attach_builtin_dataset_metadata(a0, dataset_key="gse194122", dataset_index=0)
    info0 = dataset_info(a0)
    assert info0["label_key"] == "celltype"
    assert info0["batch_key"] == "BATCH"
    assert info0["count_layer"] == "counts"
    assert info0["scib_hvg_batch_mode"] == "evaluation_batch"
    assert info0["benchmark_ready"] is True
    assert "method_hvg" not in info0
    assert "preparation" not in info0

    obs2 = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "donor_id": ["d1", "d2"],
            "assay": ["10x", "smart"],
        },
        index=["c0", "c1"],
    )
    a2 = _small_adata(obs2)
    attach_builtin_dataset_metadata(a2, dataset_key="mbdrc_renal_cortex", dataset_index=2)
    info2 = dataset_info(a2)
    assert info2["label_key"] == "cell_type"
    assert info2["batch_key"] == "scrarebench_batch"
    assert info2["batch_components"] == ["donor_id", "assay"]
    assert info2["scib_hvg_batch_mode"] == "global"
    assert info2["benchmark_ready"] is True
    assert a2.obs["scrarebench_batch"].astype(str).tolist() == [
        "d1 || 10x",
        "d2 || smart",
    ]
    assert "method_hvg" not in info2
    assert "preparation" not in info2


@pytest.mark.parametrize(
    "dataset_key,label_key,batch_key,count_layer,scib_mode,hvg_mode",
    [
        (
            "wu_breast_cancer_atlas",
            "celltype_subset",
            "donor_id",
            "counts",
            "global",
            "global",
        ),
        (
            "covid19_autoimmunity_pbmc",
            "cell_type",
            "donor_id",
            "counts",
            "evaluation_batch",
            "evaluation_batch",
        ),
        (
            "nygc_seurat_v4_pbmc",
            "celltype.l2",
            "orig.ident",
            "",
            "evaluation_batch",
            "evaluation_batch",
        ),
    ],
)
def test_datasets3_4_5_profiles_are_benchmark_ready(
    dataset_key, label_key, batch_key, count_layer, scib_mode, hvg_mode
):
    obs = pd.DataFrame(
        {label_key: ["A", "B"], batch_key: ["b1", "b2"]},
        index=["c0", "c1"],
    )
    a = _small_adata(obs)
    attach_builtin_dataset_metadata(a, dataset_key=dataset_key)
    info = dataset_info(a)
    assert info["label_key"] == label_key
    assert info["batch_key"] == batch_key
    assert info["count_layer"] == count_layer
    assert info["scib_hvg_batch_mode"] == scib_mode
    assert info["benchmark_ready"] is True
    assert info["method_hvg"] == {
        "flavor": "seurat_v3",
        "n_top_genes": 4000,
        "batch_mode": hvg_mode,
        "span": 0.3,
    }


def test_dataset3_preparation_exposes_aligned_raw_counts_without_touching_x(monkeypatch):
    key = "wu_breast_cancer_atlas"
    obs = pd.DataFrame(
        {
            "celltype_subset": ["A", "A", "B", "B"],
            "donor_id": ["d1", "d2", "d1", "d2"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    raw_counts = sparse.csr_matrix(
        np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1, 3, 5],
            ],
            dtype=float,
        )
    )
    processed = sparse.csr_matrix(np.log1p(raw_counts.toarray()))
    a = ad.AnnData(
        X=processed.copy(),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    raw = ad.AnnData(
        X=raw_counts.copy(),
        obs=obs.copy(),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    a.raw = raw

    monkeypatch.setitem(prep._EXPECTED_SOURCE_SHAPES, key, (4, 3))
    monkeypatch.setitem(
        prep._EXPECTED_ANALYSIS_CONTRACTS,
        key,
        {
            "n_obs": 4,
            "n_vars": 3,
            "label_key": "celltype_subset",
            "n_labels": 2,
            "batch_key": "donor_id",
            "n_batches": 2,
        },
    )

    before_x = a.X.copy()
    result = prep.prepare_builtin_benchmark_dataset(a, dataset_key=key)
    assert result is a
    assert np.allclose(result.X.toarray(), before_x.toarray())
    assert np.array_equal(result.layers["counts"].toarray(), raw_counts.toarray())
    assert result.uns["scrarebench_preparation"]["cell_filter"] == "none"


def test_dataset5_official_qc_preparation_is_deterministic(monkeypatch):
    key = "nygc_seurat_v4_pbmc"
    obs = pd.DataFrame(
        {
            "celltype.l2": ["A", "Doublet", "B", "B"],
            "orig.ident": ["b1", "b1", "b2", "b2"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    x = sparse.csr_matrix(
        np.array(
            [
                [1, 99],
                [1, 99],
                [2, 98],
                [20, 20],
            ],
            dtype=float,
        )
    )
    a = ad.AnnData(
        X=x,
        obs=obs,
        var=pd.DataFrame(index=["MT-test", "GENE"]),
    )
    a.obsm["protein_counts"] = np.full((4, 160), 20.0)

    retained = ["c0", "c2"]
    retained_hash = hashlib.sha256(
        "\0".join(retained).encode("utf-8")
    ).hexdigest()

    monkeypatch.setitem(prep._EXPECTED_SOURCE_SHAPES, key, (4, 2))
    monkeypatch.setitem(
        prep._EXPECTED_ANALYSIS_CONTRACTS,
        key,
        {
            "n_obs": 2,
            "n_vars": 2,
            "label_key": "celltype.l2",
            "n_labels": 2,
            "batch_key": "orig.ident",
            "n_batches": 2,
        },
    )
    monkeypatch.setattr(prep, "_DATASET5_RETAINED_CELL_SHA256", retained_hash)

    result = prep.prepare_builtin_benchmark_dataset(a, dataset_key=key)
    assert result.obs_names.tolist() == retained
    assert result.n_obs == 2
    assert "counts" not in result.layers
    info = result.uns["scrarebench_preparation"]
    assert info["source_cells"] == 4
    assert info["retained_cells"] == 2
    assert info["cell_filter"] == "official_scvi_tools_pbmc_qc"
    assert info["retained_cell_order_sha256"] == retained_hash


def test_registry_does_not_run_dataset3_5_preparation_for_dataset2(monkeypatch, tmp_path):
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "donor_id": ["d1", "d2"],
            "assay": ["10x", "smart"],
        },
        index=["c0", "c1"],
    )
    a = _small_adata(obs)
    source = tmp_path / "d2.h5ad"
    source.touch()

    monkeypatch.setattr(registry, "download_dataset", lambda *args, **kwargs: source)
    a_obj = a
    monkeypatch.setattr(
        gse,
        "_require_anndata",
        lambda: SimpleNamespace(read_h5ad=lambda *args, **kwargs: a_obj),
    )

    import scrarebench.scenarios as scenarios
    monkeypatch.setattr(
        scenarios,
        "annotate_registered_scenarios",
        lambda adata, **kwargs: adata,
    )

    import scrarebench.datasets.preparation as preparation
    monkeypatch.setattr(
        preparation,
        "prepare_builtin_benchmark_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Dataset 2 must not enter the Dataset 3-5 preparation path")
        ),
    )

    result = registry.load_dataset(2, data_dir=tmp_path)
    assert result is a
    assert dataset_info(result)["batch_key"] == "scrarebench_batch"
