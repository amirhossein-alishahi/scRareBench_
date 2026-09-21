from __future__ import annotations

import hashlib

import anndata as ad
import numpy as np
import pandas as pd

from scrarebench.datasets.metadata import (
    BUILTIN_BENCHMARK_PROFILES,
    attach_builtin_dataset_metadata,
    dataset_info,
)
import scrarebench.datasets.preparation as prep
from scrarebench.scenarios import (
    load_registered_scenario_table,
    registered_scenario_info,
)


def _ordered_hash(names) -> str:
    return hashlib.sha256("\0".join(map(str, names)).encode("utf-8")).hexdigest()


def _raw_dataset(*, label_key: str, batch_key: str) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            label_key: ["A", "A", "B", "B", "C", "C"],
            batch_key: ["b1", "b2", "b1", "b2", "b1", "b2"],
        },
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    counts = np.array(
        [
            [1, 2, 1, 0],
            [2, 1, 0, 1],
            [1, 1, 2, 0],
            [1, 0, 1, 2],
            [2, 1, 1, 1],
            [1, 2, 1, 1],
        ],
        dtype=float,
    )
    source = ad.AnnData(X=np.log1p(counts), obs=obs.copy(), var=var.copy())
    source.raw = ad.AnnData(X=counts, obs=obs.copy(), var=var.copy())
    return source


def test_datasets_0_1_2_profiles_are_frozen():
    d0 = BUILTIN_BENCHMARK_PROFILES["gse194122"]
    assert (
        d0.label_key,
        d0.batch_key,
        d0.count_layer,
        d0.scib_hvg_batch_mode,
        d0.batch_components,
        d0.benchmark_ready,
    ) == ("celltype", "BATCH", "counts", "evaluation_batch", (), True)

    d1 = BUILTIN_BENCHMARK_PROFILES["gse194122_raw"]
    assert (
        d1.label_key,
        d1.batch_key,
        d1.count_layer,
        d1.scib_hvg_batch_mode,
        d1.batch_components,
        d1.benchmark_ready,
    ) == ("celltype", "BATCH", "counts", "evaluation_batch", (), True)

    d2 = BUILTIN_BENCHMARK_PROFILES["mbdrc_renal_cortex"]
    assert (
        d2.label_key,
        d2.batch_key,
        d2.count_layer,
        d2.scib_hvg_batch_mode,
        d2.batch_components,
        d2.benchmark_ready,
    ) == (
        "cell_type",
        "scrarebench_batch",
        "counts",
        "global",
        ("donor_id", "assay"),
        True,
    )

    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "donor_id": ["d1", "d2", "d1", "d2"],
            "assay": ["rna", "rna", "cite", "cite"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    a = ad.AnnData(X=np.ones((4, 3)), obs=obs)
    a.layers["counts"] = a.X.copy()
    attach_builtin_dataset_metadata(a, dataset_key="mbdrc_renal_cortex", dataset_index=2)
    info = dataset_info(a)
    assert info["batch_key"] == "scrarebench_batch"
    assert info["scib_hvg_batch_mode"] == "global"
    assert a.obs["scrarebench_batch"].astype(str).tolist() == [
        "d1 || rna",
        "d2 || rna",
        "d1 || cite",
        "d2 || cite",
    ]


def test_datasets_3_4_5_profiles_match_audited_contracts():
    d3 = BUILTIN_BENCHMARK_PROFILES["wu_breast_cancer_atlas"]
    assert (
        d3.label_key,
        d3.batch_key,
        d3.count_layer,
        d3.scib_hvg_batch_mode,
        d3.benchmark_ready,
    ) == ("celltype_subset", "donor_id", "counts", "global", True)

    d4 = BUILTIN_BENCHMARK_PROFILES["covid19_autoimmunity_pbmc"]
    assert (
        d4.label_key,
        d4.batch_key,
        d4.count_layer,
        d4.scib_hvg_batch_mode,
        d4.benchmark_ready,
    ) == ("cell_type", "donor_id", "counts", "evaluation_batch", True)

    d5 = BUILTIN_BENCHMARK_PROFILES["nygc_seurat_v4_pbmc"]
    assert (
        d5.label_key,
        d5.batch_key,
        d5.count_layer,
        d5.scib_hvg_batch_mode,
        d5.benchmark_ready,
    ) == ("celltype.l2", "orig.ident", None, "evaluation_batch", True)


def test_dataset3_raw_count_preparation_is_in_memory_and_label_free(monkeypatch):
    a = _raw_dataset(label_key="celltype_subset", batch_key="donor_id")
    original_x = a.X.copy()
    monkeypatch.setitem(
        prep.EXTERNAL_BENCHMARK_CONTRACTS,
        "wu_breast_cancer_atlas",
        {
            "label_key": "celltype_subset",
            "batch_key": "donor_id",
            "source_n_obs": 6,
            "analysis_n_obs": 6,
            "n_vars": 4,
            "n_labels": 3,
            "n_batches": 2,
            "count_source": "raw_aligned",
        },
    )

    result = prep.prepare_external_benchmark_dataset(
        a,
        dataset_key="wu_breast_cancer_atlas",
    )
    assert result is a
    assert np.array_equal(result.X, original_x)
    assert np.array_equal(result.layers["counts"], np.asarray(result.raw.X))
    assert result.uns[prep.PREPARATION_UNS_KEY]["source_h5ad_modified"] is False
    assert result.uns[prep.PREPARATION_UNS_KEY]["count_source"].startswith("adata.raw.X")


def test_dataset4_raw_count_preparation_uses_donor_only(monkeypatch):
    a = _raw_dataset(label_key="cell_type", batch_key="donor_id")
    monkeypatch.setitem(
        prep.EXTERNAL_BENCHMARK_CONTRACTS,
        "covid19_autoimmunity_pbmc",
        {
            "label_key": "cell_type",
            "batch_key": "donor_id",
            "source_n_obs": 6,
            "analysis_n_obs": 6,
            "n_vars": 4,
            "n_labels": 3,
            "n_batches": 2,
            "count_source": "raw_aligned",
        },
    )
    prep.prepare_external_benchmark_dataset(
        a,
        dataset_key="covid19_autoimmunity_pbmc",
    )
    assert "counts" in a.layers
    assert str(a.obs["donor_id"].dtype) == "category"


def test_dataset5_official_qc_and_x_count_contract(monkeypatch):
    obs = pd.DataFrame(
        {
            "celltype.l2": ["pDC", "Doublet", "cDC1", "dnT", "ILC"],
            "orig.ident": ["b1", "b1", "b2", "b2", "b3"],
        },
        index=[f"c{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=["MT-A", "G1", "G2"])
    x = np.array(
        [
            [1, 100, 100],
            [1, 100, 100],
            [100, 1, 1],
            [1, 100, 100],
            [1, 100, 100],
        ],
        dtype=float,
    )
    a = ad.AnnData(X=x, obs=obs, var=var)

    protein = np.full((5, 151), 20.0)
    protein[3, 100:] = 0.0
    protein[4, :] = 300.0
    a.obsm["protein_counts"] = protein

    mask, reasons = prep._nygc_official_qc_mask(a)
    assert mask.tolist() == [True, False, False, False, False]
    assert reasons["doublet_label"] == 1
    assert reasons["pct_mito_ge_12"] == 1
    assert reasons["proteins_detected_le_150"] == 1
    assert reasons["protein_log_library_ge_10_3"] == 1

    monkeypatch.setitem(
        prep.EXTERNAL_BENCHMARK_CONTRACTS,
        "nygc_seurat_v4_pbmc",
        {
            "label_key": "celltype.l2",
            "batch_key": "orig.ident",
            "source_n_obs": 5,
            "analysis_n_obs": 1,
            "n_vars": 3,
            "n_labels": 1,
            "n_batches": 1,
            "count_source": "X",
            "retained_cell_order_sha256": _ordered_hash(["c0"]),
        },
    )
    result = prep.prepare_nygc_seurat_v4_pbmc(a)
    assert result.n_obs == 1
    assert result.obs_names.tolist() == ["c0"]
    assert "counts" not in result.layers
    assert result.uns[prep.PREPARATION_UNS_KEY]["count_source"] == "adata.X"


def test_dataset4_label_revision_and_dataset5_scenarios_are_registered():
    d4 = load_registered_scenario_table("covid19_autoimmunity_pbmc", include_unassigned=True)
    assert "mucosal-associated invariant T cell" in set(d4["cell_type"].astype(str))
    assert "mucosal invariant T cell" not in set(d4["cell_type"].astype(str))

    d5 = load_registered_scenario_table("nygc_seurat_v4_pbmc", include_unassigned=True)
    assert len(d5) == 9
    assert set(d5["cell_type"]) == {
        "CD4 Proliferating",
        "CD8 Proliferating",
        "ILC",
        "NK Proliferating",
        "NK_CD56bright",
        "Plasmablast",
        "cDC1",
        "dnT",
        "pDC",
    }
    assert set(d5["scenario"]) == {"GR-DL", "GR-RM", "SR-DL", "SR-RM"}
    info = registered_scenario_info("nygc_seurat_v4_pbmc")
    assert info["n_six_state_rows"] == 9
    assert info["missing_scenarios"] == ["LE-DL", "LE-RM"]
