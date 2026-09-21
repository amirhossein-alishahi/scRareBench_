from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scrarebench import benchmark_latent, dataset_info
from scrarebench.datasets.metadata import attach_builtin_dataset_metadata
from scrarebench.scenarios import (
    annotate_paper_scenarios,
    annotate_registered_scenarios,
    load_paper_scenario_table,
    load_registered_scenario_table,
)


DATASETS = (
    (0, "gse194122"),
    (1, "gse194122_raw"),
    (2, "mbdrc_renal_cortex"),
    (3, "wu_breast_cancer_atlas"),
    (4, "covid19_autoimmunity_pbmc"),
    (5, "nygc_seurat_v4_pbmc"),
)


def _labels_for(dataset_key: str) -> list[str]:
    if dataset_key == "gse194122":
        return load_paper_scenario_table()["cell_type"].astype(str).tolist() + ["common"]
    if dataset_key in {
        "mbdrc_renal_cortex",
        "wu_breast_cancer_atlas",
        "covid19_autoimmunity_pbmc",
        "nygc_seurat_v4_pbmc",
    }:
        table = load_registered_scenario_table(dataset_key, include_unassigned=True)
        return table["cell_type"].astype(str).tolist() + ["common"]
    return ["A", "B", "C"]


def _synthetic_builtin(dataset_index: int, dataset_key: str) -> ad.AnnData:
    labels = _labels_for(dataset_key)
    repeated = [label for label in labels for _ in range(3)]
    n = len(repeated)
    rng = np.random.default_rng(100 + dataset_index)
    x = rng.poisson(2.0, size=(n, 12)).astype(float)
    x[:, 0] += 1.0
    obs = pd.DataFrame(index=[f"{dataset_key}_c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(x.shape[1])])

    if dataset_key in {"gse194122", "gse194122_raw"}:
        obs["celltype"] = repeated
        obs["BATCH"] = [f"b{i % 3}" for i in range(n)]
    elif dataset_key == "mbdrc_renal_cortex":
        obs["cell_type"] = repeated
        obs["donor_id"] = [f"d{i % 3}" for i in range(n)]
        obs["assay"] = ["rna" if i % 2 == 0 else "cite" for i in range(n)]
    elif dataset_key == "wu_breast_cancer_atlas":
        obs["celltype_subset"] = repeated
        obs["donor_id"] = [f"d{i % 3}" for i in range(n)]
    elif dataset_key == "covid19_autoimmunity_pbmc":
        obs["cell_type"] = repeated
        obs["donor_id"] = [f"d{i % 3}" for i in range(n)]
    elif dataset_key == "nygc_seurat_v4_pbmc":
        obs["celltype.l2"] = repeated
        obs["orig.ident"] = [f"sample{i % 3}" for i in range(n)]
    else:  # pragma: no cover
        raise AssertionError(dataset_key)

    a = ad.AnnData(X=x, obs=obs, var=var)
    if dataset_key != "nygc_seurat_v4_pbmc":
        a.layers["counts"] = x.copy()

    if dataset_key == "gse194122":
        annotate_paper_scenarios(a, inplace=True)
    elif dataset_key in {
        "mbdrc_renal_cortex",
        "wu_breast_cancer_atlas",
        "covid19_autoimmunity_pbmc",
        "nygc_seurat_v4_pbmc",
    }:
        annotate_registered_scenarios(
            a,
            dataset_key=dataset_key,
            inplace=True,
            strict_labels=True,
        )

    attach_builtin_dataset_metadata(
        a,
        dataset_key=dataset_key,
        dataset_index=dataset_index,
        display_name=dataset_key,
    )
    return a


def _patch_evaluation(monkeypatch):
    import scrarebench.evaluation as ev

    def cluster(
        adata,
        *,
        representation_key,
        method_name,
        n_neighbors,
        metric,
        resolutions,
        random_state,
        overwrite,
        **kwargs,
    ):
        label_key = dataset_info(adata)["label_key"]
        keys = {}
        for resolution in resolutions:
            key = f"synthetic_cluster_{str(resolution).replace('.', '_')}"
            adata.obs[key] = pd.Categorical(adata.obs[label_key].astype(str))
            keys[float(resolution)] = key
        return SimpleNamespace(
            cluster_keys=keys,
            neighbors_key="synthetic_neighbors",
            leiden_flavor=kwargs.get("leiden_flavor", "igraph"),
            leiden_n_iterations=kwargs.get("leiden_n_iterations", 2),
        )

    monkeypatch.setattr(ev, "run_standard_clustering", cluster)
    monkeypatch.setattr(ev, "run_scib_evaluation", lambda *args, **kwargs: None)


@pytest.mark.parametrize(("dataset_index", "dataset_key"), DATASETS)
def test_builtin_dataset_contract_to_benchmark_html_pipeline(
    dataset_index,
    dataset_key,
    tmp_path,
    monkeypatch,
):
    _patch_evaluation(monkeypatch)
    a = _synthetic_builtin(dataset_index, dataset_key)
    info = dataset_info(a)

    # Freeze the historical D0/D1/D2 contracts while also exercising D3/D4/D5.
    expected = {
        "gse194122": ("celltype", "BATCH", "counts", "evaluation_batch"),
        "gse194122_raw": ("celltype", "BATCH", "counts", "evaluation_batch"),
        "mbdrc_renal_cortex": ("cell_type", "scrarebench_batch", "counts", "global"),
        "wu_breast_cancer_atlas": ("celltype_subset", "donor_id", "counts", "global"),
        "covid19_autoimmunity_pbmc": ("cell_type", "donor_id", "counts", "evaluation_batch"),
        "nygc_seurat_v4_pbmc": ("celltype.l2", "orig.ident", "", "evaluation_batch"),
    }[dataset_key]
    assert (
        info["label_key"],
        info["batch_key"],
        info["count_layer"],
        info["scib_hvg_batch_mode"],
    ) == expected
    assert info["benchmark_ready"] is True

    latent = np.random.default_rng(900 + dataset_index).normal(
        size=(a.n_obs, 5)
    )
    result = benchmark_latent(
        a,
        latent,
        method=f"Synthetic-{dataset_index}",
        output_dir=tmp_path / f"d{dataset_index}",
        config={
            "run_scib": False,
            "write_interactive_report": True,
            "write_pdf_report": False,
            "create_bundle": True,
            "include_cell_ids": False,
        },
    )

    assert result.report_path.exists()
    assert result.report_path.stat().st_size > 1_000
    html = result.report_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert f"Synthetic-{dataset_index}" in html
    assert result.bundle_path is not None
    assert result.bundle_path.exists()

    if dataset_key == "gse194122_raw":
        assert result.rare_metrics.empty
    else:
        assert not result.metrics.empty
