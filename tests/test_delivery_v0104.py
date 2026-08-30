from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scrarebench import (
    create_report_bundle,
    finalize_multiseed_delivery,
    validate_multiseed_report,
    write_interactive_report,
)
from scrarebench.multiseed import extract_embedded_report_payload


class MiniAdata:
    def __init__(self) -> None:
        self.obs = pd.DataFrame(
            {
                "celltype": ["A", "A", "B", "B", "C", "C"],
                "BATCH": ["x", "y", "x", "y", "x", "y"],
                "scrarebench_scenario": ["GR-DL", "GR-DL", "non_rare", "non_rare", "SR-RM", "SR-RM"],
                "cluster": pd.Categorical(["0", "0", "1", "1", "2", "2"]),
                "pred": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
            },
            index=[f"c{i}" for i in range(6)],
        )
        self.obs_names = self.obs.index
        self.n_obs = len(self.obs)
        self.var_names = pd.Index(["G1", "G2"])
        self.X = np.ones((6, 2), dtype=np.float32)
        self.layers = {"counts": self.X.copy()}
        self.obsm = {
            "X_demo": np.arange(18, dtype=np.float32).reshape(6, 3) / 10,
            "X_umap_demo": np.array([[0, 0], [0.1, 0], [1, 1], [1.1, 1], [2, 0], [2.1, 0]], dtype=float),
        }
        self.uns = {"scrarebench_dataset": {"dataset_key": "demo", "dataset_index": 0}}


class MiniResult:
    def __init__(self, root: Path) -> None:
        self.output_dir = root
        repro = root / "reproducibility"
        repro.mkdir(parents=True, exist_ok=True)
        (repro / "run_config.yaml").write_text(
            "method_name: Demo\nrepresentation_key: X_demo\nlabel_key: celltype\nbatch_key: BATCH\n"
            "benchmark_seed: 42\nrandom_state: 42\nn_neighbors: 15\n"
            "reference_n_clusters: 3\n",
            encoding="utf-8",
        )
        self.files = {"run_config": repro / "run_config.yaml"}
        self.prediction_key = "pred"
        self.cluster_keys = {1.0: "cluster"}
        self.local_recovery_key = None
        self.local_recovery_adjusted_key = None
        self.subset_metrics = pd.DataFrame([
            {"subset": "overall", "n_cells": 6, "n_cell_types": 3, "F1_macro": 0.8},
            {"subset": "rare", "n_cells": 4, "n_cell_types": 2, "F1_macro": 0.6},
        ])
        self.subset_metric_ratios = pd.DataFrame()
        self.per_type_metrics = pd.DataFrame([
            {"cell_type": "A", "support": 2, "f1": 0.8},
            {"cell_type": "C", "support": 2, "f1": 0.4},
        ])
        self.rare_metrics = pd.DataFrame([
            {"cell_type": "A", "scenario": "GR-DL", "distribution": "GR", "topology": "DL",
             "support": 2, "knn_local_recovery_adjusted": 0.5, "best_cluster_f1": 0.8,
             "inverse_purity": 1.0, "precision": 0.8, "recall": 0.8, "f1": 0.8,
             "failure_archetype": "preserved", "failure_archetype_v2": "preserved"},
            {"cell_type": "C", "scenario": "SR-RM", "distribution": "SR", "topology": "RM",
             "support": 2, "knn_local_recovery_adjusted": 0.2, "best_cluster_f1": 0.4,
             "inverse_purity": 0.5, "precision": 0.3, "recall": 0.5, "f1": 0.375,
             "failure_archetype": "lineage_assimilation", "failure_archetype_v2": "resolution_limited"},
        ])
        self.rare_summary = pd.DataFrame([
            {"metric": "knn_local_recovery_adjusted", "metric_type": "rare", "mean": 0.35, "n_valid": 2},
            {"metric": "best_cluster_f1", "metric_type": "rare", "mean": 0.6, "n_valid": 2},
        ])
        self.scenario_metrics = pd.DataFrame([
            {"scenario": "GR-DL", "distribution": "GR", "topology": "DL", "n_cell_types": 1},
            {"scenario": "SR-RM", "distribution": "SR", "topology": "RM", "n_cell_types": 1},
        ])
        self.resolution_rare_metrics = pd.DataFrame([
            {"resolution": 1.0, "cell_type": "A", "scenario": "GR-DL", "best_cluster_f1": 0.8},
            {"resolution": 1.0, "cell_type": "C", "scenario": "SR-RM", "best_cluster_f1": 0.4},
        ])
        self.rare_evaluation_status = {"status": "available"}
        self.scib_status = {"status": "disabled"}
        self.scib = None


def _make_reports(tmp_path: Path):
    adata = MiniAdata()
    result = MiniResult(tmp_path / "result")
    reports = []
    bundles = {}
    statuses = {}
    latents = {}
    for seed, delta in ((42, 0.0), (123, 0.02), (2026, -0.01)):
        adata.obsm["X_demo"] = (np.arange(18, dtype=np.float32).reshape(6, 3) / 10) + delta
        path = tmp_path / f"seed_{seed}.html"
        write_interactive_report(
            adata,
            result,
            path,
            representation_key="X_demo",
            umap_key="X_umap_demo",
            method_seed=seed,
            method_config={"seed": seed, "n_latent": 3},
            expected_seeds=[42, 123, 2026],
            include_static_figures=False,
        )
        reports.append(path)
        bundle = tmp_path / f"bundle_{seed}.zip"
        create_report_bundle(
            adata,
            result,
            bundle,
            representation_key="X_demo",
            include_latent=True,
            write_interactive=True,
            write_pdf=False,
            existing_interactive_report=path,
            method_seed=seed,
            method_config={"seed": seed, "n_latent": 3},
            expected_seeds=[42, 123, 2026],
        )
        bundles[seed] = bundle
        status = tmp_path / f"status_{seed}.json"
        status.write_text(json.dumps({"status": "complete", "method_seed": seed}), encoding="utf-8")
        statuses[seed] = status
        latent = tmp_path / f"latent_{seed}.npy"
        np.save(latent, adata.obsm["X_demo"], allow_pickle=False)
        latents[seed] = latent
    return reports, bundles, statuses, latents
