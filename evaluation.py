from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from importlib import metadata as importlib_metadata
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .clustering import run_standard_clustering
from .failures import classify_failure_archetypes, load_failure_rules
from .metrics import majority_vote_predictions, per_type_metrics, subset_metrics
from .plotting import (
    plot_failure_counts,
    plot_precision_recall,
    plot_rare_metric_heatmap,
    write_sankey_html,
)
from .reporting import write_html_report
from .scenarios import load_paper_scenario_table, scenario_table_from_adata
from .scib_backend import ScibEvaluationConfig, ScibEvaluationResult, run_scib_evaluation
from .utils import slugify, write_json


@dataclass(frozen=True)
class EvaluationConfig:
    method_name: str
    representation_key: str
    label_key: str = "celltype"
    batch_key: str = "BATCH"
    scenario_key: str = "scrarebench_scenario"
    reference_resolution: float = 1.0
    resolution_sweep: tuple[float, ...] = (1.0,)
    n_neighbors: int = 15
    distance_metric: str = "euclidean"
    random_state: int = 0
    overwrite: bool = False
    scib: ScibEvaluationConfig = field(default_factory=ScibEvaluationConfig)


@dataclass
class EvaluationResult:
    output_dir: Path
    subset_metrics: pd.DataFrame
    per_type_metrics: pd.DataFrame
    rare_metrics: pd.DataFrame
    rare_summary: pd.DataFrame
    scenario_metrics: pd.DataFrame
    cluster_keys: dict[float, str]
    prediction_key: str
    scib: ScibEvaluationResult | None
    files: dict[str, Path]


def _rare_summary(rare: pd.DataFrame) -> pd.DataFrame:
    metrics = ["precision", "recall", "f1", "inverse_purity", "within_type_batch_nmi"]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(rare[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "n_valid": int(values.notna().sum()),
            }
        )
    failures = rare["failure_archetype"].value_counts(normalize=True)
    rows.append(
        {
            "metric": "preserved_fraction",
            "mean": float(failures.get("preserved", 0.0)),
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "n_valid": int(len(rare)),
        }
    )
    return pd.DataFrame(rows)


def _scenario_summary(rare: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics while preserving the full six-scenario output schema.

    Some external datasets do not contain every GR/LE/SR × DL/RM combination.
    We still emit one row for each registered benchmark scenario so CSV/static/PDF
    outputs have the same six-slot structure as the interactive dashboard. Empty
    scenarios have zero counts and NaN metric summaries; no population is fabricated.
    """
    from .scenarios import SIX_SCENARIOS

    numeric = ["precision", "recall", "f1", "inverse_purity", "within_type_batch_nmi"]
    grouped = (
        rare.groupby(["scenario", "distribution", "topology"], dropna=False)[numeric]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]

    by_scenario = {str(row["scenario"]): row for _, row in grouped.iterrows()}
    rows = []
    for scenario in SIX_SCENARIOS:
        if scenario in by_scenario:
            row = by_scenario[scenario].to_dict()
        else:
            distribution, topology = scenario.split("-", 1)
            row = {
                "scenario": scenario,
                "distribution": distribution,
                "topology": topology,
            }
            for metric in numeric:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
                row[f"{metric}_count"] = 0
        row["is_empty"] = int(row.get("f1_count", 0) or 0) == 0
        rows.append(row)
    return pd.DataFrame(rows)


def _resolution_table(adata: Any, cluster_keys: dict[float, str]) -> pd.DataFrame:
    rows = []
    for resolution, key in cluster_keys.items():
        rows.append(
            {
                "resolution": float(resolution),
                "cluster_key": key,
                "n_clusters": int(adata.obs[key].astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)



def _write_package_versions(path: Path) -> None:
    packages = [
        "scrarebench", "scib-metrics", "scanpy", "anndata", "numpy",
        "pandas", "scikit-learn", "scipy", "matplotlib", "plotly",
        "python-igraph", "leidenalg", "scvi-tools",
    ]
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    for package in packages:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not-installed"
        lines.append(f"{package}={version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combined_summary(
    rare_summary: pd.DataFrame,
    scib_result: ScibEvaluationResult | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scib_result is not None and not scib_result.aggregate_scores.empty:
        for record in scib_result.aggregate_scores.to_dict(orient="records"):
            rows.append(
                {
                    "section": "scib",
                    "metric": str(record.get("metric", "")),
                    "value": record.get("value", np.nan),
                    "status": record.get("status", "computed"),
                    "source": f"{record.get('backend', 'scib-metrics')} {record.get('backend_version', '')}".strip(),
                }
            )
    for record in rare_summary.to_dict(orient="records"):
        rows.append(
            {
                "section": "rare_cell",
                "metric": str(record["metric"]),
                "value": record.get("mean", np.nan),
                "status": "computed",
                "source": "scRareBench",
            }
        )
    return pd.DataFrame(rows)

def evaluate_latent(
    adata: Any,
    config: EvaluationConfig,
    output_dir: str | Path,
    *,
    scenario_table: pd.DataFrame | None = None,
    failure_rules: dict[str, Any] | None = None,
) -> EvaluationResult:
    output = Path(output_dir).expanduser().resolve()
    scib_dir = output / "scib"
    rare_dir = output / "rare_cell"
    clustering_dir = output / "clustering"
    reproducibility_dir = output / "reproducibility"
    figures_dir = rare_dir / "figures"
    for directory in (output, scib_dir, rare_dir, clustering_dir, reproducibility_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for required in (config.label_key, config.batch_key):
        if required not in adata.obs.columns:
            raise KeyError(f"adata.obs['{required}'] is required")
    if config.representation_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{config.representation_key}'] is required")

    resolutions = tuple(sorted(set(config.resolution_sweep + (config.reference_resolution,))))
    clustering = run_standard_clustering(
        adata,
        representation_key=config.representation_key,
        method_name=config.method_name,
        n_neighbors=config.n_neighbors,
        metric=config.distance_metric,
        resolutions=resolutions,
        random_state=config.random_state,
        overwrite=config.overwrite,
    )
    reference_cluster_key = clustering.cluster_keys[config.reference_resolution]
    y_true = adata.obs[config.label_key].astype(str).to_numpy()
    clusters = adata.obs[reference_cluster_key].astype(str).to_numpy()
    predictions, mapping = majority_vote_predictions(y_true, clusters)
    prediction_key = f"scrarebench_prediction_{slugify(config.method_name)}"
    adata.obs[prediction_key] = pd.Categorical(predictions)

    if scenario_table is not None:
        metadata = scenario_table.copy()
    else:
        metadata = scenario_table_from_adata(adata)
        if metadata is None:
            metadata = load_paper_scenario_table()
    rare_types = metadata["cell_type"].astype(str).tolist()
    all_per_type = per_type_metrics(
        y_true,
        clusters,
        predictions,
        batch_labels=adata.obs[config.batch_key].astype(str).to_numpy(),
    )
    rare = all_per_type[all_per_type["cell_type"].isin(rare_types)].copy()
    rare = rare.merge(
        metadata[["cell_type", "scenario", "distribution", "topology", "parent_type", "curation_source"]],
        on="cell_type",
        how="left",
    )
    selected_rules = failure_rules or load_failure_rules()
    rare = classify_failure_archetypes(rare, rules=selected_rules)

    subsets = subset_metrics(
        adata.obsm[config.representation_key],
        y_true,
        clusters,
        predictions,
        rare_types,
    )
    rare_summary = _rare_summary(rare)
    scenario_summary = _scenario_summary(rare)
    resolution_summary = _resolution_table(adata, clustering.cluster_keys)

    files: dict[str, Path] = {}
    files["subset_metrics"] = output / "subset_metrics.csv"  # compatibility alias
    subsets.to_csv(files["subset_metrics"], index=False)
    files["per_type_metrics"] = output / "per_type_metrics.csv"  # compatibility alias
    all_per_type.to_csv(files["per_type_metrics"], index=False)
    files["rare_metrics"] = output / "rare_metrics.csv"  # compatibility alias
    rare.to_csv(files["rare_metrics"], index=False)

    files["rare_metrics_structured"] = rare_dir / "rare_metrics_per_type.csv"
    rare.to_csv(files["rare_metrics_structured"], index=False)
    files["rare_summary"] = rare_dir / "rare_metrics_summary.csv"
    rare_summary.to_csv(files["rare_summary"], index=False)
    files["scenario_metrics"] = rare_dir / "scenario_metrics.csv"
    scenario_summary.to_csv(files["scenario_metrics"], index=False)
    files["all_type_metrics"] = rare_dir / "all_cell_type_metrics.csv"
    all_per_type.to_csv(files["all_type_metrics"], index=False)

    files["cluster_mapping"] = clustering_dir / "cluster_majority_mapping.json"
    write_json(files["cluster_mapping"], mapping)
    files["clusters"] = clustering_dir / "clusters.csv"
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "cluster": clusters,
            "majority_vote_prediction": predictions,
        }
    ).to_csv(files["clusters"], index=False)
    files["resolution_results"] = clustering_dir / "resolution_results.csv"
    resolution_summary.to_csv(files["resolution_results"], index=False)

    run_config = asdict(config)
    run_config["resolution_sweep"] = list(resolutions)
    run_config["reference_cluster_key"] = reference_cluster_key
    run_config["neighbors_key"] = clustering.neighbors_key
    files["run_config"] = reproducibility_dir / "run_config.yaml"
    files["run_config"].write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    files["failure_rules"] = reproducibility_dir / "failure_rules.yaml"
    files["failure_rules"].write_text(yaml.safe_dump(selected_rules, sort_keys=False), encoding="utf-8")
    files["package_versions"] = reproducibility_dir / "package_versions.txt"
    _write_package_versions(files["package_versions"])

    heatmap = plot_rare_metric_heatmap(rare, figures_dir / "rare_metric_heatmap.png")
    precision_recall = plot_precision_recall(rare, figures_dir / "rare_precision_recall.png")
    failure_counts = plot_failure_counts(rare, figures_dir / "failure_archetype_counts.png")
    files["rare_metric_heatmap"] = heatmap
    files["rare_precision_recall"] = precision_recall
    files["failure_counts"] = failure_counts
    sankey = write_sankey_html(y_true, predictions, rare_dir / "sankey_all.html")
    if sankey is not None:
        files["sankey"] = sankey

    scib_result = run_scib_evaluation(
        adata,
        representation_key=config.representation_key,
        batch_key=config.batch_key,
        label_key=config.label_key,
        output_dir=scib_dir,
        config=config.scib,
    )
    if scib_result is not None:
        for name, path in scib_result.files.items():
            files[f"scib_{name}"] = path

    combined = _combined_summary(rare_summary, scib_result)
    files["combined_summary"] = output / "combined_summary.csv"
    combined.to_csv(files["combined_summary"], index=False)

    report_metadata = {
        "method": config.method_name,
        "representation_key": config.representation_key,
        "n_cells": adata.n_obs,
        "n_dimensions": adata.obsm[config.representation_key].shape[1],
        "label_key": config.label_key,
        "batch_key": config.batch_key,
        "reference_resolution": config.reference_resolution,
        "n_neighbors": config.n_neighbors,
        "distance_metric": config.distance_metric,
        "random_state": config.random_state,
        "reference_cluster_key": reference_cluster_key,
        "scib_backend": scib_result.backend if scib_result else "disabled_or_failed",
        "scib_backend_version": scib_result.backend_version if scib_result else "n/a",
    }
    report_path = output / "report.html"
    report_figures: list[Path] = [heatmap, precision_recall, failure_counts]
    if scib_result is not None:
        report_figures.insert(0, scib_result.files["metric_plot"])
    write_html_report(
        report_path,
        title=f"scRareBench report — {config.method_name}",
        metadata=report_metadata,
        global_table=subsets,
        rare_table=rare,
        figure_names=report_figures,
        scib_metrics=scib_result.metrics_long if scib_result else None,
        scib_aggregates=scib_result.aggregate_scores if scib_result else None,
        scib_status=scib_result.metric_status if scib_result else None,
        rare_summary=rare_summary,
        scenario_table=scenario_summary,
    )
    files["report"] = report_path
    return EvaluationResult(
        output_dir=output,
        subset_metrics=subsets,
        per_type_metrics=all_per_type,
        rare_metrics=rare,
        rare_summary=rare_summary,
        scenario_metrics=scenario_summary,
        cluster_keys=clustering.cluster_keys,
        prediction_key=prediction_key,
        scib=scib_result,
        files=files,
    )
