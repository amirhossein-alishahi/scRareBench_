from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
import json
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .clustering import run_standard_clustering
from .constants import DEFAULT_BENCHMARK_SEED
from .failures import classify_failure_archetypes, load_failure_rules
from .metric_registry import METRIC_REGISTRY
from .metrics import (
    knn_local_recovery_from_graph,
    majority_vote_predictions,
    per_type_metrics,
    subset_metric_ratios,
    subset_metrics,
)
from .plotting import plot_failure_counts, plot_precision_recall, plot_rare_metric_heatmap
from .reporting import write_html_report, write_results_json
from .scenarios import SIX_SCENARIOS, load_paper_scenario_table, scenario_table_from_adata
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
    random_state: int = DEFAULT_BENCHMARK_SEED
    # Method-run provenance is intentionally separate from the benchmark seed.
    # method_seed may vary across stochastic training replicates while random_state
    # remains fixed so evaluation itself is deterministic/comparable.
    method_seed: int | None = None
    method_config: dict[str, Any] = field(default_factory=dict)
    rare_types: tuple[str, ...] | None = None
    leiden_flavor: str = "igraph"
    leiden_n_iterations: int = 2
    overwrite: bool = False
    rare_evaluation: bool = True
    scenario_policy: str = "require"  # require | paper_fallback
    strict_scenario_labels: bool = True
    scib: ScibEvaluationConfig = field(default_factory=ScibEvaluationConfig)


@dataclass
class EvaluationResult:
    output_dir: Path
    subset_metrics: pd.DataFrame
    subset_metric_ratios: pd.DataFrame
    per_type_metrics: pd.DataFrame
    rare_metrics: pd.DataFrame
    rare_summary: pd.DataFrame
    scenario_metrics: pd.DataFrame
    resolution_rare_metrics: pd.DataFrame
    cluster_keys: dict[float, str]
    prediction_key: str
    local_recovery_key: str | None
    local_recovery_adjusted_key: str | None
    scib: ScibEvaluationResult | None
    rare_evaluation_status: dict[str, Any]
    scib_status: dict[str, Any]
    run_config: dict[str, Any]
    files: dict[str, Path]


def _rare_summary(rare: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "precision",
        "recall",
        "f1",
        "inverse_purity",
        "best_cluster_f1",
        "knn_same_label_fraction",
        "knn_local_recovery_adjusted",
        "knn_local_recovery",
        "within_type_batch_nmi",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(rare[metric], errors="coerce") if metric in rare.columns else pd.Series(dtype=float)
        valid = values.dropna()
        rows.append({
            "metric": metric,
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "median": float(valid.median()) if len(valid) else np.nan,
            "minimum": float(valid.min()) if len(valid) else np.nan,
            "maximum": float(valid.max()) if len(valid) else np.nan,
            "n_valid": int(len(valid)),
        })
    failures = rare["failure_archetype"].value_counts(normalize=True) if "failure_archetype" in rare.columns and len(rare) else pd.Series(dtype=float)
    rows.append({
        "metric": "preserved_fraction",
        # This is a measured zero when no rare population is classified preserved,
        # not missing data. Preserve NaN only when rare evaluation itself is absent.
        "mean": float(failures.get("preserved", 0.0)) if len(rare) else np.nan,
        "median": np.nan,
        "minimum": np.nan,
        "maximum": np.nan,
        "n_valid": int(len(rare)),
    })

    # Resolution-limited is an additive v2 interpretation outcome, not a performance
    # score. Export its measured fraction so CSV consumers see the same headline
    # context that the dashboard/PDF expose. Empty rare evaluation remains missing.
    failures_v2 = (
        rare["failure_archetype_v2"].value_counts(normalize=True)
        if "failure_archetype_v2" in rare.columns and len(rare)
        else pd.Series(dtype=float)
    )
    rows.append({
        "metric": "resolution_limited_fraction",
        "mean": float(failures_v2.get("resolution_limited", 0.0)) if len(rare) else np.nan,
        "median": np.nan,
        "minimum": np.nan,
        "maximum": np.nan,
        "n_valid": int(len(rare)),
    })
    return pd.DataFrame(rows)


def _scenario_summary(rare: pd.DataFrame) -> pd.DataFrame:
    """Return one self-contained row per canonical rare scenario.

    Numeric recovery aggregates and the resolution-aware v2 archetype breakdown are
    exported together so downstream CSV/manuscript workflows do not need to rebuild
    taxonomy counts from the per-population file. Empty canonical scenario slots are
    retained explicitly; counts are zero while fractions remain missing because no
    population was evaluated in that slot.
    """
    numeric = [
        "precision",
        "recall",
        "f1",
        "inverse_purity",
        "best_cluster_f1",
        "knn_same_label_fraction",
        "knn_local_recovery_adjusted",
        "knn_local_recovery",
        "within_type_batch_nmi",
    ]
    v2_archetypes = [
        "preserved",
        "resolution_limited",
        "batch_driven_fragmentation",
        "lineage_leakage",
        "lineage_assimilation",
        "mixed_or_uncertain",
    ]

    if rare.empty or not {"scenario", "distribution", "topology"}.issubset(rare.columns):
        grouped = pd.DataFrame()
    else:
        available = [m for m in numeric if m in rare.columns]
        grouped = (
            rare.groupby(["scenario", "distribution", "topology"], dropna=False)[available]
            .agg(["mean", "median", "min", "max", "count"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(str(part) for part in column if str(part) != "").rstrip("_")
            if isinstance(column, tuple)
            else str(column)
            for column in grouped.columns
        ]

    by_scenario = {str(row["scenario"]): row for _, row in grouped.iterrows()} if not grouped.empty else {}
    rows: list[dict[str, Any]] = []
    for scenario in SIX_SCENARIOS:
        scenario_frame = rare[rare["scenario"].astype(str) == scenario].copy() if "scenario" in rare.columns else pd.DataFrame()
        if scenario in by_scenario:
            row = by_scenario[scenario].to_dict()
        else:
            distribution, topology = scenario.split("-", 1)
            row = {"scenario": scenario, "distribution": distribution, "topology": topology}
            for metric in numeric:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
                row[f"{metric}_count"] = 0

        n_types = int(len(scenario_frame))
        row["n_cell_types"] = n_types
        row["is_empty"] = n_types == 0

        if n_types and "failure_archetype_v2" in scenario_frame.columns:
            counts = scenario_frame["failure_archetype_v2"].astype(str).value_counts()
            for archetype in v2_archetypes:
                count = int(counts.get(archetype, 0))
                row[f"v2_{archetype}_count"] = count
                row[f"v2_{archetype}_fraction"] = count / n_types
        else:
            for archetype in v2_archetypes:
                row[f"v2_{archetype}_count"] = 0
                row[f"v2_{archetype}_fraction"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _resolution_table(adata: Any, cluster_keys: dict[float, str]) -> pd.DataFrame:
    rows = [{"resolution": float(resolution), "cluster_key": key, "n_clusters": int(adata.obs[key].astype(str).nunique())} for resolution, key in cluster_keys.items()]
    return pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)


def _resolution_rare_table(
    adata: Any,
    *,
    y_true: np.ndarray,
    batch_labels: np.ndarray,
    cluster_keys: dict[float, str],
    rare_types: list[str],
    metadata: pd.DataFrame | None,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """Sensitivity table for cluster-dependent rare metrics across resolutions."""
    if not rare_types:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for resolution, key in sorted(cluster_keys.items()):
        clusters = adata.obs[key].astype(str).to_numpy()
        predictions, _ = majority_vote_predictions(y_true, clusters)
        frame = per_type_metrics(y_true, clusters, predictions, batch_labels=batch_labels)
        frame = frame[frame["cell_type"].isin(rare_types)].copy()
        if metadata is not None and len(frame):
            frame = frame.merge(
                metadata[["cell_type", "scenario", "distribution", "topology", "parent_type", "curation_source"]],
                on="cell_type",
                how="left",
            )
        if len(frame):
            frame = classify_failure_archetypes(frame, rules=rules)
        frame.insert(0, "resolution", float(resolution))
        frame.insert(1, "cluster_key", key)
        frame.insert(2, "n_clusters", int(pd.Series(clusters).nunique()))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_package_versions(path: Path) -> None:
    packages = ["scrarebench", "scib-metrics", "scanpy", "anndata", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "plotly", "python-igraph", "leidenalg", "scvi-tools"]
    lines = [f"python={sys.version.split()[0]}", f"platform={platform.platform()}"]
    for package in packages:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not-installed"
        lines.append(f"{package}={version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combined_summary(rare_summary: pd.DataFrame, scib_result: ScibEvaluationResult | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scib_result is not None and not scib_result.aggregate_scores.empty:
        for record in scib_result.aggregate_scores.to_dict(orient="records"):
            rows.append({
                "section": "scib",
                "metric": str(record.get("metric", "")),
                "value": record.get("value", np.nan),
                "status": record.get("status", "computed"),
                "source": f"{record.get('backend', 'scib-metrics')} {record.get('backend_version', '')}".strip(),
            })
    for record in rare_summary.to_dict(orient="records"):
        rows.append({"section": "rare_cell", "metric": str(record["metric"]), "value": record.get("mean", np.nan), "status": "computed", "source": "scRareBench"})
    return pd.DataFrame(rows)


def _resolve_rare_metadata(adata: Any, config: EvaluationConfig, scenario_table: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not config.rare_evaluation:
        return None, {"attempted": False, "success": False, "status": "disabled", "source": None, "message": "Rare-cell evaluation was explicitly disabled."}
    if scenario_table is not None:
        metadata = scenario_table.copy()
        source = "explicit_argument"
    else:
        metadata = scenario_table_from_adata(adata)
        source = "adata.uns/scrarebench_scenario_table" if metadata is not None else None
        if metadata is None and config.rare_types:
            # Developer-friendly custom-dataset path: rare recovery metrics do not
            # require the six-scenario paper taxonomy. Scenario-specific summaries
            # remain empty while per-type/local recovery metrics are still valid.
            metadata = pd.DataFrame({
                "cell_type": list(dict.fromkeys(map(str, config.rare_types))),
                "scenario": "UNASSIGNED",
                "distribution": "",
                "topology": "",
                "parent_type": "",
                "curation_source": "user_rare_types",
            })
            source = "explicit_rare_types"
        elif metadata is None and config.scenario_policy == "paper_fallback":
            metadata = load_paper_scenario_table()
            source = "explicit_paper_fallback"
        elif metadata is None:
            raise ValueError(
                "Rare-cell evaluation requires dataset-specific scenario metadata or explicit rare_types. "
                "Pass scenario_table=..., set EvaluationConfig.rare_types, load a registered dataset that embeds scenario metadata, "
                "set rare_evaluation=False, or explicitly use scenario_policy='paper_fallback' only for the GSE194122 paper benchmark."
            )
    required = {"cell_type", "scenario", "distribution", "topology"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Scenario table is missing required columns: {sorted(missing)}")
    for optional in ("parent_type", "curation_source"):
        if optional not in metadata.columns:
            metadata[optional] = ""
    metadata = metadata.copy()
    metadata["cell_type"] = metadata["cell_type"].astype(str)
    metadata["scenario"] = metadata["scenario"].astype(str)

    # Validate scenario codes *before* filtering.  Earlier releases filtered
    # unknown codes first, which meant a one-character curation typo could make
    # a real population disappear with no warning.  Canonical/strict evaluation
    # now fails closed; exploratory mode records and drops the invalid rows.
    n_scenario_rows_input = int(len(metadata))
    allowed_scenario_codes = set(SIX_SCENARIOS) | {"UNASSIGNED", ""}
    invalid_mask = ~metadata["scenario"].isin(allowed_scenario_codes)
    invalid_rows_frame = metadata.loc[invalid_mask, ["cell_type", "scenario"]].copy()
    invalid_rows = invalid_rows_frame.to_dict(orient="records")
    invalid_codes = sorted(invalid_rows_frame["scenario"].astype(str).unique().tolist()) if len(invalid_rows_frame) else []
    if invalid_rows and config.strict_scenario_labels:
        preview = ", ".join(f"{row['cell_type']} -> {row['scenario']}" for row in invalid_rows[:8])
        if len(invalid_rows) > 8:
            preview += " ..."
        raise ValueError(
            "Scenario metadata contains invalid scenario code(s): "
            f"{preview}. Allowed values are: {', '.join(SIX_SCENARIOS)}, UNASSIGNED. "
            "For exploratory inspection only, set strict_scenario_labels=False to continue while recording and dropping invalid rows."
        )
    metadata = metadata.loc[~invalid_mask].reset_index(drop=True)

    # A curated scenario row naming a label that no longer exists in the loaded
    # data is a strong signal of source/annotation drift.  Such rows have no
    # evaluable cells, so strict mode fails closed and exploratory mode records
    # the drift explicitly.
    observed = set(adata.obs[config.label_key].astype(str).unique()) if config.label_key in adata.obs.columns else set()
    declared = metadata["cell_type"].astype(str).tolist()
    absent = sorted(set(declared).difference(observed)) if observed else []
    if absent and config.strict_scenario_labels:
        preview = ", ".join(absent[:8]) + (" ..." if len(absent) > 8 else "")
        raise ValueError(
            "Scenario metadata contains curated cell types that are absent from the loaded data: "
            f"{preview}. This usually indicates annotation/source drift. "
            "For exploratory inspection only, set strict_scenario_labels=False to continue while recording the drift."
        )

    drift = bool(absent or invalid_rows)
    return metadata, {
        "attempted": True,
        "success": True,
        "status": "available_with_warning" if drift else "available",
        "source": source,
        "n_scenario_rows_input": n_scenario_rows_input,
        "n_scenario_rows_valid": int(len(metadata)),
        "n_scenario_cell_types": int(len(metadata)),
        "n_scenario_cell_types_present": int(len(declared) - len(absent)),
        "invalid_scenario_rows": invalid_rows,
        "invalid_scenario_codes": invalid_codes,
        "scenario_code_warning": bool(invalid_rows),
        "scenario_cell_types_absent_from_data": absent,
        "annotation_drift_warning": drift,
        "strict_scenario_labels": bool(config.strict_scenario_labels),
        "scenario_policy": config.scenario_policy,
    }


def _read_status(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _neighbor_graph(adata: Any, *, neighbors_key: str, latent: np.ndarray, n_neighbors: int, metric: str) -> Any:
    """Return the benchmark kNN graph, with a deterministic small/test fallback."""
    uns = getattr(adata, "uns", {})
    obsp = getattr(adata, "obsp", {})
    config = uns.get(neighbors_key, {}) if isinstance(uns, dict) or hasattr(uns, "get") else {}
    for graph_field in ("distances_key", "connectivities_key"):
        key = config.get(graph_field) if isinstance(config, dict) else None
        if key and key in obsp:
            return obsp[key]
    # Used mainly by light-weight synthetic tests or third-party cluster adapters.
    # scanpy stores ``n_neighbors - 1`` graph neighbors per cell in the distances
    # matrix (self excluded); match that degree so ``knn_mean_neighbors`` is
    # comparable between the real and fallback graph paths.
    k = min(max(2, int(n_neighbors) - 1), max(2, len(latent) - 1))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(latent)), metric=metric, n_jobs=-1)
    nn.fit(latent)
    _, indices = nn.kneighbors(latent)
    rows: list[int] = []
    cols: list[int] = []
    for i, row in enumerate(indices):
        for j in row:
            if int(j) != i:
                rows.append(i)
                cols.append(int(j))
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(latent), len(latent)))


def _nonself_graph_degrees(graph: Any) -> np.ndarray:
    """Return realized per-row neighbor counts with diagonal/self edges excluded.

    The scientific kNN metric explicitly removes ``i -> i`` entries. Provenance
    must use the same denominator so a third-party graph containing self-loops
    cannot make ``run_config['knn_graph']`` disagree with the metric layer.
    """
    csr = graph.tocsr() if hasattr(graph, "tocsr") else None
    if csr is None or not hasattr(csr, "indptr") or not hasattr(csr, "indices"):
        return np.asarray([], dtype=int)
    degrees = np.zeros(csr.shape[0], dtype=int)
    for i in range(csr.shape[0]):
        start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
        indices = np.asarray(csr.indices[start:end], dtype=int)
        degrees[i] = int(np.count_nonzero(indices != i))
    return degrees


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
    latent = np.asarray(adata.obsm[config.representation_key])
    if latent.ndim != 2 or latent.shape[0] != int(adata.n_obs):
        raise ValueError(f"Integrated latent must be 2D with one row per cell; observed {latent.shape}, n_obs={adata.n_obs}.")
    if not np.isfinite(latent).all():
        raise ValueError("Integrated latent contains NaN or infinite values.")

    metadata, rare_status = _resolve_rare_metadata(adata, config, scenario_table)
    write_json(reproducibility_dir / "rare_evaluation_status.json", rare_status)
    write_json(reproducibility_dir / "metric_registry.json", METRIC_REGISTRY)

    resolutions = tuple(sorted(set(config.resolution_sweep + (config.reference_resolution,))))
    clustering = run_standard_clustering(
        adata,
        representation_key=config.representation_key,
        method_name=config.method_name,
        n_neighbors=config.n_neighbors,
        metric=config.distance_metric,
        resolutions=resolutions,
        random_state=config.random_state,
        leiden_flavor=config.leiden_flavor,
        leiden_n_iterations=config.leiden_n_iterations,
        overwrite=config.overwrite,
    )
    reference_cluster_key = clustering.cluster_keys[config.reference_resolution]
    y_true = adata.obs[config.label_key].astype(str).to_numpy()
    batch_labels = adata.obs[config.batch_key].astype(str).to_numpy()
    clusters = adata.obs[reference_cluster_key].astype(str).to_numpy()
    predictions, mapping = majority_vote_predictions(y_true, clusters)
    prediction_key = f"scrarebench_prediction_{slugify(config.method_name)}"
    adata.obs[prediction_key] = pd.Categorical(predictions)

    all_per_type = per_type_metrics(y_true, clusters, predictions, batch_labels=batch_labels)

    # Resolution-free local biological recovery from the package-controlled kNN graph.
    graph = _neighbor_graph(
        adata,
        neighbors_key=clustering.neighbors_key,
        latent=latent,
        n_neighbors=config.n_neighbors,
        metric=config.distance_metric,
    )
    knn_per_type, per_cell_same_label = knn_local_recovery_from_graph(y_true, graph)
    # The realized kNN degree is NOT always ``n_neighbors - 1``: scanpy switches
    # between exact and approximate neighbor backends depending on dataset size,
    # which changes the denominator of every kNN-based rare metric.  Record what
    # was actually used so the value is auditable and comparable across runs.
    _degrees = _nonself_graph_degrees(graph)
    _neighbors_cfg = getattr(adata, "uns", {}).get(clustering.neighbors_key, {}) if hasattr(getattr(adata, "uns", {}), "get") else {}
    _graph_source = "scanpy_neighbors_graph" if isinstance(_neighbors_cfg, dict) and any(_neighbors_cfg.get(k) for k in ("distances_key", "connectivities_key")) else "sklearn_fallback"
    knn_graph_stats = {
        "source": _graph_source,
        "requested_n_neighbors": int(config.n_neighbors),
        "realized_degree_mean": float(np.mean(_degrees)) if len(_degrees) else None,
        "realized_degree_min": int(np.min(_degrees)) if len(_degrees) else None,
        "realized_degree_max": int(np.max(_degrees)) if len(_degrees) else None,
    }
    all_per_type = all_per_type.merge(knn_per_type, on="cell_type", how="left")
    expected_by_type = knn_per_type.set_index("cell_type")["knn_expected_fraction"].to_dict()
    support_by_type = pd.Series(y_true, dtype="string").value_counts().to_dict()
    expected_per_cell = np.asarray([expected_by_type.get(str(label), np.nan) for label in y_true], dtype=float)
    local_recovery_per_cell = np.divide(
        per_cell_same_label - expected_per_cell,
        1.0 - expected_per_cell,
        out=np.full_like(per_cell_same_label, np.nan, dtype=float),
        where=np.isfinite(per_cell_same_label) & np.isfinite(expected_per_cell) & ((1.0 - expected_per_cell) > 0),
    )
    degrees_per_cell = _degrees.astype(float) if len(_degrees) == len(y_true) else np.full(len(y_true), np.nan)
    supports_per_cell = np.asarray([support_by_type.get(str(label), 0) for label in y_true], dtype=float)
    achievable_per_cell = np.divide(
        np.minimum(np.maximum(supports_per_cell - 1.0, 0.0), degrees_per_cell),
        degrees_per_cell,
        out=np.full(len(y_true), np.nan, dtype=float),
        where=np.isfinite(degrees_per_cell) & (degrees_per_cell > 0),
    )
    local_recovery_adjusted_per_cell = np.divide(
        per_cell_same_label - expected_per_cell,
        achievable_per_cell - expected_per_cell,
        out=np.full(len(y_true), np.nan, dtype=float),
        where=(
            np.isfinite(per_cell_same_label)
            & np.isfinite(expected_per_cell)
            & np.isfinite(achievable_per_cell)
            & ((achievable_per_cell - expected_per_cell) > 1e-12)
        ),
    )
    local_recovery_key = f"scrarebench_knn_local_recovery_{slugify(config.method_name)}"
    local_recovery_adjusted_key = f"scrarebench_knn_local_recovery_adjusted_{slugify(config.method_name)}"
    adata.obs[local_recovery_key] = local_recovery_per_cell
    adata.obs[local_recovery_adjusted_key] = local_recovery_adjusted_per_cell

    selected_rules = failure_rules or load_failure_rules()
    if metadata is None:
        rare_types: list[str] = []
        rare = pd.DataFrame(columns=list(all_per_type.columns) + [
            "scenario", "distribution", "topology", "parent_type", "curation_source",
            "failure_archetype", "failure_rationale", "failure_matched_archetypes", "failure_match_count",
            "failure_archetype_v2", "failure_rationale_v2", "failure_matched_archetypes_v2", "failure_match_count_v2",
        ])
    else:
        metadata_types = metadata["cell_type"].astype(str).tolist()
        if config.rare_types is not None:
            rare_types = list(dict.fromkeys(map(str, config.rare_types)))
            unknown_rare = sorted(set(rare_types).difference(set(map(str, y_true))))
            if unknown_rare:
                raise ValueError(f"rare_types contains labels absent from the dataset: {unknown_rare}")
            # Preserve explicit user selection even when the scenario table contains
            # additional curated populations. Unannotated explicit rare types receive
            # an UNASSIGNED metadata row so generic rare metrics remain available.
            missing_meta = [x for x in rare_types if x not in set(metadata_types)]
            if missing_meta:
                metadata = pd.concat([metadata, pd.DataFrame({
                    "cell_type": missing_meta, "scenario": "UNASSIGNED", "distribution": "",
                    "topology": "", "parent_type": "", "curation_source": "user_rare_types",
                })], ignore_index=True)
        else:
            rare_types = metadata_types
        rare = all_per_type[all_per_type["cell_type"].isin(rare_types)].copy()
        rare = rare.merge(metadata[["cell_type", "scenario", "distribution", "topology", "parent_type", "curation_source"]], on="cell_type", how="left")
        rare = classify_failure_archetypes(rare, rules=selected_rules) if len(rare) else rare.assign(
            failure_archetype=pd.Series(dtype=str),
            failure_rationale=pd.Series(dtype=str),
            failure_matched_archetypes=pd.Series(dtype=str),
            failure_match_count=pd.Series(dtype=int),
            failure_archetype_v2=pd.Series(dtype=str),
            failure_rationale_v2=pd.Series(dtype=str),
            failure_matched_archetypes_v2=pd.Series(dtype=str),
            failure_match_count_v2=pd.Series(dtype=int),
        )

    subsets = subset_metrics(latent, y_true, clusters, predictions, rare_types, random_state=config.random_state)
    ratios = subset_metric_ratios(subsets)
    rare_summary = _rare_summary(rare)
    scenario_summary = _scenario_summary(rare)
    resolution_summary = _resolution_table(adata, clustering.cluster_keys)
    resolution_rare = _resolution_rare_table(
        adata,
        y_true=y_true,
        batch_labels=batch_labels,
        cluster_keys=clustering.cluster_keys,
        rare_types=rare_types,
        metadata=metadata,
        rules=selected_rules,
    )

    files: dict[str, Path] = {}
    files["subset_metrics"] = output / "subset_metrics.csv"
    subsets.to_csv(files["subset_metrics"], index=False)
    files["subset_metric_ratios"] = output / "subset_metric_ratios.csv"
    ratios.to_csv(files["subset_metric_ratios"], index=False)
    files["per_type_metrics"] = output / "per_type_metrics.csv"
    all_per_type.to_csv(files["per_type_metrics"], index=False)
    files["rare_metrics"] = output / "rare_metrics.csv"
    rare.to_csv(files["rare_metrics"], index=False)
    files["rare_metrics_structured"] = rare_dir / "rare_metrics_per_type.csv"
    rare.to_csv(files["rare_metrics_structured"], index=False)
    files["rare_summary"] = rare_dir / "rare_metrics_summary.csv"
    rare_summary.to_csv(files["rare_summary"], index=False)
    files["scenario_metrics"] = rare_dir / "scenario_metrics.csv"
    scenario_summary.to_csv(files["scenario_metrics"], index=False)
    files["rare_resolution_sensitivity"] = rare_dir / "rare_resolution_sensitivity.csv"
    resolution_rare.to_csv(files["rare_resolution_sensitivity"], index=False)
    files["all_type_metrics"] = rare_dir / "all_cell_type_metrics.csv"
    all_per_type.to_csv(files["all_type_metrics"], index=False)

    files["cluster_mapping"] = clustering_dir / "cluster_majority_mapping.json"
    write_json(files["cluster_mapping"], mapping)
    files["clusters"] = clustering_dir / "clusters.csv"
    pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "cluster": clusters,
        "majority_vote_prediction": predictions,
        "knn_local_recovery_adjusted_cell": local_recovery_adjusted_per_cell,
        "knn_local_recovery_cell": local_recovery_per_cell,
    }).to_csv(files["clusters"], index=False)
    files["resolution_results"] = clustering_dir / "resolution_results.csv"
    resolution_summary.to_csv(files["resolution_results"], index=False)

    run_config = asdict(config)
    run_config["resolution_sweep"] = list(resolutions)
    run_config["reference_cluster_key"] = reference_cluster_key
    run_config["neighbors_key"] = clustering.neighbors_key
    run_config["benchmark_seed"] = config.random_state
    run_config["leiden_flavor_used"] = getattr(clustering, "leiden_flavor", config.leiden_flavor)
    run_config["leiden_n_iterations_used"] = getattr(clustering, "leiden_n_iterations", config.leiden_n_iterations)
    run_config["knn_graph"] = knn_graph_stats
    run_config["reference_n_clusters"] = int(pd.Series(clusters).nunique())
    run_config["n_reference_cell_types"] = int(pd.Series(y_true).nunique())
    run_config["cluster_count_warning"] = bool(run_config["reference_n_clusters"] < run_config["n_reference_cell_types"])
    files["run_config"] = reproducibility_dir / "run_config.yaml"
    files["run_config"].write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    files["failure_rules"] = reproducibility_dir / "failure_rules.yaml"
    files["failure_rules"].write_text(yaml.safe_dump(selected_rules, sort_keys=False), encoding="utf-8")
    files["package_versions"] = reproducibility_dir / "package_versions.txt"
    _write_package_versions(files["package_versions"])

    report_figures: list[Path] = []
    if len(rare):
        heatmap = plot_rare_metric_heatmap(rare, figures_dir / "rare_metric_heatmap.png")
        precision_recall = plot_precision_recall(rare, figures_dir / "rare_precision_recall.png")
        failure_column_v2 = "failure_archetype_v2" if "failure_archetype_v2" in rare.columns else "failure_archetype"
        failure_counts = plot_failure_counts(
            rare,
            figures_dir / "failure_archetype_counts_v2.png",
            column=failure_column_v2,
            title="Resolution-aware provisional failure-archetype counts",
        )
        legacy_failure_counts = plot_failure_counts(
            rare,
            figures_dir / "failure_archetype_counts_legacy.png",
            column="failure_archetype",
            title="Legacy majority-vote failure-archetype counts",
        )
        files["rare_metric_heatmap"] = heatmap
        files["rare_precision_recall"] = precision_recall
        files["failure_counts"] = failure_counts
        files["failure_counts_legacy"] = legacy_failure_counts
        report_figures.extend([heatmap, precision_recall, failure_counts, legacy_failure_counts])
        # The interactive dashboard already contains the Sankey explorer.  Avoid a
        # second self-contained Plotly HTML that would duplicate ~5 MB of JS.

    scib_result = run_scib_evaluation(adata, representation_key=config.representation_key, batch_key=config.batch_key, label_key=config.label_key, output_dir=scib_dir, config=config.scib)
    scib_status = _read_status(scib_dir / "scib_status.json", {"attempted": bool(config.scib.enabled), "success": scib_result is not None, "status": "computed" if scib_result else "unknown"})
    if scib_result is not None:
        for name, path in scib_result.files.items():
            files[f"scib_{name}"] = path
        if "metric_plot" in scib_result.files:
            report_figures.insert(0, scib_result.files["metric_plot"])

    combined = _combined_summary(rare_summary, scib_result)
    files["combined_summary"] = output / "combined_summary.csv"
    combined.to_csv(files["combined_summary"], index=False)

    report_metadata = {
        "method": config.method_name,
        "representation_key": config.representation_key,
        "n_cells": adata.n_obs,
        "n_dimensions": latent.shape[1],
        "label_key": config.label_key,
        "batch_key": config.batch_key,
        "reference_resolution": config.reference_resolution,
        "n_neighbors": config.n_neighbors,
        "distance_metric": config.distance_metric,
        "benchmark_seed": config.random_state,
        "reference_cluster_key": reference_cluster_key,
        "reference_n_clusters": int(pd.Series(clusters).nunique()),
        "n_reference_cell_types": int(pd.Series(y_true).nunique()),
        "cluster_count_warning": bool(pd.Series(clusters).nunique() < pd.Series(y_true).nunique()),
        "leiden_flavor": getattr(clustering, "leiden_flavor", config.leiden_flavor),
        "leiden_n_iterations": getattr(clustering, "leiden_n_iterations", config.leiden_n_iterations),
        "rare_evaluation_status": rare_status.get("status"),
        "scib_status": scib_status.get("status"),
        "scib_backend": scib_result.backend if scib_result else "disabled_or_failed",
        "scib_backend_version": scib_result.backend_version if scib_result else "n/a",
    }
    report_path = output / "report.html"
    write_html_report(
        report_path,
        title=f"scRareBench report — {config.method_name}",
        metadata=report_metadata,
        global_table=subsets,
        ratio_table=ratios,
        rare_table=rare,
        figure_names=report_figures,
        scib_metrics=scib_result.metrics_long if scib_result else None,
        scib_aggregates=scib_result.aggregate_scores if scib_result else None,
        scib_status=scib_result.metric_status if scib_result else None,
        rare_summary=rare_summary,
        scenario_table=scenario_summary,
        rare_execution_status=rare_status,
        scib_execution_status=scib_status,
    )
    files["report"] = report_path

    result = EvaluationResult(
        output_dir=output,
        subset_metrics=subsets,
        subset_metric_ratios=ratios,
        per_type_metrics=all_per_type,
        rare_metrics=rare,
        rare_summary=rare_summary,
        scenario_metrics=scenario_summary,
        resolution_rare_metrics=resolution_rare,
        cluster_keys=clustering.cluster_keys,
        prediction_key=prediction_key,
        local_recovery_key=local_recovery_key,
        local_recovery_adjusted_key=local_recovery_adjusted_key,
        scib=scib_result,
        rare_evaluation_status=rare_status,
        scib_status=scib_status,
        run_config=run_config,
        files=files,
    )
    files["results_json"] = output / "results.json"
    write_results_json(
        adata, result, files["results_json"],
        representation_key=config.representation_key, label_key=config.label_key, batch_key=config.batch_key,
        benchmark_config=run_config, method_seed=config.method_seed, method_config=config.method_config,
    )
    return result
