from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from .exceptions import MissingDependencyError
from .plotting import plot_scib_metric_scores
from .utils import write_json


SCIB_METRICS_BACKEND = "scib-metrics"
SCIB_METRICS_PIN = "0.5.9"
PREINTEGRATED_KEY = "X_scrarebench_unintegrated"


def _legacy_pandas_value_counts(
    values: Any,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins: int | None = None,
    dropna: bool = True,
):
    """Compatibility implementation for the pandas <3 top-level ``value_counts`` API.

    pandas 3 removed ``pandas.value_counts`` in favor of ``Series.value_counts``.
    scib-metrics 0.5.9 still calls the removed top-level function in its graph
    connectivity metric.  This helper mirrors the legacy call using the supported
    Series API without changing the numerical definition of the metric.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.value_counts(
        normalize=normalize,
        sort=sort,
        ascending=ascending,
        bins=bins,
        dropna=dropna,
    )


@contextmanager
def _scib_runtime_compatibility() -> Iterator[list[str]]:
    """Temporarily bridge known upstream API removals during scib-metrics execution.

    The compatibility layer is deliberately scoped to the benchmark call.  It does
    not pin/downgrade pandas and it restores the pandas module afterwards.  On
    environments where the legacy API still exists, this context is a no-op.
    """
    adjustments: list[str] = []
    added_value_counts = False
    if not hasattr(pd, "value_counts"):
        pd.value_counts = _legacy_pandas_value_counts  # type: ignore[attr-defined]
        added_value_counts = True
        adjustments.append(
            "pandas>=3 compatibility: temporarily mapped pandas.value_counts to Series.value_counts "
            "for scib-metrics graph_connectivity"
        )
    try:
        yield adjustments
    finally:
        if added_value_counts and getattr(pd, "value_counts", None) is _legacy_pandas_value_counts:
            del pd.value_counts  # type: ignore[attr-defined]


def _runtime_versions(backend_version: str) -> dict[str, str]:
    """Collect compact version diagnostics for reproducibility and support."""
    import platform
    import sys

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scib_metrics": backend_version,
    }
    for package, key in (("scanpy", "scanpy"), ("anndata", "anndata"), ("jax", "jax")):
        try:
            versions[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


@dataclass(frozen=True)
class ScibEvaluationConfig:
    """Configuration for the standard scIB-compatible evaluation layer.

    ``hvg_batch_mode`` controls only the benchmark reference HVG selection; it
    does not change the batch labels used by scIB metrics themselves.  The
    default preserves the historical behavior (select HVGs within the
    evaluation batch).  ``"global"`` is useful for datasets where Seurat-v3
    within-batch LOESS is numerically singular, while keeping the same
    evaluation ``batch_key`` for all downstream batch metrics.
    """

    enabled: bool = True
    count_layer: str | None = "counts"
    n_hvg: int = 4000
    reference_n_pcs: int = 50
    target_sum: float = 10_000.0
    hvg_flavor: str = "seurat_v3"
    hvg_batch_mode: str = "evaluation_batch"
    n_jobs: int = 1
    progress_bar: bool = True
    solver: str = "arpack"
    min_max_scale: bool = False
    include_silhouette_batch: bool = True
    require_backend: bool = True
    canonical: bool = True
    allow_hvg_fallback: bool = False


@dataclass
class ScibEvaluationResult:
    backend: str
    backend_version: str
    results_wide: pd.DataFrame
    metrics_long: pd.DataFrame
    aggregate_scores: pd.DataFrame
    metric_status: pd.DataFrame
    reference_config: dict[str, Any]
    files: dict[str, Path]


def _require_scanpy_anndata():
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "The scIB-compatible layer requires anndata and scanpy."
        ) from exc
    return ad, sc


def _require_scib_metrics():
    try:
        import scib_metrics
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "The standard scIB-compatible layer requires scib-metrics. "
            "Install scrarebench with its full dependencies."
        ) from exc
    try:
        version = metadata.version("scib-metrics")
    except metadata.PackageNotFoundError:  # pragma: no cover
        version = getattr(scib_metrics, "__version__", "unknown")
    return scib_metrics, Benchmarker, BioConservation, BatchCorrection, version


def _feature_mask(adata: Any) -> np.ndarray:
    """Select gene-expression features when feature-type metadata is available."""
    for key in ("feature_types", "feature_type", "modality"):
        if key not in adata.var.columns:
            continue
        values = adata.var[key].astype(str).str.lower()
        mask = values.str.contains("gene expression|gex|rna", regex=True).to_numpy()
        if mask.any():
            return mask
    return np.ones(adata.n_vars, dtype=bool)


def _copy_matrix(matrix: Any):
    return matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)


def _resolve_hvg_batch_key(*, evaluation_batch_key: str, mode: str) -> str | None:
    """Resolve the batch key used only for scIB reference HVG selection."""
    normalized = str(mode).strip().lower()
    if normalized == "evaluation_batch":
        return evaluation_batch_key
    if normalized == "global":
        return None
    raise ValueError(
        "ScibEvaluationConfig.hvg_batch_mode must be 'evaluation_batch' or 'global'; "
        f"got {mode!r}."
    )


def prepare_scib_reference(
    adata: Any,
    *,
    representation_key: str,
    batch_key: str,
    label_key: str,
    config: ScibEvaluationConfig,
):
    """Create a benchmark-only normalized reference without changing user preprocessing.

    The returned object contains only GEX features and selected HVGs. Its ``X`` is
    normalized/log-transformed and ``obsm[PREINTEGRATED_KEY]`` stores the canonical
    unintegrated PCA used by PCR comparison. The submitted integration latent is
    copied into ``obsm[representation_key]`` with the original cell order.
    """
    ad, sc = _require_scanpy_anndata()
    for key in (batch_key, label_key):
        if key not in adata.obs.columns:
            raise KeyError(f"adata.obs['{key}'] is required for scIB metrics")
    if representation_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{representation_key}'] is required")

    mask = _feature_mask(adata)
    source_name = "X"
    source = adata.X
    if config.count_layer:
        if config.count_layer in adata.layers:
            source_name = f"layers/{config.count_layer}"
            source = adata.layers[config.count_layer]
        elif config.canonical:
            raise KeyError(
                f"Canonical scIB reference requires raw/count-like input in "
                f"adata.layers[{config.count_layer!r}], but that layer is missing. "
                "Set canonical=False only for an explicitly non-canonical exploratory run."
            )
    matrix = _copy_matrix(source[:, mask])
    var = adata.var.iloc[np.flatnonzero(mask)].copy()
    obs = adata.obs.copy()
    reference = ad.AnnData(X=matrix, obs=obs, var=var)

    n_hvg = min(int(config.n_hvg), reference.n_vars)
    hvg_batch_key = _resolve_hvg_batch_key(
        evaluation_batch_key=batch_key,
        mode=config.hvg_batch_mode,
    )
    hvg_kwargs: dict[str, Any] = {
        "n_top_genes": n_hvg,
        "flavor": config.hvg_flavor,
        "subset": False,
    }
    if hvg_batch_key is not None:
        hvg_kwargs["batch_key"] = hvg_batch_key
    try:
        sc.pp.highly_variable_genes(reference, **hvg_kwargs)
        hvg_method = config.hvg_flavor
        hvg_fallback_used = False
    except (ImportError, ModuleNotFoundError) as exc:
        if config.canonical or not config.allow_hvg_fallback:
            raise MissingDependencyError(
                "Canonical scIB reference HVG selection failed. Install scikit-misc "
                "for seurat_v3, or explicitly use canonical=False with "
                "allow_hvg_fallback=True for exploratory/non-canonical evaluation."
            ) from exc
        sc.pp.normalize_total(reference, target_sum=config.target_sum)
        sc.pp.log1p(reference)
        fallback_kwargs: dict[str, Any] = {
            "n_top_genes": n_hvg,
            "flavor": "seurat",
            "subset": False,
        }
        if hvg_batch_key is not None:
            fallback_kwargs["batch_key"] = hvg_batch_key
        sc.pp.highly_variable_genes(reference, **fallback_kwargs)
        hvg_method = "seurat_fallback"
        hvg_fallback_used = True
        already_normalized = True
    else:
        already_normalized = False

    hvg_mask = reference.var["highly_variable"].to_numpy(dtype=bool)
    if not hvg_mask.any():
        raise ValueError("HVG selection returned no features")
    reference = reference[:, hvg_mask].copy()
    reference.var["highly_variable"] = True
    if not already_normalized:
        sc.pp.normalize_total(reference, target_sum=config.target_sum)
        sc.pp.log1p(reference)

    max_pcs = min(int(config.reference_n_pcs), reference.n_obs - 1, reference.n_vars - 1)
    if max_pcs < 2:
        raise ValueError("At least two PCA components are required for scIB reference preparation")
    sc.tl.pca(reference, n_comps=max_pcs, svd_solver=config.solver, use_highly_variable=False)
    reference.obsm[PREINTEGRATED_KEY] = np.asarray(reference.obsm["X_pca"], dtype=np.float32).copy()
    integrated = np.asarray(adata.obsm[representation_key], dtype=np.float32)
    if integrated.shape[0] != reference.n_obs:
        raise ValueError(
            f"Integrated latent has {integrated.shape[0]} rows; expected {reference.n_obs}."
        )
    reference.obsm[representation_key] = integrated.copy()

    reference_config = {
        "input_matrix": source_name,
        "gex_features_before_hvg": int(mask.sum()),
        "selected_hvgs": int(reference.n_vars),
        "hvg_method": hvg_method,
        "hvg_batch_mode": config.hvg_batch_mode,
        "hvg_batch_key": hvg_batch_key,
        "evaluation_batch_key": batch_key,
        "requested_hvgs": int(config.n_hvg),
        "normalization_target_sum": float(config.target_sum),
        "log1p": True,
        "reference_pcs": int(max_pcs),
        "preintegrated_embedding_key": PREINTEGRATED_KEY,
        "integrated_embedding_key": representation_key,
        "cell_order_preserved": bool(reference.obs_names.equals(adata.obs_names)),
        "canonical_benchmark": bool(config.canonical and not hvg_fallback_used),
        "hvg_fallback_used": bool(hvg_fallback_used),
    }
    reference.uns["scrarebench_scib_reference"] = reference_config
    return reference, reference_config


def _status_catalog() -> pd.DataFrame:
    rows = [
        ("Isolated labels", "Bio conservation", "scib-metrics", "supported", "Current scib-metrics isolated-label score."),
        ("Leiden NMI", "Bio conservation", "scib-metrics", "supported", "Leiden clustering NMI from scib-metrics."),
        ("Leiden ARI", "Bio conservation", "scib-metrics", "supported", "Leiden clustering ARI from scib-metrics."),
        ("KMeans NMI", "Bio conservation", "scib-metrics", "supported", "K-means clustering NMI from scib-metrics."),
        ("KMeans ARI", "Bio conservation", "scib-metrics", "supported", "K-means clustering ARI from scib-metrics."),
        ("Silhouette label", "Bio conservation", "scib-metrics", "supported", "Cell-type silhouette score."),
        ("cLISI", "Bio conservation", "scib-metrics", "supported", "Cell-type LISI."),
        ("BRAS", "Batch correction", "scib-metrics", "supported", "Batch-removal adapted silhouette."),
        ("Silhouette batch", "Batch correction", "scib-metrics supplement", "supported", "Classic scIB batch ASW, computed separately."),
        ("iLISI", "Batch correction", "scib-metrics", "supported", "Integration LISI."),
        ("KBET", "Batch correction", "scib-metrics", "supported", "kBET per label."),
        ("Graph connectivity", "Batch correction", "scib-metrics", "supported", "Graph connectivity."),
        ("PCR comparison", "Batch correction", "scib-metrics", "supported", "PCR before versus after integration."),
        ("Bio conservation", "Aggregate score", "scib-metrics", "supported", "Mean of the configured biological-conservation metrics."),
        ("Batch correction", "Aggregate score", "scib-metrics", "supported", "Mean of the configured batch-correction metrics."),
        ("Total", "Aggregate score", "scib-metrics", "supported", "scib-metrics weighted aggregate score."),
        ("HVG overlap", "Legacy scIB", "requires corrected expression", "not_applicable", "A latent-only submission does not provide a corrected gene-expression matrix."),
        ("Cell cycle conservation", "Legacy scIB", "requires expression and organism-specific cell-cycle genes", "not_applicable", "Not identifiable from a latent alone."),
        ("Trajectory conservation", "Legacy scIB", "requires curated pseudotime", "not_applicable", "No reference pseudotime is part of the version-1 benchmark contract."),
        ("Moran's I", "Legacy scIB", "requires gene-level representation", "not_applicable", "Not identifiable from a latent alone."),
    ]
    return pd.DataFrame(rows, columns=["metric", "metric_type", "implementation", "status", "reason"])


def _parse_benchmarker_results(
    frame: pd.DataFrame,
    *,
    representation_key: str,
    backend_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if representation_key not in frame.index:
        available = [str(value) for value in frame.index]
        raise KeyError(
            f"scib-metrics results do not contain row '{representation_key}'. Available: {available}"
        )
    if "Metric Type" not in frame.index:
        raise KeyError("scib-metrics result is missing the 'Metric Type' row")
    values = frame.loc[representation_key]
    metric_types = frame.loc["Metric Type"]
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for metric in frame.columns:
        raw_value = values[metric]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        item = {
            "method": representation_key,
            "metric": str(metric),
            "value": value,
            "metric_type": str(metric_types[metric]),
            "backend": SCIB_METRICS_BACKEND,
            "backend_version": backend_version,
            "status": "computed" if np.isfinite(value) else "failed",
        }
        if str(metric_types[metric]) == "Aggregate score":
            aggregates.append(item)
        else:
            rows.append(item)
    return pd.DataFrame(rows), pd.DataFrame(aggregates)


def run_scib_evaluation(
    adata: Any,
    *,
    representation_key: str,
    batch_key: str,
    label_key: str,
    output_dir: str | Path,
    config: ScibEvaluationConfig,
) -> ScibEvaluationResult | None:
    """Run the current scib-metrics benchmark and always preserve execution status."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "scib_status.json"
    if not config.enabled:
        write_json(status_path, {
            "attempted": False,
            "success": False,
            "status": "disabled",
            "message": "scIB-compatible evaluation was explicitly disabled.",
            "config": asdict(config),
        })
        return None

    try:
        scib_metrics, Benchmarker, BioConservation, BatchCorrection, backend_version = _require_scib_metrics()
        reference, reference_config = prepare_scib_reference(
            adata,
            representation_key=representation_key,
            batch_key=batch_key,
            label_key=label_key,
            config=config,
        )
        bio = BioConservation(
            isolated_labels=True,
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=True,
        )
        batch = BatchCorrection(
            bras=True,
            ilisi_knn=True,
            kbet_per_label=True,
            graph_connectivity=True,
            pcr_comparison=True,
        )
        benchmarker = Benchmarker(
            reference,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=[representation_key],
            bio_conservation_metrics=bio,
            batch_correction_metrics=batch,
            pre_integrated_embedding_obsm_key=PREINTEGRATED_KEY,
            n_jobs=config.n_jobs,
            progress_bar=config.progress_bar,
            solver=config.solver,
        )
        with _scib_runtime_compatibility() as compatibility_adjustments:
            benchmarker.benchmark()
        wide = benchmarker.get_results(min_max_scale=config.min_max_scale, clean_names=True)
        metrics_long, aggregates = _parse_benchmarker_results(
            wide, representation_key=representation_key, backend_version=backend_version
        )

        if config.include_silhouette_batch:
            batch_asw = float(scib_metrics.silhouette_batch(
                np.asarray(reference.obsm[representation_key]),
                np.asarray(reference.obs[label_key]),
                np.asarray(reference.obs[batch_key]),
                rescale=True,
            ))
            metrics_long = pd.concat([metrics_long, pd.DataFrame([{
                "method": representation_key,
                "metric": "Silhouette batch",
                "value": batch_asw,
                "metric_type": "Batch correction supplement",
                "backend": SCIB_METRICS_BACKEND,
                "backend_version": backend_version,
                "status": "computed" if np.isfinite(batch_asw) else "failed",
            }])], ignore_index=True)

        status = _status_catalog()
        computed_names = set(metrics_long["metric"].astype(str)) | set(aggregates["metric"].astype(str))
        status.loc[status["metric"].isin(computed_names), "status"] = "computed"
        status.loc[status["metric"].isin(computed_names), "reason"] = "Computed successfully."
        status["backend_version"] = backend_version

        files: dict[str, Path] = {
            "results_wide": output / "scib_results_wide.csv",
            "metrics_long": output / "scib_metrics.csv",
            "aggregate_scores": output / "scib_aggregate_scores.csv",
            "metric_status": output / "scib_metric_status.csv",
            "reference_config": output / "scib_reference_config.yaml",
            "metric_plot": output / "scib_metric_scores.png",
            "status": status_path,
        }
        wide.to_csv(files["results_wide"], index=True)
        metrics_long.to_csv(files["metrics_long"], index=False)
        aggregates.to_csv(files["aggregate_scores"], index=False)
        status.to_csv(files["metric_status"], index=False)
        reference_payload = {
            **reference_config,
            "scib_config": asdict(config),
            "backend": SCIB_METRICS_BACKEND,
            "backend_version": backend_version,
            "backend_pin_used_by_scrarebench": SCIB_METRICS_PIN,
            "runtime_versions": _runtime_versions(backend_version),
            "runtime_compatibility_adjustments": compatibility_adjustments,
            "compatibility_note": (
                "scRareBench uses an automatic, scoped runtime compatibility bridge for known "
                "upstream API removals. The bridge is activated only when required and does not "
                "change metric definitions. Values are produced by scib-metrics and must not be "
                "numerically compared with the legacy scib repository without qualification."
            ),
        }
        files["reference_config"].write_text(yaml.safe_dump(reference_payload, sort_keys=False), encoding="utf-8")
        plot_scib_metric_scores(metrics_long, aggregates, files["metric_plot"])
        write_json(status_path, {
            "attempted": True,
            "success": True,
            "status": "computed",
            "message": "scIB-compatible evaluation completed successfully.",
            "backend": SCIB_METRICS_BACKEND,
            "backend_version": backend_version,
            "canonical_benchmark": bool(reference_config.get("canonical_benchmark", False)),
            "config": asdict(config),
        })
        return ScibEvaluationResult(
            backend=SCIB_METRICS_BACKEND,
            backend_version=backend_version,
            results_wide=wide,
            metrics_long=metrics_long,
            aggregate_scores=aggregates,
            metric_status=status,
            reference_config=reference_payload,
            files=files,
        )
    except Exception as exc:
        write_json(status_path, {
            "attempted": True,
            "success": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "message": "scIB-compatible evaluation failed; see error_type/error_message.",
            "config": asdict(config),
        })
        if config.require_backend:
            raise
        return None

