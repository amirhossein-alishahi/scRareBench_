from __future__ import annotations

import base64
import html
import json
import mimetypes
import hashlib
from pathlib import Path
from importlib.resources import files as resource_files
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import slugify
from .metric_registry import METRIC_REGISTRY
from .failures import load_failure_rules


from .constants import DEFAULT_BENCHMARK_SEED
from .multiseed import (
    canonicalize_embedded_run, dataset_contract_hash, evaluation_contract_hash,
    extract_embedded_report_payload, make_multirun_container, make_run_id,
    method_training_hash, normalize_method_seeds, validate_compatible_run,
)

def _records(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"columns": [], "rows": []}
    clean = frame.copy()
    clean.columns = [str(c) for c in clean.columns]
    clean = clean.replace({np.nan: None, np.inf: None, -np.inf: None})
    rows = []
    for record in clean.to_dict(orient="records"):
        rows.append({str(k): _json_scalar(v) for k, v in record.items()})
    return {"columns": list(clean.columns), "rows": rows}


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _category_encoding(values: Sequence[Any]) -> dict[str, Any]:
    """Compact categorical encoding for self-contained HTML payloads."""
    series = pd.Series(values, dtype="object").fillna("missing").astype(str)
    categories = list(pd.Index(series).unique())
    mapping = {value: idx for idx, value in enumerate(categories)}
    raw = np.asarray([mapping[value] for value in series], dtype=np.uint32)
    if len(categories) <= np.iinfo(np.uint8).max + 1:
        codes = raw.astype(np.uint8); dtype = "uint8"
    elif len(categories) <= np.iinfo(np.uint16).max + 1:
        codes = raw.astype(np.uint16); dtype = "uint16"
    else:
        codes = raw; dtype = "uint32"
    return {
        "categories": categories,
        "codes_b64": base64.b64encode(codes.tobytes()).decode("ascii"),
        "dtype": dtype,
        "n": int(len(codes)),
    }


def _coordinate_encoding(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"data_b64": "", "minimum": 0.0, "maximum": 0.0, "n": int(len(array)), "dtype": "uint16"}
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        codes = np.zeros(len(array), dtype=np.uint16)
    else:
        normalized = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
        codes = np.rint(normalized * 65535.0).astype(np.uint16)
    return {
        "data_b64": base64.b64encode(codes.tobytes()).decode("ascii"),
        "minimum": lo, "maximum": hi, "n": int(len(array)), "dtype": "uint16",
    }


def _string_values(obs: pd.DataFrame, column: str | None, *, missing: str = "unknown", empty_as_missing: bool = False) -> list[str]:
    if not column or column not in obs.columns:
        return [missing] * len(obs)
    # Important: convert Categorical to nullable string BEFORE fillna.
    values = obs[column].astype("string").fillna(missing)
    if empty_as_missing:
        values = values.replace("", missing)
    return values.astype(str).tolist()


def _pick_cluster_key(result: Any) -> str | None:
    cluster_keys = getattr(result, "cluster_keys", None)
    if not cluster_keys:
        return None
    try:
        # Prefer resolution 1.0 when available, otherwise the smallest resolution.
        if 1.0 in cluster_keys:
            return cluster_keys[1.0]
        return cluster_keys[sorted(cluster_keys.keys())[0]]
    except Exception:
        return next(iter(cluster_keys.values()))


def _coordinates(
    adata: Any,
    representation_key: str,
    *,
    umap_key: str | None,
    random_state: int,
) -> tuple[np.ndarray, str, str]:
    if umap_key and umap_key in adata.obsm:
        coords = np.asarray(adata.obsm[umap_key], dtype=float)
        return coords[:, :2], umap_key, "UMAP"
    # Prefer any explicitly saved scRareBench/scVI UMAP before generic X_umap.
    for key in ("X_umap_scVI", "X_umap_scrarebench_interactive", "X_umap"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=float)
            return coords[:, :2], key, "UMAP"
    try:
        import scanpy as sc  # type: ignore

        import anndata as ad  # type: ignore

        working_key = "X_umap_scrarebench_interactive"
        neighbors_key = f"scrarebench_neighbors_{slugify(representation_key)}_interactive"
        # Only the latent representation is needed to build the neighbor graph and
        # UMAP.  A full adata.copy() would double peak memory for large datasets
        # purely to render a report, so build a minimal view-free container that
        # still leaves the caller's AnnData untouched.
        representation = np.asarray(adata.obsm[representation_key], dtype=np.float32)
        names = getattr(adata, "obs_names", None)
        index = (
            pd.Index([str(value) for value in names])
            if names is not None
            else pd.RangeIndex(representation.shape[0]).astype(str)
        )
        working = ad.AnnData(obs=pd.DataFrame(index=index))
        working.obsm[representation_key] = representation
        sc.pp.neighbors(
            working,
            use_rep=representation_key,
            n_neighbors=15,
            metric="euclidean",
            key_added=neighbors_key,
            random_state=random_state,
        )
        sc.tl.umap(working, neighbors_key=neighbors_key, random_state=random_state)
        return np.asarray(working.obsm["X_umap"], dtype=float)[:, :2], working_key, "UMAP"
    except Exception as exc:
        coords = np.asarray(adata.obsm[representation_key], dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("Interactive report requires a two-dimensional embedding or a representation with at least two dimensions.") from exc
        return coords[:, :2], representation_key, "Latent projection (dims 1–2; UMAP unavailable)"


def _read_text_file(path: Any) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _image_data_uri(path: Any) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return None
    mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(candidate.read_bytes()).decode('ascii')}"






def _marker_gene_payload(adata: Any, genes: Sequence[str] | None, *, layer: str | None, max_genes: int = 50) -> dict[str, Any]:
    """Quantize selected marker-gene vectors to compact base64 Uint8 payloads."""
    if not genes:
        return {"genes": [], "source": layer or "X", "encoding": "uint8_base64", "rows": {}}
    requested = []
    seen = set()
    for gene in genes:
        value = str(gene)
        if value and value not in seen:
            requested.append(value); seen.add(value)
        if len(requested) >= int(max_genes):
            break
    var_names = pd.Index(getattr(adata, "var_names", getattr(getattr(adata, "var", None), "index", []))).astype(str)
    matrix = adata.layers[layer] if layer and hasattr(adata, "layers") and layer in adata.layers else adata.X
    rows: dict[str, Any] = {}
    for gene in requested:
        loc = np.flatnonzero(var_names.to_numpy() == gene)
        if not len(loc):
            continue
        vector = matrix[:, int(loc[0])]
        if hasattr(vector, "toarray"):
            vector = vector.toarray()
        values = np.asarray(vector, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        lo = float(np.quantile(finite, 0.01))
        hi = float(np.quantile(finite, 0.99))
        if not np.isfinite(lo): lo = float(np.nanmin(finite))
        if not np.isfinite(hi): hi = float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(values, lo, hi)
        quantized = np.rint((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        rows[gene] = {
            "data_b64": base64.b64encode(quantized.tobytes()).decode("ascii"),
            "q01": lo, "q99": hi, "n": int(len(values)),
        }
    return {"genes": list(rows), "source": layer or "X", "encoding": "uint8_base64_q01_q99", "rows": rows}


def _failure_rules_payload(result: Any) -> dict[str, Any]:
    path = getattr(result, "files", {}).get("failure_rules") if getattr(result, "files", None) else None
    if path and Path(path).exists():
        try:
            import yaml
            value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return load_failure_rules()

def _rare_category_breakdown(rare_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the six curated rare scenarios for dashboard comparison."""
    if rare_frame is None or rare_frame.empty or "scenario" not in rare_frame.columns:
        return []
    scenario_order = ["GR-DL", "GR-RM", "LE-DL", "LE-RM", "SR-DL", "SR-RM"]
    metrics = ["knn_local_recovery_adjusted", "knn_local_recovery", "knn_same_label_fraction", "knn_max_achievable_fraction", "best_cluster_f1", "precision", "recall", "f1", "inverse_purity", "within_type_batch_nmi"]
    out: list[dict[str, Any]] = []
    for scenario in scenario_order:
        frame = rare_frame[rare_frame["scenario"].astype(str) == scenario].copy()
        failures: list[dict[str, Any]] = []
        failure_column = "failure_archetype_v2" if "failure_archetype_v2" in frame.columns else "failure_archetype"
        if not frame.empty and failure_column in frame.columns:
            for failure, group in frame.groupby(failure_column, dropna=False):
                failures.append({
                    "failure_archetype": str(failure),
                    "count": int(len(group)),
                    "cell_types": sorted(group["cell_type"].astype(str).tolist()),
                })
            failures.sort(key=lambda row: (-row["count"], row["failure_archetype"]))
        row: dict[str, Any] = {
            "scenario": scenario,
            "distribution": (
                str(frame["distribution"].iloc[0])
                if not frame.empty and "distribution" in frame.columns
                else scenario.split("-")[0]
            ),
            "topology": (
                str(frame["topology"].iloc[0])
                if not frame.empty and "topology" in frame.columns
                else scenario.split("-")[1]
            ),
            "n_cell_types": int(len(frame)),
            "total_cells": int(pd.to_numeric(frame.get("support", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "cell_types": sorted(frame["cell_type"].astype(str).tolist()) if not frame.empty else [],
            "failures": failures,
            "rows": _records(frame)["rows"],
            "is_empty": bool(frame.empty),
        }
        for metric in metrics:
            if metric in frame.columns and not frame.empty:
                values = pd.to_numeric(frame[metric], errors="coerce")
                finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
                if len(finite):
                    row[f"{metric}_mean"] = _json_scalar(finite.mean())
                    row[f"{metric}_median"] = _json_scalar(finite.median())
                else:
                    # A singleton rare population legitimately has no support-adjusted
                    # kNN estimate. Avoid NumPy/Pandas all-NaN reduction warnings and
                    # preserve the scientific contract: not assessable is missing,
                    # never silently converted to zero.
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_median"] = None
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_median"] = None
        if failure_column in frame.columns and not frame.empty:
            preserved = int((frame[failure_column].astype(str) == "preserved").sum())
            row["preserved_count"] = preserved
            row["preserved_fraction"] = preserved / len(frame)
        else:
            row["preserved_count"] = 0
            row["preserved_fraction"] = None
        out.append(row)
    return out

def _collect_figures(result: Any, *, include_static_figures: bool) -> list[dict[str, str]]:
    if not include_static_figures:
        return []
    title_map = {
        "rare_metric_heatmap": "Rare-cell metric heatmap",
        "rare_precision_recall": "Precision–recall profile of curated rare populations",
        "failure_counts": "Resolution-aware failure-archetype distribution",
        "failure_counts_legacy": "Legacy majority-vote failure-archetype distribution",
        "scib_metric_plot": "scIB-compatible metric scores",
        "umap_scvi_cell_types": "UMAP colored by reference cell type",
        "umap_scvi_batches": "UMAP colored by batch",
        "umap_scvi_rare_scenarios": "UMAP colored by rare-cell scenario",
        "metric_plot": "scIB-compatible metric scores",
    }
    figures: list[dict[str, str]] = []
    seen: set[Path] = set()
    def add(key: str, path: Any, group: str) -> None:
        candidate = Path(path) if path else None
        if not candidate or candidate in seen:
            return
        uri = _image_data_uri(candidate)
        if not uri:
            return
        seen.add(candidate)
        stem_key = candidate.stem.lower()
        key_norm = str(key).replace("scib_", "", 1)
        title = title_map.get(str(key)) or title_map.get(key_norm)
        if title is None:
            if "umap" in stem_key and "batch" in stem_key:
                title = "UMAP colored by batch"
            elif "umap" in stem_key and "rare" in stem_key:
                title = "UMAP colored by rare-cell scenario"
            elif "umap" in stem_key:
                title = "UMAP embedding"
            else:
                title = candidate.stem.replace("_", " ").strip().title()
        figures.append({
            "key": str(key), "name": candidate.name, "title": title,
            "group": group, "uri": uri,
        })
    for key, path in getattr(result, "files", {}).items():
        add(str(key), path, "scIB" if str(key).startswith("scib_") else "Rare-cell / benchmark")
    scib = getattr(result, "scib", None)
    if scib is not None:
        for key, path in getattr(scib, "files", {}).items():
            add(f"scib_{key}", path, "scIB")
    return figures


def _build_payload(
    adata: Any,
    result: Any,
    *,
    representation_key: str,
    label_key: str,
    batch_key: str,
    scenario_key: str,
    umap_key: str | None,
    random_state: int,
    include_overview: bool,
    include_metrics: bool,
    include_scib: bool,
    include_rare: bool,
    include_rare_umap: bool,
    include_rare_heatmaps: bool,
    include_rare_scenario_analysis: bool,
    include_umap: bool,
    include_sankey: bool,
    include_reproducibility: bool,
    include_static_figures: bool,
    include_cell_ids: bool,
    marker_genes: Sequence[str] | None = None,
    marker_layer: str | None = None,
) -> dict[str, Any]:
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    cluster_key = _pick_cluster_key(result)
    prediction_key = getattr(result, "prediction_key", None)
    label_values = _string_values(obs, label_key)
    batch_values = _string_values(obs, batch_key)
    scenario_values = _string_values(obs, scenario_key, missing="non_rare", empty_as_missing=True)
    cluster_values = _string_values(obs, cluster_key)
    prediction_values = _string_values(obs, prediction_key)

    rare_frame = getattr(result, "rare_metrics", pd.DataFrame()).copy()
    rare_types = sorted(rare_frame["cell_type"].astype(str).unique().tolist()) if not rare_frame.empty and "cell_type" in rare_frame else []
    primary_failure_column = "failure_archetype_v2" if "failure_archetype_v2" in rare_frame.columns else "failure_archetype"
    failure_map = (
        rare_frame.set_index("cell_type")[primary_failure_column].astype(str).to_dict()
        if not rare_frame.empty and {"cell_type", primary_failure_column}.issubset(rare_frame.columns)
        else {}
    )
    legacy_failure_map = (
        rare_frame.set_index("cell_type")["failure_archetype"].astype(str).to_dict()
        if not rare_frame.empty and {"cell_type", "failure_archetype"}.issubset(rare_frame.columns)
        else {}
    )
    failure_values = [failure_map.get(value, "non_rare") for value in label_values]
    legacy_failure_values = [legacy_failure_map.get(value, "non_rare") for value in label_values]
    is_rare = [value in set(rare_types) for value in label_values]

    scib = getattr(result, "scib", None)
    payload: dict[str, Any] = {
        "sections": {
            "overview": bool(include_overview),
            "metrics": bool(include_metrics),
            "scib": bool(include_scib and scib is not None),
            "rare": bool(include_rare),
            "umap": bool(include_umap),
            "sankey": bool(include_sankey),
            "reproducibility": bool(include_reproducibility),
            "figures": bool(include_static_figures),
        },
        "features": {
            "rare_umap": bool(include_rare and include_rare_umap),
            "rare_heatmaps": bool(include_rare and include_rare_heatmaps),
            "rare_scenario_analysis": bool(include_rare and include_rare_scenario_analysis),
        },
        "meta": {
            "n_cells": int(len(obs)),
            "n_dimensions": int(np.asarray(adata.obsm[representation_key]).shape[1]),
            "representation_key": representation_key,
            "label_key": label_key,
            "batch_key": batch_key,
            "scenario_key": scenario_key,
            "cluster_key": cluster_key or "unknown",
            "prediction_key": prediction_key or "unknown",
            "n_cell_types": int(pd.Series(label_values).nunique()),
            "n_batches": int(pd.Series(batch_values).nunique()),
            "n_rare_types": int(len(rare_types)),
            "n_rare_cells": int(sum(is_rare)),
            "scib_backend": getattr(scib, "backend", "not run") if scib is not None else "not run",
            "scib_backend_version": getattr(scib, "backend_version", "n/a") if scib is not None else "n/a",
            "n_reference_clusters": int(pd.Series(cluster_values).nunique()),
            "cluster_count_warning": int(pd.Series(cluster_values).nunique()) < int(pd.Series(label_values).nunique()),
            "failure_taxonomy_primary": "resolution_aware_v2" if "failure_archetype_v2" in rare_frame.columns else "legacy",
            "failure_taxonomy_legacy_retained": bool("failure_archetype" in rare_frame.columns),
        },
        "metric_registry": METRIC_REGISTRY,
        "failure_rules": _failure_rules_payload(result),
    }

    if include_metrics or include_overview:
        payload["metrics"] = {
            "subset": _records(getattr(result, "subset_metrics", None)),
            "per_type": _records(getattr(result, "per_type_metrics", None)),
        }
    if include_rare or include_overview:
        payload["rare"] = {
            "status": getattr(result, "rare_evaluation_status", {}) or {},
            "per_type": _records(rare_frame),
            "summary": _records(getattr(result, "rare_summary", None)),
            "scenarios": _records(getattr(result, "scenario_metrics", None)),
            "resolution_sensitivity": _records(getattr(result, "resolution_rare_metrics", None)),
            "category_breakdown": _rare_category_breakdown(rare_frame) if include_rare_scenario_analysis else [],
            "rare_types": rare_types,
        }
    if include_scib and scib is not None:
        payload["scib"] = {
            "metrics": _records(getattr(scib, "metrics_long", None)),
            "aggregates": _records(getattr(scib, "aggregate_scores", None)),
            "status": _records(getattr(scib, "metric_status", None)),
            "results_wide": _records(getattr(scib, "results_wide", None).reset_index() if getattr(scib, "results_wide", None) is not None else None),
            "reference_config": getattr(scib, "reference_config", {}),
        }

    needs_points = include_umap or (include_rare and include_rare_umap)
    if needs_points:
        coords, embedding_key, embedding_label = _coordinates(
            adata,
            representation_key,
            umap_key=umap_key,
            random_state=random_state,
        )
        fields = {
            "celltype": _category_encoding(label_values),
            "batch": _category_encoding(batch_values),
            "scenario": _category_encoding(scenario_values),
            "cluster": _category_encoding(cluster_values),
            "prediction": _category_encoding(prediction_values),
            # ``failure_archetype`` is the primary resolution-aware v2 view in
            # the browser. The historical majority-vote taxonomy remains available
            # explicitly as ``failure_archetype_legacy``.
            "failure_archetype": _category_encoding(failure_values),
            "failure_archetype_legacy": _category_encoding(legacy_failure_values),
            "is_rare": _category_encoding(["rare" if value else "non_rare" for value in is_rare]),
        }
        resolution_fields = {}
        for resolution, key in sorted((getattr(result, "cluster_keys", {}) or {}).items()):
            if key in obs.columns:
                resolution_fields[str(float(resolution))] = _category_encoding(_string_values(obs, key))
        numeric_fields = {}
        adjusted_key = getattr(result, "local_recovery_adjusted_key", None)
        if adjusted_key and adjusted_key in obs.columns:
            numeric_fields["knn_local_recovery_adjusted_cell"] = [
                _json_scalar(v) for v in pd.to_numeric(obs[adjusted_key], errors="coerce").to_numpy()
            ]
        local_key = getattr(result, "local_recovery_key", None)
        if local_key and local_key in obs.columns:
            numeric_fields["knn_local_recovery_cell"] = [
                _json_scalar(v) for v in pd.to_numeric(obs[local_key], errors="coerce").to_numpy()
            ]
        payload["points"] = {
            "x": [],
            "y": [],
            "coords_q16": {
                "x": _coordinate_encoding(coords[:, 0]),
                "y": _coordinate_encoding(coords[:, 1]),
            },
            "cell_id": obs.index.tolist() if include_cell_ids else [],
            "fields": fields,
            "numeric_fields": numeric_fields,
            "resolution_clusters": resolution_fields,
            "marker_genes": _marker_gene_payload(adata, marker_genes, layer=marker_layer),
            "embedding_key": embedding_key,
            "embedding_label": embedding_label,
        }
        payload["meta"]["embedding_key"] = embedding_key
        payload["meta"]["embedding_label"] = embedding_label
        fallback = str(embedding_label).startswith("Latent projection")
        payload["meta"]["embedding_fallback"] = bool(fallback)
        payload["meta"]["embedding_warning"] = (
            "UMAP generation was unavailable; the report is displaying latent dimensions 1–2. "
            "Benchmark metrics are unchanged." if fallback else None
        )

    if include_sankey:
        frame = pd.DataFrame({"true": label_values, "predicted": prediction_values, "cluster": cluster_values})
        true_pred = frame.groupby(["true", "predicted"], dropna=False).size().reset_index(name="count")
        true_cluster = frame.groupby(["true", "cluster"], dropna=False).size().reset_index(name="count")
        rare_pred = true_pred[true_pred["true"].isin(rare_types)].reset_index(drop=True)
        payload["sankey"] = {
            "true_to_prediction": true_pred.rename(columns={"true": "source", "predicted": "target"}).to_dict(orient="records"),
            "true_to_cluster": true_cluster.rename(columns={"true": "source", "cluster": "target"}).to_dict(orient="records"),
            "rare_true_to_prediction": rare_pred.rename(columns={"true": "source", "predicted": "target"}).to_dict(orient="records"),
            "highlight_options": sorted(set(label_values)),
        }

    if include_reproducibility:
        files = getattr(result, "files", {})
        payload["reproducibility"] = {
            "run_config": _read_text_file(files.get("run_config")),
            "failure_rules": _read_text_file(files.get("failure_rules")),
            "package_versions": _read_text_file(files.get("package_versions")),
            "scib_reference_config": _read_text_file(files.get("scib_reference_config")),
        }

    if include_static_figures:
        payload["figures"] = _collect_figures(result, include_static_figures=True)
    return payload


def _sha256_string_values(values: Sequence[Any]) -> str:
    return hashlib.sha256("\n".join(str(x) for x in values).encode("utf-8")).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _run_config_dict(result: Any) -> dict[str, Any]:
    files = getattr(result, "files", {}) or {}
    path = files.get("run_config")
    if not path or not Path(path).exists():
        return {}
    try:
        import yaml
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _dashboard_run_entry(
    adata: Any,
    result: Any,
    *,
    payload: dict[str, Any],
    representation_key: str,
    method_seed: int | None = None,
    method_config: dict[str, Any] | None = None,
    run_id: str | None = None,
    included: bool = True,
) -> dict[str, Any]:
    cfg = _run_config_dict(result)
    method_cfg = dict(method_config or cfg.get("method_config") or {})
    seed = method_seed
    if seed is None:
        raw_seed = cfg.get("method_seed", cfg.get("seed"))
        if isinstance(raw_seed, (int, np.integer)) and not isinstance(raw_seed, bool):
            seed = int(raw_seed)
    method_name = str(
        cfg.get("method_name")
        or getattr(result, "prediction_key", representation_key).replace("scrarebench_prediction_", "")
        or representation_key
    )
    obs_names = list(getattr(adata, "obs_names", getattr(adata.obs, "index", [])))
    cell_hash = _sha256_string_values(obs_names)
    latent = np.asarray(adata.obsm[representation_key])
    latent_hash = _sha256_array(latent)
    config_hash = evaluation_contract_hash(cfg, method_cfg)
    training_hash = method_training_hash(method_cfg)
    _uns_get = getattr(getattr(adata, "uns", {}), "get", lambda *_: {})
    dataset_meta = _uns_get('scrarebench_dataset', {}) or _uns_get('scrarebench', {}) or {}
    dataset_key = dataset_meta.get("dataset_key") or dataset_meta.get("key") or dataset_meta.get("display_name") or dataset_meta.get("name") or "dataset"
    dataset_fingerprint = hashlib.sha256(f"{dataset_key}|{cell_hash}|{len(obs_names)}".encode("utf-8")).hexdigest()
    label_key = str(payload.get("meta", {}).get("label_key") or cfg.get("label_key") or "celltype")
    batch_key = str(payload.get("meta", {}).get("batch_key") or cfg.get("batch_key") or "BATCH")
    scenario_key = str(payload.get("meta", {}).get("scenario_key") or cfg.get("scenario_key") or "scrarebench_scenario")
    dataset_contract = dataset_contract_hash(
        adata, dataset_key=str(dataset_key), label_key=label_key, batch_key=batch_key, scenario_key=scenario_key
    )
    benchmark_seed = cfg.get("benchmark_seed", cfg.get("random_state"))
    rid = run_id or make_run_id(
        method_name=method_name,
        dataset_fingerprint=dataset_fingerprint,
        method_seed=seed,
        config_hash=config_hash,
        latent_hash=latent_hash,
    )
    payload.setdefault("meta", {})
    payload["meta"].update({
        "method_name": method_name,
        "method_seed": seed,
        "run_id": rid,
        "configuration_hash": config_hash,
        "evaluation_contract_hash": config_hash,
        "method_training_hash": training_hash,
        "method_configuration": method_cfg,
        "dataset_key": str(dataset_key),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_contract_sha256": dataset_contract,
        "cell_order_sha256": cell_hash,
        "latent_sha256": latent_hash,
        "benchmark_seed": int(benchmark_seed) if benchmark_seed is not None else None,
    })
    return {
        "run_id": rid,
        "method_seed": seed,
        "included": bool(included),
        "method_name": method_name,
        "configuration_hash": config_hash,
        "evaluation_contract_hash": config_hash,
        "method_training_hash": training_hash,
        "method_configuration": method_cfg,
        "dataset_key": str(dataset_key),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_contract_sha256": dataset_contract,
        "cell_order_sha256": cell_hash,
        "latent_sha256": latent_hash,
        "benchmark_seed": int(benchmark_seed) if benchmark_seed is not None else None,
        "payload": payload,
    }


def _render_report_container(container: dict[str, Any], output_path: str | Path, *, title: str) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.offline as plotly_offline
        plotly_js = plotly_offline.get_plotlyjs()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required for interactive report generation") from exc

    # JSON script payloads must not contain a literal closing script tag.
    report_json = json.dumps(container, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    safe_title = html.escape(title)
    asset_root = resource_files("scrarebench") / "assets"
    template = (asset_root / "dashboard_template.html").read_text(encoding="utf-8")
    css = (asset_root / "dashboard.css").read_text(encoding="utf-8")
    js = (asset_root / "dashboard.js").read_text(encoding="utf-8")
    document = (
        template
        .replace("__TITLE__", safe_title)
        .replace("__PLOTLYJS__", plotly_js)
        .replace("__DASHBOARD_CSS__", css)
        .replace("__DASHBOARD_JS__", js)
        .replace("__REPORT_JSON__", report_json)
    )
    target.write_text(document, encoding="utf-8")
    return target


def write_interactive_report(
    adata: Any,
    result: Any,
    output_path: str | Path,
    *,
    title: str | None = None,
    representation_key: str | None = None,
    label_key: str = "celltype",
    batch_key: str = "BATCH",
    scenario_key: str = "scrarebench_scenario",
    umap_key: str | None = None,
    random_state: int = DEFAULT_BENCHMARK_SEED,
    include_overview: bool = True,
    include_metrics: bool = True,
    include_scib: bool = True,
    include_rare: bool = True,
    include_rare_umap: bool = True,
    include_rare_heatmaps: bool = True,
    include_rare_scenario_analysis: bool = True,
    include_umap: bool = True,
    include_sankey: bool = True,
    include_reproducibility: bool = True,
    include_static_figures: bool = True,
    include_cell_ids: bool = True,
    marker_genes: Sequence[str] | None = None,
    marker_layer: str | None = None,
    method_seed: int | None = None,
    method_config: dict[str, Any] | None = None,
    run_id: str | None = None,
    expected_seeds: Sequence[int] | None = None,
    included_in_aggregate: bool = True,
) -> Path:
    """Write a standalone report that is natively compatible with one or many seeds.

    A single-run report is stored in the same multi-run container used by merged
    reports. This allows a later run from the same method/dataset/configuration to
    be imported directly in the browser without regenerating the original report.
    """
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    title = title or f"scRareBench benchmark dashboard — {representation_key}"
    payload = _build_payload(
        adata, result,
        representation_key=representation_key,
        label_key=label_key, batch_key=batch_key, scenario_key=scenario_key,
        umap_key=umap_key, random_state=random_state,
        include_overview=include_overview, include_metrics=include_metrics,
        include_scib=include_scib, include_rare=include_rare,
        include_rare_umap=include_rare_umap, include_rare_heatmaps=include_rare_heatmaps,
        include_rare_scenario_analysis=include_rare_scenario_analysis,
        include_umap=include_umap, include_sankey=include_sankey,
        include_reproducibility=include_reproducibility,
        include_static_figures=include_static_figures, include_cell_ids=include_cell_ids,
        marker_genes=marker_genes, marker_layer=marker_layer,
    )
    entry = _dashboard_run_entry(
        adata, result, payload=payload, representation_key=representation_key,
        method_seed=method_seed, method_config=method_config, run_id=run_id, included=included_in_aggregate,
    )
    expected = normalize_method_seeds(expected_seeds) if expected_seeds is not None else ([entry["method_seed"]] if entry["method_seed"] is not None else [])
    container = make_multirun_container([entry], expected_seeds=expected if expected else None, title=title)
    return _render_report_container(container, output_path, title=title)


def write_multiseed_interactive_report(
    reports: Sequence[str | Path | dict[str, Any]],
    output_path: str | Path,
    *,
    title: str | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> Path:
    """Merge compatible single- or multi-run scRareBench HTML payloads.

    The function preserves every complete run payload (including UMAP/Sankey data)
    and never averages coordinates or cluster identities. Metric aggregation is
    recalculated from the currently included runs in the browser.
    """
    runs: list[dict[str, Any]] = []
    inherited_expected: list[int] = []
    inherited_title: str | None = None
    for item in reports:
        data = item if isinstance(item, dict) else extract_embedded_report_payload(item)
        if data.get("report_type") == "scrarebench_multi_run":
            inherited_title = inherited_title or data.get("title")
            inherited_expected.extend(x for x in data.get("expected_seeds", []) if isinstance(x, int))
            incoming = data.get("runs", [])
        else:
            # Legacy v0.9.x interactive HTML. It remains inspectable/importable, but
            # has no reliable first-class method seed unless one is present in meta.
            meta = data.get("meta", {})
            incoming = [{
                "run_id": meta.get("run_id") or hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:20],
                "method_seed": meta.get("method_seed"),
                "included": True,
                "method_name": meta.get("method_name") or str(meta.get("representation_key", "unknown")).replace("X_", ""),
                "configuration_hash": meta.get("configuration_hash"),
                "dataset_fingerprint": meta.get("dataset_fingerprint") or meta.get("cell_order_sha256"),
                "cell_order_sha256": meta.get("cell_order_sha256"),
                "latent_sha256": meta.get("latent_sha256"),
                "payload": data,
            }]
        for run in incoming:
            # Recompute canonical configuration identity from embedded
            # reproducibility metadata. This repairs stale v0.10.0 hashes caused
            # by seed-dependent realized clustering outcomes.
            canonical_run = canonicalize_embedded_run(run)
            validate_compatible_run(runs, canonical_run)
            runs.append(canonical_run)
    if not runs:
        raise ValueError("At least one compatible report/run is required.")
    expected = normalize_method_seeds(expected_seeds) if expected_seeds is not None else sorted(set(inherited_expected or [r.get("method_seed") for r in runs if isinstance(r.get("method_seed"), int)]))
    final_title = title or inherited_title or "scRareBench multi-seed interactive report"
    container = make_multirun_container(runs, expected_seeds=expected if expected else None, title=final_title)
    return _render_report_container(container, output_path, title=final_title)


merge_interactive_reports = write_multiseed_interactive_report
