from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import slugify


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
    series = pd.Series(values, dtype="object").fillna("missing").astype(str)
    categories = list(pd.Index(series).unique())
    mapping = {value: idx for idx, value in enumerate(categories)}
    return {"categories": categories, "codes": [mapping[value] for value in series]}


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

        working_key = "X_umap_scrarebench_interactive"
        neighbors_key = f"scrarebench_neighbors_{slugify(representation_key)}_interactive"
        if working_key not in adata.obsm:
            sc.pp.neighbors(
                adata,
                use_rep=representation_key,
                n_neighbors=15,
                metric="euclidean",
                key_added=neighbors_key,
            )
            sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=random_state)
            adata.obsm[working_key] = np.asarray(adata.obsm["X_umap"], dtype=float).copy()
        return np.asarray(adata.obsm[working_key], dtype=float)[:, :2], working_key, "UMAP"
    except Exception:
        coords = np.asarray(adata.obsm[representation_key], dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("Interactive report requires a two-dimensional embedding or a representation with at least two dimensions.")
        return coords[:, :2], representation_key, "Latent projection (dims 1–2)"


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




def _rare_category_breakdown(rare_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the six curated rare scenarios for dashboard comparison."""
    if rare_frame is None or rare_frame.empty or "scenario" not in rare_frame.columns:
        return []
    scenario_order = ["GR-DL", "GR-RM", "LE-DL", "LE-RM", "SR-DL", "SR-RM"]
    metrics = ["precision", "recall", "f1", "inverse_purity", "within_type_batch_nmi"]
    out: list[dict[str, Any]] = []
    for scenario in scenario_order:
        frame = rare_frame[rare_frame["scenario"].astype(str) == scenario].copy()
        failures: list[dict[str, Any]] = []
        if not frame.empty and "failure_archetype" in frame.columns:
            for failure, group in frame.groupby("failure_archetype", dropna=False):
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
                row[f"{metric}_mean"] = _json_scalar(values.mean())
                row[f"{metric}_median"] = _json_scalar(values.median())
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_median"] = None
        if "failure_archetype" in frame.columns and not frame.empty:
            preserved = int((frame["failure_archetype"].astype(str) == "preserved").sum())
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
        "failure_counts": "Failure-archetype distribution",
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
    failure_map = (
        rare_frame.set_index("cell_type")["failure_archetype"].astype(str).to_dict()
        if not rare_frame.empty and {"cell_type", "failure_archetype"}.issubset(rare_frame.columns)
        else {}
    )
    failure_values = [failure_map.get(value, "non_rare") for value in label_values]
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
        },
    }

    if include_metrics or include_overview:
        payload["metrics"] = {
            "subset": _records(getattr(result, "subset_metrics", None)),
            "per_type": _records(getattr(result, "per_type_metrics", None)),
        }
    if include_rare or include_overview:
        payload["rare"] = {
            "per_type": _records(rare_frame),
            "summary": _records(getattr(result, "rare_summary", None)),
            "scenarios": _records(getattr(result, "scenario_metrics", None)),
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
            "failure_archetype": _category_encoding(failure_values),
            "is_rare": _category_encoding(["rare" if value else "non_rare" for value in is_rare]),
        }
        payload["points"] = {
            "x": np.asarray(coords[:, 0], dtype=float).round(6).tolist(),
            "y": np.asarray(coords[:, 1], dtype=float).round(6).tolist(),
            "cell_id": obs.index.tolist() if include_cell_ids else [],
            "fields": fields,
            "embedding_key": embedding_key,
            "embedding_label": embedding_label,
        }
        payload["meta"]["embedding_key"] = embedding_key
        payload["meta"]["embedding_label"] = embedding_label

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
    random_state: int = 0,
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
) -> Path:
    """Write the complete standalone scRareBench HTML dashboard.

    All section flags default to ``True``. Turning a section off omits its payload
    from the generated HTML (not just its tab), which can substantially reduce file
    size for large datasets. Rare-cell subfeatures can be independently disabled
    with ``include_rare_umap``, ``include_rare_heatmaps``, and
    ``include_rare_scenario_analysis``. ``include_cell_ids=False`` keeps embedding
    plots but removes barcode strings from hover payloads.
    """
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    title = title or f"scRareBench benchmark dashboard — {representation_key}"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(
        adata,
        result,
        representation_key=representation_key,
        label_key=label_key,
        batch_key=batch_key,
        scenario_key=scenario_key,
        umap_key=umap_key,
        random_state=random_state,
        include_overview=include_overview,
        include_metrics=include_metrics,
        include_scib=include_scib,
        include_rare=include_rare,
        include_rare_umap=include_rare_umap,
        include_rare_heatmaps=include_rare_heatmaps,
        include_rare_scenario_analysis=include_rare_scenario_analysis,
        include_umap=include_umap,
        include_sankey=include_sankey,
        include_reproducibility=include_reproducibility,
        include_static_figures=include_static_figures,
        include_cell_ids=include_cell_ids,
    )
    try:
        import plotly.offline as plotly_offline
        plotly_js = plotly_offline.get_plotlyjs()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required for interactive report generation") from exc

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    safe_title = html.escape(title)
    document = _DASHBOARD_TEMPLATE.replace("__TITLE__", safe_title).replace("__PLOTLYJS__", plotly_js).replace("__PAYLOAD__", payload_json)
    target.write_text(document, encoding="utf-8")
    return target


_DASHBOARD_TEMPLATE = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#dce3ee;--accent:#3157d5;--accent2:#6d4aff;--good:#079669;--warn:#d97706;--bad:#dc2626;--soft:#f7f9fe;--shadow:0 10px 28px rgba(31,55,93,.07);--focus:0 0 0 3px rgba(49,87,213,.22)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
button,input,select{font:inherit}button:focus-visible,input:focus-visible,select:focus-visible,[role=button]:focus-visible{outline:2px solid var(--accent);outline-offset:2px;box-shadow:var(--focus)}
header{background:linear-gradient(120deg,#13264f,#314db6 58%,#6846c7);color:white;padding:28px 30px 24px}.head{max-width:1680px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.74rem;opacity:.72}.head h1{margin:5px 0 6px;font-size:1.85rem}.head p{margin:0;opacity:.86;max-width:1050px;line-height:1.5}
.shell{max-width:1680px;margin:auto;padding:18px}.tabs{display:flex;gap:8px;flex-wrap:wrap;background:white;border:1px solid var(--line);border-radius:15px;padding:7px;position:sticky;top:8px;z-index:50;box-shadow:var(--shadow);scroll-margin-top:8px}.tab-btn{appearance:none;border:0;background:transparent;color:#42526a;padding:10px 14px;border-radius:10px;font-weight:650;cursor:pointer;transition:.16s}.tab-btn:hover{background:#f1f4fb}.tab-btn.active{background:#e9eeff;color:#263fa7}.tab-panel{display:none;padding-top:17px}.tab-panel.active{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin-bottom:14px}.metric-card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}.metric-card{padding:14px 16px}.metric-card .k{font-size:.76rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}.metric-card .v{font-size:1.35rem;font-weight:760;margin-top:4px;word-break:break-word}.metric-card .s{font-size:.8rem;color:var(--muted);margin-top:3px;line-height:1.4}
.panel{margin-bottom:14px;overflow:hidden}.panel-title{padding:14px 17px;border-bottom:1px solid var(--line);font-weight:720;display:flex;align-items:center;justify-content:space-between;gap:10px}.panel-title-main{min-width:0}.panel-subtitle{font-weight:400;color:var(--muted);font-size:.82rem;margin-top:2px}.panel-actions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.panel-body{padding:16px}.grid2{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:14px;align-items:start}.grid-half{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.plot{min-height:570px;width:100%}.plot.smallplot{min-height:390px}.controls{display:flex;flex-direction:column;gap:12px}.control-label,label{font-size:.83rem;color:var(--muted);display:block;margin-bottom:5px}select,input[type=number],input[type=range],input[type=text],input[type=search]{width:100%;border:1px solid var(--line);border-radius:9px;padding:9px 10px;background:white;color:var(--ink)}select[multiple]{min-height:145px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.btn{border:0;border-radius:9px;padding:9px 12px;font-weight:650;cursor:pointer;background:var(--accent);color:white;transition:.14s}.btn:hover{filter:brightness(.97)}.btn:disabled{opacity:.45;cursor:not-allowed;filter:none}.btn.light{background:#e9eef8;color:#31405e}.btn.good{background:var(--good)}.btn.ghost{background:transparent;color:#42526a;border:1px solid var(--line)}.btn.compact{padding:6px 9px;font-size:.78rem}.note{font-size:.84rem;color:var(--muted);line-height:1.55}.callout{background:var(--soft);border:1px dashed #c8d3e8;border-radius:10px;padding:11px}.interpret-note{background:#fffaf1;border:1px solid #f0d8aa;padding:11px 13px;border-radius:10px;font-size:.86rem;color:#684719;margin-bottom:13px;line-height:1.5}
.empty-state{border:1px dashed #cbd5e1;border-radius:11px;background:#fbfcfe;padding:24px;text-align:center;color:var(--muted);line-height:1.5}.empty-state b{display:block;color:var(--ink);margin-bottom:4px}.status-line{display:flex;align-items:center;gap:7px;flex-wrap:wrap;font-size:.8rem;color:var(--muted)}.filter-chip,.match-chip,.scenario-chip{border:1px solid #d5deef;background:#f6f8fd;color:#34486d;border-radius:999px;padding:5px 9px;font-size:.76rem;cursor:pointer}.filter-chip:hover,.match-chip:hover,.scenario-chip:hover{border-color:#93a8d4;background:#eef3ff}.match-chip.active,.scenario-chip.active{background:#e6edff;border-color:#728ce0;color:#263f9c}.scenario-chip{font-weight:700;padding:7px 12px}.match-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:7px}.scenario-selector{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 12px}
.table-tools{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:center}.table-tools input{flex:1 1 260px}.export-group{margin-left:auto;display:flex;gap:6px}.table-wrap{overflow:auto;max-height:650px;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:.82rem;background:white}th,td{padding:8px 9px;border-bottom:1px solid #edf1f6;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#f7f9fc;z-index:2;cursor:pointer;color:#45536b;user-select:none}th:hover{color:#243a80}th[data-sort="asc"]::after{content:" ▲";font-size:.65rem}th[data-sort="desc"]::after{content:" ▼";font-size:.65rem}tr:hover td{background:#f8faff}.clickrow{cursor:pointer}.status{display:inline-block;padding:3px 7px;border-radius:999px;background:#eef2ff;color:#3949a5;font-size:.74rem}.section-heading{margin:3px 0 12px;font-size:1.15rem}
.fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.figure-box{position:relative;border:1px solid var(--line);border-radius:12px;background:#fff;padding:10px;transition:.16s}.figure-box:hover{border-color:#b8c7e2;box-shadow:0 8px 22px rgba(37,57,91,.08)}.figure-media{position:relative;border-radius:8px;overflow:hidden;background:#f8fafc;min-height:80px}.figure-box img{width:100%;height:auto;display:block;border-radius:8px;cursor:zoom-in}.fig-actions{position:absolute;right:8px;top:8px;display:flex;gap:6px;opacity:0;transition:.16s}.figure-box:hover .fig-actions{opacity:1}.icon-btn{width:34px;height:34px;border-radius:9px;border:1px solid rgba(255,255,255,.55);background:rgba(18,29,50,.78);color:#fff;cursor:pointer;font-size:1rem;display:grid;place-items:center;backdrop-filter:blur(5px)}.figure-box .cap{font-size:.82rem;color:var(--muted);margin-top:8px;line-height:1.4}.figure-box .cap b{color:var(--ink);font-weight:650}.figure-toolbar{display:grid;grid-template-columns:minmax(180px,1fr) minmax(180px,280px);gap:10px;margin-bottom:13px}
.pre{white-space:pre-wrap;word-break:break-word;background:#101827;color:#dbe7ff;padding:14px;border-radius:10px;max-height:520px;overflow:auto;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.hidden{display:none!important}.rare-focus{border:1px solid #ccd7ee;border-radius:11px;padding:13px;background:#fbfcff}.rare-focus h3{margin:0 0 8px}.pill{display:inline-block;margin:2px 4px 2px 0;padding:4px 8px;border-radius:999px;background:#eef3ff;color:#304ca9;font-size:.75rem}.failure-preserved{background:#eaf8f2;color:#087b5a}.failure-lineage_assimilation{background:#fff1e8;color:#a14c18}.failure-lineage_leakage{background:#fff0f2;color:#ae3247}.failure-batch_driven_fragmentation{background:#f3efff;color:#6844aa}.failure-mixed_or_uncertain{background:#eef1f5;color:#586474}
.scenario-guide,.scenario-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.guide-card,.scenario-card{border:1px solid var(--line);border-radius:12px;background:#fff;padding:13px}.guide-card h4,.scenario-card h4{margin:0 0 5px}.guide-card p{margin:0;color:var(--muted);font-size:.82rem;line-height:1.45}.scenario-card{cursor:pointer;transition:.16s}.scenario-card:hover{border-color:#aebee0;box-shadow:0 7px 20px rgba(37,57,91,.07)}.scenario-card.active{border-color:#7189df;background:#f8faff}.scenario-meta{font-size:.78rem;color:var(--muted);margin-bottom:9px}.mini-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:9px 0}.mini-metric{background:#f6f8fc;border-radius:8px;padding:7px;text-align:center}.mini-metric b{display:block;font-size:.92rem}.mini-metric span{font-size:.67rem;color:var(--muted)}.failure-list{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}.cell-mini-list{border-top:1px solid #edf1f6;margin-top:9px;padding-top:6px}.cell-mini{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;padding:6px 4px;font-size:.76rem;border-bottom:1px solid #f1f3f7;border-radius:5px;cursor:pointer}.cell-mini:hover{background:#f5f8ff}.cell-mini:last-child{border-bottom:0}.cell-mini .ct{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.cell-mini .score{color:var(--muted);white-space:nowrap}
.selection-detail{margin-top:12px}.selection-detail h3{margin:0 0 8px}.outcome-summary{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.zoom-readout{min-width:58px;text-align:center;color:#cbd5e1;font-size:.75rem}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.detail-block{border:1px solid #e4e9f2;border-radius:9px;padding:10px;background:#fff}.detail-block .dk{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.detail-block .dv{font-size:1rem;font-weight:700;margin-top:3px}.glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}.gloss-item{border:1px solid #e4e9f2;border-radius:9px;padding:9px;background:#fff}.gloss-item b{font-size:.82rem}.gloss-item p{margin:4px 0 0;font-size:.76rem;color:var(--muted);line-height:1.4}.heatmap-note{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}.heatmap-note .callout{font-size:.8rem}.rare-layout-no-umap{display:block}
.modal{position:fixed;inset:0;background:rgba(8,15,28,.88);z-index:200;display:none;align-items:center;justify-content:center;padding:18px}.modal.open{display:flex}.modal-card{width:min(96vw,1500px);height:min(94vh,1000px);background:#0f1725;border:1px solid rgba(255,255,255,.15);border-radius:14px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.45)}.modal-head{display:flex;align-items:center;gap:8px;padding:10px 12px;color:#fff;border-bottom:1px solid rgba(255,255,255,.12)}.modal-title{font-size:.9rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.modal-head button{background:#1c2940;color:#fff;border:1px solid #32435f;border-radius:8px;padding:7px 10px;cursor:pointer}.modal-stage{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:flex-start;padding:20px;background:#0b1220}.modal-stage-inner{min-width:100%;min-height:100%;display:flex;align-items:center;justify-content:center}.modal-stage img{display:block;max-width:none;max-height:none;height:auto;transition:width .1s ease}.modal-foot{color:#aebbd0;font-size:.76rem;padding:8px 12px;border-top:1px solid rgba(255,255,255,.1)}
.toast{position:fixed;right:18px;bottom:18px;z-index:300;background:#172033;color:#fff;border-radius:10px;padding:10px 13px;box-shadow:0 10px 30px rgba(0,0,0,.25);font-size:.84rem;opacity:0;transform:translateY(8px);pointer-events:none;transition:.18s}.toast.show{opacity:1;transform:none}.toast.error{background:#8f2735}.loading{padding:30px;text-align:center;color:var(--muted)}
@media(max-width:1150px){.grid2,.grid-half{grid-template-columns:1fr}.scenario-guide,.scenario-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.tabs{position:static}.plot{min-height:480px}.heatmap-note{grid-template-columns:1fr}}@media(max-width:650px){header{padding:20px}.shell{padding:10px}.head h1{font-size:1.4rem}.tabs{flex-wrap:nowrap;overflow-x:auto}.tab-btn{white-space:nowrap}.plot{min-height:390px}.scenario-guide,.scenario-grid{grid-template-columns:1fr}.detail-grid{grid-template-columns:1fr}.fig-actions{opacity:1}.figure-toolbar{grid-template-columns:1fr}.mini-metrics{grid-template-columns:repeat(2,1fr)}.modal{padding:4px}.modal-card{width:99vw;height:97vh}.modal-head{flex-wrap:wrap}.export-group{margin-left:0}}
</style>
</head>
<body>
<script>__PLOTLYJS__</script>
<header><div class="head"><div class="eyebrow">scRareBench</div><h1>__TITLE__</h1><p>Standalone benchmark explorer for integration quality, scIB-compatible scores, rare-population preservation, clustering outcomes, and failure patterns. Interactive filters change visual exploration only; benchmark scores remain unchanged.</p></div></header>
<div class="shell">
<div id="warning" class="interpret-note"><b>Interpretation note.</b> Failure-archetype assignments use provisional scRareBench thresholds. Treat them as diagnostic annotations; threshold sensitivity should be assessed before manuscript-level conclusions.</div>
<nav id="tabs" class="tabs" role="tablist" aria-label="Benchmark report sections"></nav>
<div id="panels"></div>
</div>
<div id="figModal" class="modal" role="dialog" aria-modal="true" aria-hidden="true" aria-label="Expanded figure"><div class="modal-card"><div class="modal-head"><button id="modalPrev" title="Previous figure" aria-label="Previous figure">←</button><div id="modalTitle" class="modal-title">Figure</div><button id="zoomOut" title="Zoom out">−</button><button id="zoomFit" title="Fit figure to viewer">Fit</button><button id="zoomActual" title="Show figure at 100% of original pixels">100%</button><span id="zoomReadout" class="zoom-readout" aria-live="polite">Fit</span><button id="zoomIn" title="Zoom in">+</button><button id="modalDownload" title="Download original figure">Download original</button><button id="modalNext" title="Next figure" aria-label="Next figure">→</button><button id="modalClose" title="Close" aria-label="Close figure">×</button></div><div id="modalStage" class="modal-stage"><div class="modal-stage-inner"><img id="modalImg" alt="Expanded figure"></div></div><div id="modalFoot" class="modal-foot"></div></div></div>
<div id="toast" class="toast" role="status" aria-live="polite"></div>
<script>
const P=__PAYLOAD__;
const S=P.sections||{};
const F=P.features||{};
let activeFlow=null;
let selectedScenario=null;
let modalIndex=-1,modalZoom=1,modalLastFocus=null,modalFitScale=1,modalNavIndices=[];
const rendered=new Set();
const rareState={scenario:'',outcome:'',failure:'',query:'',otherMode:'gray',focus:null};
let rareSearchTimer=null;

function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fmt(v){if(v===null||v===undefined||Number.isNaN(v))return '—';if(typeof v==='number'){if(!Number.isFinite(v))return '—';const a=Math.abs(v);if(a!==0&&a<1e-4)return v.toExponential(3);if(a>=1000)return v.toLocaleString(undefined,{maximumFractionDigits:2});return v.toLocaleString(undefined,{maximumFractionDigits:4});}return String(v)}
function pct(v){return typeof v==='number'&&Number.isFinite(v)?(100*v).toLocaleString(undefined,{maximumFractionDigits:1})+'%':'—'}
function card(k,v,s=''){return `<div class="metric-card"><div class="k">${esc(k)}</div><div class="v">${esc(fmt(v))}</div>${s?`<div class="s">${esc(s)}</div>`:''}</div>`}
function toast(message,type='info'){const el=document.getElementById('toast');if(!el)return;el.textContent=message;el.className='toast show'+(type==='error'?' error':'');clearTimeout(toast._timer);toast._timer=setTimeout(()=>el.className='toast',2200)}
function debounce(fn,delay=160){let t;return(...args)=>{clearTimeout(t);t=setTimeout(()=>fn(...args),delay)}}

const COLUMN_LABELS={ASW_true_on_latent:'Cell-type silhouette (latent)',ARI_true_vs_cluster:'ARI: labels vs clusters',AMI_true_vs_cluster:'AMI: labels vs clusters',Accuracy:'Majority-vote accuracy',F1_macro:'Macro F1',F1_weighted:'Weighted F1',G_Mean:'G-Mean',cell_type:'Cell type',predicted_count:'Predicted count',true_positive:'True positive',false_positive:'False positive',false_negative:'False negative',inverse_purity:'Dominant-cluster capture',within_type_batch_nmi:'Within-type batch NMI',n_clusters_found_in:'Clusters containing target',n_clusters_assigned_to:'Clusters assigned to target',dominant_wrong_label:'Dominant wrong label',dominant_wrong_fraction:'Dominant wrong fraction',failure_archetype:'Failure archetype',failure_rationale:'Failure rationale',curation_source:'Curation source',parent_type:'Parent type',metric_type:'Metric group',n_valid:'Valid populations',support:'Support',precision:'Precision',recall:'Recall',f1:'F1',scenario:'Scenario',distribution:'Distribution',topology:'Topology'};
const COLUMN_HELP={inverse_purity:'Fraction of this true cell type contained in its dominant cluster. Higher is better. Internal column name: inverse_purity.',within_type_batch_nmi:'NMI between batch labels and clusters within the target cell type. Lower values indicate less batch-associated cluster structure; interpret separately from recovery metrics.',ARI_true_vs_cluster:'Adjusted Rand Index between reference cell-type labels and Leiden clusters.',AMI_true_vs_cluster:'Adjusted Mutual Information between reference cell-type labels and Leiden clusters.',F1_macro:'Unweighted mean F1 across cell types.',F1_weighted:'F1 averaged with cell-type support as weights.',G_Mean:'Geometric mean of per-class recall.',precision:'For a target cell type, the fraction of predicted target cells that are true positives.',recall:'For a target cell type, the fraction of true target cells recovered by the majority-vote prediction.',f1:'Harmonic mean of precision and recall.',support:'Number of true cells in the target population.'};
function colLabel(c){return COLUMN_LABELS[c]||String(c).replaceAll('_',' ')}
function failureLabel(x){return ({preserved:'Preserved',lineage_assimilation:'Lineage assimilation',lineage_leakage:'Lineage leakage',batch_driven_fragmentation:'Batch-driven fragmentation',mixed_or_uncertain:'Mixed / uncertain'})[x]||String(x||'Unknown')}
function failurePill(x,count=''){return `<span class="pill failure-${esc(x)}">${esc(failureLabel(x))}${count!==''?' · '+esc(count):''}</span>`}
function rareOutcome(row){return String(row?.failure_archetype||'')==='preserved'?'preserved':'not_preserved'}
function outcomeLabel(x){return x==='preserved'?'Preserved':x==='not_preserved'?'Not preserved':'All outcomes'}
function outcomePill(x,count=''){const cls=x==='preserved'?'failure-preserved':'failure-mixed_or_uncertain';return `<span class="pill ${cls}">${esc(outcomeLabel(x))}${count!==''?' · '+esc(count):''}</span>`}
function cellHTML(col,value){if(col==='failure_archetype'&&value)return failurePill(value);if(col==='status'&&value)return `<span class="status">${esc(String(value).replaceAll('_',' '))}</span>`;return esc(fmt(value))}
function csvCell(v){const s=String(v??'');return /[",\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}
function downloadText(name,text,type='text/plain'){try{const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1200);toast(`Saved ${name}`)}catch(err){console.error(err);toast('Export failed in this browser.','error')}}
function downloadPlot(id,name){const el=document.getElementById(id);if(!el||!el.data||!el.data.length){toast('No rendered plot is available to export.','error');return Promise.resolve(false)}return Plotly.downloadImage(el,{format:'png',filename:name,width:1700,height:1100,scale:2}).then(()=>{toast(`Exported ${name}.png`);return true}).catch(err=>{console.error(err);toast('Plot export failed in this browser.','error');return false})}
function field(name,i){const f=P.points?.fields?.[name];return f?f.categories[f.codes[i]]:'unknown'}
function palette(n){const b=['#3157d5','#e04d4d','#0b9b72','#df941d','#714bd3','#d94c9b','#13a2ae','#e66c28','#4788e8','#78a827','#1598d4','#9b55d7','#34a853','#d83c64','#69778e'];return Array.from({length:n},(_,i)=>b[i%b.length])}
function hover(indices){return indices.map(i=>{const id=P.points?.cell_id&&P.points.cell_id.length?`Cell: ${P.points.cell_id[i]}<br>`:'';return id+`True type: ${field('celltype',i)}<br>Batch: ${field('batch',i)}<br>Scenario: ${field('scenario',i)}<br>Cluster: ${field('cluster',i)}<br>Prediction: ${field('prediction',i)}<br>Failure: ${field('failure_archetype',i)}`})}
function emptyHTML(title,text=''){return `<div class="empty-state"><b>${esc(title)}</b>${text?esc(text):''}</div>`}

function tableHTML(data,id,options={}){if(!data||!data.columns||!data.columns.length)return emptyHTML('No table data','No rows were produced for this section.');const showSearch=options.showSearch!==false,clickable=!!options.clickable;return `<div class="table-tools">${showSearch?`<input id="${id}-search" type="search" aria-label="Filter ${esc(id)} rows" placeholder="Filter rows…">`:''}<div class="export-group"><button class="btn ghost compact" data-export-current="${id}" title="Export the currently visible rows as CSV">Export view</button><button class="btn ghost compact" data-export-full="${id}" title="Export the complete source table as CSV">Export all</button></div></div><div class="table-wrap"><table id="${id}"><thead><tr>${data.columns.map(c=>`<th data-col="${esc(c)}" title="${esc(COLUMN_HELP[c]||String(c))}" aria-sort="none">${esc(colLabel(c))}</th>`).join('')}</tr></thead><tbody>${data.rows.map((r,i)=>`<tr data-i="${i}" class="${clickable?'clickrow':''}">${data.columns.map(c=>`<td>${cellHTML(c,r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function exportTable(viewData,fullData,id,current){const table=document.getElementById(id);if(!table)return;const cols=viewData.columns||[];let rows,headers;if(current){headers=cols.map(colLabel);rows=[...table.tBodies[0].rows].filter(r=>r.style.display!=='none').map(r=>[...r.cells].map(c=>c.textContent.trim()))}else{const src=fullData||viewData;headers=cols;rows=(src.rows||[]).map(r=>cols.map(c=>r[c]))}const lines=[headers,...rows].map(r=>r.map(csvCell).join(','));downloadText(id+(current?'_view':'_all_raw')+'.csv',lines.join('\n'),'text/csv;charset=utf-8')}
function wireTable(data,id,onRow=null,fullData=null){const t=document.getElementById(id),q=document.getElementById(id+'-search');if(!t)return;const sort={idx:-1,dir:1};t.querySelectorAll('th').forEach((th,idx)=>th.addEventListener('click',()=>{if(sort.idx===idx)sort.dir*=-1;else{sort.idx=idx;sort.dir=1}t.querySelectorAll('th').forEach(h=>{h.dataset.sort='';h.setAttribute('aria-sort','none')});th.dataset.sort=sort.dir===1?'asc':'desc';th.setAttribute('aria-sort',sort.dir===1?'ascending':'descending');const rows=[...t.tBodies[0].rows];rows.sort((a,b)=>{const av=a.cells[idx].textContent.trim(),bv=b.cells[idx].textContent.trim(),an=Number(av.replace(/,/g,'')),bn=Number(bv.replace(/,/g,''));const cmp=Number.isFinite(an)&&Number.isFinite(bn)?an-bn:av.localeCompare(bv,undefined,{numeric:true,sensitivity:'base'});return sort.dir*cmp});rows.forEach(r=>t.tBodies[0].appendChild(r))}));if(q)q.addEventListener('input',()=>{const z=q.value.trim().toLowerCase();[...t.tBodies[0].rows].forEach(r=>r.style.display=!z||r.textContent.toLowerCase().includes(z)?'':'none')});if(onRow)[...t.tBodies[0].rows].forEach(r=>r.addEventListener('click',()=>onRow(data.rows[Number(r.dataset.i)])));document.querySelector(`[data-export-current="${id}"]`)?.addEventListener('click',()=>exportTable(data,fullData||data,id,true));document.querySelector(`[data-export-full="${id}"]`)?.addEventListener('click',()=>exportTable(data,fullData||data,id,false))}
function findMetric(table,name){if(!table||!table.rows)return null;const r=table.rows.find(x=>String(x.metric||'').toLowerCase()===name.toLowerCase());return r?r.value??r.mean:null}

const tabs=[];
function addTab(id,label){if(!S[id])return;tabs.push([id,label])}
addTab('overview','Overview');addTab('metrics','Metrics');addTab('scib','scIB');addTab('rare','Rare-cell Explorer');addTab('umap','UMAP');addTab('sankey','Sankey');addTab('reproducibility','Reproducibility');addTab('figures','Figures');
const tabBar=document.getElementById('tabs'),panels=document.getElementById('panels');
tabs.forEach(([id,label],i)=>{const b=document.createElement('button');b.className='tab-btn'+(i===0?' active':'');b.textContent=label;b.dataset.tab=id;b.id='tabbtn-'+id;b.setAttribute('role','tab');b.setAttribute('aria-controls','tab-'+id);b.setAttribute('aria-selected',i===0?'true':'false');tabBar.appendChild(b);const p=document.createElement('section');p.id='tab-'+id;p.className='tab-panel'+(i===0?' active':'');p.setAttribute('role','tabpanel');p.setAttribute('aria-labelledby',b.id);p.innerHTML='<div class="loading">Open this tab to render its contents.</div>';panels.appendChild(p);b.addEventListener('click',()=>openTab(id))});
function ensureRendered(id){if(rendered.has(id))return;const fn=renderers[id];if(fn){try{fn();rendered.add(id)}catch(err){console.error(`Failed to render ${id}`,err);document.getElementById('tab-'+id).innerHTML=emptyHTML('This section could not be rendered',String(err?.message||err));toast(`${id} section encountered an error.`,'error')}}}
function resizeVisiblePlots(){['umapPlot','rareUmap','sankeyPlot','scibPlot','rareRecoveryHeatmap','rareBatchHeatmap','rareScenarioMetrics','rareFailurePlot','overviewScibPlot','overviewRarePlot'].forEach(x=>{const el=document.getElementById(x);if(el&&el.data)try{Plotly.Plots.resize(el)}catch(_){}})}
function openTab(id,{scroll=true}={}){if(!S[id])return;ensureRendered(id);document.querySelectorAll('.tab-btn').forEach(x=>{const active=x.dataset.tab===id;x.classList.toggle('active',active);x.setAttribute('aria-selected',active?'true':'false')});document.querySelectorAll('.tab-panel').forEach(x=>x.classList.toggle('active',x.id==='tab-'+id));if(scroll){const top=Math.max(0,document.querySelector('.shell').offsetTop-6);window.scrollTo({top,behavior:'smooth'})}setTimeout(resizeVisiblePlots,80)}
tabBar.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const buttons=[...tabBar.querySelectorAll('.tab-btn')],cur=buttons.indexOf(document.activeElement);if(cur<0)return;e.preventDefault();let next=cur;if(e.key==='ArrowRight')next=(cur+1)%buttons.length;if(e.key==='ArrowLeft')next=(cur-1+buttons.length)%buttons.length;if(e.key==='Home')next=0;if(e.key==='End')next=buttons.length-1;buttons[next].focus();openTab(buttons[next].dataset.tab,{scroll:false})});

function figureCards(group){const figs=(P.figures||[]).map((f,i)=>({...f,_i:i})).filter(f=>!group||f.group===group);if(!figs.length)return '';return `<div class="fig-grid">${figs.map(f=>`<div class="figure-box" data-figure-card="${f._i}" data-figure-group="${esc(f.group)}" data-figure-search="${esc(((f.title||'')+' '+(f.name||'')+' '+(f.group||'')).toLowerCase())}"><div class="figure-media"><img class="zoomable-figure" data-fig-index="${f._i}" src="${f.uri}" alt="${esc(f.title||f.name)}"><div class="fig-actions"><button class="icon-btn fig-open" data-fig-index="${f._i}" title="Enlarge figure" aria-label="Enlarge figure">⤢</button><button class="icon-btn fig-download" data-fig-index="${f._i}" title="Download original figure" aria-label="Download original figure">↓</button></div></div><div class="cap"><b>${esc(f.title||f.name)}</b><br>${esc(f.group)} · ${esc(f.name)}</div></div>`).join('')}</div>`}
function bindFigureActions(scope=document){scope.querySelectorAll('.zoomable-figure,.fig-open').forEach(el=>el.addEventListener('click',()=>{modalLastFocus=el;const galleryCard=el.closest('#figureGallery [data-figure-card]');modalNavIndices=galleryCard?[...document.querySelectorAll('#figureGallery [data-figure-card]')].filter(c=>c.style.display!=='none').map(c=>Number(c.dataset.figureCard)):(P.figures||[]).map((_,i)=>i);openFigure(Number(el.dataset.figIndex))}));scope.querySelectorAll('.fig-download').forEach(el=>el.addEventListener('click',e=>{e.stopPropagation();downloadFigure(Number(el.dataset.figIndex))}))}
function openFigure(i){const figs=P.figures||[],f=figs[i];if(!f)return;modalIndex=i;modalZoom=1;const img=document.getElementById('modalImg');img.alt=f.title||f.name;document.getElementById('modalTitle').textContent=f.title||f.name;document.getElementById('modalFoot').textContent=f.group+' · '+f.name;document.getElementById('figModal').classList.add('open');document.getElementById('figModal').setAttribute('aria-hidden','false');img.onload=()=>fitModalImage();img.src=f.uri;if(img.complete)requestAnimationFrame(fitModalImage);document.getElementById('modalClose').focus()}
function fitModalImage(){const img=document.getElementById('modalImg'),stage=document.getElementById('modalStage');if(!img.naturalWidth||!stage.clientWidth)return;const maxW=Math.max(100,stage.clientWidth-48),maxH=Math.max(100,stage.clientHeight-48);modalFitScale=Math.min(1,maxW/img.naturalWidth,maxH/img.naturalHeight);applyModalSize()}
function applyModalSize(){const img=document.getElementById('modalImg');if(!img.naturalWidth)return;const scale=modalFitScale*modalZoom;img.style.width=Math.max(1,Math.round(img.naturalWidth*scale))+'px';const readout=document.getElementById('zoomReadout');if(readout)readout.textContent=Math.abs(scale-1)<.015?'100%':Math.round(scale*100)+'%'}
function setModalZoom(z){modalZoom=Math.min(5,Math.max(.5,z));applyModalSize()}
function navigateFigure(step){const figs=P.figures||[],nav=(modalNavIndices||[]).filter(i=>figs[i]);if(!nav.length)return;let pos=nav.indexOf(modalIndex);if(pos<0)pos=0;openFigure(nav[(pos+step+nav.length)%nav.length])}
function closeFigure(){const modal=document.getElementById('figModal');modal.classList.remove('open');modal.setAttribute('aria-hidden','true');modalLastFocus?.focus?.()}
function downloadFigure(i){const f=(P.figures||[])[i];if(!f)return;try{const a=document.createElement('a');a.href=f.uri;a.download=f.name||'scrarebench_figure.png';document.body.appendChild(a);a.click();a.remove();toast(`Saved ${f.name||'figure'}`)}catch(err){console.error(err);toast('Figure download failed.','error')}}
document.getElementById('modalClose').onclick=closeFigure;document.getElementById('zoomIn').onclick=()=>setModalZoom(modalZoom+.25);document.getElementById('zoomOut').onclick=()=>setModalZoom(modalZoom-.25);document.getElementById('zoomFit').onclick=()=>setModalZoom(1);document.getElementById('zoomActual').onclick=()=>setModalZoom(modalFitScale>0?1/modalFitScale:1);document.getElementById('modalDownload').onclick=()=>downloadFigure(modalIndex);document.getElementById('modalPrev').onclick=()=>navigateFigure(-1);document.getElementById('modalNext').onclick=()=>navigateFigure(1);document.getElementById('figModal').addEventListener('click',e=>{if(e.target.id==='figModal')closeFigure()});document.addEventListener('keydown',e=>{if(!document.getElementById('figModal').classList.contains('open'))return;if(e.key==='Escape')closeFigure();if(e.key==='ArrowLeft')navigateFigure(-1);if(e.key==='ArrowRight')navigateFigure(1)});window.addEventListener('resize',()=>{if(document.getElementById('figModal').classList.contains('open'))fitModalImage()});

function renderOverview(){if(!S.overview)return;const m=P.meta,r=P.rare||{},sc=P.scib||{};let cards=card('Cells',m.n_cells)+card('Cell types',m.n_cell_types)+card('Batches',m.n_batches)+card('Latent dimensions',m.n_dimensions)+card('Rare populations',m.n_rare_types)+card('Rare cells',m.n_rare_cells);if(sc.aggregates){['Bio conservation','Batch correction','Total'].forEach(x=>{const v=findMetric(sc.aggregates,x);if(v!==null)cards+=card('scIB '+x,v)});}if(r.summary){['precision','recall','f1'].forEach(x=>{const v=findMetric(r.summary,x);if(v!==null)cards+=card('Mean rare '+x,v,'Unweighted mean across curated rare populations')});const pf=findMetric(r.summary,'preserved_fraction');if(pf!==null)cards+=card('Preserved rare populations',pct(pf),`${Math.round(pf*(m.n_rare_types||0))} of ${m.n_rare_types||0} populations`)}const rows=r.per_type?.rows||[],preserved=rows.filter(x=>rareOutcome(x)==='preserved').length,notPreserved=rows.length-preserved,failCounts={};rows.filter(x=>rareOutcome(x)==='not_preserved').forEach(x=>{const k=x.failure_archetype||'mixed_or_uncertain';failCounts[k]=(failCounts[k]||0)+1});const failureHtml=Object.entries(failCounts).sort((a,b)=>b[1]-a[1]).map(([k,v])=>failurePill(k,v)).join('');document.getElementById('tab-overview').innerHTML=`<div class="cards">${cards}</div><div class="panel"><div class="panel-title"><div class="panel-title-main">Run identity<div class="panel-subtitle">Keys used for evaluation and visualization</div></div></div><div class="panel-body"><div class="rare-focus"><span class="pill">${esc(m.representation_key)}</span><span class="pill">${esc(m.cluster_key)}</span><span class="pill">${esc(m.scib_backend)} ${esc(m.scib_backend_version)}</span><p class="note">The standard scIB-compatible layer and the scRareBench rare-cell layer are intentionally reported separately rather than collapsed into one unvalidated composite score.</p>${rows.length?`<div class="outcome-summary"><b>Rare outcomes:</b>${outcomePill('preserved',preserved)}${outcomePill('not_preserved',notPreserved)}</div>${failureHtml?`<div class="status-line" style="margin-top:7px"><b>Failure modes among not-preserved populations:</b> ${failureHtml}</div>`:''}`:''}</div></div></div><div class="grid-half"><div class="panel"><div class="panel-title"><div class="panel-title-main">scIB aggregate scores<div class="panel-subtitle">Standard scIB-compatible layer</div></div><div class="panel-actions"><button class="btn ghost compact" id="overviewScibPng">Export PNG</button></div></div><div class="panel-body"><div id="overviewScibPlot" class="plot smallplot"></div></div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">Rare-cell summary scores<div class="panel-subtitle">scRareBench rare-population layer; not a composite with scIB</div></div><div class="panel-actions"><button class="btn ghost compact" id="overviewRarePng">Export PNG</button></div></div><div class="panel-body"><div id="overviewRarePlot" class="plot smallplot"></div></div></div></div>`;const scRows=(sc.aggregates?.rows||[]).filter(x=>typeof x.value==='number'&&Number.isFinite(x.value));if(scRows.length)Plotly.newPlot('overviewScibPlot',[{type:'bar',x:scRows.map(x=>x.value),y:scRows.map(x=>x.metric),orientation:'h',marker:{color:'#4866d5'}}],{margin:{l:150,r:20,t:15,b:40},xaxis:{range:[0,1],title:'Score'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});else document.getElementById('overviewScibPlot').innerHTML=emptyHTML('No scIB aggregate scores');const rareRows=(r.summary?.rows||[]).filter(x=>['precision','recall','f1','preserved_fraction'].includes(x.metric)&&typeof x.mean==='number'&&Number.isFinite(x.mean));if(rareRows.length)Plotly.newPlot('overviewRarePlot',[{type:'bar',x:rareRows.map(x=>x.mean),y:rareRows.map(x=>x.metric==='preserved_fraction'?'Preserved fraction':colLabel(x.metric)),orientation:'h',marker:{color:'#6d4aff'}}],{margin:{l:150,r:20,t:15,b:40},xaxis:{range:[0,1],title:'Score / fraction'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});else document.getElementById('overviewRarePlot').innerHTML=emptyHTML('No rare summary scores');document.getElementById('overviewScibPng').onclick=()=>downloadPlot('overviewScibPlot','scrarebench_overview_scib_scores');document.getElementById('overviewRarePng').onclick=()=>downloadPlot('overviewRarePlot','scrarebench_overview_rare_scores')}

function renderMetrics(){if(!S.metrics)return;const M=P.metrics||{};document.getElementById('tab-metrics').innerHTML=`<div class="panel"><div class="panel-title"><div class="panel-title-main">Overall / rare / non-rare benchmark metrics<div class="panel-subtitle">Global comparison by evaluation subset</div></div></div><div class="panel-body">${tableHTML(M.subset,'subsetTable')}</div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">All cell-type metrics<div class="panel-subtitle">Sort, filter, and export without leaving the report</div></div></div><div class="panel-body"><p class="note">Headers are human-readable; hover a header for metric details where available. These metrics are benchmark outputs and are not changed by dashboard filters.</p>${tableHTML(M.per_type,'perTypeTable')}</div></div><div class="panel"><div class="panel-title">Metric glossary</div><div class="panel-body"><div class="glossary">${Object.entries(COLUMN_HELP).map(([k,v])=>`<div class="gloss-item"><b>${esc(colLabel(k))}</b><p>${esc(v)}</p></div>`).join('')}</div></div></div>`;wireTable(M.subset,'subsetTable');wireTable(M.per_type,'perTypeTable')}

function renderScib(){if(!S.scib)return;const X=P.scib||{};const metrics=X.metrics||{columns:[],rows:[]},aggregates=X.aggregates||{rows:[]};document.getElementById('tab-scib').innerHTML=`<div class="callout note" style="margin-bottom:14px">The scIB-compatible section is reported independently from scRareBench rare-cell diagnostics. Metrics requiring gene-level corrected expression, curated trajectories, or other unavailable inputs are explicitly marked <b>not applicable</b> rather than silently omitted.</div><div class="cards">${(aggregates.rows||[]).map(r=>card(r.metric,r.value,r.metric_type||'Aggregate score')).join('')||card('scIB status','No aggregate scores')}</div><div class="grid-half"><div class="panel"><div class="panel-title"><div class="panel-title-main">Interactive scIB score profile<div class="panel-subtitle">Computed metrics plus reported aggregate scores</div></div><div class="panel-actions"><button class="btn ghost compact" id="scibPng">Export PNG</button></div></div><div class="panel-body"><div id="scibPlot" class="plot smallplot"></div></div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">Backend / reference configuration<div class="panel-subtitle">Configuration used by the scIB-compatible layer</div></div><div class="panel-actions"><button class="btn ghost compact" id="scibConfigDownload">Export JSON</button></div></div><div class="panel-body"><div class="pre">${esc(JSON.stringify(X.reference_config||{},null,2))}</div></div></div></div><div class="panel"><div class="panel-title">All scIB-compatible metrics</div><div class="panel-body">${tableHTML(metrics,'scibMetrics')}</div></div><div class="panel"><div class="panel-title">Metric availability and applicability</div><div class="panel-body">${tableHTML(X.status,'scibStatus')}</div></div><div class="panel"><div class="panel-title">Raw benchmarker result table</div><div class="panel-body">${tableHTML(X.results_wide,'scibWide')}</div></div>${figureCards('scIB')?`<div class="panel"><div class="panel-title">Backend-produced scIB figures</div><div class="panel-body">${figureCards('scIB')}</div></div>`:''}`;wireTable(metrics,'scibMetrics');wireTable(X.status,'scibStatus');wireTable(X.results_wide,'scibWide');const rows=[...(metrics.rows||[]),...(aggregates.rows||[])].filter(r=>typeof r.value==='number'&&Number.isFinite(r.value));if(rows.length)Plotly.newPlot('scibPlot',[{type:'bar',x:rows.map(r=>r.value),y:rows.map(r=>r.metric),orientation:'h',marker:{color:rows.map(r=>((r.metric_type||'').toLowerCase().includes('batch')||String(r.metric||'').toLowerCase()==='batch correction')?'#0f9b75':'#4968d8')}}],{height:Math.max(390,rows.length*28),margin:{l:200,r:20,t:20,b:40},xaxis:{title:'Score'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});else document.getElementById('scibPlot').innerHTML=emptyHTML('No numeric scIB scores','Check metric availability and applicability below.');document.getElementById('scibPng').onclick=()=>downloadPlot('scibPlot','scrarebench_scib_metrics');document.getElementById('scibConfigDownload').onclick=()=>downloadText('scib_reference_config.json',JSON.stringify(X.reference_config||{},null,2),'application/json');bindFigureActions(document.getElementById('tab-scib'))}

function scenarioGuide(){return `<div class="scenario-guide"><div class="guide-card"><h4>GR — Globally Rare</h4><p>Rare globally while represented across multiple batches in the benchmark taxonomy.</p></div><div class="guide-card"><h4>LE — Locally Enriched</h4><p>Concentrated in a limited subset of batches or conditions.</p></div><div class="guide-card"><h4>SR — Sporadic Rare</h4><p>Sparse and limited to a small subset of batches or conditions.</p></div><div class="guide-card"><h4>DL — Distinct lineage</h4><p>A biologically distinct rare lineage or population rather than a rare state nested near a related abundant population.</p></div><div class="guide-card"><h4>RM — Related manifold / state</h4><p>A rare state or subtype biologically related to a more abundant or neighboring population; parent-lineage metadata is shown when curated.</p></div><div class="guide-card"><h4>Six scenarios</h4><p>GR-DL, GR-RM, LE-DL, LE-RM, SR-DL, and SR-RM combine distribution and topology classes.</p></div></div>`}
function scenarioCard(c){const metrics=[['P',c.precision_mean],['R',c.recall_mean],['F1',c.f1_mean],['Capture',c.inverse_purity_mean]],failures=(c.failures||[]).map(f=>failurePill(f.failure_archetype,f.count)).join(''),cells=(c.rows||[]).map(r=>`<div class="cell-mini" role="button" tabindex="0" data-cell="${esc(r.cell_type)}"><span class="ct" title="${esc(r.cell_type)}">${esc(r.cell_type)}</span><span class="score">F1 ${esc(fmt(r.f1))} · ${esc(failureLabel(r.failure_archetype))}</span></div>`).join(''),empty=c.is_empty?'<span class="note">No registered rare populations in this dataset.</span>':'';return `<div class="scenario-card ${selectedScenario===c.scenario?'active':''}" role="button" tabindex="0" data-scenario-card="${esc(c.scenario)}"><div class="row" style="justify-content:space-between"><h4>${esc(c.scenario)}</h4><span class="status">${esc(c.n_cell_types)} types</span></div><div class="scenario-meta">${esc(fmt(c.total_cells))} cells · Preserved ${esc(pct(c.preserved_fraction))}</div><div class="mini-metrics">${metrics.map(([k,v])=>`<div class="mini-metric"><b>${esc(fmt(v))}</b><span>${esc(k)}</span></div>`).join('')}</div><div class="failure-list">${c.is_empty?empty:(failures||'<span class="note">No failure labels</span>')}</div><div class="cell-mini-list">${cells}</div></div>`}
function renderScenarioSelector(){const box=document.getElementById('scenarioSelector');if(!box)return;const cats=P.rare.category_breakdown||[];box.innerHTML=cats.map(c=>`<button class="scenario-chip ${selectedScenario===c.scenario?'active':''}" data-scenario-select="${esc(c.scenario)}">${esc(c.scenario)}</button>`).join('');box.querySelectorAll('[data-scenario-select]').forEach(b=>b.onclick=()=>{selectedScenario=b.dataset.scenarioSelect;renderScenarioBreakdown();renderScenarioSelector();renderScenarioDetail()})}
function renderScenarioBreakdown(){const box=document.getElementById('scenarioBreakdown');if(!box)return;const cats=P.rare.category_breakdown||[];if(!cats.length){box.innerHTML=emptyHTML('No scenario-level breakdown','Scenario analysis was disabled or no curated scenario metadata was available.');return}box.innerHTML=`<div class="scenario-grid">${cats.map(scenarioCard).join('')}</div>`;box.querySelectorAll('[data-scenario-card]').forEach(el=>{const act=e=>{if(e.target.closest('[data-cell]'))return;selectedScenario=el.dataset.scenarioCard;renderScenarioBreakdown();renderScenarioSelector();renderScenarioDetail()};el.addEventListener('click',act);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act(e)}})});box.querySelectorAll('[data-cell]').forEach(el=>{const act=e=>{e.stopPropagation();const row=P.rare.per_type.rows.find(r=>String(r.cell_type)===el.dataset.cell);if(row)selectRare(row,true)};el.addEventListener('click',act);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act(e)}})})}
function renderScenarioDetail(){const box=document.getElementById('scenarioDetail');if(!box)return;const cats=P.rare.category_breakdown||[],c=cats.find(x=>x.scenario===selectedScenario)||cats[0];if(!c){box.innerHTML=emptyHTML('No scenario breakdown','No cell-type-level scenario data is available.');return}selectedScenario=c.scenario;const data={columns:P.rare.per_type.columns,rows:c.rows||[]},notPreserved=(c.rows||[]).filter(r=>rareOutcome(r)==='not_preserved'),failureCounts={};notPreserved.forEach(r=>{const k=r.failure_archetype||'mixed_or_uncertain';failureCounts[k]=(failureCounts[k]||0)+1});const ranked=Object.entries(failureCounts).sort((a,b)=>b[1]-a[1]),top=ranked[0],topCells=top?(c.rows||[]).filter(r=>r.failure_archetype===top[0]).map(r=>r.cell_type):[];const outcomeNote=`<div class="callout note" style="margin-bottom:10px"><b>Outcome summary:</b> ${c.preserved_count||0} preserved · ${notPreserved.length} not preserved.${top?`<br><b>Most frequent failure mode among not-preserved populations:</b> ${esc(failureLabel(top[0]))} in ${esc(top[1])} cell type(s): ${esc(topCells.join(', '))}`:'<br>No failure mode was observed in this scenario.'}</div>`;box.innerHTML=`<div class="cards">${card('Scenario',c.scenario,`${c.distribution||''} × ${c.topology||''}`)+card('Rare cell types',c.n_cell_types,`${fmt(c.total_cells)} cells`)+card('Mean F1',c.f1_mean)+card('Preserved',pct(c.preserved_fraction),`${c.preserved_count||0} of ${c.n_cell_types||0}`)}</div>${outcomeNote}<div class="row" style="justify-content:space-between;margin-bottom:9px"><div class="note">Inspect the cell types that drive performance and failure in this scenario.</div><button class="btn light compact" id="applyScenarioFilter">Apply ${esc(c.scenario)} to Rare Explorer</button></div>${tableHTML(data,'scenarioDetailTable',{clickable:true})}`;wireTable(data,'scenarioDetailTable',row=>selectRare(row,true));document.getElementById('applyScenarioFilter').onclick=()=>{rareState.scenario=c.scenario;rareState.outcome='';rareState.failure='';rareState.query='';syncRareControls();refreshRareViews();document.getElementById('rareSearch')?.focus()}}
function renderRareScenarioPlots(){const cats=P.rare.category_breakdown||[];const p1=document.getElementById('rareScenarioMetrics'),p2=document.getElementById('rareFailurePlot');if(!cats.length){if(p1)p1.innerHTML=emptyHTML('No scenario metrics');if(p2)p2.innerHTML=emptyHTML('No outcome profile');return}const metrics=[['precision_mean','Precision'],['recall_mean','Recall'],['f1_mean','F1'],['inverse_purity_mean','Dominant-cluster capture']];Plotly.react('rareScenarioMetrics',metrics.map(([key,label])=>({type:'bar',name:label,x:cats.map(c=>c.scenario),y:cats.map(c=>c[key])})),{barmode:'group',margin:{l:50,r:15,t:20,b:55},yaxis:{range:[0,1],title:'Mean score'},xaxis:{title:'Rare scenario'},legend:{orientation:'h',y:1.12},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});const failOrder=['preserved','lineage_assimilation','lineage_leakage','batch_driven_fragmentation','mixed_or_uncertain'],cols=['#079669','#e57b36','#d94c64','#7652c8','#8090a5'];Plotly.react('rareFailurePlot',failOrder.map((f,j)=>({type:'bar',name:f==='preserved'?'Preserved':failureLabel(f),x:cats.map(c=>c.scenario),y:cats.map(c=>(c.failures||[]).find(x=>x.failure_archetype===f)?.count||0),marker:{color:cols[j]},customdata:cats.map(c=>({failure:f,cellTypes:(c.failures||[]).find(x=>x.failure_archetype===f)?.cell_types||[]})),hovertemplate:'%{x}<br>%{fullData.name}: %{y}<br>Cell types: %{customdata.cellTypes}<extra></extra>'})),{barmode:'stack',margin:{l:50,r:15,t:20,b:55},yaxis:{title:'Number of rare cell types',dtick:1},xaxis:{title:'Rare scenario'},legend:{orientation:'h',y:1.18},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});const metricEl=document.getElementById('rareScenarioMetrics'),failureEl=document.getElementById('rareFailurePlot');metricEl.removeAllListeners?.('plotly_click');metricEl.on?.('plotly_click',e=>{const scenario=e.points?.[0]?.x;if(scenario){selectedScenario=String(scenario);renderScenarioBreakdown();renderScenarioSelector();renderScenarioDetail()}});failureEl.removeAllListeners?.('plotly_click');failureEl.on?.('plotly_click',e=>{const p=e.points?.[0],scenario=p?.x,failure=p?.customdata?.failure;if(!scenario)return;selectedScenario=String(scenario);renderScenarioBreakdown();renderScenarioSelector();renderScenarioDetail();rareState.scenario=String(scenario);rareState.query='';rareState.focus=null;if(failure==='preserved'){rareState.outcome='preserved';rareState.failure=''}else{rareState.outcome='not_preserved';rareState.failure=String(failure||'')}syncRareControls();refreshRareViews()})}

function rareFilteredRows(){const rows=P.rare?.per_type?.rows||[],q=rareState.query.trim().toLowerCase();return rows.filter(r=>(!rareState.scenario||String(r.scenario||'')===rareState.scenario)&&(!rareState.outcome||rareOutcome(r)===rareState.outcome)&&(!rareState.failure||String(r.failure_archetype||'')===rareState.failure)&&(!q||String(r.cell_type||'').toLowerCase().includes(q)))}
function rareFilteredCellCount(rows=rareFilteredRows()){if(P.points){const types=new Set(rows.map(r=>String(r.cell_type)));let n=0;for(let i=0;i<P.points.x.length;i++)if(types.has(field('celltype',i)))n++;return n}return rows.reduce((a,r)=>a+(Number(r.support)||0),0)}
function syncRareControls(){const s=document.getElementById('rareScenario'),o1=document.getElementById('rareOutcome'),f=document.getElementById('rareFailure'),q=document.getElementById('rareSearch'),o=document.getElementById('rareOtherMode');if(s)s.value=rareState.scenario;if(o1)o1.value=rareState.outcome;if(f){f.value=rareState.failure;f.disabled=rareState.outcome==='preserved'}if(q)q.value=rareState.query;if(o)o.value=rareState.otherMode}
function renderRareFilterStatus(){const box=document.getElementById('rareFilterStatus'),matches=document.getElementById('rareMatches');if(!box||!matches)return;const rows=rareFilteredRows(),total=P.rare?.per_type?.rows?.length||0,cells=rareFilteredCellCount(rows),chips=[];if(rareState.scenario)chips.push(`<button class="filter-chip" data-clear-rare="scenario">Scenario: ${esc(rareState.scenario)} ×</button>`);if(rareState.outcome)chips.push(`<button class="filter-chip" data-clear-rare="outcome">Outcome: ${esc(outcomeLabel(rareState.outcome))} ×</button>`);if(rareState.failure)chips.push(`<button class="filter-chip" data-clear-rare="failure">Failure mode: ${esc(failureLabel(rareState.failure))} ×</button>`);if(rareState.query)chips.push(`<button class="filter-chip" data-clear-rare="query">Search: ${esc(rareState.query)} ×</button>`);box.innerHTML=`<div class="status-line"><b>${rows.length} of ${total}</b> rare populations · <b>${fmt(cells)}</b> cells ${chips.join('')}</div>`;matches.innerHTML=rows.length?`<div class="control-label">Matching populations</div><div class="match-list">${rows.map(r=>`<button class="match-chip ${rareState.focus===String(r.cell_type)?'active':''}" data-rare-match="${esc(r.cell_type)}">${esc(r.cell_type)}</button>`).join('')}</div>`:emptyHTML('No matching rare populations','Adjust the scenario, outcome, failure mode, or cell-type search.');box.querySelectorAll('[data-clear-rare]').forEach(b=>b.onclick=()=>{rareState[b.dataset.clearRare]='';if(b.dataset.clearRare==='outcome'&&rareState.failure)rareState.failure='';syncRareControls();refreshRareViews()});matches.querySelectorAll('[data-rare-match]').forEach(b=>b.onclick=()=>{const row=P.rare.per_type.rows.find(r=>String(r.cell_type)===b.dataset.rareMatch);if(row)selectRare(row,false)})}
function renderRareDetail(){const d=document.getElementById('rareDetail');if(!d)return;const rows=rareFilteredRows(),row=rows.find(r=>String(r.cell_type)===rareState.focus);if(!row){let msg='Select a matching population above, click a row in the rare-metrics table, heatmap, or rare UMAP.';if(rareState.query&&rows.length>1)msg=`${rows.length} populations match “${rareState.query}”. Select one of the matching population chips above.`;if(!rows.length)msg='No rare population matches the current filters.';d.innerHTML=`<h3>Rare population details</h3><div class="note">${esc(msg)}</div>`;return}const outcome=rareOutcome(row),actions=[S.umap?'<button id="toMainUmap" class="btn">View on main UMAP</button>':'',S.sankey?'<button id="toSankey" class="btn light">View in Sankey</button>':''].filter(Boolean).join('');d.innerHTML=`<div class="row" style="justify-content:space-between"><h3>${esc(row.cell_type)}</h3><div>${outcomePill(outcome)} ${failurePill(row.failure_archetype)}</div></div><div class="scenario-meta">${esc(row.scenario||'')} · ${esc(row.curation_source||'curated population')}${row.parent_type?` · Parent: ${esc(row.parent_type)}`:''}</div><div class="detail-grid"><div class="detail-block"><div class="dk">Precision</div><div class="dv">${esc(fmt(row.precision))}</div></div><div class="detail-block"><div class="dk">Recall</div><div class="dv">${esc(fmt(row.recall))}</div></div><div class="detail-block"><div class="dk">F1</div><div class="dv">${esc(fmt(row.f1))}</div></div><div class="detail-block"><div class="dk">Dominant-cluster capture</div><div class="dv">${esc(fmt(row.inverse_purity))}</div></div><div class="detail-block"><div class="dk">Support</div><div class="dv">${esc(fmt(row.support))}</div></div><div class="detail-block"><div class="dk">Within-type batch NMI</div><div class="dv">${esc(fmt(row.within_type_batch_nmi))}</div></div></div><div class="callout note" style="margin-top:9px"><b>${outcome==='preserved'?'Preservation assessment':'Failure assessment'}:</b> ${esc(row.failure_rationale||'No rationale recorded.')}<br>${row.dominant_wrong_label?`Dominant wrong label: <b>${esc(row.dominant_wrong_label)}</b> (${esc(fmt(row.dominant_wrong_fraction))}). `:''}${row.n_clusters_found_in!==undefined?`Target cells appear in ${esc(fmt(row.n_clusters_found_in))} cluster(s).`:''}</div>${actions?`<div class="row" style="margin-top:10px">${actions}</div>`:''}`;if(S.umap&&document.getElementById('toMainUmap'))document.getElementById('toMainUmap').onclick=()=>{openTab('umap');const cb=document.getElementById('colorBy');cb.value='celltype';activeFlow=null;umapState.selected=new Set([String(row.cell_type)]);refreshHighlight();renderUmapSelectionDetail('celltype',String(row.cell_type));renderUmap()};if(S.sankey&&document.getElementById('toSankey'))document.getElementById('toSankey').onclick=()=>{openTab('sankey');document.getElementById('highlightLabel').value=row.cell_type;renderSankey()}}
function selectRare(row,sync=false){if(!row)return;rareState.focus=String(row.cell_type||'');if(sync){rareState.scenario=String(row.scenario||'');rareState.outcome=rareOutcome(row);rareState.failure=rareOutcome(row)==='not_preserved'?String(row.failure_archetype||''):'';rareState.query=String(row.cell_type||'');syncRareControls()}refreshRareViews({autoFocus:false})}
function renderRareTable(){const box=document.getElementById('rareTableBox');if(!box)return;const base=P.rare.per_type||{columns:[],rows:[]},data={columns:base.columns,rows:rareFilteredRows()};box.innerHTML=tableHTML(data,'rareTable',{clickable:true,showSearch:false});wireTable(data,'rareTable',row=>selectRare(row,false),base)}
function renderRareUmap(){if(!F.rare_umap||!P.points||!document.getElementById('rareUmap'))return;const rows=rareFilteredRows(),matchSet=new Set(rows.map(r=>String(r.cell_type))),allRare=new Set(P.rare.rare_types||[]),context=[],groups={};for(let i=0;i<P.points.x.length;i++){const ct=field('celltype',i);if(matchSet.has(ct))(groups[ct]??=[]).push(i);else if(rareState.otherMode==='gray')context.push(i)}const cats=Object.keys(groups),cols=palette(cats.length),tr=[];if(context.length)tr.push({type:'scattergl',mode:'markers',name:'Other cells',x:context.map(i=>P.points.x[i]),y:context.map(i=>P.points.y[i]),hoverinfo:'skip',marker:{size:3,color:'rgba(150,160,175,.13)'}});cats.forEach((ct,j)=>{const idx=groups[ct],focused=!rareState.focus||rareState.focus===ct;tr.push({type:'scattergl',mode:'markers',name:ct,x:idx.map(i=>P.points.x[i]),y:idx.map(i=>P.points.y[i]),text:hover(idx),hovertemplate:'%{text}<extra></extra>',marker:{size:focused?6:4,color:focused?cols[j]:'rgba(155,165,180,.18)',line:focused?{width:.3,color:'#23324d'}:undefined}})});const el=document.getElementById('rareUmap');if(!cats.length){try{Plotly.purge(el)}catch(_){}el.innerHTML=emptyHTML('No rare populations match the current filters','The gray context is hidden when no target population is selected.');return}Plotly.react(el,tr,{margin:{l:45,r:20,t:18,b:45},xaxis:{title:(P.points.embedding_label||'Embedding')+' 1'},yaxis:{title:(P.points.embedding_label||'Embedding')+' 2',scaleanchor:'x'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});el.removeAllListeners?.('plotly_click');el.on?.('plotly_click',e=>{const name=e.points?.[0]?.data?.name;if(name&&name!=='Other cells'){const row=P.rare.per_type.rows.find(r=>String(r.cell_type)===String(name));if(row)selectRare(row,false)}})}
function renderRareHeatmaps(){if(!F.rare_heatmaps)return;const recoveryEl=document.getElementById('rareRecoveryHeatmap'),batchEl=document.getElementById('rareBatchHeatmap');if(!recoveryEl||!batchEl)return;const R=P.rare.per_type||{columns:[]},rows=rareFilteredRows(),recovery=['precision','recall','f1','inverse_purity'].filter(x=>R.columns.includes(x));if(!rows.length){[recoveryEl,batchEl].forEach(el=>{try{Plotly.purge(el)}catch(_){}el.innerHTML=emptyHTML('No matching rare populations','Change the active Rare-cell filters.')});return}if(recovery.length)Plotly.react(recoveryEl,[{type:'heatmap',z:rows.map(r=>recovery.map(m=>r[m])),x:recovery.map(colLabel),y:rows.map(r=>r.cell_type),zmin:0,zmax:1,colorscale:'Blues',hovertemplate:'%{y}<br>%{x}: %{z:.3f}<br>Click to inspect this population<extra></extra>'}],{height:Math.max(390,rows.length*25),margin:{l:190,r:20,t:20,b:85},paper_bgcolor:'white'},{responsive:true,displaylogo:false});else recoveryEl.innerHTML=emptyHTML('Recovery metrics unavailable');if(R.columns.includes('within_type_batch_nmi'))Plotly.react(batchEl,[{type:'heatmap',z:rows.map(r=>[r.within_type_batch_nmi]),x:['Within-type batch NMI'],y:rows.map(r=>r.cell_type),zmin:0,zmax:1,colorscale:'RdYlGn',reversescale:true,hovertemplate:'%{y}<br>Within-type batch NMI: %{z:.3f}<br>Click to inspect this population<extra></extra>'}],{height:Math.max(390,rows.length*25),margin:{l:190,r:20,t:20,b:85},paper_bgcolor:'white'},{responsive:true,displaylogo:false});else batchEl.innerHTML=emptyHTML('Batch-dependence metric unavailable');[recoveryEl,batchEl].forEach(el=>{el.removeAllListeners?.('plotly_click');el.on?.('plotly_click',e=>{const ct=e.points?.[0]?.y,row=P.rare.per_type.rows.find(r=>String(r.cell_type)===String(ct));if(row)selectRare(row,false)})})}
function refreshRareViews({autoFocus=true}={}){const rows=rareFilteredRows();if(rareState.focus&&!rows.some(r=>String(r.cell_type)===rareState.focus))rareState.focus=null;if(autoFocus&&rareState.query.trim()&&rows.length===1)rareState.focus=String(rows[0].cell_type);renderRareFilterStatus();renderRareDetail();renderRareTable();renderRareUmap();renderRareHeatmaps()}

function renderRare(){if(!S.rare)return;const R=P.rare||{},scenarios=(R.category_breakdown||[]).map(x=>x.scenario),allScenarios=[...new Set((R.per_type?.rows||[]).map(r=>r.scenario).filter(Boolean))],scenarioOptions=scenarios.length?scenarios:allScenarios,fails=[...new Set((R.per_type?.rows||[]).map(r=>r.failure_archetype).filter(x=>x&&x!=='preserved'))].sort(),summaryCards=(R.summary?.rows||[]).map(r=>r.metric==='preserved_fraction'?card('Preserved populations',pct(r.mean),`${Math.round((r.mean||0)*(P.meta.n_rare_types||0))} of ${P.meta.n_rare_types||0}`):card('Mean rare '+colLabel(r.metric),r.mean,`Median ${fmt(r.median)} · n=${fmt(r.n_valid)}`)).join('');const scenarioAnalysis=F.rare_scenario_analysis?`<div class="panel"><div class="panel-title"><div class="panel-title-main">Rare-scenario taxonomy<div class="panel-subtitle">Distribution × topology classes used by benchmark curation</div></div></div><div class="panel-body">${scenarioGuide()}</div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">Performance and outcome by rare scenario<div class="panel-subtitle">Shows which rare categories and cell types are preserved or drive specific failure modes</div></div></div><div class="panel-body"><div id="scenarioBreakdown"></div></div></div><div class="grid-half"><div class="panel"><div class="panel-title"><span>Mean metrics across rare scenarios</span><div class="panel-actions"><button class="btn ghost compact" id="rareScenarioPng">Export PNG</button></div></div><div class="panel-body"><div id="rareScenarioMetrics" class="plot smallplot"></div></div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">Outcome and failure-mode profile across rare scenarios<div class="panel-subtitle">Preserved is an outcome; the remaining stacks partition not-preserved populations by failure mode</div></div><div class="panel-actions"><button class="btn ghost compact" id="rareFailurePng">Export PNG</button></div></div><div class="panel-body"><div id="rareFailurePlot" class="plot smallplot"></div></div></div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">Cell-type performance within selected rare scenario<div class="panel-subtitle">Choose any scenario below; GR-DL is only the initial selection</div></div></div><div class="panel-body"><div id="scenarioSelector" class="scenario-selector"></div><div id="scenarioDetail"></div></div></div>`:'';const rareUmapPanel=F.rare_umap?`<div class="panel"><div class="panel-title"><div class="panel-title-main">Rare-cell UMAP<div class="panel-subtitle">Uses the same Scenario / Outcome / Failure mode / Cell-type filters as the table and heatmaps</div></div><div class="panel-actions"><button class="btn ghost compact" id="rarePng">Export PNG</button></div></div><div class="panel-body"><div id="rareUmap" class="plot"></div></div></div>`:'';const heatmaps=F.rare_heatmaps?`<div class="panel"><div class="panel-title"><div class="panel-title-main">Filtered rare-cell metric heatmaps<div class="panel-subtitle">Both heatmaps follow exactly the same active Rare-cell filters; click a heatmap row/cell to inspect that population</div></div></div><div class="panel-body"><div class="heatmap-note"><div class="callout"><b>Recovery quality</b><br>Precision, recall, F1, and dominant-cluster capture. Higher values indicate better target recovery.</div><div class="callout"><b>Batch dependence</b><br>Within-type batch NMI is shown separately because lower values indicate less batch-associated cluster structure.</div></div><div class="grid-half"><div><div class="row" style="justify-content:space-between"><b>Recovery metrics</b><button class="btn ghost compact" id="rareRecoveryPng">Export PNG</button></div><div id="rareRecoveryHeatmap" class="plot smallplot"></div></div><div><div class="row" style="justify-content:space-between"><b>Batch-dependence diagnostic</b><button class="btn ghost compact" id="rareBatchPng">Export PNG</button></div><div id="rareBatchHeatmap" class="plot smallplot"></div></div></div></div></div>`:'';document.getElementById('tab-rare').innerHTML=`<div class="cards">${summaryCards||card('Rare populations',P.meta.n_rare_types)}</div>${scenarioAnalysis}<div class="grid2 ${F.rare_umap?'':'rare-layout-no-umap'}"><div><div class="panel"><div class="panel-title"><div class="panel-title-main">Rare-cell filters<div class="panel-subtitle">One shared filter state controls the rare table, UMAP, heatmaps, and population details</div></div><div class="panel-actions"><button class="btn ghost compact" id="rareReset">Reset filters</button></div></div><div class="panel-body controls"><div><label for="rareScenario">Scenario</label><select id="rareScenario"><option value="">All scenarios</option>${scenarioOptions.map(x=>`<option>${esc(x)}</option>`).join('')}</select></div><div><label for="rareOutcome">Outcome</label><select id="rareOutcome"><option value="">All outcomes</option><option value="preserved">Preserved</option><option value="not_preserved">Not preserved</option></select></div><div><label for="rareFailure">Failure mode <span class="note">(applies to not-preserved populations)</span></label><select id="rareFailure"><option value="">All failure modes</option>${fails.map(x=>`<option value="${esc(x)}">${esc(failureLabel(x))}</option>`).join('')}</select></div><div><label for="rareSearch">Search rare population</label><input id="rareSearch" type="search" placeholder="Cell type name, e.g. ILC or pDC" autocomplete="off"></div>${F.rare_umap?`<div><label for="rareOtherMode">Other cells in Rare UMAP</label><select id="rareOtherMode"><option value="gray">Show in gray for context</option><option value="hide">Hide non-matching cells</option></select></div>`:''}<div id="rareFilterStatus"></div><div id="rareMatches"></div><div id="rareDetail" class="rare-focus"></div></div></div><div class="panel"><div class="panel-title">Aggregated rare scenario summary</div><div class="panel-body">${tableHTML(R.scenarios,'rareScenarioTable')}</div></div></div>${rareUmapPanel}</div>${heatmaps}${figureCards('Rare-cell / benchmark')?`<div class="panel"><div class="panel-title">Rare-cell / benchmark static figures</div><div class="panel-body">${figureCards('Rare-cell / benchmark')}</div></div>`:''}<div class="panel"><div class="panel-title"><div class="panel-title-main">Rare metrics per population<div class="panel-subtitle">The displayed rows follow the shared Rare-cell filters; Export all downloads every curated rare population with raw columns/full precision</div></div></div><div class="panel-body"><div id="rareTableBox"></div></div></div>`;selectedScenario=scenarioOptions[0]||null;wireTable(R.scenarios,'rareScenarioTable');if(F.rare_scenario_analysis){renderScenarioBreakdown();renderScenarioSelector();renderScenarioDetail();renderRareScenarioPlots();document.getElementById('rareScenarioPng').onclick=()=>downloadPlot('rareScenarioMetrics','scrarebench_rare_scenario_metrics');document.getElementById('rareFailurePng').onclick=()=>downloadPlot('rareFailurePlot','scrarebench_rare_outcome_profile')}syncRareControls();document.getElementById('rareScenario').onchange=e=>{rareState.scenario=e.target.value;refreshRareViews()};document.getElementById('rareOutcome').onchange=e=>{rareState.outcome=e.target.value;if(rareState.outcome==='preserved')rareState.failure='';syncRareControls();refreshRareViews()};document.getElementById('rareFailure').onchange=e=>{rareState.failure=e.target.value;if(rareState.failure)rareState.outcome='not_preserved';syncRareControls();refreshRareViews()};document.getElementById('rareSearch').oninput=e=>{rareState.query=e.target.value;clearTimeout(rareSearchTimer);rareSearchTimer=setTimeout(()=>refreshRareViews(),140)};if(F.rare_umap)document.getElementById('rareOtherMode').onchange=e=>{rareState.otherMode=e.target.value;renderRareUmap()};document.getElementById('rareReset').onclick=()=>{rareState.scenario='';rareState.outcome='';rareState.failure='';rareState.query='';rareState.focus=null;rareState.otherMode='gray';syncRareControls();refreshRareViews()};if(F.rare_umap)document.getElementById('rarePng').onclick=()=>downloadPlot('rareUmap','scrarebench_rare_umap');if(F.rare_heatmaps){document.getElementById('rareRecoveryPng').onclick=()=>downloadPlot('rareRecoveryHeatmap','scrarebench_rare_recovery_heatmap');document.getElementById('rareBatchPng').onclick=()=>downloadPlot('rareBatchHeatmap','scrarebench_rare_batch_nmi_heatmap')}bindFigureActions(document.getElementById('tab-rare'));refreshRareViews()}

const umapState={selected:new Set(),focus:null};
function refreshHighlight(){if(!P.points)return;const sel=document.getElementById('colorBy'),box=document.getElementById('highlightValues');if(!sel||!box)return;const f=P.points.fields[sel.value];umapState.selected=new Set([...umapState.selected].filter(x=>f.categories.includes(x)));box.innerHTML=f.categories.map(x=>`<option value="${esc(x)}" ${umapState.selected.has(x)?'selected':''}>${esc(x)}</option>`).join('');updateUmapStatus()}
function updateUmapStatus(){const el=document.getElementById('umapStatus');if(!el)return;if(activeFlow){const n=flowIndices()?.size||0;el.textContent=`Sankey selection: ${activeFlow.source} → ${activeFlow.target} · ${fmt(n)} cells`;return}el.textContent=umapState.selected.size?`${umapState.selected.size} highlighted categor${umapState.selected.size===1?'y':'ies'}`:'All categories colored'}
function flowIndices(){if(!activeFlow||!P.points)return null;const out=new Set;for(let i=0;i<P.points.x.length;i++){if(field('celltype',i)!==activeFlow.source)continue;const t=activeFlow.mode==='true_to_cluster'?field('cluster',i):field('prediction',i);if(t===activeFlow.target)out.add(i)}return out}
function umapCategoryCount(fieldName,value){if(!P.points)return 0;let n=0;for(let i=0;i<P.points.x.length;i++)if(field(fieldName,i)===value)n++;return n}
function renderUmapSelectionDetail(fieldName=null,value=null){const box=document.getElementById('umapSelectionDetail');if(!box)return;if(activeFlow){const n=flowIndices()?.size||0;box.innerHTML=`<h3>Sankey-linked selection</h3><div class="scenario-meta">${esc(activeFlow.source)} → ${esc(activeFlow.target)}</div><div class="detail-grid"><div class="detail-block"><div class="dk">Selected cells</div><div class="dv">${esc(fmt(n))}</div></div><div class="detail-block"><div class="dk">Flow mode</div><div class="dv">${esc(activeFlow.mode.replaceAll('_',' '))}</div></div></div>`;return}if(!fieldName||!value){box.innerHTML=`<h3>Selection details</h3><div class="note">Click a colored population/category in the embedding to focus it and inspect its details.</div>`;return}const count=umapCategoryCount(fieldName,value),label=({celltype:'True cell type',batch:'Batch',scenario:'Rare scenario',cluster:'Leiden cluster',prediction:'Predicted cell type',failure_archetype:'Failure archetype',is_rare:'Rare / non-rare'})[fieldName]||fieldName,rareRow=fieldName==='celltype'?(P.rare?.per_type?.rows||[]).find(r=>String(r.cell_type)===String(value)):null;const actions=[];if(rareRow&&S.rare)actions.push('<button class="btn" id="umapToRare">Open in Rare-cell Explorer</button>');if(fieldName==='celltype'&&S.sankey)actions.push('<button class="btn light" id="umapToSankey">Highlight in Sankey</button>');box.innerHTML=`<div class="row" style="justify-content:space-between"><h3>${esc(value)}</h3>${rareRow?outcomePill(rareOutcome(rareRow)):''}</div><div class="scenario-meta">${esc(label)}</div><div class="detail-grid"><div class="detail-block"><div class="dk">Cells</div><div class="dv">${esc(fmt(count))}</div></div>${rareRow?`<div class="detail-block"><div class="dk">Rare scenario</div><div class="dv">${esc(rareRow.scenario||'—')}</div></div><div class="detail-block"><div class="dk">Rare F1</div><div class="dv">${esc(fmt(rareRow.f1))}</div></div><div class="detail-block"><div class="dk">Failure mode</div><div class="dv">${esc(failureLabel(rareRow.failure_archetype))}</div></div>`:''}</div>${actions.length?`<div class="row" style="margin-top:10px">${actions.join('')}</div>`:''}`;document.getElementById('umapToRare')?.addEventListener('click',()=>{openTab('rare');selectRare(rareRow,true)});document.getElementById('umapToSankey')?.addEventListener('click',()=>{openTab('sankey');document.getElementById('highlightLabel').value=value;renderSankey()})}

function renderUmap(){if(!S.umap||!P.points||!document.getElementById('umapPlot'))return;const by=document.getElementById('colorBy').value,mode=document.getElementById('otherMode').value,flow=flowIndices(),f=P.points.fields[by],cols=palette(f.categories.length),tr=[];if(flow){const a=[],o=[];for(let i=0;i<P.points.x.length;i++)(flow.has(i)?a:o).push(i);if(mode==='gray'&&o.length)tr.push({type:'scattergl',mode:'markers',name:'Other cells',x:o.map(i=>P.points.x[i]),y:o.map(i=>P.points.y[i]),hoverinfo:'skip',marker:{size:3,color:'rgba(150,160,175,.18)'}});tr.push({type:'scattergl',mode:'markers',name:`${activeFlow.source} → ${activeFlow.target}`,x:a.map(i=>P.points.x[i]),y:a.map(i=>P.points.y[i]),text:hover(a),hovertemplate:'%{text}<extra></extra>',marker:{size:6,color:'#ed762d'}})}else{const other=[];f.categories.forEach((cat,j)=>{const idx=[];for(let i=0;i<P.points.x.length;i++)if(field(by,i)===cat)idx.push(i);if(umapState.selected.size&&!umapState.selected.has(cat)){other.push(...idx);return}tr.push({type:'scattergl',mode:'markers',name:cat,x:idx.map(i=>P.points.x[i]),y:idx.map(i=>P.points.y[i]),text:hover(idx),hovertemplate:'%{text}<extra></extra>',marker:{size:umapState.selected.size?5:4,color:cols[j]}})});if(umapState.selected.size&&mode==='gray'&&other.length)tr.unshift({type:'scattergl',mode:'markers',name:'Other cells',x:other.map(i=>P.points.x[i]),y:other.map(i=>P.points.y[i]),hoverinfo:'skip',marker:{size:3,color:'rgba(150,160,175,.20)'}})}const el=document.getElementById('umapPlot');Plotly.react(el,tr,{margin:{l:50,r:20,t:20,b:50},xaxis:{title:(P.points.embedding_label||'Embedding')+' 1'},yaxis:{title:(P.points.embedding_label||'Embedding')+' 2',scaleanchor:'x'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false});el.removeAllListeners?.('plotly_click');el.on?.('plotly_click',e=>{if(activeFlow)return;const name=e.points?.[0]?.data?.name;if(!name||name==='Other cells')return;umapState.focus=String(name);umapState.selected=new Set([String(name)]);[...document.getElementById('highlightValues').options].forEach(o=>o.selected=o.value===name);renderUmapSelectionDetail(by,String(name));renderUmap()});updateUmapStatus();if(activeFlow)renderUmapSelectionDetail();else if(umapState.focus)renderUmapSelectionDetail(by,umapState.focus)}
function renderUmapTab(){if(!S.umap)return;const sankeyControls=S.sankey?`<button class="btn light" id="clearFlow">Clear Sankey selection</button><div class="callout note">A Sankey link click highlights the exact cells belonging to that flow in this embedding.</div>`:`<div class="callout note">This embedding is standalone because the Sankey section was excluded from this report.</div>`;document.getElementById('tab-umap').innerHTML=`<div class="grid2"><div><div class="panel"><div class="panel-title"><div class="panel-title-main">Embedding controls<div class="panel-subtitle" id="umapStatus">All categories colored</div></div><div class="panel-actions"><button class="btn ghost compact" id="umapPng">Export PNG</button></div></div><div class="panel-body controls"><div><label for="colorBy">Color by</label><select id="colorBy">${[['celltype','True cell type'],['batch','Batch'],['scenario','Rare scenario'],['cluster','Leiden cluster'],['prediction','Predicted cell type'],['failure_archetype','Failure archetype'],['is_rare','Rare / non-rare']].map(x=>`<option value="${x[0]}">${x[1]}</option>`).join('')}</select></div><div><label for="highlightValues">Keep selected categories colored</label><select id="highlightValues" multiple></select><div class="row" style="margin-top:6px"><button class="btn ghost compact" id="selectAllHighlights">Select all</button><button class="btn ghost compact" id="clearHighlights">Clear selection</button></div></div><div><label for="otherMode">Other cells</label><select id="otherMode"><option value="gray">Show in gray</option><option value="hide">Hide</option></select></div>${sankeyControls}</div></div><div id="umapSelectionDetail" class="rare-focus selection-detail"><h3>Selection details</h3><div class="note">Click a colored population/category in the embedding to focus it and inspect its details.</div></div></div><div class="panel"><div class="panel-title"><div class="panel-title-main">UMAP / embedding view<div class="panel-subtitle">Click a colored category to focus it; this affects visualization only, not benchmark scores.</div></div></div><div class="panel-body"><div id="umapPlot" class="plot"></div></div></div></div>`;refreshHighlight();document.getElementById('colorBy').onchange=()=>{umapState.selected.clear();umapState.focus=null;refreshHighlight();activeFlow=null;renderUmapSelectionDetail();renderUmap()};document.getElementById('highlightValues').onchange=e=>{umapState.selected=new Set([...e.target.selectedOptions].map(o=>o.value));umapState.focus=umapState.selected.size===1?[...umapState.selected][0]:null;activeFlow=null;renderUmapSelectionDetail(document.getElementById('colorBy').value,umapState.focus);renderUmap()};document.getElementById('otherMode').onchange=renderUmap;document.getElementById('selectAllHighlights').onclick=()=>{const opts=[...document.getElementById('highlightValues').options];umapState.selected=new Set(opts.map(o=>o.value));umapState.focus=null;opts.forEach(o=>o.selected=true);activeFlow=null;renderUmapSelectionDetail();renderUmap()};document.getElementById('clearHighlights').onclick=()=>{umapState.selected.clear();umapState.focus=null;[...document.getElementById('highlightValues').options].forEach(o=>o.selected=false);activeFlow=null;renderUmapSelectionDetail();renderUmap()};document.getElementById('umapPng').onclick=()=>downloadPlot('umapPlot','scrarebench_umap');if(S.sankey)document.getElementById('clearFlow').onclick=()=>{activeFlow=null;renderUmapSelectionDetail(document.getElementById('colorBy').value,umapState.focus);renderUmap()};renderUmap()}

function sankeyRows(){const mode=document.getElementById('sankeyMode').value,th=Math.max(1,Number(document.getElementById('flowNumber').value||1));return (P.sankey?.[mode]||[]).filter(r=>r.count>=th)}
function updateSankeyRange(){const mode=document.getElementById('sankeyMode').value,all=P.sankey?.[mode]||[],max=Math.max(1,...all.map(r=>Number(r.count)||0)),range=document.getElementById('flowRange'),num=document.getElementById('flowNumber');range.max=String(max);num.max=String(max);if(Number(num.value)>max){num.value=String(max);range.value=String(max)}}
function renderSankey(){if(!S.sankey||!document.getElementById('sankeyPlot'))return;const mode=document.getElementById('sankeyMode').value,hi=document.getElementById('highlightLabel').value,rows=sankeyRows(),summary=document.getElementById('sankeyStatus'),el=document.getElementById('sankeyPlot');if(summary)summary.textContent=`${rows.length} visible flows · ${fmt(rows.reduce((a,r)=>a+(Number(r.count)||0),0))} cells represented across links`;if(!rows.length){try{Plotly.purge(el)}catch(_){}el.innerHTML=emptyHTML('No Sankey flows match the threshold','Lower the minimum flow count or change the flow definition.');return}const src=[...new Set(rows.map(r=>r.source))],tgt=[...new Set(rows.map(r=>r.target))],sm=new Map(src.map((x,i)=>[x,i])),tm=new Map(tgt.map((x,i)=>[x,i+src.length])),custom=rows.map(r=>[r.source,r.target,r.count]);const data=[{type:'sankey',arrangement:'snap',node:{label:src.map(x=>'True: '+x).concat(tgt.map(x=>(mode==='true_to_cluster'?'Cluster: ':'Target: ')+x)),pad:12,thickness:15,color:src.map(()=>"rgba(67,88,194,.9)").concat(tgt.map(()=>"rgba(92,113,207,.62)"))},link:{source:rows.map(r=>sm.get(r.source)),target:rows.map(r=>tm.get(r.target)),value:rows.map(r=>r.count),customdata:custom,color:rows.map(r=>!hi?'rgba(128,140,160,.34)':r.source===hi?'rgba(237,118,45,.76)':r.target===hi?'rgba(49,87,213,.74)':'rgba(160,170,185,.10)'),hovertemplate:'Source: %{customdata[0]}<br>Target: %{customdata[1]}<br>Count: %{customdata[2]}<extra></extra>'}}];Plotly.react(el,data,{margin:{l:15,r:15,t:20,b:20},paper_bgcolor:'white'},{responsive:true,displaylogo:false});el.removeAllListeners?.('plotly_click');el.on?.('plotly_click',e=>{const p=e.points?.[0];if(!p?.customdata)return;activeFlow={mode,source:p.customdata[0],target:p.customdata[1]};if(S.umap){openTab('umap');renderUmapSelectionDetail();renderUmap()}else toast('The UMAP section was not included in this report.','error')})}
function renderSankeyTab(){if(!S.sankey)return;document.getElementById('tab-sankey').innerHTML=`<div class="grid2"><div class="panel"><div class="panel-title"><div class="panel-title-main">Sankey controls<div class="panel-subtitle" id="sankeyStatus">Flow explorer</div></div><div class="panel-actions"><button class="btn ghost compact" id="sankeyPng">Export PNG</button></div></div><div class="panel-body controls"><div><label for="sankeyMode">Flow definition</label><select id="sankeyMode"><option value="true_to_prediction">True type → predicted type</option><option value="true_to_cluster">True type → Leiden cluster</option><option value="rare_true_to_prediction">Rare true type → predicted type</option></select></div><div><label for="flowRange">Minimum flow count</label><input id="flowRange" type="range" min="1" value="1"><input id="flowNumber" type="number" min="1" value="1" style="margin-top:6px"></div><div><label for="highlightLabel">Highlight cell type</label><select id="highlightLabel"><option value="">No special highlight</option>${(P.sankey.highlight_options||[]).map(x=>`<option>${esc(x)}</option>`).join('')}</select></div><button class="btn light" id="resetSankey">Reset Sankey controls</button><div class="callout note">Click a connection to inspect the exact contributing cells in the UMAP tab when UMAP is included.</div></div></div><div class="panel"><div class="panel-title">Ground-truth flow explorer</div><div class="panel-body"><div id="sankeyPlot" class="plot"></div></div></div></div>`;updateSankeyRange();document.getElementById('sankeyMode').onchange=()=>{updateSankeyRange();renderSankey()};document.getElementById('highlightLabel').onchange=renderSankey;document.getElementById('flowRange').oninput=e=>{document.getElementById('flowNumber').value=e.target.value;renderSankey()};document.getElementById('flowNumber').oninput=e=>{const range=document.getElementById('flowRange'),max=Number(range.max||1),raw=Number(e.target.value),v=Math.min(max,Math.max(1,Number.isFinite(raw)?raw:1));e.target.value=String(v);range.value=String(v);renderSankey()};document.getElementById('sankeyPng').onclick=()=>downloadPlot('sankeyPlot','scrarebench_sankey');document.getElementById('resetSankey').onclick=()=>{document.getElementById('sankeyMode').value='true_to_prediction';document.getElementById('highlightLabel').value='';document.getElementById('flowNumber').value='1';activeFlow=null;updateSankeyRange();document.getElementById('flowRange').value='1';renderSankey()};renderSankey()}

function renderRepro(){if(!S.reproducibility)return;const R=P.reproducibility||{},blocks=[['Run config',R.run_config,'run_config.yaml'],['Failure rules',R.failure_rules,'failure_rules.yaml'],['Package versions',R.package_versions,'package_versions.txt'],['scIB reference config',R.scib_reference_config,'scib_reference_config.yaml']].filter(x=>x[1]);document.getElementById('tab-reproducibility').innerHTML=`<div class="cards">${card('Representation',P.meta.representation_key)+card('Cluster key',P.meta.cluster_key)+card('Prediction key',P.meta.prediction_key)+card('Embedding',P.meta.embedding_key||'not embedded')}</div>${blocks.length?blocks.map(([t,v,name],i)=>`<div class="panel"><div class="panel-title"><span>${esc(t)}</span><div class="panel-actions"><button class="btn ghost compact" data-repro="${i}">Download ${esc(name)}</button></div></div><div class="panel-body"><div id="repro-${i}" class="pre">${esc(v)}</div></div></div>`).join(''):emptyHTML('No reproducibility text artifacts were embedded','The benchmark may have been constructed without these optional files.')}`;document.querySelectorAll('[data-repro]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.repro),name=blocks[i][2],el=document.getElementById('repro-'+i);downloadText(name,el.textContent,'text/plain;charset=utf-8')})}

function applyFigureFilters(){const group=document.getElementById('figureGroup')?.value||'',q=(document.getElementById('figureSearch')?.value||'').trim().toLowerCase();let visible=0;document.querySelectorAll('#figureGallery [data-figure-card]').forEach(card=>{const okGroup=!group||card.dataset.figureGroup===group,okSearch=!q||card.dataset.figureSearch.includes(q),show=okGroup&&okSearch;card.style.display=show?'':'none';if(show)visible++});const status=document.getElementById('figureStatus');if(status)status.textContent=`${visible} figure${visible===1?'':'s'} shown`}
function renderFigures(){if(!S.figures)return;const figs=P.figures||[],groups=[...new Set(figs.map(f=>f.group).filter(Boolean))];document.getElementById('tab-figures').innerHTML=`<div class="panel"><div class="panel-title"><div class="panel-title-main">Embedded static figures<div class="panel-subtitle" id="figureStatus">${figs.length} figures</div></div></div><div class="panel-body"><p class="note">Click a figure to enlarge it. Lightbox zoom preserves a scrollable high-resolution view, and Download original saves the embedded image rather than a screenshot.</p><div class="figure-toolbar"><input id="figureSearch" type="search" placeholder="Search figure title or filename" aria-label="Search figures"><select id="figureGroup" aria-label="Filter figures by group"><option value="">All figure groups</option>${groups.map(g=>`<option>${esc(g)}</option>`).join('')}</select></div><div id="figureGallery">${figureCards(null)||emptyHTML('No static figures were embedded')}</div></div></div>`;bindFigureActions(document.getElementById('tab-figures'));document.getElementById('figureSearch').oninput=applyFigureFilters;document.getElementById('figureGroup').onchange=applyFigureFilters;applyFigureFilters()}

const renderers={overview:renderOverview,metrics:renderMetrics,scib:renderScib,rare:renderRare,umap:renderUmapTab,sankey:renderSankeyTab,reproducibility:renderRepro,figures:renderFigures};
if(tabs.length){const requested=location.hash?.replace('#','');const initial=tabs.some(([id])=>id===requested)?requested:tabs[0][0];openTab(initial,{scroll:false})}
</script>
</body>
</html>
'''
