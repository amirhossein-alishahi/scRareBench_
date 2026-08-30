from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ._version import __version__
from .constants import BUNDLE_SCHEMA_VERSION, RESULTS_SCHEMA_VERSION
from .metric_registry import METRIC_REGISTRY
from .failures import FAILURE_PRECEDENCE, FAILURE_PRECEDENCE_V2
from .utils import slugify
from .multiseed import dataset_contract_hash, evaluation_contract_hash, make_run_id, method_training_hash


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        items = [_json_safe(v) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return value


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _string_array(values: Any) -> np.ndarray:
    return np.asarray([str(x) for x in list(values)], dtype=np.str_)


def _cell_order_sha256(values: Any) -> str:
    arr = _string_array(values)
    return _sha256_bytes("\n".join(arr.tolist()).encode("utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("ascii"))
    h.update(str(tuple(contiguous.shape)).encode("ascii"))
    h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def dataframe_html(frame: pd.DataFrame | None, max_rows: int = 200) -> str:
    if frame is None or frame.empty:
        return '<p class="note">No rows were produced for this section.</p>'
    return frame.head(max_rows).to_html(index=False, border=0, classes="dataframe", float_format=lambda value: f"{value:.4f}")


def _resolve_figure_path(report_path: Path, figure: str | Path) -> Path:
    candidate = Path(figure)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = report_path.parent / candidate
    if direct.exists():
        return direct
    for folder in (report_path.parent / "figures", report_path.parent / "scib", report_path.parent / "rare_cell" / "figures"):
        path = folder / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Report figure was not found: {figure!s}")


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _figure_html(report_path: Path, figures: Iterable[str | Path]) -> str:
    blocks: list[str] = []
    for figure in figures:
        path = _resolve_figure_path(report_path, figure)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            continue
        blocks.append("<figure>" f'<img src="{_data_uri(path)}" alt="{html.escape(path.name)}">' f"<figcaption>{html.escape(path.name)}</figcaption>" "</figure>")
    return "\n".join(blocks)


def _section(title: str, body: str, *, identifier: str | None = None) -> str:
    anchor = f' id="{html.escape(identifier)}"' if identifier else ""
    return f"<section{anchor}><h2>{html.escape(title)}</h2>{body}</section>"


def write_html_report(
    output_path: str | Path,
    *,
    title: str,
    metadata: dict[str, Any],
    global_table: pd.DataFrame,
    rare_table: pd.DataFrame,
    figure_names: list[str | Path],
    ratio_table: pd.DataFrame | None = None,
    scib_metrics: pd.DataFrame | None = None,
    scib_aggregates: pd.DataFrame | None = None,
    scib_status: pd.DataFrame | None = None,
    rare_summary: pd.DataFrame | None = None,
    scenario_table: pd.DataFrame | None = None,
    rare_execution_status: dict[str, Any] | None = None,
    scib_execution_status: dict[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figures = _figure_html(target, figure_names)
    meta_rows = "\n".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in metadata.items())
    scib_exec = html.escape(json.dumps(_json_safe(scib_execution_status or {}), ensure_ascii=False))
    rare_exec = html.escape(json.dumps(_json_safe(rare_execution_status or {}), ensure_ascii=False))
    scib_body = (
        f'<p class="note">Execution status: <code>{scib_exec}</code></p>'
        + ('<p class="note">The standard layer was not available for this run.</p>' if scib_metrics is None else (
            '<p class="note">Scores in this section are produced by the pinned <code>scib-metrics</code> backend and remain separate from rare-cell-specific outputs.</p>'
            '<h3>Aggregate scores</h3>' + dataframe_html(scib_aggregates) + '<h3>Individual metrics</h3>' + dataframe_html(scib_metrics) + '<h3>Metric availability and applicability</h3>' + dataframe_html(scib_status)
        ))
    )
    rare_body = f'<p class="note">Execution status: <code>{rare_exec}</code></p>' + '<h3>Rare-cell summary</h3>' + dataframe_html(rare_summary) + '<h3>Six-scenario summary</h3>' + dataframe_html(scenario_table) + '<h3>Per-cell-type results</h3>' + dataframe_html(rare_table)
    ratio_body = '<p class="note">Ratios are diagnostic and intentionally separated from ordinary benchmark scores. Near-zero denominators are marked unstable instead of producing extreme values.</p>' + dataframe_html(ratio_table)
    document = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; }} body {{ font-family: Arial,sans-serif; max-width:1350px; margin:2rem auto; padding:0 1rem; line-height:1.45; }}
h1,h2,h3 {{ margin-top:1.6rem; }} nav {{ position:sticky; top:0; padding:.65rem; background:Canvas; border-bottom:1px solid #aaa; z-index:2; }} nav a {{ margin-right:1rem; }}
table {{ border-collapse:collapse; width:100%; font-size:.84rem; display:block; overflow-x:auto; }} th,td {{ border:1px solid #bbb; padding:.35rem; text-align:left; white-space:nowrap; }}
figure {{ margin:1.5rem 0; page-break-inside:avoid; }} img {{ max-width:100%; height:auto; display:block; margin:0 auto; }} figcaption {{ text-align:center; opacity:.75; margin-top:.35rem; font-size:.85rem; }}
.warning {{ padding:.8rem; background:#fff4d6; color:#2b2200; border-left:4px solid #d99b00; }} .note {{ opacity:.78; font-size:.9rem; }} code {{ white-space:pre-wrap; }} section {{ scroll-margin-top:4rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<nav><a href="#metadata">Metadata</a><a href="#scib">scIB-compatible</a><a href="#paper">Core</a><a href="#ratios">Ratios</a><a href="#rare">Rare-cell</a><a href="#figures">Figures</a></nav>
<p class="warning">Failure-archetype thresholds remain provisional and must be sensitivity-tested before manuscript claims.</p>
<p class="note">This HTML is self-contained: static figures are embedded as base64 data URIs.</p>
{_section("Run metadata", f'<table>{meta_rows}</table>', identifier="metadata")}
{_section("Standard scIB-compatible evaluation", scib_body, identifier="scib")}
{_section("Global and subset metrics", dataframe_html(global_table), identifier="paper")}
{_section("Subset ratio diagnostics", ratio_body, identifier="ratios")}
{_section("Rare-cell-specific evaluation", rare_body, identifier="rare")}
{_section("Figures", figures, identifier="figures")}
</body></html>'''
    target.write_text(document, encoding="utf-8")
    return target


def _metadata_from_result(adata: Any, result: Any, representation_key: str) -> dict[str, Any]:
    cluster_key = next(iter(result.cluster_keys.values())) if getattr(result, "cluster_keys", None) else "n/a"
    return {
        "method": getattr(result, "prediction_key", representation_key).replace("scrarebench_prediction_", ""),
        "representation_key": representation_key,
        "n_cells": int(getattr(adata, "n_obs", len(getattr(adata, "obs", [])))),
        "n_dimensions": int(np.asarray(adata.obsm[representation_key]).shape[1]),
        "reference_cluster_key": cluster_key,
        "rare_types": int(len(getattr(result, "rare_metrics", pd.DataFrame()))),
        "interactive_report": "Yes",
    }


def _write_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white"); plt.axis("off")
    fig.text(0.06, 0.96, title, fontsize=18, weight="bold", va="top")
    y = 0.92
    for line in lines:
        fig.text(0.06, y, line, fontsize=10.2, va="top", family="monospace")
        y -= 0.026
        if y < 0.06:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white"); plt.axis("off")
            fig.text(0.06, 0.96, title + " (cont.)", fontsize=18, weight="bold", va="top"); y = 0.92
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _compact_table(frame: pd.DataFrame | None, preferred: Sequence[str], max_rows: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = [c for c in preferred if c in frame.columns]
    if not cols:
        cols = list(frame.columns[:8])
    return frame.loc[:, cols].head(max_rows).copy()


def _render_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, *, max_rows: int = 25) -> None:
    shown = frame.head(max_rows).copy() if frame is not None else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    if shown.empty:
        ax.text(0.02, 0.9, "No rows available.", fontsize=12, transform=ax.transAxes)
    else:
        formatted = shown.copy()
        for column in formatted.columns:
            formatted[column] = formatted[column].map(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("—" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else str(v)))
        table = ax.table(cellText=formatted.values, colLabels=list(formatted.columns), loc="upper left", cellLoc="left", colLoc="left", bbox=[0.0, 0.0, 1.0, 0.9])
        table.auto_set_font_size(False); table.set_fontsize(7.2); table.scale(1, 1.18)
        if len(frame) > max_rows:
            ax.text(0.0, -0.03, f"Showing first {max_rows} of {len(frame)} rows. Full table is available in CSV/HTML.", fontsize=9, transform=ax.transAxes)
    fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _add_figure_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white"); ax.axis("off"); ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=10)
    ax.imshow(plt.imread(str(image_path))); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_pdf_report(adata: Any, result: Any, output_path: str | Path, *, title: str | None = None, representation_key: str | None = None) -> Path:
    """Write a readable scientific-summary PDF; full raw tables remain in CSV/HTML."""
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    title = title or f"scRareBench summary report — {representation_key}"
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    meta = _metadata_from_result(adata, result, representation_key)
    with PdfPages(target) as pdf:
        _write_text_page(pdf, title, ["scRareBench scientific summary", "", *[f"{k}: {v}" for k,v in meta.items()], "", "Full-width raw tables are intentionally kept in CSV/HTML.", "Failure-archetype thresholds and the legacy preserved-rule outcome are provisional.", "Primary local rare-recovery metric: support-adjusted kNN local recovery."])
        _render_table_page(pdf, "Core / subset metrics", _compact_table(
            getattr(result, "subset_metrics", pd.DataFrame()),
            ["subset", "n_cells", "n_cell_types", "ASW_true_on_latent", "ASW_selected_cells_in_full_latent", "ARI_true_vs_cluster", "AMI_true_vs_cluster", "F1_macro", "F1_weighted", "G_Mean"],
            10,
        ), max_rows=10)
        ratios = getattr(result, "subset_metric_ratios", pd.DataFrame())
        _render_table_page(pdf, "Subset ratio diagnostics", _compact_table(ratios, ["metric","numerator","denominator","ratio","status"], 20), max_rows=20)
        _render_table_page(pdf, "Rare-cell summary", _compact_table(getattr(result, "rare_summary", pd.DataFrame()), ["metric","mean","median","minimum","maximum","n_valid"], 20), max_rows=20)
        scenario = getattr(result, "scenario_metrics", pd.DataFrame())
        scenario_pdf = _compact_table(
            scenario,
            ["scenario", "distribution", "topology", "knn_local_recovery_adjusted_mean", "best_cluster_f1_mean", "inverse_purity_mean", "knn_local_recovery_mean", "f1_mean"],
            12,
        ).rename(columns={
            "distribution": "dist",
            "topology": "topo",
            "knn_local_recovery_adjusted_mean": "adj_kNN",
            "best_cluster_f1_mean": "bestF1",
            "inverse_purity_mean": "capture",
            "knn_local_recovery_mean": "raw_kNN",
            "f1_mean": "legacyF1",
        })
        _render_table_page(pdf, "Six-scenario summary", scenario_pdf, max_rows=12)
        rare = getattr(result, "rare_metrics", pd.DataFrame())
        if rare is not None and not rare.empty:
            ranked = rare.copy()
            rank_key = "knn_local_recovery_adjusted" if "knn_local_recovery_adjusted" in ranked.columns else ("best_cluster_f1" if "best_cluster_f1" in ranked.columns else ("knn_local_recovery" if "knn_local_recovery" in ranked.columns else "f1"))
            if rank_key in ranked.columns:
                ranked = ranked.sort_values(rank_key, ascending=False)
            primary_failure_column = "failure_archetype_v2" if "failure_archetype_v2" in ranked.columns else "failure_archetype"
            rare_pdf = _compact_table(
                pd.concat([ranked.head(8), ranked.tail(8)]).drop_duplicates(),
                ["cell_type", "scenario", "support", "knn_local_recovery_adjusted", "best_cluster_f1", "inverse_purity", "f1", primary_failure_column],
                16,
            )
            if primary_failure_column in rare_pdf.columns:
                rare_pdf[primary_failure_column] = rare_pdf[primary_failure_column].replace({
                    "resolution_limited": "resolution limited",
                    "lineage_assimilation": "lineage assimilation",
                    "lineage_leakage": "lineage leakage",
                    "batch_driven_fragmentation": "batch fragmentation",
                    "mixed_or_uncertain": "mixed / uncertain",
                })
            rare_pdf = rare_pdf.rename(columns={
                "cell_type": "cell type",
                "knn_local_recovery_adjusted": "adj_kNN",
                "best_cluster_f1": "bestF1",
                "inverse_purity": "capture",
                "f1": "legacyF1",
                primary_failure_column: "v2 failure",
            })
            _render_table_page(
                pdf,
                "Rare populations — strongest/weakest recovery",
                rare_pdf,
                max_rows=16,
            )
            if "failure_archetype_v2" in rare.columns:
                counts_v2 = rare["failure_archetype_v2"].value_counts(dropna=False).rename_axis("failure_archetype_v2").reset_index(name="count")
                _render_table_page(pdf, "Rare-cell failure summary — resolution-aware v2", counts_v2, max_rows=20)
            if "failure_archetype" in rare.columns:
                counts_legacy = rare["failure_archetype"].value_counts(dropna=False).rename_axis("failure_archetype").reset_index(name="count")
                _render_table_page(pdf, "Rare-cell failure summary — legacy majority-vote taxonomy", counts_legacy, max_rows=20)
        resolution = getattr(result, "resolution_rare_metrics", pd.DataFrame())
        if resolution is not None and not resolution.empty:
            agg_cols = [c for c in ("f1", "best_cluster_f1", "inverse_purity") if c in resolution.columns]
            if agg_cols:
                # Older/external result objects may omit ``n_clusters``.  Group only by
                # keys that are actually present instead of relying on pandas' legacy
                # behavior of interpreting a missing key-like label as an external grouper.
                # This keeps PDF export warning-free under ``-W error`` and preserves
                # backward compatibility with resolution-only summaries.
                group_keys = [c for c in ("resolution", "n_clusters") if c in resolution.columns]
                if group_keys:
                    resolution_summary = resolution.groupby(group_keys, as_index=False)[agg_cols].mean(numeric_only=True)
                    _render_table_page(
                        pdf,
                        "Rare recovery across Leiden resolution sweep",
                        _compact_table(resolution_summary, ["resolution", "n_clusters", "best_cluster_f1", "f1", "inverse_purity"], 30),
                        max_rows=30,
                    )
        scib_result = getattr(result, "scib", None)
        if scib_result is not None:
            _render_table_page(pdf, "scIB-compatible aggregate scores", _compact_table(scib_result.aggregate_scores, ["metric","value","status"], 20), max_rows=20)
            _render_table_page(pdf, "Selected scIB-compatible metrics", _compact_table(scib_result.metrics_long, ["metric","value","metric_type","status"], 30), max_rows=30)
        for label in ("rare_metric_heatmap", "rare_precision_recall", "failure_counts", "scib_metric_plot"):
            path = getattr(result, "files", {}).get(label)
            if path and Path(path).exists(): _add_figure_page(pdf, f"Figure: {Path(path).name}", Path(path))
        run_config = getattr(result, "run_config", {}) or {}
        status_lines = [
            "Rare status: " + json.dumps(_json_safe(getattr(result, "rare_evaluation_status", {})), ensure_ascii=False),
            "scIB status: " + json.dumps(_json_safe(getattr(result, "scib_status", {})), ensure_ascii=False),
            "Leiden flavor: " + str(run_config.get("leiden_flavor", "—")),
            "Leiden n_iterations: " + str(run_config.get("leiden_n_iterations", "—")),
            "Reference clusters: " + str(run_config.get("reference_n_clusters", "—")),
            "Reference cell types: " + str(run_config.get("n_cell_types", "—")),
            "Cluster-count warning: " + str(run_config.get("cluster_count_warning", "—")),
            "kNN graph: " + json.dumps(_json_safe(run_config.get("knn_graph", {})), ensure_ascii=False),
        ]
        _write_text_page(pdf, "Reproducibility / execution status", status_lines)
    return target


def _dataset_metadata(adata: Any) -> dict[str, Any]:
    uns = getattr(adata, "uns", {}) or {}
    meta = {}
    if hasattr(uns, "get"):
        meta = uns.get("scrarebench_dataset", {}) or uns.get("scrarebench", {}) or {}
    return _json_safe(meta if isinstance(meta, dict) else {})


def build_results_payload(
    adata: Any,
    result: Any,
    *,
    representation_key: str,
    label_key: str = "celltype",
    batch_key: str = "BATCH",
    benchmark_config: dict[str, Any] | None = None,
    method_seed: int | None = None,
    method_config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    latent = np.asarray(adata.obsm[representation_key])
    obs = getattr(adata, "obs", None)
    obs_names = getattr(adata, "obs_names", getattr(obs, "index", []))
    scib = getattr(result, "scib", None)
    cfg = dict(benchmark_config or {})
    resolved_seed = method_seed
    if resolved_seed is None:
        raw_seed = cfg.get("method_seed", cfg.get("seed"))
        if isinstance(raw_seed, (int, np.integer)) and not isinstance(raw_seed, bool):
            resolved_seed = int(raw_seed)
    method_name = cfg.get("method_name") or getattr(result, "prediction_key", representation_key).replace("scrarebench_prediction_", "")
    cell_hash = _cell_order_sha256(obs_names)
    latent_hash = _array_sha256(latent)
    method_cfg = dict(method_config or cfg.get("method_config") or {})
    config_hash = evaluation_contract_hash(cfg, method_cfg)
    training_hash = method_training_hash(method_cfg)
    dataset_meta = _dataset_metadata(adata)
    dataset_key = dataset_meta.get("dataset_key") or dataset_meta.get("key") or dataset_meta.get("display_name") or dataset_meta.get("name") or "dataset"
    dataset_fingerprint = hashlib.sha256(f"{dataset_key}|{cell_hash}|{len(obs_names)}".encode("utf-8")).hexdigest()
    scenario_key = str(cfg.get("scenario_key") or "scrarebench_scenario")
    dataset_contract = dataset_contract_hash(
        adata, dataset_key=str(dataset_key), label_key=label_key, batch_key=batch_key, scenario_key=scenario_key
    )
    resolved_run_id = run_id or make_run_id(method_name=str(method_name), dataset_fingerprint=dataset_fingerprint, method_seed=resolved_seed, config_hash=config_hash, latent_hash=latent_hash)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_by": {"package": "scrarebench", "version": __version__},
        "method": {
            "name": method_name,
            "representation_key": representation_key,
            "configuration_hash": config_hash,
            "evaluation_contract_hash": config_hash,
            "method_training_hash": training_hash,
            "configuration": _json_safe(method_cfg),
        },
        "run": {
            "run_id": resolved_run_id,
            "method_seed": resolved_seed,
            "configuration_hash": config_hash,
            "evaluation_contract_hash": config_hash,
            "method_training_hash": training_hash,
            "dataset_key": str(dataset_key),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_contract_sha256": dataset_contract,
        },
        "dataset": dataset_meta,
        "benchmark": {
            "n_cells": int(getattr(adata, "n_obs", len(obs_names))),
            "n_dimensions": int(latent.shape[1]),
            "label_key": label_key,
            "batch_key": batch_key,
            "n_cell_types": int(obs[label_key].astype(str).nunique()) if obs is not None and label_key in obs.columns else None,
            "n_batches": int(obs[batch_key].astype(str).nunique()) if obs is not None and batch_key in obs.columns else None,
            "benchmark_seed": cfg.get("benchmark_seed", cfg.get("random_state")),
            "config": _json_safe(cfg),
            "method_config": _json_safe(method_cfg),
        },
        "metrics": {
            "subset": _records(getattr(result, "subset_metrics", None)),
            "subset_ratios": _records(getattr(result, "subset_metric_ratios", None)),
            "per_type": _records(getattr(result, "per_type_metrics", None)),
        },
        "rare": {
            "status": _json_safe(getattr(result, "rare_evaluation_status", {})),
            "methodology": {
                "historical_cluster_label_metrics_retained": True,
                "primary_local_recovery_metric": "knn_local_recovery_adjusted",
                "local_recovery_adjusted": "kNN same-label neighborhood enrichment normalized against both the global-abundance null and the support/realized-degree achievable ceiling",
                "local_recovery_raw": "knn_local_recovery is retained for backward comparison but can have a support-dependent ceiling when population support is smaller than realized graph degree",
                "local_recovery_is_leiden_resolution_independent": True,
                "local_recovery_depends_on_knn_graph_contract": True,
                "legacy_preserved_fraction_is_primary_endpoint": False,
                "legacy_preserved_fraction_note": "preserved_fraction remains majority-vote dependent and provisional; retain for continuity, not as the primary preservation endpoint",
                "full_space_asw_metric": "ASW_selected_cells_in_full_latent",
                "failure_taxonomy_primary": "resolution_aware_v2",
                "failure_taxonomy_legacy_columns_retained": True,
                "failure_rule_precedence_legacy": list(FAILURE_PRECEDENCE),
                "failure_rule_precedence_v2": list(FAILURE_PRECEDENCE_V2),
                "resolution_limited_definition": "A legacy lineage-assimilation match is relabeled resolution_limited in v2 when support-adjusted kNN recovery and dominant-cluster capture remain high; the legacy lineage_assimilation match remains preserved in legacy columns and in v2 matched-rule provenance.",
                "failure_rules_are_provisional": True,
            },
            "rows": _records(getattr(result, "rare_metrics", None)),
            "summary": _records(getattr(result, "rare_summary", None)),
            "scenarios": _records(getattr(result, "scenario_metrics", None)),
            "resolution_sensitivity": _records(getattr(result, "resolution_rare_metrics", None)),
        },
        "scib": {
            "status": _json_safe(getattr(result, "scib_status", {})),
            "metrics": _records(getattr(scib, "metrics_long", None)) if scib is not None else [],
            "aggregates": _records(getattr(scib, "aggregate_scores", None)) if scib is not None else [],
            "metric_status": _records(getattr(scib, "metric_status", None)) if scib is not None else [],
            "reference_config": _json_safe(getattr(scib, "reference_config", {})) if scib is not None else {},
        },
        "metric_registry": _json_safe(METRIC_REGISTRY),
        "provenance": {
            "cell_order_sha256": cell_hash,
            "dataset_key": str(dataset_key),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_contract_sha256": dataset_contract,
            "configuration_hash": config_hash,
            "evaluation_contract_hash": config_hash,
            "method_training_hash": training_hash,
            "run_id": resolved_run_id,
            "method_seed": resolved_seed,
            "latent": {"shape": list(latent.shape), "dtype": str(latent.dtype), "sha256": latent_hash},
        },
    }
    return _json_safe(payload)


def write_results_json(
    adata: Any,
    result: Any,
    output_path: str | Path,
    *,
    representation_key: str,
    label_key: str = "celltype",
    batch_key: str = "BATCH",
    benchmark_config: dict[str, Any] | None = None,
    method_seed: int | None = None,
    method_config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_results_payload(
        adata, result, representation_key=representation_key, label_key=label_key, batch_key=batch_key,
        benchmark_config=benchmark_config, method_seed=method_seed, method_config=method_config, run_id=run_id,
    )
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return target


from .dashboard import write_interactive_report


def _read_run_config(result: Any) -> dict[str, Any]:
    path = getattr(result, "files", {}).get("run_config") if getattr(result, "files", None) else None
    if path and Path(path).exists():
        try:
            import yaml
            return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def create_report_bundle(
    adata: Any,
    result: Any,
    output_zip: str | Path,
    *,
    representation_key: str | None = None,
    include_latent: bool = False,
    write_interactive: bool = True,
    write_pdf: bool = True,
    existing_interactive_report: str | Path | None = None,
    existing_pdf_report: str | Path | None = None,
    copy_existing_results: bool = True,
    interactive_report_options: dict[str, Any] | None = None,
    label_key: str | None = None,
    batch_key: str | None = None,
    extra_provenance_files: dict[str, str | Path] | None = None,
    method_seed: int | None = None,
    method_config: dict[str, Any] | None = None,
    expected_seeds: Sequence[int] | None = None,
    run_id: str | None = None,
) -> Path:
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    target_zip = Path(output_zip); target_zip.parent.mkdir(parents=True, exist_ok=True); slug = slugify(representation_key)
    run_config = _read_run_config(result)
    opts = dict(interactive_report_options or {})
    label_key = label_key or opts.get("label_key") or run_config.get("label_key") or "celltype"
    batch_key = batch_key or opts.get("batch_key") or run_config.get("batch_key") or "BATCH"
    opts.setdefault("label_key", label_key); opts.setdefault("batch_key", batch_key)

    with tempfile.TemporaryDirectory(prefix="scrarebench_bundle_") as temp_dir:
        root = Path(temp_dir) / "scrarebench_bundle"; root.mkdir(parents=True, exist_ok=True)
        reports_dir = root / "reports"; reports_dir.mkdir()
        repro_dir = root / "reproducibility"; repro_dir.mkdir()
        created_files: dict[str, str] = {}
        if copy_existing_results and getattr(result, "output_dir", None):
            source_dir = Path(result.output_dir)
            if source_dir.exists():
                def ignore(_d: str, names: list[str]) -> set[str]: return {n for n in names if n in {"interactive_report.html", "summary_report.pdf"}}
                shutil.copytree(source_dir, root / "benchmark_results", dirs_exist_ok=True, ignore=ignore)
                created_files["benchmark_results"] = "benchmark_results/"
        benchmark_dir = root / "benchmark_results"; benchmark_dir.mkdir(exist_ok=True)
        results_json = benchmark_dir / "results.json"
        write_results_json(adata, result, results_json, representation_key=representation_key, label_key=label_key, batch_key=batch_key, benchmark_config=run_config, method_seed=method_seed, method_config=method_config, run_id=run_id)
        created_files["results_json"] = str(results_json.relative_to(root))

        if write_interactive:
            p = reports_dir / "interactive_report.html"
            if existing_interactive_report is not None:
                src = Path(existing_interactive_report)
                if not src.exists():
                    raise FileNotFoundError(f"Existing interactive report not found: {src}")
                shutil.copy2(src, p)
            else:
                write_interactive_report(adata, result, p, representation_key=representation_key, method_seed=method_seed, method_config=method_config, run_id=run_id, expected_seeds=expected_seeds, **opts)
            created_files["interactive_report"] = str(p.relative_to(root))
        if write_pdf:
            p = reports_dir / "summary_report.pdf"
            if existing_pdf_report is not None:
                src = Path(existing_pdf_report)
                if not src.exists():
                    raise FileNotFoundError(f"Existing PDF report not found: {src}")
                shutil.copy2(src, p)
            else:
                write_pdf_report(adata, result, p, representation_key=representation_key)
            created_files["summary_pdf"] = str(p.relative_to(root))
        if getattr(result, "files", None) and "report" in result.files:
            static = Path(result.files["report"])
            if static.exists() and not (benchmark_dir / static.name).exists():
                copied = reports_dir / "static_report.html"; shutil.copy2(static, copied); created_files["static_report"] = str(copied.relative_to(root))

        dataset_manifest = repro_dir / "dataset_manifest.json"
        dataset_manifest.write_text(json.dumps(_dataset_metadata(adata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        created_files["dataset_manifest"] = str(dataset_manifest.relative_to(root))
        method_dir = repro_dir / "method"; method_dir.mkdir()
        for name, source in (extra_provenance_files or {}).items():
            src = Path(source)
            if src.exists() and src.is_file():
                safe_name = f"{slugify(str(name))}{src.suffix}" if not str(name).endswith(src.suffix) else slugify(str(name).removesuffix(src.suffix)) + src.suffix
                dest = method_dir / safe_name; shutil.copy2(src, dest); created_files[f"method_provenance_{name}"] = str(dest.relative_to(root))
        if include_latent:
            latent_dir = root / "latent"; latent_dir.mkdir()
            latent_path = latent_dir / f"{slug}_latent.npy"; barcodes_path = latent_dir / f"{slug}_barcodes.npy"
            np.save(latent_path, np.asarray(adata.obsm[representation_key]), allow_pickle=False)
            np.save(barcodes_path, _string_array(getattr(adata, "obs_names", getattr(adata.obs, "index", []))), allow_pickle=False)
            created_files["latent"] = str(latent_path.relative_to(root)); created_files["latent_barcodes"] = str(barcodes_path.relative_to(root))

        latent = np.asarray(adata.obsm[representation_key])
        resolved_method_cfg = dict(method_config or run_config.get("method_config") or {})
        resolved_method_name = run_config.get("method_name") or getattr(result, "prediction_key", representation_key).replace("scrarebench_prediction_", "")
        config_hash = evaluation_contract_hash(run_config, resolved_method_cfg)
        training_hash = method_training_hash(resolved_method_cfg)
        dataset_meta = _dataset_metadata(adata)
        dataset_key = dataset_meta.get("dataset_key") or dataset_meta.get("key") or dataset_meta.get("display_name") or dataset_meta.get("name") or "dataset"
        cell_hash = _cell_order_sha256(getattr(adata, "obs_names", getattr(adata.obs, "index", [])))
        dataset_fingerprint = hashlib.sha256(f"{dataset_key}|{cell_hash}|{int(getattr(adata, 'n_obs', len(getattr(adata, 'obs', []))))}".encode("utf-8")).hexdigest()
        dataset_contract = dataset_contract_hash(
            adata, dataset_key=str(dataset_key), label_key=label_key, batch_key=batch_key,
            scenario_key=str(run_config.get("scenario_key") or "scrarebench_scenario")
        )
        latent_hash = _array_sha256(latent)
        resolved_run_id = run_id or make_run_id(
            method_name=str(resolved_method_name), dataset_fingerprint=dataset_fingerprint,
            method_seed=method_seed, config_hash=config_hash, latent_hash=latent_hash,
        )
        benchmark_seed = run_config.get("benchmark_seed", run_config.get("random_state"))
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "scrarebench_version": __version__,
            "method_name": resolved_method_name,
            "method_seed": method_seed,
            "method_configuration": _json_safe(resolved_method_cfg),
            "method_training_hash": training_hash,
            "configuration_hash": config_hash,
            "evaluation_contract_hash": config_hash,
            "benchmark_seed": int(benchmark_seed) if benchmark_seed is not None else None,
            "expected_seeds": list(expected_seeds) if expected_seeds is not None else ([method_seed] if method_seed is not None else []),
            "run_id": resolved_run_id,
            "representation_key": representation_key,
            "dataset": dataset_meta,
            "dataset_key": str(dataset_key),
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_contract_sha256": dataset_contract,
            "n_cells": int(getattr(adata, "n_obs", len(getattr(adata, "obs", [])))),
            "n_dimensions": int(latent.shape[1]),
            "benchmark_config": _json_safe(run_config),
            "cell_order_sha256": cell_hash,
            "latent": {"included": include_latent, "shape": list(latent.shape), "dtype": str(latent.dtype), "sha256": latent_hash},
            "files": created_files,
        }
        (root / "bundle_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (root / "README.md").write_text("# scRareBench result bundle\n\nThis bundle contains machine-readable results, reports, provenance, and optional latent data. Start with `benchmark_results/results.json` for programmatic comparison and `reports/interactive_report.html` for interactive inspection. Missing values are not encoded as zero. See `bundle_manifest.json` and `artifact_hashes.json` for provenance.\n", encoding="utf-8")
        hashes: dict[str, dict[str, Any]] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "artifact_hashes.json":
                rel = str(path.relative_to(root)); hashes[rel] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        (root / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(target_zip.with_suffix("")), "zip", root)
    return Path(archive)
