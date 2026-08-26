from __future__ import annotations

import base64
import html
import json
import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .utils import slugify


def dataframe_html(frame: pd.DataFrame | None, max_rows: int = 200) -> str:
    if frame is None or frame.empty:
        return '<p class="note">No rows were produced for this section.</p>'
    return frame.head(max_rows).to_html(
        index=False,
        border=0,
        classes="dataframe",
        float_format=lambda value: f"{value:.4f}",
    )


def _resolve_figure_path(report_path: Path, figure: str | Path) -> Path:
    candidate = Path(figure)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = report_path.parent / candidate
    if direct.exists():
        return direct
    for folder in (
        report_path.parent / "figures",
        report_path.parent / "scib",
        report_path.parent / "rare_cell" / "figures",
    ):
        path = folder / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Report figure was not found: {figure!s}")


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _figure_html(report_path: Path, figures: Iterable[str | Path]) -> str:
    blocks: list[str] = []
    for figure in figures:
        path = _resolve_figure_path(report_path, figure)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            continue
        blocks.append(
            "<figure>"
            f'<img src="{_data_uri(path)}" alt="{html.escape(path.name)}">'
            f"<figcaption>{html.escape(path.name)}</figcaption>"
            "</figure>"
        )
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
    scib_metrics: pd.DataFrame | None = None,
    scib_aggregates: pd.DataFrame | None = None,
    scib_status: pd.DataFrame | None = None,
    rare_summary: pd.DataFrame | None = None,
    scenario_table: pd.DataFrame | None = None,
) -> Path:
    """Write a portable static HTML report containing standard scIB and rare-cell layers."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figures = _figure_html(target, figure_names)
    meta_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata.items()
    )

    scib_body = (
        '<p class="note">The standard layer was not run.</p>'
        if scib_metrics is None
        else (
            '<p class="note">Scores in this section are produced by the pinned '
            '<code>scib-metrics</code> backend. They are kept separate from the '
            'rare-cell-specific outputs and from the legacy scIB implementation.</p>'
            '<h3>Aggregate scores</h3>'
            f"{dataframe_html(scib_aggregates)}"
            '<h3>Individual metrics</h3>'
            f"{dataframe_html(scib_metrics)}"
            '<h3>Metric availability and applicability</h3>'
            f"{dataframe_html(scib_status)}"
        )
    )

    rare_body = (
        '<h3>Rare-cell summary</h3>'
        f"{dataframe_html(rare_summary)}"
        '<h3>Six-scenario summary</h3>'
        f"{dataframe_html(scenario_table)}"
        '<h3>Per-cell-type results</h3>'
        f"{dataframe_html(rare_table)}"
    )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: Arial, sans-serif; max-width: 1350px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
h1,h2,h3 {{ margin-top: 1.6rem; }}
nav {{ position: sticky; top: 0; padding: .65rem; background: Canvas; border-bottom: 1px solid #aaa; z-index: 2; }}
nav a {{ margin-right: 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; display: block; overflow-x: auto; }}
th,td {{ border: 1px solid #bbb; padding: 0.35rem; text-align: left; white-space: nowrap; }}
th {{ background: color-mix(in srgb, Canvas 90%, #888 10%); }}
figure {{ margin: 1.5rem 0; page-break-inside: avoid; }} img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
figcaption {{ text-align: center; opacity: .75; margin-top: .35rem; font-size: .85rem; }}
.warning {{ padding: 0.8rem; background: #fff4d6; color: #2b2200; border-left: 4px solid #d99b00; }}
.note {{ opacity: .78; font-size: .9rem; }} code {{ white-space: nowrap; }}
section {{ scroll-margin-top: 4rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<nav><a href="#metadata">Metadata</a><a href="#scib">Standard scIB-compatible</a><a href="#paper">Paper-style</a><a href="#rare">Rare-cell</a><a href="#figures">Figures</a></nav>
<p class="warning">Failure-archetype thresholds remain provisional and must be sensitivity-tested before final manuscript claims.</p>
<p class="note">This HTML is self-contained: static figures are embedded as base64 data URIs.</p>
{_section("Run metadata", f'<table>{meta_rows}</table>', identifier="metadata")}
{_section("Standard scIB-compatible evaluation", scib_body, identifier="scib")}
{_section("Paper-style global and subset metrics", dataframe_html(global_table), identifier="paper")}
{_section("Rare-cell-specific evaluation", rare_body, identifier="rare")}
{_section("Figures", figures, identifier="figures")}
</body></html>"""
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
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.06, 0.96, title, fontsize=18, weight="bold", va="top")
    y = 0.92
    for line in lines:
        fig.text(0.06, y, line, fontsize=10.5, va="top", family="monospace")
        y -= 0.026
        if y < 0.06:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            fig.text(0.06, 0.96, title + " (cont.)", fontsize=18, weight="bold", va="top")
            y = 0.92
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, *, max_rows: int = 25) -> None:
    shown = frame.head(max_rows).copy() if frame is not None else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    if shown.empty:
        ax.text(0.02, 0.9, "No rows available.", fontsize=12, transform=ax.transAxes)
    else:
        formatted = shown.copy()
        for column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            )
        table = ax.table(
            cellText=formatted.values,
            colLabels=list(formatted.columns),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.0, 1.0, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)
        if len(frame) > max_rows:
            ax.text(
                0.0,
                -0.03,
                f"Showing first {max_rows} of {len(frame)} rows.",
                fontsize=9,
                transform=ax.transAxes,
            )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_figure_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=10)
    img = plt.imread(str(image_path))
    ax.imshow(img)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    adata: Any,
    result: Any,
    output_path: str | Path,
    *,
    title: str | None = None,
    representation_key: str | None = None,
) -> Path:
    """Write a compact, user-friendly PDF summary report from an evaluation result."""
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    title = title or f"scRareBench summary report — {representation_key}"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = _metadata_from_result(adata, result, representation_key)
    with PdfPages(target) as pdf:
        _write_text_page(
            pdf,
            title,
            [
                "scRareBench summary report",
                "",
                *[f"{key}: {value}" for key, value in meta.items()],
                "",
                "Note: failure-archetype thresholds are provisional and should be sensitivity-tested.",
            ],
        )
        _render_table_page(pdf, "Paper-style global and subset metrics", getattr(result, "subset_metrics", pd.DataFrame()), max_rows=20)
        _render_table_page(pdf, "Rare-cell summary", getattr(result, "rare_summary", pd.DataFrame()), max_rows=20)
        _render_table_page(pdf, "Rare-cell per-type metrics", getattr(result, "rare_metrics", pd.DataFrame()), max_rows=35)
        _render_table_page(pdf, "Scenario summary", getattr(result, "scenario_metrics", pd.DataFrame()), max_rows=20)
        scib_result = getattr(result, "scib", None)
        if scib_result is not None:
            _render_table_page(pdf, "Standard scIB-compatible aggregate scores", scib_result.aggregate_scores, max_rows=20)
            _render_table_page(pdf, "Standard scIB-compatible individual metrics", scib_result.metrics_long, max_rows=35)
        for label in ("rare_metric_heatmap", "rare_precision_recall", "failure_counts"):
            path = getattr(result, "files", {}).get(label)
            if path and Path(path).exists():
                _add_figure_page(pdf, f"Figure: {Path(path).name}", Path(path))
        scib_metric_plot = getattr(result, "files", {}).get("scib_metric_plot")
        if scib_metric_plot and Path(scib_metric_plot).exists():
            _add_figure_page(pdf, f"Figure: {Path(scib_metric_plot).name}", Path(scib_metric_plot))
    return target


from .dashboard import write_interactive_report


def create_report_bundle(
    adata: Any,
    result: Any,
    output_zip: str | Path,
    *,
    representation_key: str | None = None,
    include_latent: bool = False,
    write_interactive: bool = True,
    write_pdf: bool = True,
    copy_existing_results: bool = True,
    interactive_report_options: dict[str, Any] | None = None,
) -> Path:
    """Create a ZIP bundle with reports and optional latent data.

    ``interactive_report_options`` is forwarded to ``write_interactive_report``
    so the same section flags can be used for the bundled HTML.
    """
    representation_key = representation_key or next(iter(getattr(adata, "obsm", {})))
    target_zip = Path(output_zip)
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    slug = slugify(representation_key)

    with tempfile.TemporaryDirectory(prefix="scrarebench_bundle_") as temp_dir:
        root = Path(temp_dir) / "scrarebench_bundle"
        root.mkdir(parents=True, exist_ok=True)
        reports_dir = root / "reports"
        reports_dir.mkdir(exist_ok=True)

        created_files: dict[str, str] = {}
        if copy_existing_results and getattr(result, "output_dir", None):
            source_dir = Path(result.output_dir)
            if source_dir.exists():
                def _ignore_generated_reports(_directory: str, names: list[str]) -> set[str]:
                    return {name for name in names if name in {"interactive_report.html", "summary_report.pdf"}}

                shutil.copytree(
                    source_dir,
                    root / "benchmark_results",
                    dirs_exist_ok=True,
                    ignore=_ignore_generated_reports,
                )
                created_files["benchmark_results"] = "benchmark_results/"

        if write_interactive:
            interactive_path = reports_dir / "interactive_report.html"
            write_interactive_report(
                adata,
                result,
                interactive_path,
                representation_key=representation_key,
                **(interactive_report_options or {}),
            )
            created_files["interactive_report"] = str(interactive_path.relative_to(root))
        if write_pdf:
            pdf_path = reports_dir / "summary_report.pdf"
            write_pdf_report(adata, result, pdf_path, representation_key=representation_key)
            created_files["summary_pdf"] = str(pdf_path.relative_to(root))
        static_path = None
        if getattr(result, "files", None) and "report" in result.files:
            static_path = Path(result.files["report"])
            if static_path.exists():
                if copy_existing_results and getattr(result, "output_dir", None):
                    try:
                        relative_static = static_path.resolve().relative_to(Path(result.output_dir).resolve())
                        created_files["static_report"] = str(Path("benchmark_results") / relative_static)
                    except ValueError:
                        copied = reports_dir / "static_report.html"
                        shutil.copy2(static_path, copied)
                        created_files["static_report"] = str(copied.relative_to(root))
                else:
                    copied = reports_dir / "static_report.html"
                    shutil.copy2(static_path, copied)
                    created_files["static_report"] = str(copied.relative_to(root))

        if include_latent:
            latent_dir = root / "latent"
            latent_dir.mkdir(exist_ok=True)
            latent_path = latent_dir / f"{slug}_latent.npy"
            barcodes_path = latent_dir / f"{slug}_barcodes.npy"
            np.save(latent_path, np.asarray(adata.obsm[representation_key], dtype=np.float32))
            np.save(barcodes_path, np.asarray(getattr(adata, "obs_names", getattr(adata.obs, "index", []))).astype(str))
            created_files["latent"] = str(latent_path.relative_to(root))
            created_files["latent_barcodes"] = str(barcodes_path.relative_to(root))

        manifest = {
            "representation_key": representation_key,
            "include_latent": include_latent,
            "files": created_files,
            "n_cells": int(getattr(adata, "n_obs", len(getattr(adata, "obs", [])))),
        }
        (root / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        archive = shutil.make_archive(str(target_zip.with_suffix("")), "zip", root)
    return Path(archive)
