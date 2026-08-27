from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rare_metric_heatmap(
    rare_metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "inverse_purity"),
) -> Path:
    table = rare_metrics.set_index("cell_type").loc[:, list(metrics)]
    fig_width = max(8.0, 0.42 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    image = ax.imshow(table.T.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(table.index)), labels=table.index, rotation=65, ha="right")
    ax.set_yticks(np.arange(len(metrics)), labels=metrics)
    ax.set_title("Rare-cell metrics by curated population")
    for row in range(len(metrics)):
        for col in range(len(table.index)):
            ax.text(col, row, f"{table.iloc[col, row]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Score")
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_precision_recall(rare_metrics: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(rare_metrics["recall"], rare_metrics["precision"])
    for _, row in rare_metrics.iterrows():
        ax.annotate(str(row["cell_type"]), (row["recall"], row["precision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Rare-cell precision–recall profile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_failure_counts(rare_metrics: pd.DataFrame, output_path: str | Path) -> Path:
    counts = rare_metrics["failure_archetype"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of rare cell types")
    ax.set_title("Provisional failure-archetype counts")
    for index, value in enumerate(counts.values):
        ax.text(value, index, f" {value}", va="center")
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def write_sankey_html(
    y_true: Any,
    y_pred: Any,
    output_path: str | Path,
    *,
    highlight_label: str | None = None,
) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    frame = pd.DataFrame({"true": np.asarray(y_true).astype(str), "predicted": np.asarray(y_pred).astype(str)})
    counts = frame.groupby(["true", "predicted"]).size().reset_index(name="count")
    true_nodes = counts["true"].drop_duplicates().tolist()
    pred_nodes = counts["predicted"].drop_duplicates().tolist()
    labels = [f"True: {value}" for value in true_nodes] + [f"Pred: {value}" for value in pred_nodes]
    source_map = {value: idx for idx, value in enumerate(true_nodes)}
    target_map = {value: idx + len(true_nodes) for idx, value in enumerate(pred_nodes)}
    colors: list[str] = []
    for row in counts.itertuples(index=False):
        if highlight_label is None:
            colors.append("rgba(150,150,150,0.35)")
        elif row.true == highlight_label and row.predicted == highlight_label:
            colors.append("rgba(0,150,0,0.65)")
        elif row.true == highlight_label:
            colors.append("rgba(200,0,0,0.65)")
        elif row.predicted == highlight_label:
            colors.append("rgba(0,80,210,0.65)")
        else:
            colors.append("rgba(150,150,150,0.15)")
    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=12, thickness=15),
            link=dict(
                source=[source_map[value] for value in counts["true"]],
                target=[target_map[value] for value in counts["predicted"]],
                value=counts["count"].tolist(),
                color=colors,
            ),
        )
    )
    fig.update_layout(title="Ground truth to majority-vote prediction", font_size=10)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(target, include_plotlyjs=True, full_html=True)
    return target


def plot_scib_metric_scores(
    metrics: pd.DataFrame,
    aggregates: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot all validated scIB-compatible metrics and their aggregate scores."""
    metric_frame = metrics.copy()
    aggregate_frame = aggregates.copy()
    metric_frame = metric_frame[np.isfinite(pd.to_numeric(metric_frame["value"], errors="coerce"))]
    aggregate_frame = aggregate_frame[np.isfinite(pd.to_numeric(aggregate_frame["value"], errors="coerce"))]
    plot_frame = pd.concat(
        [
            metric_frame.assign(section=metric_frame["metric_type"]),
            aggregate_frame.assign(section="Aggregate score"),
        ],
        ignore_index=True,
    )
    if plot_frame.empty:
        raise ValueError("No finite scIB-compatible scores are available for plotting")
    plot_frame["label"] = plot_frame["metric"].astype(str)
    plot_frame = plot_frame.sort_values(["section", "value", "label"], ascending=[True, True, True])
    height = max(5.0, 0.36 * len(plot_frame) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, height))
    y = np.arange(len(plot_frame))
    ax.barh(y, plot_frame["value"].to_numpy(dtype=float))
    ax.set_yticks(y, labels=plot_frame["label"])
    ax.set_xlim(0, max(1.02, float(plot_frame["value"].max()) * 1.08))
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("Standard scIB-compatible metrics")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(plot_frame["value"].to_numpy(dtype=float)):
        ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target
