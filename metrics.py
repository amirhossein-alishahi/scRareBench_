from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    recall_score,
    silhouette_score,
)


def majority_vote_mapping(y_true: Iterable[Any], clusters: Iterable[Any]) -> dict[str, str]:
    frame = pd.DataFrame({"true": pd.Series(y_true, dtype="string"), "cluster": pd.Series(clusters, dtype="string")})
    table = pd.crosstab(frame["cluster"], frame["true"])
    # idxmax is deterministic because crosstab columns are sorted.
    return {str(cluster): str(label) for cluster, label in table.idxmax(axis=1).items()}


def majority_vote_predictions(y_true: Iterable[Any], clusters: Iterable[Any]) -> tuple[np.ndarray, dict[str, str]]:
    cluster_array = np.asarray(list(clusters)).astype(str)
    mapping = majority_vote_mapping(y_true, cluster_array)
    predicted = np.asarray([mapping[value] for value in cluster_array], dtype=str)
    return predicted, mapping


def multiclass_gmean(y_true: Iterable[Any], y_pred: Iterable[Any], *, epsilon: float = 1e-12) -> float:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    recalls = recall_score(
        np.asarray(list(y_true)).astype(str),
        np.asarray(list(y_pred)).astype(str),
        labels=labels,
        average=None,
        zero_division=0,
    )
    return float(np.exp(np.mean(np.log(recalls + epsilon))))


def safe_silhouette(
    X: np.ndarray,
    labels: Iterable[Any],
    *,
    max_cells: int = 10_000,
    random_state: int = 0,
) -> float:
    """Compute silhouette width with deterministic subsampling for large datasets.

    Exact silhouette computation is quadratic in the number of cells. The BMMC
    benchmark has nearly 90,000 cells, so version 1 uses at most 10,000 cells
    for this metric while keeping all other metrics exact.
    """
    matrix = np.asarray(X)
    label_array = np.asarray(list(labels)).astype(str)
    n_labels = len(np.unique(label_array))
    if n_labels < 2 or n_labels >= len(label_array):
        return float("nan")
    sample_size = min(int(max_cells), len(label_array))
    return float(
        silhouette_score(
            matrix,
            label_array,
            sample_size=sample_size if sample_size < len(label_array) else None,
            random_state=random_state,
        )
    )


def global_metrics(
    X: np.ndarray,
    y_true: Iterable[Any],
    clusters: Iterable[Any],
    y_pred: Iterable[Any],
) -> dict[str, float]:
    true = np.asarray(list(y_true)).astype(str)
    cluster = np.asarray(list(clusters)).astype(str)
    pred = np.asarray(list(y_pred)).astype(str)
    return {
        "ASW_true_on_latent": safe_silhouette(X, true),
        "ARI_true_vs_cluster": float(adjusted_rand_score(true, cluster)),
        "AMI_true_vs_cluster": float(adjusted_mutual_info_score(true, cluster)),
        "Accuracy": float(accuracy_score(true, pred)),
        "F1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "G_Mean": multiclass_gmean(true, pred),
    }


def per_type_metrics(
    y_true: Iterable[Any],
    clusters: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    batch_labels: Iterable[Any] | None = None,
) -> pd.DataFrame:
    true = np.asarray(list(y_true)).astype(str)
    cluster = np.asarray(list(clusters)).astype(str)
    pred = np.asarray(list(y_pred)).astype(str)
    labels = sorted(set(true))
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    contingency = pd.crosstab(pd.Series(true, name="true"), pd.Series(cluster, name="cluster"))
    rows: list[dict[str, Any]] = []
    batch = np.asarray(list(batch_labels)).astype(str) if batch_labels is not None else None
    for idx, cell_type in enumerate(labels):
        target = true == cell_type
        predicted_target = pred == cell_type
        tp = int(np.sum(target & predicted_target))
        fp = int(np.sum(~target & predicted_target))
        fn = int(np.sum(target & ~predicted_target))
        cluster_counts = contingency.loc[cell_type] if cell_type in contingency.index else pd.Series(dtype=int)
        inverse_purity = float(cluster_counts.max() / support[idx]) if support[idx] else 0.0
        dominant_cluster = str(cluster_counts.idxmax()) if len(cluster_counts) else ""
        n_clusters_found = int(np.sum(cluster_counts > 0))
        assigned_clusters = pd.Series(cluster[predicted_target]).nunique() if np.any(predicted_target) else 0
        wrong_predictions = pd.Series(pred[target & ~predicted_target])
        if len(wrong_predictions):
            dominant_wrong_label = str(wrong_predictions.value_counts().index[0])
            dominant_wrong_fraction = float(wrong_predictions.value_counts().iloc[0] / support[idx])
        else:
            dominant_wrong_label = ""
            dominant_wrong_fraction = 0.0
        if batch is not None and target.sum() >= 2 and len(np.unique(batch[target])) >= 2 and len(np.unique(cluster[target])) >= 2:
            within_batch_nmi = float(normalized_mutual_info_score(batch[target], cluster[target]))
        else:
            within_batch_nmi = float("nan")
        rows.append(
            {
                "cell_type": cell_type,
                "support": int(support[idx]),
                "predicted_count": int(predicted_target.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "inverse_purity": inverse_purity,
                "dominant_cluster": dominant_cluster,
                "n_clusters_found_in": n_clusters_found,
                "n_clusters_assigned_to": int(assigned_clusters),
                "dominant_wrong_label": dominant_wrong_label,
                "dominant_wrong_fraction": dominant_wrong_fraction,
                "within_type_batch_nmi": within_batch_nmi,
            }
        )
    return pd.DataFrame(rows)


def subset_metrics(
    X: np.ndarray,
    y_true: Iterable[Any],
    clusters: Iterable[Any],
    y_pred: Iterable[Any],
    rare_types: Iterable[str],
) -> pd.DataFrame:
    true = np.asarray(list(y_true)).astype(str)
    cluster = np.asarray(list(clusters)).astype(str)
    pred = np.asarray(list(y_pred)).astype(str)
    rare_set = set(map(str, rare_types))
    masks = {
        "overall": np.ones(len(true), dtype=bool),
        "rare": np.asarray([value in rare_set for value in true]),
        "non_rare": np.asarray([value not in rare_set for value in true]),
    }
    rows: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        if mask.sum() == 0:
            continue
        values = global_metrics(X[mask], true[mask], cluster[mask], pred[mask])
        values.update({"subset": subset, "n_cells": int(mask.sum()), "n_cell_types": int(len(np.unique(true[mask])))})
        rows.append(values)
    result = pd.DataFrame(rows)
    if {"rare", "non_rare"}.issubset(set(result["subset"])):
        rare = result.set_index("subset").loc["rare"]
        common = result.set_index("subset").loc["non_rare"]
        ratio: dict[str, Any] = {"subset": "non_rare_to_rare_ratio", "n_cells": np.nan, "n_cell_types": np.nan}
        for column in [c for c in result.columns if c not in {"subset", "n_cells", "n_cell_types"}]:
            denominator = float(rare[column])
            ratio[column] = float(common[column] / denominator) if denominator != 0 else float("inf")
        result = pd.concat([result, pd.DataFrame([ratio])], ignore_index=True)
    return result
