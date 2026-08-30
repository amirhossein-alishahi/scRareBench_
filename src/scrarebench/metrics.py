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
    silhouette_samples,
    silhouette_score,
)

from .constants import DEFAULT_BENCHMARK_SEED


def majority_vote_mapping(y_true: Iterable[Any], clusters: Iterable[Any]) -> dict[str, str]:
    frame = pd.DataFrame({"true": pd.Series(y_true, dtype="string"), "cluster": pd.Series(clusters, dtype="string")})
    table = pd.crosstab(frame["cluster"], frame["true"])
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
    random_state: int = DEFAULT_BENCHMARK_SEED,
) -> float:
    """Return the conventional ASW on the supplied matrix/labels.

    This function intentionally preserves the historical scRareBench semantics:
    callers that pass a subset get a silhouette *within that subset*.  The newer
    :func:`selected_cells_silhouette_in_full_space` metric is additive and should
    be used when the scientific question is whether selected cells remain
    separated from all other populations.
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


def selected_cells_silhouette_in_full_space(
    X: np.ndarray,
    labels: Iterable[Any],
    selected_mask: Iterable[bool],
    *,
    max_cells: int = 10_000,
    random_state: int = DEFAULT_BENCHMARK_SEED,
    min_selected_sample: int = 64,
) -> float:
    """Average silhouette of selected cells while retaining full-space competitors.

    The historical rare-subset ASW first removed non-rare cells and therefore
    could report a high value even when rare populations were embedded inside an
    abundant lineage.  This metric instead samples from the *full* latent space,
    computes per-cell silhouette values there, and averages only the selected
    cells' scores.

    For large datasets a deterministic global sample is used.  When the global
    sample would contain too few selected cells, selected examples are forced into
    the sample by replacing non-selected cells; this keeps the computation bounded
    while ensuring the requested subset is represented.
    """
    matrix = np.asarray(X)
    label_array = np.asarray(list(labels)).astype(str)
    mask = np.asarray(list(selected_mask), dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != len(label_array) or len(mask) != len(label_array):
        raise ValueError("X, labels, and selected_mask must describe the same cells")
    if not mask.any():
        return float("nan")
    if len(np.unique(label_array)) < 2 or len(np.unique(label_array)) >= len(label_array):
        return float("nan")

    n = len(label_array)
    if n <= int(max_cells):
        indices = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(random_state)
        indices = np.asarray(rng.choice(n, size=int(max_cells), replace=False), dtype=int)
        selected_in_sample = indices[mask[indices]]
        desired = min(int(min_selected_sample), int(mask.sum()), int(max_cells) // 2)
        if len(selected_in_sample) < desired:
            selected_all = np.flatnonzero(mask)
            missing_pool = np.setdiff1d(selected_all, selected_in_sample, assume_unique=False)
            add_n = min(desired - len(selected_in_sample), len(missing_pool))
            if add_n:
                additions = np.asarray(rng.choice(missing_pool, size=add_n, replace=False), dtype=int)
                replace_positions = np.flatnonzero(~mask[indices])[:add_n]
                if len(replace_positions) < add_n:
                    replace_positions = np.arange(add_n)
                indices[replace_positions] = additions
        indices = np.unique(indices)

    sample_labels = label_array[indices]
    if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(sample_labels):
        return float("nan")
    values = silhouette_samples(matrix[indices], sample_labels)
    selected_values = values[mask[indices]]
    return float(np.mean(selected_values)) if len(selected_values) else float("nan")



def full_space_silhouette_group_means(
    X: np.ndarray,
    labels: Iterable[Any],
    masks: dict[str, np.ndarray],
    *,
    max_cells: int = 10_000,
    random_state: int = DEFAULT_BENCHMARK_SEED,
    min_group_sample: int = 64,
) -> dict[str, float]:
    """Compute full-space silhouette once and summarize several cell masks.

    This avoids repeating the expensive pairwise silhouette calculation for rare
    and non-rare subsets. Historical subset ASWs are still computed separately for
    backward compatibility.
    """
    matrix = np.asarray(X)
    label_array = np.asarray(list(labels)).astype(str)
    n = len(label_array)
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ValueError("X and labels must describe the same cells")
    if n == 0 or len(np.unique(label_array)) < 2 or len(np.unique(label_array)) >= n:
        return {name: float("nan") for name in masks}
    if n <= int(max_cells):
        indices = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(random_state)
        indices = np.asarray(rng.choice(n, size=int(max_cells), replace=False), dtype=int)
        # Ensure small requested groups are represented without growing the sample.
        for mask in masks.values():
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                continue
            have = int(mask[indices].sum())
            desired = min(int(min_group_sample), int(mask.sum()), max(1, int(max_cells) // max(2, len(masks))))
            if have >= desired:
                continue
            pool = np.setdiff1d(np.flatnonzero(mask), indices[mask[indices]], assume_unique=False)
            add_n = min(desired - have, len(pool))
            if add_n <= 0:
                continue
            additions = np.asarray(rng.choice(pool, size=add_n, replace=False), dtype=int)
            replace_positions = np.flatnonzero(~mask[indices])[:add_n]
            if len(replace_positions) < add_n:
                continue
            indices[replace_positions] = additions
        indices = np.unique(indices)
    sample_labels = label_array[indices]
    if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(sample_labels):
        return {name: float("nan") for name in masks}
    scores = silhouette_samples(matrix[indices], sample_labels)
    result: dict[str, float] = {}
    for name, mask in masks.items():
        selected = np.asarray(mask, dtype=bool)[indices]
        result[name] = float(np.mean(scores[selected])) if selected.any() else float("nan")
    return result

def global_metrics(
    X: np.ndarray,
    y_true: Iterable[Any],
    clusters: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    random_state: int = DEFAULT_BENCHMARK_SEED,
) -> dict[str, float]:
    true = np.asarray(list(y_true)).astype(str)
    cluster = np.asarray(list(clusters)).astype(str)
    pred = np.asarray(list(y_pred)).astype(str)
    return {
        "ASW_true_on_latent": safe_silhouette(X, true, random_state=random_state),
        "ARI_true_vs_cluster": float(adjusted_rand_score(true, cluster)),
        "AMI_true_vs_cluster": float(adjusted_mutual_info_score(true, cluster)),
        "Accuracy": float(accuracy_score(true, pred)),
        "F1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "G_Mean": multiclass_gmean(true, pred),
    }


def _best_cluster_recovery(
    contingency: pd.DataFrame,
    cluster_sizes: pd.Series,
    cell_type: str,
    support: int,
) -> tuple[str, float, float, float]:
    """Best one-cluster-vs-rest recovery for a true cell type.

    Unlike majority-vote label transfer, this diagnostic does not require the cell
    type to win a cluster's majority label.  It therefore avoids the structural
    ceiling caused by having fewer clusters than true cell types, while remaining
    explicitly cluster/resolution dependent.
    """
    if support <= 0 or cell_type not in contingency.index:
        return "", 0.0, 0.0, 0.0
    intersections = contingency.loc[cell_type].astype(float)
    best = ("", 0.0, 0.0, 0.0)
    best_f1 = -1.0
    for cluster_name, tp in intersections.items():
        tp = float(tp)
        if tp <= 0:
            continue
        cluster_size = float(cluster_sizes.get(cluster_name, 0.0))
        precision = tp / cluster_size if cluster_size > 0 else 0.0
        recall = tp / float(support)
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best = (str(cluster_name), float(precision), float(recall), float(f1))
    return best


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
        true, pred, labels=labels, average=None, zero_division=0
    )
    contingency = pd.crosstab(pd.Series(true, name="true"), pd.Series(cluster, name="cluster"))
    cluster_sizes = pd.Series(cluster).value_counts()
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
            counts = wrong_predictions.value_counts()
            dominant_wrong_label = str(counts.index[0])
            dominant_wrong_fraction = float(counts.iloc[0] / support[idx])
        else:
            dominant_wrong_label = ""
            dominant_wrong_fraction = 0.0
        if batch is not None and target.sum() >= 2 and len(np.unique(batch[target])) >= 2 and len(np.unique(cluster[target])) >= 2:
            within_batch_nmi = float(normalized_mutual_info_score(batch[target], cluster[target]))
        else:
            within_batch_nmi = float("nan")
        best_cluster, best_precision, best_recall, best_f1 = _best_cluster_recovery(
            contingency, cluster_sizes, cell_type, int(support[idx])
        )
        rows.append({
            "cell_type": cell_type,
            "support": int(support[idx]),
            "predicted_count": int(predicted_target.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            # Historical majority-vote metrics are intentionally retained.
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
            # Additive cluster-recovery diagnostic without majority-vote competition.
            "best_cluster": best_cluster,
            "best_cluster_precision": best_precision,
            "best_cluster_recall": best_recall,
            "best_cluster_f1": best_f1,
        })
    return pd.DataFrame(rows)


def knn_local_recovery_from_graph(
    y_true: Iterable[Any],
    neighbor_graph: Any,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute Leiden-resolution-independent local recovery from a kNN graph.

    Parameters
    ----------
    y_true:
        Ground-truth cell labels in graph row order.
    neighbor_graph:
        Sparse matrix whose non-zero entries identify kNN neighbors.  Values are
        not used; the metric is intentionally based on neighborhood membership.

    Returns
    -------
    per_type, per_cell
        ``knn_same_label_fraction`` is the observed fraction of graph neighbors
        sharing the cell's true label. ``knn_expected_fraction`` is the null
        expectation from global abundance (excluding the query cell), and
        ``knn_local_recovery`` is the historical/raw abundance-normalized score
        ``(observed - expected) / (1 - expected)``.  It is retained for backward
        comparison but has a support-dependent ceiling for populations smaller
        than the realized graph degree. ``knn_local_recovery_adjusted`` rescales
        against ``knn_max_achievable_fraction`` so 0 is the abundance-null
        expectation and 1 is the maximum local coherence achievable given the
        population support and realized graph degree. Negative values indicate
        below-null same-label neighborhoods.
    """
    labels = np.asarray(list(y_true)).astype(str)
    n = len(labels)
    if getattr(neighbor_graph, "shape", None) != (n, n):
        raise ValueError(f"neighbor_graph must have shape {(n, n)}")
    graph = neighbor_graph.tocsr() if hasattr(neighbor_graph, "tocsr") else None
    if graph is None:
        raise TypeError("neighbor_graph must be a scipy-compatible sparse matrix")

    per_cell = np.full(n, np.nan, dtype=float)
    neighbor_counts = np.zeros(n, dtype=int)
    for i in range(n):
        start, end = int(graph.indptr[i]), int(graph.indptr[i + 1])
        indices = np.asarray(graph.indices[start:end], dtype=int)
        indices = indices[indices != i]
        if len(indices) == 0:
            continue
        neighbor_counts[i] = len(indices)
        per_cell[i] = float(np.mean(labels[indices] == labels[i]))

    counts = pd.Series(labels).value_counts()
    rows: list[dict[str, Any]] = []
    for cell_type in sorted(counts.index.astype(str)):
        mask = labels == cell_type
        observed_values = per_cell[mask]
        valid = observed_values[np.isfinite(observed_values)]
        support = int(mask.sum())
        expected = float((support - 1) / (n - 1)) if n > 1 else float("nan")
        observed = float(np.mean(valid)) if len(valid) else float("nan")
        if np.isfinite(observed) and np.isfinite(expected) and expected < 1.0:
            normalized = float((observed - expected) / (1.0 - expected))
        else:
            normalized = float("nan")

        # Support-aware ceiling.  A cell of a population with ``support`` members
        # has at most ``support - 1`` same-label cells available, so with a graph
        # degree of ``d`` its same-label fraction cannot exceed ``min(support-1,
        # d)/d``.  A perfectly isolated population smaller than the graph degree
        # therefore cannot reach 1.0 on the unadjusted score, which penalizes the
        # smallest populations purely for being small.  The adjusted score
        # rescales against the achievable maximum so that "perfectly preserved"
        # is 1.0 at every support level.
        degrees = neighbor_counts[mask].astype(float)
        usable = degrees[np.isfinite(observed_values) & (degrees > 0)]
        if len(usable):
            achievable = float(np.mean(np.minimum(support - 1, usable) / usable))
        else:
            achievable = float("nan")
        if (
            np.isfinite(observed)
            and np.isfinite(expected)
            and np.isfinite(achievable)
            and (achievable - expected) > 1e-12
        ):
            adjusted = float((observed - expected) / (achievable - expected))
        else:
            adjusted = float("nan")

        rows.append({
            "cell_type": cell_type,
            "knn_same_label_fraction": observed,
            "knn_expected_fraction": expected,
            "knn_max_achievable_fraction": achievable,
            "knn_local_recovery": normalized,
            "knn_local_recovery_adjusted": adjusted,
            "knn_mean_neighbors": float(np.mean(neighbor_counts[mask])) if support else np.nan,
            "knn_valid_cells": int(len(valid)),
        })
    return pd.DataFrame(rows), per_cell


def subset_metrics(
    X: np.ndarray,
    y_true: Iterable[Any],
    clusters: Iterable[Any],
    y_pred: Iterable[Any],
    rare_types: Iterable[str],
    *,
    random_state: int = DEFAULT_BENCHMARK_SEED,
) -> pd.DataFrame:
    """Return historical subset metrics plus additive full-space ASW diagnostics.

    ``ASW_true_on_latent`` retains the original scRareBench semantics and is
    computed *within each selected subset*.  The new
    ``ASW_selected_cells_in_full_latent`` computes silhouette in the full latent
    space before averaging the selected cells and is therefore the preferred
    rare-preservation ASW interpretation.
    """
    true = np.asarray(list(y_true)).astype(str)
    cluster = np.asarray(list(clusters)).astype(str)
    pred = np.asarray(list(y_pred)).astype(str)
    rare_set = set(map(str, rare_types))
    masks = {
        "overall": np.ones(len(true), dtype=bool),
        "rare": np.asarray([value in rare_set for value in true]),
        "non_rare": np.asarray([value not in rare_set for value in true]),
    }
    full_space_means = full_space_silhouette_group_means(
        X, true, masks, random_state=random_state
    )
    rows: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        values = global_metrics(X[mask], true[mask], cluster[mask], pred[mask], random_state=random_state)
        values["ASW_selected_cells_in_full_latent"] = full_space_means.get(subset, np.nan)
        values.update({"subset": subset, "n_cells": int(mask.sum()), "n_cell_types": int(len(np.unique(true[mask])))})
        rows.append(values)
    return pd.DataFrame(rows)


def subset_metric_ratios(
    subset_frame: pd.DataFrame,
    *,
    numerator_subset: str = "non_rare",
    denominator_subset: str = "rare",
    denominator_epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Return ratio diagnostics separately from bounded/absolute metrics.

    Near-zero denominators are reported as missing with status
    ``unstable_denominator`` rather than producing huge misleading values.
    """
    if subset_frame is None or subset_frame.empty or "subset" not in subset_frame.columns:
        return pd.DataFrame(columns=["metric", "numerator_subset", "denominator_subset", "numerator", "denominator", "ratio", "status"])
    indexed = subset_frame.set_index("subset")
    if numerator_subset not in indexed.index or denominator_subset not in indexed.index:
        return pd.DataFrame(columns=["metric", "numerator_subset", "denominator_subset", "numerator", "denominator", "ratio", "status"])
    excluded = {"subset", "n_cells", "n_cell_types"}
    rows: list[dict[str, Any]] = []
    for column in [c for c in subset_frame.columns if c not in excluded]:
        numerator = pd.to_numeric(pd.Series([indexed.loc[numerator_subset, column]]), errors="coerce").iloc[0]
        denominator = pd.to_numeric(pd.Series([indexed.loc[denominator_subset, column]]), errors="coerce").iloc[0]
        status = "computed"
        ratio = np.nan
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            status = "non_finite_input"
        elif abs(float(denominator)) <= float(denominator_epsilon):
            status = "unstable_denominator"
        else:
            ratio = float(numerator) / float(denominator)
        rows.append({
            "metric": str(column),
            "numerator_subset": numerator_subset,
            "denominator_subset": denominator_subset,
            "numerator": float(numerator) if np.isfinite(numerator) else np.nan,
            "denominator": float(denominator) if np.isfinite(denominator) else np.nan,
            "ratio": ratio,
            "status": status,
        })
    return pd.DataFrame(rows)
