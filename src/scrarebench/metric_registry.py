from __future__ import annotations

from typing import Any

# Canonical metric semantics used by machine-readable reports and dashboards.
# Descriptions are deliberately plain-language enough to surface in the UI.
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "ASW_true_on_latent": {
        "label": "ASW true labels on latent",
        "direction": "maximize",
        "family": "core",
        "description": "Historical silhouette width computed within the cells supplied to the metric. For the rare subset this measures rare-vs-rare separation, not separation from abundant populations.",
    },
    "ASW_selected_cells_in_full_latent": {
        "label": "ASW selected cells in full latent",
        "direction": "maximize",
        "family": "rare_recovery",
        "description": "Silhouette values are computed with abundant and rare competitors present, then averaged over the selected cells. Preferred ASW diagnostic for rare preservation.",
    },
    "ARI_true_vs_cluster": {"label": "ARI true vs cluster", "direction": "maximize", "family": "core", "description": "Adjusted Rand index between reference labels and the evaluated Leiden partition. In a subset row it is computed on that subset only; rare-row ARI therefore describes rare-internal partition agreement, not confusion with excluded abundant cells."},
    "AMI_true_vs_cluster": {"label": "AMI true vs cluster", "direction": "maximize", "family": "core", "description": "Adjusted mutual information between reference labels and the evaluated Leiden partition. In a subset row it is subset-internal and should be interpreted together with full-space/local rare-recovery diagnostics."},
    "Accuracy": {"label": "Accuracy", "direction": "maximize", "family": "cluster_label_transfer", "description": "Cell-level accuracy after each Leiden cluster is assigned its majority reference label."},
    "F1_macro": {"label": "F1 macro", "direction": "maximize", "family": "cluster_label_transfer", "description": "Macro F1 after majority-vote cluster-to-label transfer. It is resolution dependent and can be limited when there are fewer clusters than cell types."},
    "F1_weighted": {"label": "F1 weighted", "direction": "maximize", "family": "cluster_label_transfer", "description": "Support-weighted F1 after majority-vote cluster-to-label transfer."},
    "G_Mean": {"label": "G-Mean", "direction": "maximize", "family": "cluster_label_transfer", "description": "Geometric mean of class recalls after majority-vote cluster-to-label transfer."},
    "precision": {"label": "Majority-vote precision", "direction": "maximize", "family": "cluster_label_transfer", "description": "Per-type precision after cluster majority-vote label transfer; retained for backward compatibility and resolution sensitivity analysis."},
    "recall": {"label": "Majority-vote recall", "direction": "maximize", "family": "cluster_label_transfer", "description": "Per-type recall after cluster majority-vote label transfer; retained for backward compatibility and resolution sensitivity analysis."},
    "f1": {"label": "Majority-vote F1", "direction": "maximize", "family": "cluster_label_transfer", "description": "Per-type F1 after cluster majority-vote label transfer. A zero can reflect cluster-count/resolution limitations, so interpret with the Leiden-resolution-independent local recovery metrics and the recorded kNN graph contract."},
    "inverse_purity": {"label": "Inverse purity / capture", "direction": "maximize", "family": "rare_recovery", "description": "Fraction of a true population captured by its single dominant Leiden cluster. It does not require that the population win the cluster majority label."},
    "best_cluster_precision": {"label": "Best-cluster precision", "direction": "maximize", "family": "rare_recovery", "description": "Precision of the single Leiden cluster that gives the best one-cluster-vs-rest F1 for the true population."},
    "best_cluster_recall": {"label": "Best-cluster recall", "direction": "maximize", "family": "rare_recovery", "description": "Recall of the cluster that best recovers the true population without requiring majority-vote ownership."},
    "best_cluster_f1": {"label": "Best-cluster F1", "direction": "maximize", "family": "rare_recovery", "description": "Best one-cluster-vs-rest F1 for a true population. Resolution dependent, but not subject to competition for the cluster majority label."},
    "knn_same_label_fraction": {"label": "kNN same-label fraction", "direction": "maximize", "family": "rare_recovery", "description": "Mean fraction of package-controlled kNN neighbors sharing the cell's true label. This is independent of Leiden resolution."},
    "knn_expected_fraction": {"label": "kNN abundance null", "direction": "context", "family": "rare_recovery", "description": "Expected same-label neighbor fraction from global abundance alone. This is a null/reference quantity, not a score to rank directly."},
    "knn_local_recovery": {"label": "kNN local recovery (raw, support-limited)", "direction": "context", "family": "rare_recovery", "description": "Historical/raw kNN local coherence normalized against global abundance. For populations smaller than the graph degree, the score has a support-dependent ceiling below 1; retain it for backward comparison, but use the support-adjusted metric for cross-population or cross-scenario interpretation."},
    "within_type_batch_nmi": {"label": "Within-type batch NMI", "direction": "minimize", "family": "batch_dependence", "description": "Association between batch and Leiden cluster assignment within one true cell type. Lower indicates less batch-driven fragmentation."},
    "dominant_wrong_fraction": {"label": "Dominant wrong fraction", "direction": "minimize", "family": "failure_diagnostic", "description": "Fraction of a true population assigned to its most common wrong majority-vote label. Lower is better."},
    "knn_local_recovery_adjusted": {"label": "kNN local recovery (support-adjusted)", "direction": "maximize", "family": "rare_recovery", "description": "Primary local-recovery metric. It rescales observed same-label neighborhood enrichment against both the global-abundance null and the maximum achievable same-label fraction given population support and realized graph degree. A perfectly isolated population can reach 1.0 even when support is smaller than k."},
    # Diagnostics, not scores. Registered explicitly so metric_info() cannot
    # default them to "maximize" if one is ever chosen as a ranking axis.
    "knn_max_achievable_fraction": {"label": "kNN achievable ceiling", "direction": "context", "family": "rare_recovery", "description": "Maximum same-label neighbor fraction attainable given this population's support and the graph degree. Context for interpreting the unadjusted score."},
    "knn_mean_neighbors": {"label": "Mean kNN graph neighbors", "direction": "context", "family": "rare_recovery", "description": "Average number of kNN graph neighbors per cell of this population. Graph-degree context, not a quality score."},
    "knn_valid_cells": {"label": "Cells with kNN neighbors", "direction": "context", "family": "rare_recovery", "description": "Number of cells of this population that had at least one graph neighbor. Coverage context, not a quality score."},
    "failure_match_count": {"label": "Matched failure rules", "direction": "context", "family": "failure_diagnostic", "description": "Number of provisional failure rules matched by this population. Overlap context, not a quality score."},
    "failure_match_count_v2": {"label": "Matched failure rules (resolution-aware v2)", "direction": "context", "family": "failure_diagnostic", "description": "Number of provisional resolution-aware v2 failure rules matched by this population. Context only, not a performance score."},
    "support": {"label": "Population support", "direction": "context", "family": "population_context", "description": "Number of cells in the true population. Interpret very small populations with wider uncertainty; support is context, not a performance score."},
    "preserved_fraction": {"label": "Legacy preserved fraction (majority-vote rule)", "direction": "context", "family": "failure_diagnostic", "description": "Historical rule-based fraction of evaluated rare populations satisfying provisional precision/recall/dominant-cluster thresholds. Precision and recall depend on majority-vote label transfer, so this value is retained for continuity but should not be the primary rare-preservation endpoint. A measured absence is 0, not missing."},
    "resolution_limited_fraction": {"label": "Resolution-limited fraction (v2)", "direction": "context", "family": "failure_diagnostic", "description": "Fraction of evaluated rare populations whose resolution-aware v2 interpretation is resolution_limited. This is diagnostic context, not evidence that the population is preserved and not a performance score for ranking methods."},
    "Isolated labels": {"label": "Isolated labels", "direction": "maximize", "family": "scib", "description": "scIB-compatible isolated-label conservation score."},
    "Leiden NMI": {"label": "Leiden NMI", "direction": "maximize", "family": "scib", "description": "scIB-compatible NMI label-conservation score using Leiden clustering."},
    "Leiden ARI": {"label": "Leiden ARI", "direction": "maximize", "family": "scib", "description": "scIB-compatible ARI label-conservation score using Leiden clustering."},
    "KMeans NMI": {"label": "KMeans NMI", "direction": "maximize", "family": "scib", "description": "scIB-compatible NMI label-conservation score using KMeans."},
    "KMeans ARI": {"label": "KMeans ARI", "direction": "maximize", "family": "scib", "description": "scIB-compatible ARI label-conservation score using KMeans."},
    "Silhouette label": {"label": "Silhouette label", "direction": "maximize", "family": "scib", "description": "scIB-compatible biological silhouette score."},
    "cLISI": {"label": "cLISI", "direction": "maximize", "family": "scib", "description": "Cell-type local inverse Simpson index score after scIB normalization."},
    "BRAS": {"label": "BRAS", "direction": "maximize", "family": "scib", "description": "Batch-removal adjusted silhouette score from the pinned scib-metrics backend."},
    "Silhouette batch": {"label": "Silhouette batch", "direction": "maximize", "family": "scib", "description": "scIB-normalized batch mixing silhouette score; the normalized score is higher-is-better."},
    "iLISI": {"label": "iLISI", "direction": "maximize", "family": "scib", "description": "Integration local inverse Simpson index score after scIB normalization."},
    "KBET": {"label": "KBET", "direction": "maximize", "family": "scib", "description": "scIB-normalized kBET batch-mixing score."},
    "Graph connectivity": {"label": "Graph connectivity", "direction": "maximize", "family": "scib", "description": "Connectivity of cells sharing biological labels in the integrated graph."},
    "PCR comparison": {"label": "PCR comparison", "direction": "maximize", "family": "scib", "description": "scIB principal-component regression comparison score for batch removal."},
    "Bio conservation": {"label": "Bio conservation", "direction": "maximize", "family": "scib_aggregate", "description": "Aggregate scIB biological-conservation score."},
    "Batch correction": {"label": "Batch correction", "direction": "maximize", "family": "scib_aggregate", "description": "Aggregate scIB batch-correction score."},
    "Total": {"label": "scIB Total", "direction": "maximize", "family": "scib_aggregate", "description": "Pinned scIB aggregate total derived from biological-conservation and batch-correction components."},
}


def metric_info(metric: str) -> dict[str, Any]:
    return dict(METRIC_REGISTRY.get(str(metric), {"label": str(metric), "direction": "maximize", "family": "unknown", "description": "No project-specific explanation is registered for this metric."}))


def metric_direction(metric: str) -> str:
    return str(metric_info(metric)["direction"])
