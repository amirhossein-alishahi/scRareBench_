from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# Historical precedence is frozen for backward compatibility.  The columns
# ``failure_archetype`` / ``failure_matched_archetypes`` continue to use this
# exact contract so results from older scRareBench releases remain comparable.
FAILURE_PRECEDENCE = (
    "preserved",
    "batch_driven_fragmentation",
    "lineage_leakage",
    "lineage_assimilation",
)

# Resolution-aware taxonomy introduced in v0.9.6.  It is additive: the legacy
# taxonomy above remains in every result.  ``resolution_limited`` intercepts an
# apparent legacy assimilation only when cluster-label transfer is poor while
# independent/local evidence indicates that the population remains coherent.
FAILURE_PRECEDENCE_V2 = (
    "preserved",
    "resolution_limited",
    "batch_driven_fragmentation",
    "lineage_leakage",
    "lineage_assimilation",
)


def load_failure_rules(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else Path(files("scrarebench") / "config" / "default_failure_rules.yaml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def _float(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value) if pd.notna(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def matched_failure_archetypes(row: pd.Series, rules: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every *legacy* provisional rule matched by a population.

    This function intentionally preserves the pre-v0.9.6 majority-vote rule
    contract.  Use :func:`matched_failure_archetypes_v2` for the additive
    resolution-aware interpretation.
    """
    precision = _float(row, "precision", 0.0)
    recall = _float(row, "recall", 0.0)
    inverse_purity = _float(row, "inverse_purity", 0.0)
    nmi = _float(row, "within_type_batch_nmi")
    n_clusters = int(_float(row, "n_clusters_found_in", 0.0) or 0)
    dominant_wrong_fraction = _float(row, "dominant_wrong_fraction", 0.0)
    matches: list[tuple[str, str]] = []

    preserved = rules["preserved"]
    if (
        precision >= preserved["min_precision"]
        and recall >= preserved["min_recall"]
        and inverse_purity >= preserved["min_inverse_purity"]
    ):
        matches.append(("preserved", "High majority-vote precision and recall with high dominant-cluster capture"))

    fragmentation = rules["batch_fragmentation"]
    if (
        inverse_purity <= fragmentation["max_inverse_purity"]
        and n_clusters >= fragmentation["min_clusters"]
        and pd.notna(nmi)
        and nmi >= fragmentation["min_within_type_batch_nmi"]
    ):
        matches.append(("batch_driven_fragmentation", "Low completeness with cluster-batch association within the cell type"))

    leakage = rules["lineage_leakage"]
    if recall >= leakage["min_recall"] and precision <= leakage["max_precision"]:
        matches.append(("lineage_leakage", "Target cells are recovered but the predicted population contains many false positives"))

    assimilation = rules["lineage_assimilation"]
    if (
        recall <= assimilation["max_recall"]
        and dominant_wrong_fraction >= assimilation["min_dominant_wrong_fraction"]
    ):
        matches.append(("lineage_assimilation", "Most target cells receive one non-target majority-vote label"))

    order = {name: index for index, name in enumerate(FAILURE_PRECEDENCE)}
    matches.sort(key=lambda item: order.get(item[0], len(order)))
    return matches


def _resolution_limited_match(row: pd.Series, rules: dict[str, Any]) -> tuple[str, str] | None:
    """Return the v2 resolution-limited guard when independent evidence supports it.

    The guard is deliberately narrow.  It only applies to a row that already
    satisfies the historical lineage-assimilation rule.  A high support-adjusted
    local-recovery score plus high dominant-cluster capture indicates that the
    population remains locally coherent/captured even though majority-vote label
    transfer failed at the canonical Leiden resolution.

    Missing/undefined adjusted kNN values (including singleton populations) never
    trigger this rule.
    """
    rule = rules.get("resolution_limited")
    if not isinstance(rule, dict):
        return None

    recall = _float(row, "recall")
    dominant_wrong_fraction = _float(row, "dominant_wrong_fraction")
    adjusted = _float(row, "knn_local_recovery_adjusted")
    inverse_purity = _float(row, "inverse_purity")
    assimilation = rules.get("lineage_assimilation", {})

    required = (
        recall,
        dominant_wrong_fraction,
        adjusted,
        inverse_purity,
        _float(pd.Series(assimilation), "max_recall"),
        _float(pd.Series(assimilation), "min_dominant_wrong_fraction"),
        _float(pd.Series(rule), "min_knn_local_recovery_adjusted"),
        _float(pd.Series(rule), "min_inverse_purity"),
    )
    if not all(np.isfinite(x) for x in required):
        return None

    assimilation_candidate = (
        recall <= float(assimilation["max_recall"])
        and dominant_wrong_fraction >= float(assimilation["min_dominant_wrong_fraction"])
    )
    if (
        assimilation_candidate
        and adjusted >= float(rule["min_knn_local_recovery_adjusted"])
        and inverse_purity >= float(rule["min_inverse_purity"])
    ):
        return (
            "resolution_limited",
            "Majority-vote label transfer suggests assimilation, but support-adjusted local recovery and dominant-cluster capture remain high; interpret as resolution/cluster-ownership limited rather than confirmed biological assimilation",
        )
    return None


def matched_failure_archetypes_v2(row: pd.Series, rules: dict[str, Any]) -> list[tuple[str, str]]:
    """Return additive resolution-aware failure matches in v2 precedence order.

    Legacy matches are retained, including ``lineage_assimilation`` when its
    historical rule fires.  If the resolution-limited guard also fires it is
    placed earlier in the v2 precedence, so the primary v2 label is
    ``resolution_limited`` while the underlying legacy match remains auditable.
    """
    matches = list(matched_failure_archetypes(row, rules))
    guarded = _resolution_limited_match(row, rules)
    if guarded is not None:
        matches.append(guarded)
    order = {name: index for index, name in enumerate(FAILURE_PRECEDENCE_V2)}
    matches.sort(key=lambda item: order.get(item[0], len(order)))
    return matches


def classify_failure_row(row: pd.Series, rules: dict[str, Any]) -> tuple[str, str]:
    """Historical/legacy primary classification."""
    matches = matched_failure_archetypes(row, rules)
    if matches:
        return matches[0]
    return "mixed_or_uncertain", "No single provisional legacy rule dominates"


def classify_failure_row_v2(row: pd.Series, rules: dict[str, Any]) -> tuple[str, str]:
    """Resolution-aware v2 primary classification."""
    matches = matched_failure_archetypes_v2(row, rules)
    if matches:
        return matches[0]
    return "mixed_or_uncertain", "No single provisional resolution-aware rule dominates"


def classify_failure_archetypes(
    metrics: pd.DataFrame,
    *,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add both legacy and resolution-aware failure-taxonomy columns.

    Backward-compatible columns:
      - ``failure_archetype``
      - ``failure_rationale``
      - ``failure_matched_archetypes``
      - ``failure_match_count``

    Additive v2 columns:
      - ``failure_archetype_v2``
      - ``failure_rationale_v2``
      - ``failure_matched_archetypes_v2``
      - ``failure_match_count_v2``
    """
    selected_rules = rules or load_failure_rules()
    output = metrics.copy()

    legacy_labels: list[str] = []
    legacy_rationales: list[str] = []
    legacy_matched_labels: list[str] = []
    legacy_match_counts: list[int] = []
    v2_labels: list[str] = []
    v2_rationales: list[str] = []
    v2_matched_labels: list[str] = []
    v2_match_counts: list[int] = []

    for _, row in output.iterrows():
        legacy = matched_failure_archetypes(row, selected_rules)
        if legacy:
            legacy_label, legacy_rationale = legacy[0]
        else:
            legacy_label, legacy_rationale = "mixed_or_uncertain", "No single provisional legacy rule dominates"
        legacy_labels.append(legacy_label)
        legacy_rationales.append(legacy_rationale)
        legacy_matched_labels.append(";".join(label for label, _ in legacy))
        legacy_match_counts.append(len(legacy))

        v2 = matched_failure_archetypes_v2(row, selected_rules)
        if v2:
            v2_label, v2_rationale = v2[0]
        else:
            v2_label, v2_rationale = "mixed_or_uncertain", "No single provisional resolution-aware rule dominates"
        v2_labels.append(v2_label)
        v2_rationales.append(v2_rationale)
        v2_matched_labels.append(";".join(label for label, _ in v2))
        v2_match_counts.append(len(v2))

    output["failure_archetype"] = legacy_labels
    output["failure_rationale"] = legacy_rationales
    output["failure_matched_archetypes"] = legacy_matched_labels
    output["failure_match_count"] = legacy_match_counts
    output["failure_archetype_v2"] = v2_labels
    output["failure_rationale_v2"] = v2_rationales
    output["failure_matched_archetypes_v2"] = v2_matched_labels
    output["failure_match_count_v2"] = v2_match_counts
    return output
