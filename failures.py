from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_failure_rules(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else Path(files("scrarebench") / "config" / "default_failure_rules.yaml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def classify_failure_row(row: pd.Series, rules: dict[str, Any]) -> tuple[str, str]:
    precision = float(row["precision"])
    recall = float(row["recall"])
    inverse_purity = float(row["inverse_purity"])
    nmi = float(row["within_type_batch_nmi"]) if pd.notna(row["within_type_batch_nmi"]) else np.nan
    n_clusters = int(row["n_clusters_found_in"])
    dominant_wrong_fraction = float(row["dominant_wrong_fraction"])

    preserved = rules["preserved"]
    if (
        precision >= preserved["min_precision"]
        and recall >= preserved["min_recall"]
        and inverse_purity >= preserved["min_inverse_purity"]
    ):
        return "preserved", "High precision, recall, and inverse purity"

    fragmentation = rules["batch_fragmentation"]
    if (
        inverse_purity <= fragmentation["max_inverse_purity"]
        and n_clusters >= fragmentation["min_clusters"]
        and pd.notna(nmi)
        and nmi >= fragmentation["min_within_type_batch_nmi"]
    ):
        return "batch_driven_fragmentation", "Low completeness with cluster-batch association within the cell type"

    leakage = rules["lineage_leakage"]
    if recall >= leakage["min_recall"] and precision <= leakage["max_precision"]:
        return "lineage_leakage", "Target cells are recovered but the predicted population contains many false positives"

    assimilation = rules["lineage_assimilation"]
    if (
        recall <= assimilation["max_recall"]
        and dominant_wrong_fraction >= assimilation["min_dominant_wrong_fraction"]
    ):
        return "lineage_assimilation", "Most target cells are absorbed into one non-target majority label"

    return "mixed_or_uncertain", "No single provisional rule dominates"


def classify_failure_archetypes(
    metrics: pd.DataFrame,
    *,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    selected_rules = rules or load_failure_rules()
    output = metrics.copy()
    labels: list[str] = []
    rationales: list[str] = []
    for _, row in output.iterrows():
        label, rationale = classify_failure_row(row, selected_rules)
        labels.append(label)
        rationales.append(rationale)
    output["failure_archetype"] = labels
    output["failure_rationale"] = rationales
    return output
