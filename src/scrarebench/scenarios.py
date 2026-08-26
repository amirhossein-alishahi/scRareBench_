from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIX_SCENARIOS: tuple[str, ...] = (
    "GR-DL",
    "GR-RM",
    "LE-DL",
    "LE-RM",
    "SR-DL",
    "SR-RM",
)

REGISTERED_SCENARIO_TABLES: dict[str, dict[str, str]] = {
    "gse194122": {
        "resource": "paper_scenarios.csv",
        "label_key": "celltype",
        "status": "validated_curated",
        "description": "Paper-main six-scenario curation.",
    },
    "mbdrc_renal_cortex": {
        "resource": "registered_scenarios_mbdrc_renal_cortex.csv",
        "label_key": "cell_type",
        "status": "provisional_annotation_driven",
        "description": (
            "Provisional DL/RM assignments derived from the rare-cell benchmark report. "
            "One broad 'lymphocyte' label remains topology-ambiguous and is excluded "
            "from six-state scenario summaries."
        ),
    },
    "wu_breast_cancer_atlas": {
        "resource": "registered_scenarios_wu_breast_cancer_atlas.csv",
        "label_key": "celltype_subset",
        "status": "provisional_annotation_driven",
        "description": (
            "Provisional DL/RM assignments derived from annotation identity/state semantics "
            "in the rare-cell benchmark report."
        ),
    },
    "covid19_autoimmunity_pbmc": {
        "resource": "registered_scenarios_covid19_autoimmunity_pbmc.csv",
        "label_key": "cell_type",
        "status": "provisional_annotation_driven",
        "description": (
            "Provisional DL/RM assignments derived from annotation identity/state semantics. "
            "Disease/group confounding noted in the benchmark report must be preserved in interpretation."
        ),
    },
}


def _scenario_resource(name: str) -> Path:
    return Path(files("scrarebench") / "config" / name)


def _validate_scenario_table(table: pd.DataFrame, *, allow_unassigned: bool) -> pd.DataFrame:
    required = {"cell_type", "scenario", "distribution", "topology"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Scenario table is missing columns: {sorted(missing)}")

    clean = table.copy()
    for column in ("cell_type", "scenario", "distribution", "topology", "parent_type", "curation_source"):
        if column not in clean.columns:
            clean[column] = ""
        clean[column] = clean[column].fillna("").astype(str)

    if clean["cell_type"].eq("").any():
        raise ValueError("Scenario table contains an empty cell_type.")
    if clean["cell_type"].duplicated().any():
        duplicated = sorted(clean.loc[clean["cell_type"].duplicated(keep=False), "cell_type"].unique())
        raise ValueError(f"Scenario table contains duplicate cell types: {duplicated}")

    valid = clean["scenario"].isin(SIX_SCENARIOS)
    if not allow_unassigned:
        invalid = clean.loc[~valid & clean["scenario"].ne(""), "scenario"].unique().tolist()
        if invalid:
            raise ValueError(f"Unknown six-state scenario labels: {sorted(map(str, invalid))}")
        clean = clean.loc[valid].copy()
    return clean.reset_index(drop=True)


def load_paper_scenario_table(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else _scenario_resource("paper_scenarios.csv")
    table = pd.read_csv(source).fillna("")
    return _validate_scenario_table(table, allow_unassigned=False)


def load_registered_scenario_table(
    dataset_key: str,
    *,
    include_unassigned: bool = False,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load registered benchmark scenario metadata for one dataset.

    Dataset 0 is validated/paper-curated. Dataset 2/3/4 DL/RM labels are
    provisional annotation-driven assignments from the supplied benchmark report.
    ``include_unassigned=True`` also returns explicitly tracked rare populations
    whose topology is intentionally left ambiguous.
    """
    key = str(dataset_key)
    if key not in REGISTERED_SCENARIO_TABLES:
        available = ", ".join(sorted(REGISTERED_SCENARIO_TABLES))
        raise KeyError(f"No registered scenario table for {key!r}. Available: {available}")
    spec = REGISTERED_SCENARIO_TABLES[key]
    source = Path(path) if path is not None else _scenario_resource(spec["resource"])
    table = pd.read_csv(source).fillna("")
    if "include_in_six_state" not in table.columns:
        table["include_in_six_state"] = True
    table = _validate_scenario_table(table, allow_unassigned=True)

    # CSV readers can return bools or strings depending on mixed/missing content.
    include = table["include_in_six_state"].map(
        lambda x: x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in {"1", "true", "yes", "y"}
    )
    if include_unassigned:
        return table.reset_index(drop=True)
    return table.loc[include & table["scenario"].isin(SIX_SCENARIOS)].reset_index(drop=True)


def registered_scenario_info(dataset_key: str) -> dict[str, Any]:
    key = str(dataset_key)
    if key not in REGISTERED_SCENARIO_TABLES:
        raise KeyError(f"No registered scenario metadata for {key!r}.")
    table_all = load_registered_scenario_table(key, include_unassigned=True)
    table_six = load_registered_scenario_table(key, include_unassigned=False)
    spec = dict(REGISTERED_SCENARIO_TABLES[key])
    spec.update(
        {
            "dataset_key": key,
            "n_registered_rows": int(len(table_all)),
            "n_six_state_rows": int(len(table_six)),
            "scenario_coverage": [s for s in SIX_SCENARIOS if s in set(table_six["scenario"])],
            "missing_scenarios": [s for s in SIX_SCENARIOS if s not in set(table_six["scenario"])],
            "unassigned_cell_types": table_all.loc[
                ~table_all["scenario"].isin(SIX_SCENARIOS), "cell_type"
            ].astype(str).tolist(),
        }
    )
    return spec


def _annotate_from_table(
    adata: Any,
    *,
    label_key: str,
    metadata: pd.DataFrame,
    status: str,
    dataset_key: str,
    description: str,
    prefix: str,
    inplace: bool,
    strict_labels: bool,
):
    target = adata if inplace else adata.copy()
    if label_key not in target.obs.columns:
        raise KeyError(f"adata.obs[{label_key!r}] is required for dataset {dataset_key!r}.")

    all_metadata = _validate_scenario_table(metadata, allow_unassigned=True)
    six_metadata = all_metadata.loc[all_metadata["scenario"].isin(SIX_SCENARIOS)].copy()
    observed = set(target.obs[label_key].astype(str).unique())
    expected = set(all_metadata["cell_type"].astype(str))
    missing_labels = sorted(expected.difference(observed))
    if strict_labels and missing_labels:
        raise ValueError(
            f"Registered scenario labels are missing from dataset {dataset_key!r}: "
            f"{missing_labels}. This may indicate a source/annotation revision."
        )

    indexed = all_metadata.drop_duplicates("cell_type").set_index("cell_type")
    labels = target.obs[label_key].astype(str)

    mapping_columns = {
        "scenario": "scenario",
        "distribution": "distribution",
        "topology": "topology",
        "parent_type": "parent_type",
        "curation_source": "curation_source",
        "evidence_note": "topology_evidence",
    }
    for source_col, suffix in mapping_columns.items():
        if source_col in indexed.columns:
            target.obs[f"{prefix}{suffix}"] = labels.map(indexed[source_col]).fillna("")

    for source_col, suffix in (
        ("threshold_sensitive", "threshold_sensitive"),
        ("low_confidence", "topology_low_confidence"),
        ("protected_group_restricted", "protected_group_restricted"),
        ("include_in_six_state", "is_six_state"),
    ):
        if source_col in indexed.columns:
            values = labels.map(indexed[source_col])
            target.obs[f"{prefix}{suffix}"] = values.map(
                lambda x: (
                    x
                    if isinstance(x, (bool, np.bool_))
                    else str(x).strip().lower() in {"1", "true", "yes", "y"}
                )
            ).fillna(False).astype(bool)

    if f"{prefix}is_six_state" not in target.obs.columns:
        target.obs[f"{prefix}is_six_state"] = target.obs[f"{prefix}scenario"].isin(SIX_SCENARIOS)

    target.obs[f"{prefix}is_rare"] = target.obs[f"{prefix}distribution"].isin({"GR", "LE", "SR"})
    target.obs[f"{prefix}topology_status"] = np.where(
        target.obs[f"{prefix}is_six_state"],
        status,
        np.where(target.obs[f"{prefix}is_rare"], "ambiguous_or_unassigned", ""),
    )

    # Evaluation consumes only valid six-state rows. The all-row table preserves
    # explicitly ambiguous rare populations for provenance/audit.
    target.uns[f"{prefix}scenario_table"] = six_metadata.reset_index(drop=True).to_dict(orient="list")
    target.uns[f"{prefix}scenario_annotation_table"] = all_metadata.reset_index(drop=True).to_dict(orient="list")
    target.uns[f"{prefix}scenario_registry"] = {
        "dataset_key": dataset_key,
        "label_key": label_key,
        "status": status,
        "description": description,
        "six_scenarios": list(SIX_SCENARIOS),
        "scenario_coverage": [s for s in SIX_SCENARIOS if s in set(six_metadata["scenario"])],
        "missing_scenarios": [s for s in SIX_SCENARIOS if s not in set(six_metadata["scenario"])],
        "n_six_state_cell_types": int(len(six_metadata)),
        "n_registered_rare_cell_types": int(all_metadata["distribution"].isin({"GR", "LE", "SR"}).sum()),
        "unassigned_cell_types": all_metadata.loc[
            ~all_metadata["scenario"].isin(SIX_SCENARIOS), "cell_type"
        ].astype(str).tolist(),
        "missing_registered_labels_in_loaded_data": missing_labels,
    }
    return target


def annotate_registered_scenarios(
    adata: Any,
    *,
    dataset_key: str,
    label_key: str | None = None,
    prefix: str = "scrarebench_",
    inplace: bool = True,
    strict_labels: bool = True,
):
    """Attach registered six-state metadata without expression preprocessing.

    This only annotates ``obs``/``uns``. It never normalizes, selects HVGs,
    scales, computes PCA/neighbors, or edits the source H5AD on disk.
    """
    key = str(dataset_key)
    if key not in REGISTERED_SCENARIO_TABLES:
        raise KeyError(f"No registered scenario table for {key!r}.")
    spec = REGISTERED_SCENARIO_TABLES[key]
    metadata = load_registered_scenario_table(key, include_unassigned=True)
    return _annotate_from_table(
        adata,
        label_key=label_key or spec["label_key"],
        metadata=metadata,
        status=spec["status"],
        dataset_key=key,
        description=spec["description"],
        prefix=prefix,
        inplace=inplace,
        strict_labels=strict_labels,
    )


def scenario_table_from_adata(
    adata: Any,
    *,
    prefix: str = "scrarebench_",
) -> pd.DataFrame | None:
    """Return the dataset-specific six-state table embedded in ``adata.uns``."""
    payload = getattr(adata, "uns", {}).get(f"{prefix}scenario_table")
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        table = payload.copy()
    else:
        table = pd.DataFrame(payload)
    if table.empty:
        return None
    return _validate_scenario_table(table, allow_unassigned=False)


def annotate_paper_scenarios(
    adata: Any,
    *,
    label_key: str = "celltype",
    table: pd.DataFrame | None = None,
    prefix: str = "scrarebench_",
    inplace: bool = True,
):
    metadata = table.copy() if table is not None else load_paper_scenario_table()
    return _annotate_from_table(
        adata,
        label_key=label_key,
        metadata=metadata,
        status="validated_curated",
        dataset_key="gse194122",
        description="Paper-main six-scenario curation.",
        prefix=prefix,
        inplace=inplace,
        strict_labels=False,
    )


def infer_distribution_classes(
    obs: pd.DataFrame,
    *,
    batch_key: str,
    label_key: str,
    global_abundance_threshold: float = 0.01,
    batch_fraction_threshold: float = 0.25,
    local_abundance_threshold: float = 0.01,
) -> pd.DataFrame:
    """Infer only GR/LE/SR from distributions; DL/RM remains biological metadata.

    GR: global abundance < threshold and present in more than the batch threshold.
    LE: present in <= batch threshold and reaches local abundance >= threshold.
    SR: present in <= batch threshold and remains locally < threshold.
    Other populations are reported as COMMON_OR_UNASSIGNED.

    Note: registered Dataset 2/3/4 scenario tables are fixed metadata derived from
    the supplied benchmark report and should not be regenerated by this helper.
    """
    counts = pd.crosstab(obs[batch_key].astype(str), obs[label_key].astype(str))
    batch_sizes = counts.sum(axis=1)
    fractions = counts.div(batch_sizes, axis=0)
    total_cells = len(obs)
    total_batches = len(counts)
    rows: list[dict[str, Any]] = []
    for cell_type in counts.columns:
        cell_counts = counts[cell_type]
        present = cell_counts.gt(0)
        n_present = int(present.sum())
        batch_fraction = n_present / total_batches if total_batches else np.nan
        global_abundance = float(cell_counts.sum() / total_cells) if total_cells else np.nan
        max_local = float(fractions.loc[present, cell_type].max()) if n_present else 0.0
        mean_local = float(fractions.loc[present, cell_type].mean()) if n_present else 0.0
        if global_abundance < global_abundance_threshold and batch_fraction > batch_fraction_threshold:
            distribution = "GR"
        elif batch_fraction <= batch_fraction_threshold and max_local >= local_abundance_threshold:
            distribution = "LE"
        elif batch_fraction <= batch_fraction_threshold and max_local < local_abundance_threshold:
            distribution = "SR"
        else:
            distribution = "COMMON_OR_UNASSIGNED"
        rows.append(
            {
                "cell_type": cell_type,
                "distribution": distribution,
                "global_abundance": global_abundance,
                "n_present_batches": n_present,
                "n_total_batches": total_batches,
                "batch_fraction": batch_fraction,
                "max_local_abundance": max_local,
                "mean_local_abundance_when_present": mean_local,
            }
        )
    return pd.DataFrame(rows).sort_values(["distribution", "cell_type"]).reset_index(drop=True)
