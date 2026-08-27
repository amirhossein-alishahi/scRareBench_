from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..scenarios import SIX_SCENARIOS

SCRAREBENCH_UNS_KEY = "scrarebench"
DEFAULT_SCENARIO_KEY = "scrarebench_scenario"


@dataclass(frozen=True)
class BenchmarkDatasetProfile:
    dataset_key: str
    label_key: str | None
    batch_key: str | None
    count_layer: str | None = "counts"
    scenario_key: str = DEFAULT_SCENARIO_KEY
    scib_hvg_batch_mode: str = "evaluation_batch"
    batch_components: tuple[str, ...] = ()
    batch_separator: str = " || "
    benchmark_ready: bool = False
    note: str = ""


BUILTIN_BENCHMARK_PROFILES: dict[str, BenchmarkDatasetProfile] = {
    "gse194122": BenchmarkDatasetProfile("gse194122", "celltype", "BATCH", benchmark_ready=True,
        note="Curated GSE194122 paper benchmark."),
    "gse194122_raw": BenchmarkDatasetProfile("gse194122_raw", "celltype", "BATCH", benchmark_ready=True,
        note="Original GSE194122 source."),
    "mbdrc_renal_cortex": BenchmarkDatasetProfile(
        "mbdrc_renal_cortex", "cell_type", "scrarebench_batch", scib_hvg_batch_mode="global",
        batch_components=("donor_id", "assay"), benchmark_ready=True,
        note="Evaluation batch is donor_id × assay."),
    "wu_breast_cancer_atlas": BenchmarkDatasetProfile("wu_breast_cancer_atlas", "celltype_subset", None,
        benchmark_ready=False, note="Register/choose the biological batch before benchmarking."),
    "covid19_autoimmunity_pbmc": BenchmarkDatasetProfile("covid19_autoimmunity_pbmc", "cell_type", None,
        benchmark_ready=False, note="Register/choose the biological batch before benchmarking."),
    "nygc_seurat_v4_pbmc": BenchmarkDatasetProfile("nygc_seurat_v4_pbmc", None, None,
        benchmark_ready=False, note="Register label and batch fields before benchmarking."),
}


def default_data_dir() -> Path:
    value = os.environ.get("SCRAREBENCH_DATA_DIR")
    return Path(value).expanduser().resolve() if value else (Path.home()/".cache"/"scrarebench"/"datasets").resolve()


def dataset_info(adata: Any) -> dict[str, Any]:
    value = getattr(adata, "uns", {}).get(SCRAREBENCH_UNS_KEY, {})
    return dict(value) if isinstance(value, dict) else {}


def _ensure_batch(adata: Any, profile: BenchmarkDatasetProfile) -> None:
    if not profile.batch_components or not profile.batch_key:
        return
    missing = [x for x in profile.batch_components if x not in adata.obs.columns]
    if missing:
        raise KeyError(f"Dataset {profile.dataset_key!r} requires obs columns {missing}.")
    values = adata.obs[profile.batch_components[0]].astype(str)
    for key in profile.batch_components[1:]:
        values = values + profile.batch_separator + adata.obs[key].astype(str)
    adata.obs[profile.batch_key] = pd.Categorical(values)


def attach_builtin_dataset_metadata(adata: Any, *, dataset_key: str, dataset_index: int | None = None,
                                    display_name: str | None = None, source_path: str | Path | None = None) -> Any:
    profile = BUILTIN_BENCHMARK_PROFILES.get(str(dataset_key))
    if profile is None:
        return adata
    _ensure_batch(adata, profile)
    payload = asdict(profile)
    payload.update({
        "name": display_name or profile.dataset_key,
        "dataset_key": profile.dataset_key,
        "dataset_index": int(dataset_index) if dataset_index is not None else -1,
        "registered_dataset": True,
        "source_path": str(source_path) if source_path is not None else "",
    })
    payload = {k: ("" if v is None else v) for k, v in payload.items()}
    payload["batch_components"] = list(profile.batch_components)
    adata.uns[SCRAREBENCH_UNS_KEY] = payload
    return adata


def _read_scenario_table(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    return pd.DataFrame(value)


def register_dataset(adata: Any, *, label_key: str, batch_key: str, name: str = "custom_dataset",
                     count_layer: str | None = "counts", rare_types: Iterable[str] | None = None,
                     scenario_table: Any = None, scib_hvg_batch_mode: str = "evaluation_batch",
                     copy: bool = False) -> Any:
    """Record the evaluation contract for a user AnnData; never preprocess/run a method."""
    target = adata.copy() if copy else adata
    for key in (label_key, batch_key):
        if key not in target.obs.columns:
            raise KeyError(f"adata.obs[{key!r}] is required to register this dataset.")
    if not target.obs_names.is_unique:
        raise ValueError("adata.obs_names must be unique before benchmarking.")
    if count_layer is not None and count_layer not in target.layers:
        raise KeyError(f"count_layer={count_layer!r} is not present in adata.layers. Use count_layer=None to use adata.X.")
    mode = str(scib_hvg_batch_mode).strip().lower()
    if mode not in {"evaluation_batch", "global"}:
        raise ValueError("scib_hvg_batch_mode must be 'evaluation_batch' or 'global'.")

    observed = set(target.obs[label_key].astype(str).unique())
    rare = list(dict.fromkeys(str(x) for x in (rare_types or [])))
    missing = sorted(set(rare).difference(observed))
    if missing:
        raise ValueError(f"rare_types contains labels absent from label_key: {missing}")

    table = _read_scenario_table(scenario_table)
    if table is not None:
        required = {"cell_type", "scenario", "distribution", "topology"}
        missing_cols = required.difference(table.columns)
        if missing_cols:
            raise ValueError(f"scenario_table is missing required columns: {sorted(missing_cols)}")
        table = table.copy()
        for col in ("cell_type", "scenario", "distribution", "topology", "parent_type", "curation_source"):
            if col not in table.columns:
                table[col] = ""
            table[col] = table[col].fillna("").astype(str)
        if table["cell_type"].duplicated().any():
            raise ValueError("scenario_table contains duplicate cell types.")
        unknown_types = sorted(set(table["cell_type"]).difference(observed))
        if unknown_types:
            raise ValueError(f"scenario_table contains cell types absent from label_key: {unknown_types}")
        allowed = set(SIX_SCENARIOS) | {"", "UNASSIGNED", "GR", "LE", "SR"}
        invalid = sorted(set(table["scenario"]).difference(allowed))
        if invalid:
            raise ValueError(f"Unknown scenario values: {invalid}")
        invalid_distribution = sorted(set(table["distribution"]).difference({"", "GR", "LE", "SR"}))
        if invalid_distribution:
            raise ValueError(f"Unknown distribution values: {invalid_distribution}")
        invalid_topology = sorted(set(table["topology"]).difference({"", "DL", "RM"}))
        if invalid_topology:
            raise ValueError(f"Unknown topology values: {invalid_topology}")
        for row in table.itertuples(index=False):
            scenario = str(row.scenario)
            distribution = str(row.distribution)
            topology = str(row.topology)
            if scenario in SIX_SCENARIOS:
                expected_distribution, expected_topology = scenario.split("-", 1)
                if distribution and distribution != expected_distribution:
                    raise ValueError(f"Scenario {scenario!r} conflicts with distribution {distribution!r}.")
                if topology and topology != expected_topology:
                    raise ValueError(f"Scenario {scenario!r} conflicts with topology {topology!r}.")
            elif scenario in {"GR", "LE", "SR"} and distribution and distribution != scenario:
                raise ValueError(
                    f"Distribution-only scenario {scenario!r} conflicts with distribution {distribution!r}."
                )
        rare = list(dict.fromkeys([*rare, *table["cell_type"].tolist()]))
        indexed = table.set_index("cell_type")
        labels = target.obs[label_key].astype(str)
        for src, dst in (("scenario","scrarebench_scenario"),("distribution","scrarebench_distribution"),
                         ("topology","scrarebench_topology"),("parent_type","scrarebench_parent_type"),
                         ("curation_source","scrarebench_curation_source")):
            target.obs[dst] = labels.map(indexed[src]).fillna("").astype(str)
        target.uns["scrarebench_custom_scenario_table"] = table.to_dict(orient="list")
        six = table[table["scenario"].isin(SIX_SCENARIOS)].reset_index(drop=True)
        target.uns["scrarebench_scenario_table"] = six.to_dict(orient="list")
    if rare:
        target.obs["scrarebench_is_rare"] = target.obs[label_key].astype(str).isin(set(rare)).astype(bool)

    target.uns[SCRAREBENCH_UNS_KEY] = {
        "name": str(name), "dataset_key": "", "dataset_index": -1, "registered_dataset": False,
        "benchmark_ready": True, "label_key": str(label_key), "batch_key": str(batch_key),
        "count_layer": "" if count_layer is None else str(count_layer), "scenario_key": DEFAULT_SCENARIO_KEY,
        "scib_hvg_batch_mode": mode, "rare_types": rare, "source_path": "",
        "note": "User-registered dataset; method preprocessing remains user-controlled.",
    }
    return target


__all__ = ["BenchmarkDatasetProfile", "BUILTIN_BENCHMARK_PROFILES", "SCRAREBENCH_UNS_KEY",
           "default_data_dir", "dataset_info", "attach_builtin_dataset_metadata", "register_dataset"]
