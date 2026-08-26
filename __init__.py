"""scRareBench: standard and rare-cell-aware benchmarking for scRNA-seq integration.

The public API is loaded lazily so lightweight helpers such as
``scrarebench.runtime`` can be imported before the scientific dependency stack
is installed.  Existing ``from scrarebench import ...`` imports remain valid.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.9.2"

_EXPORTS: dict[str, tuple[str, str]] = {
    "DATASET_REGISTRY": (".datasets", "DATASET_REGISTRY"),
    "download_dataset": (".datasets", "download_dataset"),
    "list_datasets": (".datasets", "list_datasets"),
    "load_dataset": (".datasets", "load_dataset"),
    "load_gse194122_benchmark": (".datasets", "load_gse194122_benchmark"),
    "resolve_dataset": (".datasets", "resolve_dataset"),
    "EvaluationConfig": (".evaluation", "EvaluationConfig"),
    "EvaluationResult": (".evaluation", "EvaluationResult"),
    "evaluate_latent": (".evaluation", "evaluate_latent"),
    "attach_latent": (".latent", "attach_latent"),
    "load_latent": (".latent", "load_latent"),
    "create_report_bundle": (".reporting", "create_report_bundle"),
    "write_interactive_report": (".reporting", "write_interactive_report"),
    "write_pdf_report": (".reporting", "write_pdf_report"),
    "SIX_SCENARIOS": (".scenarios", "SIX_SCENARIOS"),
    "REGISTERED_SCENARIO_TABLES": (".scenarios", "REGISTERED_SCENARIO_TABLES"),
    "annotate_paper_scenarios": (".scenarios", "annotate_paper_scenarios"),
    "annotate_registered_scenarios": (".scenarios", "annotate_registered_scenarios"),
    "infer_distribution_classes": (".scenarios", "infer_distribution_classes"),
    "load_registered_scenario_table": (".scenarios", "load_registered_scenario_table"),
    "registered_scenario_info": (".scenarios", "registered_scenario_info"),
    "scenario_table_from_adata": (".scenarios", "scenario_table_from_adata"),
    "ScibEvaluationConfig": (".scib_backend", "ScibEvaluationConfig"),
    "ScibEvaluationResult": (".scib_backend", "ScibEvaluationResult"),
    "prepare_scib_reference": (".scib_backend", "prepare_scib_reference"),
    "run_scib_evaluation": (".scib_backend", "run_scib_evaluation"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
