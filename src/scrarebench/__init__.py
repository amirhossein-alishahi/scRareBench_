"""scRareBench: standard and rare-cell-aware benchmarking for scRNA-seq integration.

The public API is loaded lazily so runtime helpers can be imported before the
full scientific dependency stack is installed. Integration methods remain user
owned; scRareBench consumes their latent representations.
"""
from __future__ import annotations
from importlib import import_module
from typing import Any
from ._version import __version__

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_BENCHMARK_SEED": (".constants", "DEFAULT_BENCHMARK_SEED"),
    "METRIC_REGISTRY": (".metric_registry", "METRIC_REGISTRY"),
    "metric_direction": (".metric_registry", "metric_direction"),
    "metric_info": (".metric_registry", "metric_info"),
    "setup_runtime": (".runtime", "setup_runtime"),
    "BenchmarkConfig": (".benchmark", "BenchmarkConfig"),
    "BenchmarkResult": (".benchmark", "BenchmarkResult"),
    "MethodOutput": (".benchmark", "MethodOutput"),
    "MethodSpec": (".benchmark", "MethodSpec"),
    "MultiSeedBenchmarkResult": (".benchmark", "MultiSeedBenchmarkResult"),
    "install_method_dependencies": (".benchmark", "install_method_dependencies"),
    "benchmark": (".benchmark", "benchmark"),
    "benchmark_latent": (".benchmark", "benchmark_latent"),
    "benchmark_method": (".benchmark", "benchmark_method"),
    "DATASET_REGISTRY": (".datasets", "DATASET_REGISTRY"),
    "register_dataset": (".datasets", "register_dataset"),
    "dataset_info": (".datasets", "dataset_info"),
    "default_data_dir": (".datasets", "default_data_dir"),
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
    "build_results_payload": (".reporting", "build_results_payload"),
    "create_report_bundle": (".reporting", "create_report_bundle"),
    "write_results_json": (".reporting", "write_results_json"),
    "write_interactive_report": (".dashboard", "write_interactive_report"),
    "write_multiseed_interactive_report": (".dashboard", "write_multiseed_interactive_report"),
    "merge_interactive_reports": (".dashboard", "merge_interactive_reports"),
    "write_pdf_report": (".reporting", "write_pdf_report"),
    "normalize_method_seeds": (".multiseed", "normalize_method_seeds"),
    "configuration_hash": (".multiseed", "configuration_hash"),
    "method_training_hash": (".multiseed", "method_training_hash"),
    "evaluation_contract": (".multiseed", "evaluation_contract"),
    "evaluation_contract_hash": (".multiseed", "evaluation_contract_hash"),
    "dataset_contract_hash": (".multiseed", "dataset_contract_hash"),
    "canonicalize_embedded_run": (".multiseed", "canonicalize_embedded_run"),
    "aggregate_dashboard_runs": (".multiseed", "aggregate_dashboard_runs"),
    "mean_sd": (".multiseed", "mean_sd"),
    "consensus": (".multiseed", "consensus"),
    "MultiseedDelivery": (".delivery", "MultiseedDelivery"),
    "finalize_multiseed_delivery": (".delivery", "finalize_multiseed_delivery"),
    "validate_dashboard_run_payload": (".delivery", "validate_dashboard_run_payload"),
    "validate_multiseed_report": (".delivery", "validate_multiseed_report"),
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

__all__ = ["__version__", *_EXPORTS]

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
