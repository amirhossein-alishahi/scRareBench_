from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import DEFAULT_BENCHMARK_SEED
from .datasets.metadata import dataset_info
from .delivery import MultiseedDelivery, finalize_multiseed_delivery
from .evaluation import EvaluationConfig, EvaluationResult, evaluate_latent
from .latent import attach_latent, load_latent
from .multiseed import normalize_method_seeds
from .reporting import create_report_bundle, write_interactive_report, write_pdf_report
from .scenarios import scenario_table_from_adata
from .scib_backend import ScibEvaluationConfig
from .utils import slugify


@dataclass(frozen=True)
class BenchmarkConfig:
    """User-facing benchmark controls.

    Method training randomness is intentionally not stored here. ``random_state``
    is the fixed benchmark/evaluation seed; method seeds belong to ``MethodSpec``
    / ``benchmark_method`` or the ``method_seed`` argument of ``benchmark_latent``.
    """

    reference_resolution: float = 1.0
    resolution_sweep: tuple[float, ...] = (1.0,)
    n_neighbors: int = 15
    distance_metric: str = "euclidean"
    random_state: int = DEFAULT_BENCHMARK_SEED
    leiden_flavor: str = "igraph"
    leiden_n_iterations: int = 2
    rare_evaluation: bool | None = None
    scenario_policy: str = "require"
    strict_scenario_labels: bool = True
    run_scib: bool = True
    scib_n_hvg: int = 4000
    scib_reference_n_pcs: int = 50
    scib_n_jobs: int = 1
    scib_progress_bar: bool = True
    scib_include_silhouette_batch: bool = True
    scib_require_success: bool = True
    scib_hvg_batch_mode: str | None = None
    write_interactive_report: bool = True
    write_pdf_report: bool = True
    create_bundle: bool = True
    include_latent_in_bundle: bool = False
    include_cell_ids: bool = True


@dataclass
class BenchmarkResult:
    evaluation: EvaluationResult
    method: str
    representation_key: str
    alignment: dict[str, Any]
    dataset: dict[str, Any]
    files: dict[str, Path]
    method_seed: int | None = None
    method_config: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self): return self.evaluation.output_dir
    @property
    def metrics(self): return self.evaluation.subset_metrics
    @property
    def metric_ratios(self): return self.evaluation.subset_metric_ratios
    @property
    def per_type_metrics(self): return self.evaluation.per_type_metrics
    @property
    def rare_metrics(self): return self.evaluation.rare_metrics
    @property
    def rare_summary(self): return self.evaluation.rare_summary
    @property
    def scenario_metrics(self): return self.evaluation.scenario_metrics
    @property
    def resolution_rare_metrics(self): return self.evaluation.resolution_rare_metrics
    @property
    def scib(self): return self.evaluation.scib
    @property
    def scib_metrics(self): return self.scib.metrics_long if self.scib is not None else pd.DataFrame()
    @property
    def scib_aggregates(self): return self.scib.aggregate_scores if self.scib is not None else pd.DataFrame()
    @property
    def report_path(self): return self.files.get("report")
    @property
    def interactive_report_path(self): return self.files.get("interactive_report")
    @property
    def pdf_path(self): return self.files.get("pdf_report")
    @property
    def bundle_path(self): return self.files.get("bundle")

    def summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if self.scib is not None:
            for row in self.scib.aggregate_scores.to_dict(orient="records"):
                rows.append({"section": "scIB", "metric": str(row.get("metric", "")), "value": row.get("value", np.nan)})
        if "subset" in self.metrics:
            for subset in ("overall", "rare", "non_rare"):
                part = self.metrics[self.metrics["subset"].eq(subset)]
                if part.empty:
                    continue
                row = part.iloc[0]
                for metric in ("ARI_true_vs_cluster", "AMI_true_vs_cluster", "F1_macro", "ASW_selected_cells_in_full_latent"):
                    if metric in row.index:
                        rows.append({"section": subset, "metric": metric, "value": row[metric]})
        for metric in ("best_cluster_f1", "knn_local_recovery_adjusted", "f1"):
            part = self.rare_summary[self.rare_summary["metric"].eq(metric)] if not self.rare_summary.empty else pd.DataFrame()
            if not part.empty:
                rows.append({"section": "rare_cell", "metric": f"mean_{metric}", "value": part.iloc[0]["mean"]})
        return pd.DataFrame(rows, columns=["section", "metric", "value"])


@dataclass(frozen=True)
class MethodOutput:
    """Output contract for a user-owned integration/batch-correction method."""

    latent: Any
    barcodes: Any | None = None
    representation_key: str | None = None
    provenance_files: Mapping[str, str | Path] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    """A thin adapter around any user-owned integration method.

    ``runner`` must accept ``(adata, seed, config)`` and return either a latent
    matrix/DataFrame/obsm key or :class:`MethodOutput`. scRareBench does not
    implement the integration method itself.
    """

    name: str
    runner: Callable[[Any, int, Mapping[str, Any]], Any]
    config: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    installer: Callable[[Sequence[str]], Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("MethodSpec.name must be non-empty.")
        if not callable(self.runner):
            raise TypeError("MethodSpec.runner must be callable.")
        if self.installer is not None and not callable(self.installer):
            raise TypeError("MethodSpec.installer must be callable or None.")


@dataclass
class MultiSeedBenchmarkResult:
    method: str
    seeds: tuple[int, ...]
    runs: dict[int, BenchmarkResult]
    delivery: MultiseedDelivery | None = None

    @property
    def report_path(self) -> Path | None:
        return self.delivery.report_path if self.delivery else next(iter(self.runs.values())).interactive_report_path

    @property
    def archive_path(self) -> Path | None:
        return self.delivery.archive_path if self.delivery else next(iter(self.runs.values())).bundle_path

    @property
    def summary_path(self) -> Path | None:
        return self.delivery.summary_path if self.delivery else None

    def summary(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for seed, result in self.runs.items():
            frame = result.summary().copy()
            frame.insert(0, "method_seed", seed)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["method_seed", "section", "metric", "value"])


def install_method_dependencies(
    requirements: Sequence[str],
    *,
    installer: Callable[[Sequence[str]], Any] | None = None,
    quiet: bool = False,
) -> None:
    """Install method-owned dependencies only when the caller explicitly asks.

    This helper is deliberately opt-in. A project may instead preinstall its own
    environment or provide a custom ``installer`` (conda/uv/container policy).
    """
    reqs = tuple(str(x).strip() for x in requirements if str(x).strip())
    if not reqs:
        return
    if installer is not None:
        installer(reqs)
        return
    command = [sys.executable, "-m", "pip", "install"]
    if quiet:
        command.append("-q")
    command.extend(reqs)
    subprocess.check_call(command)


def _config(value: BenchmarkConfig | Mapping[str, Any] | None) -> BenchmarkConfig:
    if value is None:
        return BenchmarkConfig()
    if isinstance(value, BenchmarkConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("config must be BenchmarkConfig, a mapping, or None.")
    allowed = {f.name for f in fields(BenchmarkConfig)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown BenchmarkConfig option(s): {unknown}. Available options: {sorted(allowed)}")
    data = dict(value)
    if "resolution_sweep" in data:
        data["resolution_sweep"] = tuple(data["resolution_sweep"])
    return BenchmarkConfig(**data)


def _latent(adata: Any, source: Any) -> tuple[np.ndarray, Any | None]:
    if isinstance(source, str) and source in adata.obsm:
        return load_latent(source, source_type="obsm", key=source, adata=adata)
    return load_latent(source, source_type="auto")


def _custom_scenario(adata: Any) -> pd.DataFrame | None:
    value = getattr(adata, "uns", {}).get("scrarebench_custom_scenario_table")
    if value is None:
        return None
    table = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return None if table.empty else table


_UNSET = object()


def benchmark_latent(
    adata: Any,
    latent: Any,
    *,
    method: str,
    barcodes: Any | None = None,
    output_dir: str | Path | None = None,
    representation_key: str | None = None,
    label_key: str | None = None,
    batch_key: str | None = None,
    count_layer: Any = _UNSET,
    rare_types: list[str] | tuple[str, ...] | None = None,
    scenario_table: pd.DataFrame | None = None,
    config: BenchmarkConfig | Mapping[str, Any] | None = None,
    method_seed: int | None = None,
    method_config: Mapping[str, Any] | None = None,
    expected_seeds: Sequence[int] | None = None,
    extra_provenance_files: Mapping[str, str | Path] | None = None,
    allow_reorder: bool = False,
    overwrite: bool = False,
) -> BenchmarkResult:
    """Benchmark any user-generated latent; scRareBench never trains the method."""
    if not str(method).strip():
        raise ValueError("method must be a non-empty display name.")
    cfg = _config(config)
    meta = dataset_info(adata)
    label = label_key or meta.get("label_key")
    batch = batch_key or meta.get("batch_key")
    if not label or not batch:
        raise ValueError("Dataset evaluation metadata is incomplete. Call register_dataset(...), or pass label_key and batch_key.")
    for key in (label, batch):
        if key not in adata.obs.columns:
            raise KeyError(f"adata.obs[{key!r}] is required.")
    if count_layer is _UNSET:
        stored_count_layer = meta.get("count_layer", "counts")
        counts = stored_count_layer or None
    else:
        counts = count_layer
    if counts is not None and counts not in adata.layers:
        if cfg.run_scib:
            source = "dataset metadata" if count_layer is _UNSET else "benchmark_latent(count_layer=...)"
            raise KeyError(
                f"count_layer={counts!r} from {source} is required for scIB evaluation but is not present "
                "in adata.layers. Provide a validated raw-count layer, pass count_layer=None to use adata.X "
                "explicitly, or disable scIB with config={'run_scib': False}."
            )
        counts = None

    name = str(method).strip()
    rep = representation_key or f"X_{slugify(name)}"
    matrix, embedded_barcodes = _latent(adata, latent)
    alignment = attach_latent(
        adata,
        matrix,
        key=rep,
        latent_barcodes=barcodes if barcodes is not None else embedded_barcodes,
        allow_reorder=allow_reorder,
        overwrite=overwrite,
    )

    table = scenario_table if scenario_table is not None else _custom_scenario(adata)
    if table is None:
        table = scenario_table_from_adata(adata)
    selected = rare_types
    if selected is None:
        stored = meta.get("rare_types")
        if stored:
            selected = [str(x) for x in stored]
        elif table is not None and "cell_type" in table:
            selected = table["cell_type"].astype(str).tolist()
    if selected is not None:
        selected = list(dict.fromkeys(map(str, selected)))

    if cfg.rare_evaluation is None:
        rare_enabled = bool(selected) or table is not None
    else:
        rare_enabled = bool(cfg.rare_evaluation)

    mode = cfg.scib_hvg_batch_mode or meta.get("scib_hvg_batch_mode", "evaluation_batch")
    if mode not in {"evaluation_batch", "global"}:
        raise ValueError("scIB HVG batch mode must be 'evaluation_batch' or 'global'.")
    dataset_name = meta.get("dataset_key") or meta.get("name") or "custom_dataset"
    out = Path(output_dir).expanduser().resolve() if output_dir else (Path.cwd() / "scrarebench_results" / slugify(str(dataset_name)) / slugify(name)).resolve()
    method_cfg = dict(method_config or {})
    if method_seed is not None:
        method_cfg.setdefault("seed", int(method_seed))

    ec = EvaluationConfig(
        method_name=name,
        representation_key=rep,
        label_key=str(label),
        batch_key=str(batch),
        scenario_key=str(meta.get("scenario_key") or "scrarebench_scenario"),
        reference_resolution=cfg.reference_resolution,
        resolution_sweep=tuple(cfg.resolution_sweep),
        n_neighbors=cfg.n_neighbors,
        distance_metric=cfg.distance_metric,
        random_state=cfg.random_state,
        method_seed=method_seed,
        method_config=method_cfg,
        rare_types=tuple(selected) if selected is not None else None,
        leiden_flavor=cfg.leiden_flavor,
        leiden_n_iterations=cfg.leiden_n_iterations,
        overwrite=overwrite,
        rare_evaluation=rare_enabled,
        scenario_policy=cfg.scenario_policy,
        strict_scenario_labels=cfg.strict_scenario_labels,
        scib=ScibEvaluationConfig(
            enabled=cfg.run_scib,
            count_layer=counts,
            n_hvg=cfg.scib_n_hvg,
            hvg_batch_mode=str(mode),
            reference_n_pcs=cfg.scib_reference_n_pcs,
            n_jobs=cfg.scib_n_jobs,
            progress_bar=cfg.scib_progress_bar,
            include_silhouette_batch=cfg.scib_include_silhouette_batch,
            require_backend=cfg.scib_require_success,
        ),
    )
    ev = evaluate_latent(adata, ec, out, scenario_table=table)
    files = dict(ev.files)
    expected = list(expected_seeds) if expected_seeds is not None else ([int(method_seed)] if method_seed is not None else None)

    if cfg.write_interactive_report:
        p = out / "interactive_report.html"
        write_interactive_report(
            adata,
            ev,
            p,
            representation_key=rep,
            label_key=str(label),
            batch_key=str(batch),
            scenario_key=ec.scenario_key,
            random_state=cfg.random_state,
            include_cell_ids=cfg.include_cell_ids,
            method_seed=method_seed,
            method_config=method_cfg,
            expected_seeds=expected,
        )
        files["interactive_report"] = p
    if cfg.write_pdf_report:
        p = out / "summary_report.pdf"
        write_pdf_report(adata, ev, p, representation_key=rep)
        files["pdf_report"] = p
    if cfg.create_bundle:
        p = out / "scrarebench_bundle.zip"
        create_report_bundle(
            adata,
            ev,
            p,
            representation_key=rep,
            include_latent=cfg.include_latent_in_bundle,
            write_interactive=cfg.write_interactive_report,
            write_pdf=cfg.write_pdf_report,
            existing_interactive_report=files.get("interactive_report"),
            existing_pdf_report=files.get("pdf_report"),
            interactive_report_options={
                "label_key": str(label),
                "batch_key": str(batch),
                "scenario_key": ec.scenario_key,
                "random_state": cfg.random_state,
                "include_cell_ids": cfg.include_cell_ids,
            },
            method_seed=method_seed,
            method_config=method_cfg,
            expected_seeds=expected,
            extra_provenance_files=dict(extra_provenance_files or {}),
        )
        files["bundle"] = p
    return BenchmarkResult(ev, name, rep, alignment, meta, files, method_seed=method_seed, method_config=method_cfg)


def benchmark_method(
    adata: Any,
    method: MethodSpec,
    *,
    seeds: int | Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    benchmark_config: BenchmarkConfig | Mapping[str, Any] | None = None,
    install_dependencies: bool = False,
    dependency_quiet: bool = False,
    finalize: bool = True,
    copy_input_per_seed: bool = True,
    label_key: str | None = None,
    batch_key: str | None = None,
    rare_types: list[str] | tuple[str, ...] | None = None,
    scenario_table: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> MultiSeedBenchmarkResult:
    """Run a user-supplied method adapter for one or more seeds, then benchmark it.

    The method implementation and its dependencies remain user-owned. The package
    only orchestrates seed isolation, latent alignment, evaluation, reporting and
    optional multi-seed finalization.
    """
    method_seeds = normalize_method_seeds(seeds, default=42)
    if install_dependencies:
        install_method_dependencies(method.dependencies, installer=method.installer, quiet=dependency_quiet)
    cfg = _config(benchmark_config)
    root = Path(output_dir).expanduser().resolve() if output_dir else (Path.cwd() / "scrarebench_results" / slugify(method.name)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs: dict[int, BenchmarkResult] = {}
    reports: list[Path] = []
    bundles: dict[int, Path] = {}

    if finalize and len(method_seeds) > 1 and (not cfg.write_interactive_report or not cfg.create_bundle):
        raise ValueError("Multi-seed finalization requires write_interactive_report=True and create_bundle=True.")

    for seed in method_seeds:
        method_adata = adata.copy() if copy_input_per_seed else adata
        run_config = dict(method.config)
        run_config["seed"] = int(seed)
        raw_output = method.runner(method_adata, int(seed), run_config)
        if isinstance(raw_output, MethodOutput):
            output = raw_output
        else:
            output = MethodOutput(latent=raw_output)

        latent_source = output.latent
        output_barcodes = output.barcodes
        if isinstance(latent_source, str) and latent_source in getattr(method_adata, "obsm", {}):
            latent_source, embedded = _latent(method_adata, latent_source)
            if output_barcodes is None:
                output_barcodes = embedded if embedded is not None else getattr(method_adata, "obs_names", None)
        elif output_barcodes is None:
            try:
                if len(latent_source) == int(getattr(method_adata, "n_obs", len(method_adata.obs))):
                    output_barcodes = getattr(method_adata, "obs_names", None)
            except TypeError:
                pass

        eval_adata = adata.copy() if copy_input_per_seed else adata
        seed_dir = root / f"seed_{seed}"
        result = benchmark_latent(
            eval_adata,
            latent_source,
            method=method.name,
            barcodes=output_barcodes,
            output_dir=seed_dir,
            representation_key=output.representation_key,
            label_key=label_key,
            batch_key=batch_key,
            rare_types=rare_types,
            scenario_table=scenario_table,
            config=cfg,
            method_seed=int(seed),
            method_config=run_config,
            expected_seeds=method_seeds,
            extra_provenance_files=output.provenance_files,
            overwrite=overwrite,
        )
        runs[int(seed)] = result
        if result.interactive_report_path is not None:
            reports.append(result.interactive_report_path)
        if result.bundle_path is not None:
            bundles[int(seed)] = result.bundle_path

    delivery = None
    if finalize and len(method_seeds) > 1:
        meta = dataset_info(adata)
        dataset_key = str(meta.get("dataset_key") or meta.get("name") or "dataset")
        delivery = finalize_multiseed_delivery(
            reports,
            root / "multi_seed",
            method_name=method.name,
            dataset_key=dataset_key,
            expected_seeds=method_seeds,
            title=f"scRareBench multi-seed report — {method.name}",
            bundles_by_seed=bundles,
            include_latents=False,
            require_all_expected=True,
        )
    return MultiSeedBenchmarkResult(method.name, tuple(method_seeds), runs, delivery)


benchmark = benchmark_latent

__all__ = [
    "BenchmarkConfig", "BenchmarkResult", "MethodOutput", "MethodSpec", "MultiSeedBenchmarkResult",
    "install_method_dependencies", "benchmark", "benchmark_latent", "benchmark_method",
]
