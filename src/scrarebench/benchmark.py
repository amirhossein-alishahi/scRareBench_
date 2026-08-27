from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from .datasets.metadata import dataset_info
from .evaluation import EvaluationConfig, EvaluationResult, evaluate_latent
from .latent import attach_latent, load_latent
from .reporting import create_report_bundle, write_interactive_report, write_pdf_report
from .scenarios import scenario_table_from_adata
from .scib_backend import ScibEvaluationConfig
from .utils import slugify


@dataclass(frozen=True)
class BenchmarkConfig:
    reference_resolution: float = 1.0
    resolution_sweep: tuple[float, ...] = (1.0,)
    n_neighbors: int = 15
    distance_metric: str = "euclidean"
    random_state: int = 0
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

    @property
    def output_dir(self): return self.evaluation.output_dir
    @property
    def metrics(self): return self.evaluation.subset_metrics
    @property
    def per_type_metrics(self): return self.evaluation.per_type_metrics
    @property
    def rare_metrics(self): return self.evaluation.rare_metrics
    @property
    def rare_summary(self): return self.evaluation.rare_summary
    @property
    def scenario_metrics(self): return self.evaluation.scenario_metrics
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
        rows=[]
        if self.scib is not None:
            for r in self.scib.aggregate_scores.to_dict(orient="records"):
                rows.append({"section":"scIB","metric":str(r.get("metric","")),"value":r.get("value",np.nan)})
        if "subset" in self.metrics:
            for subset in ("overall","rare","non_rare"):
                part=self.metrics[self.metrics["subset"].eq(subset)]
                if part.empty: continue
                row=part.iloc[0]
                for metric in ("ARI_true_vs_cluster","AMI_true_vs_cluster","F1_macro"):
                    if metric in row.index: rows.append({"section":subset,"metric":metric,"value":row[metric]})
        f1=self.rare_summary[self.rare_summary["metric"].eq("f1")] if not self.rare_summary.empty else pd.DataFrame()
        if not f1.empty: rows.append({"section":"rare_cell","metric":"mean_f1","value":f1.iloc[0]["mean"]})
        return pd.DataFrame(rows, columns=["section","metric","value"])


def _config(value):
    if value is None: return BenchmarkConfig()
    if isinstance(value, BenchmarkConfig): return value
    if not isinstance(value, Mapping): raise TypeError("config must be BenchmarkConfig, a mapping, or None.")
    allowed={f.name for f in fields(BenchmarkConfig)}
    unknown=sorted(set(value)-allowed)
    if unknown: raise ValueError(f"Unknown BenchmarkConfig option(s): {unknown}. Available options: {sorted(allowed)}")
    data=dict(value)
    if "resolution_sweep" in data: data["resolution_sweep"]=tuple(data["resolution_sweep"])
    return BenchmarkConfig(**data)


def _latent(adata, source):
    if isinstance(source, str) and source in adata.obsm:
        return load_latent(source, source_type="obsm", key=source, adata=adata)
    return load_latent(source, source_type="auto")


def _custom_scenario(adata):
    value=getattr(adata,"uns",{}).get("scrarebench_custom_scenario_table")
    if value is None: return None
    table=value.copy() if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    return None if table.empty else table

_UNSET=object()


def benchmark_latent(adata: Any, latent: Any, *, method: str, barcodes: Any | None=None,
                     output_dir: str|Path|None=None, representation_key: str|None=None,
                     label_key: str|None=None, batch_key: str|None=None, count_layer: Any=_UNSET,
                     rare_types: list[str]|tuple[str,...]|None=None, scenario_table: pd.DataFrame|None=None,
                     config: BenchmarkConfig|Mapping[str,Any]|None=None, allow_reorder: bool=False,
                     overwrite: bool=False) -> BenchmarkResult:
    """Benchmark a user-generated latent. scRareBench never runs the integration method."""
    if not str(method).strip(): raise ValueError("method must be a non-empty display name.")
    cfg=_config(config); meta=dataset_info(adata)
    label=label_key or meta.get("label_key"); batch=batch_key or meta.get("batch_key")
    if not label or not batch:
        raise ValueError("Dataset evaluation metadata is incomplete. Call register_dataset(...), or pass label_key and batch_key.")
    for key in (label,batch):
        if key not in adata.obs.columns: raise KeyError(f"adata.obs[{key!r}] is required.")
    if count_layer is _UNSET: counts=meta.get("count_layer","counts") or None
    else: counts=count_layer
    if counts is not None and counts not in adata.layers: counts=None

    name=str(method).strip(); rep=representation_key or f"X_{slugify(name)}"
    matrix, embedded=_latent(adata,latent)
    alignment=attach_latent(adata,matrix,key=rep,latent_barcodes=barcodes if barcodes is not None else embedded,
                            allow_reorder=allow_reorder,overwrite=overwrite)
    table=scenario_table if scenario_table is not None else _custom_scenario(adata)
    if table is None: table=scenario_table_from_adata(adata)
    selected=rare_types
    if selected is None:
        stored=meta.get("rare_types")
        if stored: selected=[str(x) for x in stored]
        elif table is not None and "cell_type" in table: selected=table["cell_type"].astype(str).tolist()
    mode=cfg.scib_hvg_batch_mode or meta.get("scib_hvg_batch_mode","evaluation_batch")
    if mode not in {"evaluation_batch","global"}: raise ValueError("scIB HVG batch mode must be 'evaluation_batch' or 'global'.")
    dataset_name=meta.get("dataset_key") or meta.get("name") or "custom_dataset"
    out=Path(output_dir).expanduser().resolve() if output_dir else (Path.cwd()/"scrarebench_results"/slugify(str(dataset_name))/slugify(name)).resolve()
    ec=EvaluationConfig(method_name=name,representation_key=rep,label_key=str(label),batch_key=str(batch),
        scenario_key=str(meta.get("scenario_key") or "scrarebench_scenario"),reference_resolution=cfg.reference_resolution,
        resolution_sweep=tuple(cfg.resolution_sweep),n_neighbors=cfg.n_neighbors,distance_metric=cfg.distance_metric,
        random_state=cfg.random_state,overwrite=overwrite,rare_types=tuple(selected) if selected is not None else None,
        scib=ScibEvaluationConfig(enabled=cfg.run_scib,count_layer=counts,n_hvg=cfg.scib_n_hvg,
            hvg_batch_mode=str(mode),reference_n_pcs=cfg.scib_reference_n_pcs,n_jobs=cfg.scib_n_jobs,
            progress_bar=cfg.scib_progress_bar,include_silhouette_batch=cfg.scib_include_silhouette_batch,
            require_backend=cfg.scib_require_success))
    ev=evaluate_latent(adata,ec,out,scenario_table=table); files=dict(ev.files)
    if cfg.write_interactive_report:
        p=out/"interactive_report.html"; write_interactive_report(adata,ev,p,representation_key=rep,label_key=str(label),
            batch_key=str(batch),scenario_key=ec.scenario_key,random_state=cfg.random_state,include_cell_ids=cfg.include_cell_ids); files["interactive_report"]=p
    if cfg.write_pdf_report:
        p=out/"summary_report.pdf"; write_pdf_report(adata,ev,p,representation_key=rep); files["pdf_report"]=p
    if cfg.create_bundle:
        p=out/"scrarebench_bundle.zip"; create_report_bundle(adata,ev,p,representation_key=rep,include_latent=cfg.include_latent_in_bundle,
            write_interactive=cfg.write_interactive_report,write_pdf=cfg.write_pdf_report,interactive_report_options={"label_key":str(label),"batch_key":str(batch),
            "scenario_key":ec.scenario_key,"random_state":cfg.random_state,"include_cell_ids":cfg.include_cell_ids}); files["bundle"]=p
    return BenchmarkResult(ev,name,rep,alignment,meta,files)

benchmark=benchmark_latent
__all__=["BenchmarkConfig","BenchmarkResult","benchmark","benchmark_latent"]
