from __future__ import annotations
import ast, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from scrarebench import MethodOutput, MethodSpec, benchmark_latent, benchmark_method, dataset_info, register_dataset
from scrarebench.datasets.metadata import attach_builtin_dataset_metadata

ROOT = Path(__file__).parents[1]

class Fake:
    def __init__(self,n=60,p=20):
        r=np.random.default_rng(1); self.X=r.poisson(2,(n,p)).astype(float)
        self.obs=pd.DataFrame({'cell_type':['A','B','Rare']*(n//3),'batch':['b1','b2']*(n//2)},index=[f'c{i}' for i in range(n)])
        self.layers={'counts':self.X.copy()}; self.obsm={}; self.obsp={}; self.uns={}; self.var=pd.DataFrame(index=[f'g{i}' for i in range(p)])
    @property
    def n_obs(self): return len(self.obs)
    @property
    def n_vars(self): return len(self.var)
    @property
    def obs_names(self): return self.obs.index
    @property
    def var_names(self): return self.var.index
    def copy(self):
        import copy; return copy.deepcopy(self)

def patch_core(monkeypatch):
    import scrarebench.evaluation as ev
    def cluster(adata,*,representation_key,method_name,n_neighbors,metric,resolutions,random_state,overwrite,**kwargs):
        keys={}
        for r in resolutions:
            k=f'cluster_{r}'; adata.obs[k]=pd.Categorical(adata.obs['cell_type'].astype(str)); keys[float(r)]=k
        return SimpleNamespace(cluster_keys=keys,neighbors_key='fake_neighbors',leiden_flavor=kwargs.get('leiden_flavor','igraph'),leiden_n_iterations=kwargs.get('leiden_n_iterations',2))
    monkeypatch.setattr(ev,'run_standard_clustering',cluster)
    monkeypatch.setattr(ev,'run_scib_evaluation',lambda *a,**k:None)

def compact_cfg(**extra):
    cfg={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False}
    cfg.update(extra); return cfg

def test_register_dataset_is_metadata_only():
    a=Fake(); x=a.X.copy(); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    assert dataset_info(a)['rare_types']==['Rare']; assert np.array_equal(a.X,x); assert not a.obsm

def test_dataset2_profile_constructs_batch():
    a=Fake(); a.obs['donor_id']=['d1','d2','d3']*20; a.obs['assay']=['10x','10x','smart']*20
    attach_builtin_dataset_metadata(a,dataset_key='mbdrc_renal_cortex',dataset_index=2)
    info=dataset_info(a); assert info['batch_key']=='scrarebench_batch'; assert info['scib_hvg_batch_mode']=='global'; assert 'scrarebench_batch' in a.obs

def test_config_typo_rejected():
    a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    with pytest.raises(ValueError,match='Unknown BenchmarkConfig'):
        benchmark_latent(a,np.zeros((a.n_obs,3)),method='M',config={'randm_state':1})

def test_custom_without_rare_runs_standard_benchmark(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    z=pd.DataFrame(np.random.default_rng(2).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='M',output_dir=tmp_path,config=compact_cfg())
    assert r.rare_metrics.empty; assert set(r.metrics['subset'])=={'overall','non_rare'}; assert r.report_path.exists()

def test_custom_rare_types_do_not_require_paper_taxonomy(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    z=pd.DataFrame(np.random.default_rng(3).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='M',output_dir=tmp_path,config=compact_cfg())
    assert r.rare_metrics.cell_type.tolist()==['Rare']
    assert r.rare_metrics.scenario.tolist()==['UNASSIGNED']
    assert 'knn_local_recovery_adjusted' in r.rare_metrics.columns
    assert 'ASW_selected_cells_in_full_latent' in r.metrics.columns
    assert not r.summary().empty

def test_dataframe_alignment_is_verified(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    z=pd.DataFrame(np.zeros((a.n_obs,3)),index=list(reversed(a.obs_names)))
    with pytest.raises(Exception,match='order differs'):
        benchmark_latent(a,z,method='M',output_dir=tmp_path,config=compact_cfg())

def test_method_spec_is_generic_and_multiseed(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    seen=[]
    def runner(method_adata,seed,config):
        seen.append((seed,config['alpha'],config['seed']))
        return MethodOutput(np.random.default_rng(seed).normal(size=(method_adata.n_obs,4)),barcodes=method_adata.obs_names)
    method=MethodSpec('CompletelyCustom',runner,config={'alpha':0.2},dependencies=('user-package==1.2',))
    result=benchmark_method(a,method,seeds=[11,22],output_dir=tmp_path,benchmark_config=compact_cfg(),finalize=False)
    assert result.seeds==(11,22); assert sorted(result.runs)==[11,22]
    assert seen==[(11,0.2,11),(22,0.2,22)]
    assert all(run.method=='CompletelyCustom' for run in result.runs.values())

def test_method_dependency_install_is_explicit_opt_in(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    calls=[]
    method=MethodSpec('M',lambda x,s,c: np.zeros((x.n_obs,3)),dependencies=('x==1',),installer=lambda reqs:calls.append(tuple(reqs)))
    benchmark_method(a,method,seeds=1,output_dir=tmp_path/'one',benchmark_config=compact_cfg(),finalize=False,install_dependencies=False)
    assert calls==[]
    benchmark_method(a,method,seeds=2,output_dir=tmp_path/'two',benchmark_config=compact_cfg(),finalize=False,install_dependencies=True)
    assert calls==[('x==1',)]

def test_high_level_multiseed_final_delivery(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(36,10); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    method=MethodSpec('CustomFinal',lambda x,s,c: MethodOutput(np.random.default_rng(s).normal(size=(x.n_obs,4)),barcodes=x.obs_names),config={'latent_dim':4})
    result=benchmark_method(a,method,seeds=[5,6],output_dir=tmp_path,benchmark_config={'run_scib':False,'write_pdf_report':False,'include_cell_ids':False})
    assert result.delivery is not None and result.delivery.completed_seeds==(5,6)
    assert result.report_path.exists() and result.archive_path.exists() and result.summary_path.exists()

def test_public_api_remains_method_agnostic():
    import scrarebench
    for name in ['benchmark_latent','benchmark_method','MethodSpec','MethodOutput','BenchmarkConfig','register_dataset','dataset_info','normalize_method_seeds','finalize_multiseed_delivery','METRIC_REGISTRY']:
        assert hasattr(scrarebench,name)
    assert not (ROOT/'src/scrarebench/methods').exists()

def test_high_level_and_low_level_templates_compile_and_keep_method_user_side():
    paths=[
        ROOT/'notebooks/scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb',
        ROOT/'notebooks/scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb',
        ROOT/'notebooks/scRareBench_CustomMethod_HighLevel_Colab.ipynb',
        ROOT/'notebooks/scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb',
    ]
    for path in paths:
        nb=json.loads(path.read_text(encoding='utf-8')); code_text=[]
        for i,cell in enumerate(nb['cells']):
            if cell.get('cell_type')=='code':
                source=''.join(cell.get('source',[])); ast.parse(source,filename=f'{path}:{i}'); code_text.append(source)
        joined='\n'.join(code_text)
        assert 'scrarebench.methods' not in joined
    high=(ROOT/'notebooks/scRareBench_CustomMethod_HighLevel_Colab.ipynb').read_text(encoding='utf-8')
    assert 'MethodSpec' in high and 'benchmark_method' in high and 'METHOD_DEPENDENCIES' in high
