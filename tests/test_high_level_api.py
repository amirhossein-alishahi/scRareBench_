from __future__ import annotations
import hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scrarebench import BenchmarkConfig, benchmark_latent, dataset_info, register_dataset
from scrarebench.datasets.metadata import attach_builtin_dataset_metadata

ROOT=Path(__file__).parents[1]
class Fake:
    def __init__(self,n=60,p=20):
        r=np.random.default_rng(1); self.X=r.poisson(2,(n,p)).astype(float)
        self.obs=pd.DataFrame({'cell_type':['A','B','Rare']*(n//3),'batch':['b1','b2']*(n//2)},index=[f'c{i}' for i in range(n)])
        self.layers={'counts':self.X.copy()}; self.obsm={}; self.uns={}; self.var=pd.DataFrame(index=[f'g{i}' for i in range(p)])
    @property
    def n_obs(self): return len(self.obs)
    @property
    def n_vars(self): return len(self.var)
    @property
    def obs_names(self): return self.obs.index
    def copy(self):
        import copy; return copy.deepcopy(self)

def patch_core(monkeypatch):
    import scrarebench.evaluation as ev
    def cluster(adata,*,representation_key,method_name,n_neighbors,metric,resolutions,random_state,overwrite):
        keys={}
        for r in resolutions:
            k=f'cluster_{r}'; adata.obs[k]=pd.Categorical(adata.obs['cell_type'].astype(str)); keys[float(r)]=k
        adata.uns['fake_neighbors']={}; return SimpleNamespace(cluster_keys=keys,neighbors_key='fake_neighbors')
    monkeypatch.setattr(ev,'run_standard_clustering',cluster)
    monkeypatch.setattr(ev,'run_scib_evaluation',lambda *a,**k:None)

def test_register_dataset_is_metadata_only():
    a=Fake(); x=a.X.copy(); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    assert dataset_info(a)['rare_types']==['Rare']; assert np.array_equal(a.X,x); assert not a.obsm

def test_custom_scenario_registration():
    a=Fake(); t=pd.DataFrame([{'cell_type':'Rare','scenario':'GR-DL','distribution':'GR','topology':'DL'}])
    register_dataset(a,label_key='cell_type',batch_key='batch',scenario_table=t)
    assert a.obs.loc[a.obs.cell_type.eq('Rare'),'scrarebench_scenario'].eq('GR-DL').all()

def test_dataset2_profile_constructs_batch():
    a=Fake(); a.obs['donor_id']=['d1','d2','d3']*20; a.obs['assay']=['10x','10x','smart']*20
    attach_builtin_dataset_metadata(a,dataset_key='mbdrc_renal_cortex',dataset_index=2)
    info=dataset_info(a); assert info['batch_key']=='scrarebench_batch'; assert info['scib_hvg_batch_mode']=='global'; assert 'scrarebench_batch' in a.obs

def test_config_typo_rejected():
    a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    with pytest.raises(ValueError,match='Unknown BenchmarkConfig'):
        benchmark_latent(a,np.zeros((a.n_obs,3)),method='M',config={'randm_state':1})

def test_custom_without_rare_never_inherits_paper(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    z=pd.DataFrame(np.random.default_rng(2).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='M',output_dir=tmp_path,config={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False})
    assert r.rare_metrics.empty; assert set(r.metrics['subset'])=={'overall','non_rare'}; assert r.report_path.exists()

def test_custom_rare_metrics_flow(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    z=pd.DataFrame(np.random.default_rng(3).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='M',output_dir=tmp_path,config={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False})
    assert r.rare_metrics.cell_type.tolist()==['Rare']; assert 'rare' in set(r.metrics['subset']); assert not r.summary().empty

def test_dataframe_alignment_is_verified(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    z=pd.DataFrame(np.zeros((a.n_obs,3)),index=list(reversed(a.obs_names)))
    with pytest.raises(Exception,match='order differs'):
        benchmark_latent(a,z,method='M',output_dir=tmp_path,config={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False})

def test_obsm_input_supported(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch'); a.obsm['X_user']=np.zeros((a.n_obs,3))
    r=benchmark_latent(a,'X_user',method='M',representation_key='X_copy',output_dir=tmp_path,config={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False})
    assert r.alignment['barcodes_provided'] is False

def test_default_artifacts_generated(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(36,10); register_dataset(a,label_key='cell_type',batch_key='batch',rare_types=['Rare'])
    z=pd.DataFrame(np.random.default_rng(4).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='Full',output_dir=tmp_path,config={'run_scib':False})
    assert all(p and p.exists() for p in [r.report_path,r.interactive_report_path,r.pdf_path,r.bundle_path])

def test_public_api_and_no_methods_package():
    import scrarebench
    for n in ['benchmark_latent','benchmark','BenchmarkConfig','BenchmarkResult','register_dataset','dataset_info','load_dataset','attach_latent','evaluate_latent']:
        assert hasattr(scrarebench,n)
    assert not (ROOT/'src/scrarebench/methods').exists()

def test_new_notebooks_compile_and_method_is_user_side():
    ps=sorted((ROOT/'notebooks').glob('*HighLevel*.ipynb')); assert len(ps)==2
    for p in ps:
        nb=json.loads(p.read_text()); code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c.get('cell_type')=='code')
        for i,c in enumerate(nb['cells']):
            if c.get('cell_type')=='code': compile(''.join(c.get('source',[])),f'{p}:{i}','exec')
        assert 'scvi.model.SCVI' in code and 'benchmark_latent(' in code and 'scrarebench.methods' not in code and 'run_benchmark' not in code

def test_scrarep_is_accepted_as_arbitrary_benchmark_method(tmp_path,monkeypatch):
    patch_core(monkeypatch); a=Fake(); register_dataset(a,label_key='cell_type',batch_key='batch')
    z=pd.DataFrame(np.random.default_rng(9).normal(size=(a.n_obs,4)),index=a.obs_names)
    r=benchmark_latent(a,z,method='scRareP',output_dir=tmp_path,config={'run_scib':False,'write_interactive_report':False,'write_pdf_report':False,'create_bundle':False})
    assert r.method == 'scRareP'
    assert r.report_path.exists()

def test_custom_dataset_with_paper_label_names_does_not_inherit_paper_taxonomy(tmp_path, monkeypatch):
    """Label-name overlap alone must never activate GSE194122 rare metadata."""
    patch_core(monkeypatch)
    paper = pd.read_csv(ROOT / 'src/scrarebench/config/paper_scenarios.csv')
    labels = paper['cell_type'].astype(str).tolist()
    n = len(labels) * 2
    a = Fake(n=n, p=8)
    a.obs = pd.DataFrame(
        {
            'cell_type': labels * 2,
            'batch': ['b1'] * len(labels) + ['b2'] * len(labels),
        },
        index=[f'paper_name_{i}' for i in range(n)],
    )
    a.X = np.random.default_rng(11).poisson(2, (n, 8)).astype(float)
    a.layers = {'counts': a.X.copy()}
    a.obsm = {}
    a.uns = {}
    register_dataset(a, label_key='cell_type', batch_key='batch')
    z = pd.DataFrame(np.random.default_rng(12).normal(size=(n, 4)), index=a.obs_names)
    result = benchmark_latent(
        a,
        z,
        method='scRareP',
        output_dir=tmp_path,
        config={
            'run_scib': False,
            'write_interactive_report': False,
            'write_pdf_report': False,
            'create_bundle': False,
        },
    )
    assert result.rare_metrics.empty
    assert set(result.metrics['subset']) == {'overall', 'non_rare'}

def test_register_dataset_rejects_inconsistent_scenario_metadata():
    a = Fake()
    bad = pd.DataFrame([
        {
            'cell_type': 'Rare',
            'scenario': 'GR-DL',
            'distribution': 'LE',
            'topology': 'DL',
        }
    ])
    with pytest.raises(ValueError, match='conflicts with distribution'):
        register_dataset(a, label_key='cell_type', batch_key='batch', scenario_table=bad)
