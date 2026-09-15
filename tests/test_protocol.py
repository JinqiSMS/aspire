"""Checks whose failure changes the scientific interpretation of a run."""
from copy import deepcopy
import ast
from pathlib import Path
import mpmath as mp
import numpy as np
import pytest
from aspire.architecture import Architecture
from aspire.config import resolve,read_config
from aspire.instances import generate
from aspire.numeric import array_mp
from aspire.oracles.suffix import SuffixOracle
from aspire.oracles.interpolation import directional_derivative
from aspire.recovery.subspace import recover_subspace
from aspire.recovery.sampling import hr_step
from aspire.query_ledger import QueryLedger,CountedRealOracle
from aspire.experiments.budget import estimate
from aspire.experiments.runner import expanded
from aspire.io import inside,write_json
from aspire import ROOT

def test_nested_suffix_direction_cost():
    cfg=resolve({'architecture':{'d':5,'hidden_widths':[4,3,2]},
        'oracle':{'backend':'mpmath','precision_retry':{'enabled':False}},'sampler':{'xi_by_layer':[0.,.001]}})
    arch=Architecture(**cfg['architecture']);teacher,_=generate(arch,cfg['teacher'],np.random.default_rng(16))
    ledger=QueryLedger(budget=10000,cache_size=0);real=CountedRealOracle(teacher.forward,5,ledger)
    suffix=SuffixOracle(real,arch,2,teacher.weights[:1],cfg['oracle'],ledger)
    with mp.workdps(80):
        x=array_mp([-.4,-.5,-.6,-.7]);direction=array_mp([1.,0.,0.,0.])
        result=directional_derivative(suffix,x,direction,degree=arch.suffix_degree(2),step=.02,dps=80,ledger=ledger)
        expected=teacher.gradient(np.array(x,float),layer=2)[0]
        assert np.isclose(float(result),expected,rtol=1e-8)
    assert ledger.counts['n_real_calls_total']==17*129

def test_real_directional_rectangular_middle_span():
    cfg=resolve({'architecture':{'d':6,'hidden_widths':[4,3,2]},'oracle':{'backend':'mpmath'},
        'recovery':{'intermediate_span_points':6},'sampler':{'xi_by_layer':[0.,.001]}})
    arch=Architecture(**cfg['architecture']);teacher,_=generate(arch,cfg['teacher'],np.random.default_rng(11))
    ledger=QueryLedger(budget=10000,cache_size=0)
    basis,diag=recover_subspace(CountedRealOracle(teacher.forward,6,ledger),arch,2,teacher.weights[:1],cfg,np.random.default_rng(23),ledger)
    truth=np.linalg.qr(teacher.weights[1])[0]
    assert np.linalg.norm(basis@basis.T-truth@truth.T)<1e-7
    assert diag['span_method']=='original_directional_jacobian'
    assert ledger.counts['n_real_calls_total']==6*4*65

def test_latent_fixed_across_environment_dimensions():
    cfg=resolve({})
    t1,_=generate(Architecture(8,(3,3),4),cfg['teacher'],np.random.default_rng(5))
    t2,_=generate(Architecture(64,(3,3),4),cfg['teacher'],np.random.default_rng(5))
    assert np.array_equal(t1.weights[1],t2.weights[1]) and np.array_equal(t1.a,t2.a)

def test_fuzzy_no_interior_rejection_or_reset():
    class Deterministic:
        def normal(self,size):return np.array([0.,1.])
        def uniform(self,lo,hi):return (lo+hi)/2
    calls=[]
    def value(x):calls.append(x.copy());return 1.1*np.linalg.norm(x)**4
    z=np.array([.995,0.])
    next_,_=hr_step(z,value,xi=.1,outer_radius=1,tolerance=.01,max_iterations=20,rng=Deterministic())
    assert np.array_equal(z,next_)
    assert len(calls)==18  # nine midpoint queries per ray, no extra query at new point

def test_recovery_has_no_truth_imports_or_parameters():
    import inspect
    from aspire.recovery.network import recover_network
    assert not any('true' in name or 'teacher' in name for name in inspect.signature(recover_network).parameters)
    for folder in (ROOT/'src/aspire/recovery',ROOT/'src/aspire/oracles'):
        for path in folder.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    assert not any(s in (node.module or '') for s in ('teacher','diagnostics','evaluation'))

def test_paths_and_json():
    with pytest.raises(ValueError):inside('../outside.json')
    path=ROOT/'tmp/test_finite.json'
    write_json(path,{'nan':np.nan,'inf':np.inf,'a':np.array([1.])})
    import json
    assert json.loads(path.read_text())=={'nan':None,'inf':None,'a':[1.]}

def test_all_named_recipes_resolve_and_dry_run_without_queries():
    # These protocols have dedicated drivers, not the general run_one schema.
    for path in (ROOT/'configs').glob('*.yaml'):
        if path.name == 'reproduction.yaml':
            assert (ROOT/'reproduce.py').is_file()
            continue
        for cfg in expanded(read_config(path)):
            result=estimate(cfg)
            assert result['Q']>=16
    assert estimate(resolve({}))['generic_learning_query_estimate']==118588

def test_clustered_statistics_equal_teacher_weights():
    from aspire.experiments.aggregate import cluster_summary
    runs=[{'kind':'recovery','status':'complete','experiment_name':'test','run_id':str(i),'seed':i,'teacher_seed':0 if i<3 else 1,
           'parameter_success':i<3} for i in range(4)]
    summary=cluster_summary(runs,replicates=100)[0]
    assert summary['independent_teacher_count']==2
    assert summary['parameter_success']['teacher_weighted_mean']==.5

def test_stage_checkpoint_restores_rng_and_charges_repeated_queries():
    from aspire.recovery.network import recover_network
    cfg=resolve({'architecture':{'d':2,'hidden_widths':[2,2]},'recovery':{'moment_samples_by_layer':[16]},
                 'sampler':{'burn_in':4,'thin':1},'oracle':{'max_real_learning_queries':20000}})
    arch=Architecture(**cfg['architecture']);teacher,_=generate(arch,cfg['teacher'],np.random.default_rng(12))
    ledger=QueryLedger(budget=20000,cache_size=0);saved={}
    def checkpoint(weights,stages,state,counts,artifacts):
        saved.update(weights=[w.copy() for w in weights],stages=deepcopy(stages),state=deepcopy(state),counts=counts)
        ledger.budget=counts['n_real_calls_total']  # deterministic exhaustion at next stage
    first=recover_network(CountedRealOracle(teacher.forward,2,ledger),architecture=arch,public_bounds={'kappa0':10,'mu':.01},config=cfg,rng=np.random.default_rng(4),checkpoint=checkpoint)
    assert first['failure_reason']=='budget_exhausted' and len(first['hidden_weights'])==1
    restored_ledger=QueryLedger(budget=20000,cache_size=0,restored=ledger.snapshot())
    rng=np.random.default_rng();rng.bit_generator.state=saved['state']
    second=recover_network(CountedRealOracle(teacher.forward,2,restored_ledger),architecture=arch,public_bounds={'kappa0':10,'mu':.01},config=cfg,rng=rng,restored=saved)
    assert restored_ledger.counts['n_real_calls_total']>ledger.counts['n_real_calls_total']
    assert second['failure_reason']!='budget_exhausted'
    assert np.array_equal(first['hidden_weights'][0],second['hidden_weights'][0])

def test_zero_prefix_perturbation_does_not_introduce_column_sign_error():
    from aspire.diagnostics.experiments import prefix_error
    cfg=resolve({'kind':'prefix_error','oracle':{'mode':'exact_prefix_debug'},
                 'diagnostic':{'layer':2,'points':2,'prefix_perturbation':0.}})
    arch=Architecture(**cfg['architecture']);teacher,_=generate(arch,cfg['teacher'],np.random.default_rng(13))
    ledger=QueryLedger(budget=10000)
    metrics,_=prefix_error(CountedRealOracle(teacher.forward,arch.d,ledger),teacher,arch,cfg,np.random.default_rng(9),ledger)
    assert metrics['actual_prefix_operator_error']<1e-14
