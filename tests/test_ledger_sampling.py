import mpmath as mp
import numpy as np
import pytest
from aspire.query_ledger import CountedRealOracle,QueryLedger
from aspire.status import NumericalFailure
from aspire.config import resolve
from aspire.oracles.interpolation import directional_derivative
from aspire.recovery.sampling import sample_body,hr_step

def test_budget_cache_batch_scopes_precision():
    ledger=QueryLedger(budget=5);oracle=CountedRealOracle(lambda x:sum(t*t for t in x),2,ledger)
    with ledger.scope('membership'):
        oracle.batch(np.array([[1.,2.],[2.,3.]]));oracle(np.array([1.,2.]))
    with mp.workdps(80):oracle(np.array([mp.mpf(1),mp.mpf(2)],object))
    with mp.workdps(110):oracle(np.array([mp.mpf(1),mp.mpf(2)],object))
    oracle(np.array([4.,5.]))
    with pytest.raises(NumericalFailure,match='budget_exhausted'):oracle(np.array([6.,5.]))
    assert ledger.counts['n_real_calls_total']==5 and ledger.counts['n_cache_hits']==1
    assert ledger.counts['n_real_calls_membership']==2
    with pytest.raises(TypeError):oracle(np.array([1+0j,2+0j]))

def test_directional_cost():
    ledger=QueryLedger(cache_size=0);oracle=CountedRealOracle(lambda x:x[0]**16,1,ledger)
    directional_derivative(oracle,np.array([.5]),np.ones(1),degree=16,ledger=ledger)
    assert ledger.counts['n_real_calls_total']==17

def test_sampling_ball_moments():
    cfg=resolve({})['sampler'];cfg.update(burn_in=100,thin=2,chains=4,bisection_absolute_tolerance=1e-6)
    z,ids,diag=sample_body(lambda x:np.linalg.norm(x)**4,2,4000,config=cfg,xi=0,outer_radius=1,rng=np.random.default_rng(9),ledger=QueryLedger())
    assert np.max(np.linalg.norm(z,axis=1))<=1+1e-8
    assert np.allclose(z.T@z/len(z),np.eye(2)/4,atol=.025)
    assert abs(np.mean(np.linalg.norm(z,axis=1)**2)-.5)<.04
    assert len(z)==4000 and len(set(ids))==4

def test_fuzzy_points_remain_in_true_body():
    rng=np.random.default_rng(31);z=np.zeros(2);xi=.1
    def noisy(x):return np.linalg.norm(x)**4*(1+xi*np.sin(35*x[0]+23*x[1]))
    for _ in range(200):
        z,_=hr_step(z,noisy,xi=xi,outer_radius=1,tolerance=1e-6,max_iterations=40,rng=rng)
        assert np.linalg.norm(z)<=1+1e-10

def test_information_boundary_actual_stage():
    from aspire.architecture import Architecture
    from aspire.recovery.network import recover_network
    # The ONLY capability is a real scalar callable. No teacher object is supplied.
    def value(x):
        s=np.asarray(x)**4
        return .3*(.8*s[0]+.2*s[1])**4+.7*(.1*s[0]+.9*s[1])**4
    cfg=resolve({'architecture':{'d':2,'hidden_widths':[2,2]},
                 'recovery':{'moment_samples_by_layer':[32]},'sampler':{'burn_in':16,'thin':1}})
    ledger=QueryLedger(budget=20000)
    real=CountedRealOracle(value,2,ledger)
    result=recover_network(real,architecture=Architecture(**cfg['architecture']),public_bounds={'kappa0':10,'mu':.01},config=cfg,rng=np.random.default_rng(3))
    assert result['status'] in ('complete','failed')
    assert len(result['hidden_weights'])>=1
    assert ledger.counts['n_real_calls_total']>0
    purposes=sum(v for k,v in ledger.counts.items() if k.startswith('n_real_calls_') and k!='n_real_calls_total')
    assert purposes==ledger.counts['n_real_calls_total']
