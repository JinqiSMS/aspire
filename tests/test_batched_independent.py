import numpy as np
import pytest
from aspire.query_ledger import QueryLedger, CountedRealOracle
from aspire.status import NumericalFailure
from aspire.config import resolve
from aspire.recovery.sampling import hr_step, hr_step_batch, sample_body
from aspire.oracles.interpolation import gradient, gradient_batch


def test_batched_teacher_values_agree_with_scalar_for_original_inputs():
    from aspire.teacher import Teacher
    rng=np.random.default_rng(93)
    w1=np.linalg.qr(rng.normal(size=(8,3)))[0]
    w2=np.array([[.92,.005,.075],[.005,.845,.15],[.075,.15,.775]])
    teacher=Teacher([w1,w2],np.ones(3)/3,4)
    x=rng.normal(size=(1000,8))*rng.uniform(.01,4,size=(1000,1))
    assert np.allclose(teacher.forward(x),[teacher.forward(p) for p in x],rtol=2e-13,atol=1e-14)


def test_batch_value_count_and_atomic_budget():
    ledger=QueryLedger(budget=7,cache_size=0)
    oracle=CountedRealOracle(lambda x:np.sum(x**4),2,ledger,
                             batch_value=lambda x:np.sum(x**4,axis=1))
    x=np.array([[1.,2.],[1.,2.],[3.,4.]])
    with ledger.scope('membership'):
        assert np.array_equal(oracle.batch(x),np.sum(x**4,axis=1))
    assert ledger.counts['n_real_calls_membership']==3
    oracle.batch(x)
    with pytest.raises(NumericalFailure,match='budget_exhausted'):oracle.batch(x)
    assert ledger.counts['n_real_calls_total']==6
    with pytest.raises(TypeError):oracle.batch(x.astype(complex))
    with pytest.raises(NumericalFailure,match='nonfinite_query'):oracle.batch(np.array([[np.nan,0.]]))


def test_vectorized_chords_match_scalar_with_identical_randomness():
    rng=np.random.default_rng(410)
    z=rng.normal(size=(20,3))*.1; directions=rng.normal(size=z.shape); fractions=rng.uniform(size=len(z))
    def value(x):return np.sum(x**4)+.3*np.sum(x*x)**2
    def batch(x):return np.sum(x**4,axis=1)+.3*np.sum(x*x,axis=1)**2
    options=dict(xi=.03,outer_radius=2.,tolerance=1e-7,max_iterations=40)
    actual,zeros=hr_step_batch(z,batch,directions=directions,fractions=fractions,**options)
    class FixedRandom:
        def __init__(self,d,u):self.d,self.u=d,u
        def normal(self,size):return self.d.copy()
        def uniform(self,lo,hi):return lo+self.u*(hi-lo)
    expected=np.array([hr_step(p,value,rng=FixedRandom(d,u),**options)[0]
                       for p,d,u in zip(z,directions,fractions)])
    assert np.allclose(actual,expected,rtol=0,atol=1e-14)
    assert zeros==0 and np.max(batch(actual))<=1


def test_batch_gradients_use_only_values_and_same_interpolation():
    def values(x):
        a=x[...,0]**4+.2*x[...,1]**4
        b=.3*x[...,0]**4+.8*x[...,1]**4
        return .4*a**4+.6*b**4
    points=np.random.default_rng(31).uniform(-.7,.7,size=(23,2))
    ledger=QueryLedger(cache_size=0)
    oracle=CountedRealOracle(values,2,ledger,batch_value=values)
    with ledger.scope('moment_gradient'):
        got=gradient_batch(oracle.batch,points,degree=16,step=.0625,ledger=ledger,batch_size=7)
    expected=np.array([gradient(values,p,degree=16,step=.0625) for p in points])
    assert np.allclose(got,expected,rtol=1e-11,atol=1e-12)
    assert ledger.counts['n_real_calls_moment_gradient']==23*2*17
    assert ledger.counts['n_directional_calls']==46


def test_independent_endpoints_ball_distribution_and_accounting():
    cfg=resolve({})['sampler'];cfg.update(mode='independent_endpoints',batch_size=256,endpoint_steps=32)
    ledger=QueryLedger(budget=10**8,cache_size=0)
    oracle=CountedRealOracle(lambda x:np.sum(x*x)**2,3,ledger,
                             batch_value=lambda x:np.sum(x*x,axis=1)**2)
    z,ids,diag=sample_body(oracle,3,4096,config=cfg,xi=0,outer_radius=1.,
        rng=np.random.default_rng(120),ledger=ledger,batch_value=oracle.batch)
    assert len(np.unique(ids))==4096
    assert diag['transitions']==4096*32
    assert ledger.counts['n_hr_transitions']==4096*32
    assert ledger.counts['n_kept_samples']==4096
    assert ledger.counts['n_real_calls_membership']==4096*32*2*25
    assert np.max(np.linalg.norm(z,axis=1))<=1
    assert np.allclose(z.T@z/len(z),np.eye(3)/5,atol=.012)
