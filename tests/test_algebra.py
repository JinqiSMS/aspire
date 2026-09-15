import mpmath as mp
import numpy as np
import pytest
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.instances import generate
from aspire.teacher import Teacher
from aspire.query_ledger import CountedRealOracle, QueryLedger
from aspire.oracles.interpolation import directional_derivative, gradient, hessian
from aspire.oracles.complex_value import complex_value_from_real
from aspire.oracles.suffix import SuffixOracle
from aspire.oracles.prefix_jacobian import prefix_value_and_jacobian
from aspire.numeric import array_mp
from aspire.recovery.moments import moment_directions
from aspire.recovery.final_layer import recover_from_hessians
from aspire.evaluation.alignment import align

def instance(widths=(3,3), d=6):
    cfg=resolve({"architecture":{"d":d,"hidden_widths":list(widths)},"sampler":{"xi_by_layer":[0.]+[1e-4]*(len(widths)-2)}})
    arch=Architecture(**cfg["architecture"])
    teacher,_=generate(arch,cfg["teacher"],np.random.default_rng(19))
    return cfg,arch,teacher

def test_architecture():
    a=Architecture(8,(4,3,2),4)
    assert a.L==4 and a.Q==64
    assert [a.suffix_degree(i) for i in (1,2,3)]==[64,16,4]
    for widths,k in [((3,4),4),((3,3),2),((3,3),5),((3,1),4)]:
        with pytest.raises(ValueError): Architecture(8,widths,k)

@pytest.mark.parametrize("degree",[4,16,64])
def test_directional_and_hessian(degree):
    w=np.array([.6,-.8]);x=np.array([.4,-.7]);v=np.array([2.,-.3])
    def f(z): return (w@z)**degree
    exact=degree*(w@x)**(degree-1)*(w@v)
    assert np.isclose(float(directional_derivative(f,x,v,degree=degree,dps=80)),exact,rtol=1e-10,atol=1e-12)
    expected=degree*(degree-1)*(w@x)**(degree-2)*np.outer(w,w)
    assert np.allclose(np.array(hessian(f,x,degree=degree,step=.25,dps=80),float),expected,rtol=1e-10,atol=1e-10)
    assert directional_derivative(f,x,np.zeros(2),degree=degree)==0

def test_complex_phase_precision_and_counts():
    with mp.workdps(80):
        w=array_mp([.6,-.8]);z=np.array([mp.mpc(.4,.2),mp.mpc(-.7,.3)],object)
        ledger=QueryLedger(cache_size=0)
        f=CountedRealOracle(lambda x:(w@x)**64,2,ledger)
        value,_=complex_value_from_real(f,z,total_degree=64,dps=80,ledger=ledger)
        assert abs(value-(w@z)**64)<mp.mpf('1e-55')
        assert ledger.counts['n_real_calls_total']==129
        complex_value_from_real(f,array_mp([.2,.3]),total_degree=64,dps=80,ledger=ledger)
        assert ledger.counts['n_real_calls_total']==130

def test_forward_and_true_prefix_suffix_rectangular():
    cfg,arch,t=instance((4,3,2),7)
    cfg['oracle']['backend']='mpmath';cfg['oracle']['dps']=80
    ledger=QueryLedger(budget=10000)
    real=CountedRealOracle(t.forward,arch.d,ledger)
    with mp.workdps(80):
        for layer in (2,3):
            y=array_mp(np.linspace(-.2,.3,arch.widths[layer-1]));y[0]=mp.mpf(0)
            suffix=SuffixOracle(real,arch,layer,t.weights[:layer-1],cfg['oracle'],ledger)
            z=suffix.invert(y,80)
            state=z
            for w in t.weights[:layer-1]: state=(array_mp(w).T@state)**arch.k
            assert max(abs(a-b) for a,b in zip(state,y))<mp.mpf('1e-60')
            actual=suffix(y);expected=t.suffix(layer,y)
            assert abs(actual-expected)<mp.mpf('1e-45')
        x=np.array([.2]*arch.d)
        assert np.isclose(t.forward(x),t.forward(x.astype(complex)))

def test_jacobian_chain_rule_and_directional_pullback():
    cfg,arch,t=instance((4,3,2),7)
    x=np.linspace(.1,.5,arch.d)
    y,jac=prefix_value_and_jacobian(x,t.weights[:2],4)
    g=t.gradient(x)
    assert np.allclose(jac.T@t.gradient(y,layer=3),g,rtol=1e-9,atol=1e-80)
    b=np.array([directional_derivative(t.forward,x,v,degree=arch.Q,step=.1,dps=80) for v in jac],float)
    assert np.allclose(np.linalg.solve(jac@jac.T,b),t.gradient(y,layer=3),rtol=1e-7,atol=1e-70)

def test_moment_pencil_and_degenerate_gap():
    from aspire.status import NumericalFailure
    rng=np.random.default_rng(4);c=np.eye(3)+.15*rng.normal(size=(3,3));ci=np.linalg.inv(c)
    sx=ci.T@np.diag([1.,2.,3.])@ci;sg=c@np.diag([2.,3.,5.])@c.T
    _,v,_=moment_directions(sx,sg)
    normalized=c/np.linalg.norm(c,axis=0)
    assert np.min(np.max(abs(normalized.T@v),axis=1))>1-1e-12
    with pytest.raises(NumericalFailure,match='moment_gap_unresolved'):
        moment_directions(np.eye(3),np.eye(3))

@pytest.mark.parametrize('n,s',[(3,3),(5,3)])
def test_two_hessians_rectangular(n,s):
    rng=np.random.default_rng(13)
    w=rng.uniform(.1,1,(n,s));w/=w.sum(axis=0);a=np.arange(1,s+1,dtype=float);a/=sum(a)
    t=Teacher([w],a,4)
    recovered,coef,_=recover_from_hessians(t.hessian(np.ones(n)+.2*rng.normal(size=n)),t.hessian(np.ones(n)),s,4)
    from scipy.optimize import linear_sum_assignment
    _,order=linear_sum_assignment(np.sum((w[:,:,None]-recovered[:,None,:])**2,axis=0))
    assert np.allclose(recovered[:,order],w,atol=1e-9)
    assert np.allclose(coef[order],a,atol=1e-9)

def test_consistent_alignment():
    _,arch,t=instance((4,3,2),7)
    rng=np.random.default_rng(8);est=[w.copy() for w in t.weights];a=t.a.copy()
    for l,w in enumerate(est):
        order=rng.permutation(w.shape[1]);est[l]=w[:,order]
        if l==0:est[l]*=rng.choice([-1,1],w.shape[1])
        if l+1<len(est):est[l+1]=est[l+1][order,:]
        else:a=a[order]
    x=rng.normal(size=(12,arch.d))
    assert np.allclose(Teacher(est,a,4).forward(x),t.forward(x))
    aligned,coef,_=align(t.weights,est,a)
    assert all(np.allclose(w,truth) for w,truth in zip(aligned,t.weights))
    assert np.allclose(coef,t.a)

