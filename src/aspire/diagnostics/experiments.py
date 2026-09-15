"""Independent mechanisms E0/E1/E4 and population-gap reference integration."""
import numpy as np
import mpmath as mp
from scipy.special import logsumexp
from ..teacher import Teacher
from ..oracles.suffix import SuffixOracle
from ..oracles.complex_value import complex_value_from_real
from ..oracles.interpolation import gradient
from ..numeric import matmul,array_mp
from ..recovery.network import _recover
from ..recovery.sampling import sample_body,radius_bounds
from ..recovery.moments import raw_moments,moment_directions,normalize_directions,moment_ess,relative_gap
from ..evaluation.alignment import align
from ..status import NumericalFailure

def complex_debug_recovery(real,teacher,arch,bounds,cfg,rng,checkpoint=None,restored=None):
    def builder(layer,prefix):
        return SuffixOracle(real,arch,layer,prefix,cfg['oracle'],real.ledger,direct_complex=teacher.forward)
    return _recover(real,arch,bounds,cfg,rng,builder,checkpoint,restored)

def population_moments(teacher,layer,count,batches,rng):
    """Spec §10.5 angular integration. Not a formal population-gap certificate."""
    latent=teacher.latent(layer);n=latent.weights[0].shape[0]
    q=teacher.k**(len(teacher.weights)-layer+1);results=[]
    for _ in range(batches):
        omega=rng.normal(size=(count,n));omega/=np.linalg.norm(omega,axis=1)[:,None]
        values=latent.forward(omega)
        grads=np.array([latent.gradient(w) for w in omega])
        if np.any(values<=0) or not np.isfinite(grads).all():raise NumericalFailure('population_integration_unstable')
        logr=-np.log(values)/q;logden=logsumexp(n*logr)
        with np.errstate(divide='ignore'):
            lam=np.exp(np.log(n/(n+2))+logsumexp(2*np.log(abs(omega))+(n+2)*logr[:,None],axis=0)-logden)
            dd=np.exp(np.log(n/(n+2*q-2))+logsumexp(2*np.log(abs(grads))+(n+2*q-2)*logr[:,None],axis=0)-logden)
        theta=lam*dd
        results.append({'lambda':lam,'D':dd,'theta':theta,'relative_gap':relative_gap(theta)})
    return {'estimated_population_gap':float(np.median([r['relative_gap'] for r in results])),
            'gap_reference_batches':results,'gap_reference_directions_per_batch':count,'gap_certificate':'not_certified'}

def single_layer(teacher,arch,bounds,cfg,rng,ledger):
    layer=cfg['diagnostic']['layer']
    if not 1<=layer<=arch.L-2:raise ValueError('Moment diagnostic is for non-final layers')
    basis=np.linalg.qr(teacher.weights[layer-1])[0]
    def value(z):ledger.tick('n_debug_suffix_calls');return teacher.suffix(layer,basis@z)
    inner,outer=radius_bounds(arch,layer,bounds)
    count=cfg['recovery']['moment_samples_by_layer'][layer-1]
    z,ids,sd=sample_body(value,basis.shape[1],count,config=cfg['sampler'],xi=0,outer_radius=outer,rng=rng,ledger=ledger)
    g=np.array([basis.T@teacher.gradient(basis@p,layer=layer) for p in z]);ledger.tick('n_debug_gradient_calls',len(z))
    sx,sg=raw_moments(z,g)
    theta,v,md=moment_directions(sx,sg,cfg['recovery']['moment_ridge'],cfg['recovery']['gap_floor'])
    w=normalize_directions(basis@v,layer)
    truth=teacher.weights[layer-1]
    from scipy.optimize import linear_sum_assignment
    distances=np.sum((truth[:,:,None]-w[:,None,:])**2,axis=0)
    if layer==1:distances=np.minimum(distances,np.sum((truth[:,:,None]+w[:,None,:])**2,axis=0))
    _,order=linear_sum_assignment(distances);aligned=w[:,order]
    if layer==1:aligned*=np.where(np.sum(truth*aligned,axis=0)<0,-1,1)
    metrics={'layer':layer,'single_layer_operator_error':float(np.linalg.norm(aligned-truth,2)),**sd,**md,**moment_ess(z,g,ids,cfg['sampler']['mode'])}
    refcount=cfg['diagnostic']['reference_directions']
    if refcount:
        pop=population_moments(teacher,layer,refcount,2,rng);metrics.update(pop)
        lam=np.mean([b['lambda'] for b in pop['gap_reference_batches']],axis=0)
        dd=np.mean([b['D'] for b in pop['gap_reference_batches']],axis=0)
        c=basis.T@truth;ci=np.linalg.inv(c)
        true_sx=ci.T@np.diag(lam)@ci;true_sg=c@np.diag(dd)@c.T
        metrics.update(position_moment_relative_error=float(np.linalg.norm(sx-true_sx)/np.linalg.norm(true_sx)),
                       gradient_moment_relative_error=float(np.linalg.norm(sg-true_sg)/np.linalg.norm(true_sg)))
    radial=np.array([value(p) for p in z])**(basis.shape[1]/arch.suffix_degree(layer))
    return metrics,{'samples':z,'gradients':g,'Sx':sx,'Sg':sg,'eigenvalues':theta,'radial_uniform':radial,'true_weights':truth,'recovered_weights':w}

def prefix_error(real,teacher,arch,cfg,rng,ledger):
    layer=cfg['diagnostic']['layer']
    if not 2<=layer<arch.L:raise ValueError('Prefix diagnostic needs layer>=2')
    amplitude=cfg['diagnostic']['prefix_perturbation']
    prefix=[]
    for i,w in enumerate(teacher.weights[:layer-1]):
        perturbed=w+amplitude*rng.normal(size=w.shape)
        # Preserve the reference column signs. Canonical eigendirection signs
        # would create a spurious size-two perturbation even at amplitude zero.
        if i==0:perturbed=perturbed/np.linalg.norm(perturbed,axis=0)
        else:perturbed=normalize_directions(perturbed,i+1)
        prefix.append(perturbed)
    actual=max(np.linalg.norm(w-t,2) for w,t in zip(prefix,teacher.weights))
    estimated=SuffixOracle(real,arch,layer,prefix,cfg['oracle'],ledger)
    # This explicitly debug comparator isolates prefix error from Fourier error.
    direct=SuffixOracle(real,arch,layer,prefix,cfg['oracle'],ledger,direct_complex=teacher.forward)
    points=rng.uniform(-.5,.5,(cfg['diagnostic']['points'],arch.widths[layer-1]))
    rows=[]
    for point in points:
        true=teacher.suffix(layer,point);value=estimated(point);debug=direct(point)
        g=gradient(estimated,point,degree=arch.suffix_degree(layer),step=.05,ledger=ledger)
        gt=teacher.gradient(point,layer=layer)
        rows.append([float(abs(value-true)),float(abs(value-true)/abs(true)) if abs(true)>1e-20 else np.nan,
                     float(abs(value-debug)),float(np.linalg.norm(g-gt))])
    rows=np.asarray(rows)
    return {'actual_prefix_operator_error':float(actual),'suffix_absolute_error_median':float(np.median(rows[:,0])),
        'suffix_relative_error_median':float(np.nanmedian(rows[:,1])) if np.any(np.isfinite(rows[:,1])) else None,
        'fourier_absolute_error_median':float(np.median(rows[:,2])),'gradient_error_median':float(np.median(rows[:,3])),
        'suffix_diagnostics':estimated.diagnostics,'debug_comparator_diagnostics':direct.diagnostics}, {'points':points,'errors':rows}

def oracle_accuracy(cfg,rng):
    rows=[]
    from ..query_ledger import QueryLedger,CountedRealOracle
    for q in cfg['diagnostic']['degrees']:
        for dps in cfg['diagnostic']['precisions']:
            with mp.workdps(max(110,dps+30)):
                z=np.array([mp.mpc(.4,.2),mp.mpc(-.7,.3)],object);w=array_mp([.6,-.8])
                expected=(w@z)**q
            ledger=QueryLedger(cache_size=0)
            def polynomial(x):return (.6*x[0]-.8*x[1])**q
            oracle=CountedRealOracle(polynomial,2,ledger)
            actual,chi=complex_value_from_real(oracle,z if dps else np.array([complex(t) for t in z]),total_degree=q,dps=dps,ledger=ledger)
            with mp.workdps(max(110,dps+30)):
                absolute=float(abs(actual-expected));relative=float(abs(actual-expected)/abs(expected))
            rows.append({'degree':q,'dps':dps,'absolute_error':absolute,'relative_error':relative,'cancellation':float(chi),'real_queries':ledger.counts['n_real_calls_total']})
    return {'oracle_accuracy_rows':rows},{}

def sampler_check(cfg,rng,ledger):
    from scipy.stats import kstest
    n=cfg['architecture']['hidden_widths'][0];q=cfg['architecture']['k']
    xi=cfg['sampler']['xi_by_layer'][0]
    def value(x):return np.linalg.norm(x)**q*(1+xi*np.sin(np.dot(np.arange(1,n+1)*35,x)))
    z,ids,diag=sample_body(value,n,cfg['recovery']['moment_samples_by_layer'][0],
        config=cfg['sampler'],xi=xi,outer_radius=1,rng=rng,ledger=ledger)
    covariance=z.T@z/len(z);radial=np.linalg.norm(z,axis=1)**n
    stat=kstest(radial,'uniform')
    return {**diag,'noise_model':'bounded_oscillatory_multiplicative','true_body_max_norm':float(np.max(np.linalg.norm(z,axis=1))),
            'ball_moment_error':float(np.linalg.norm(covariance-np.eye(n)/(n+2))),
            'radial_ks_statistic':float(stat.statistic),'ks_pvalue_interpretation':'correlated_chain_descriptive_only',
            'radial_mean':float(radial.mean())}, {'samples':z,'chain_ids':ids,'radial_uniform':radial,'Sx':covariance}
