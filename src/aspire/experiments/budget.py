"""Dry-run arithmetic estimates, never oracle calls or teacher construction."""
import math
from ..architecture import Architecture
from ..recovery.sampling import radius_bounds

def estimate(cfg):
    arch=Architecture(**cfg['architecture']);stages=[]
    bounds={'kappa0':cfg['teacher']['kappa0_max'],'mu':cfg['teacher']['mu_min']}
    sampler=cfg['sampler']
    for l in range(1,arch.L-1):
        n=arch.widths[l];m=cfg['recovery']['moment_samples_by_layer'][l-1]
        inner,outer=radius_bounds(arch,l,bounds)
        bis=math.ceil(math.log2(3*outer/sampler['bisection_absolute_tolerance']))
        transitions=m*sampler['endpoint_steps'] if sampler['mode']=='independent_endpoints' else min(m,sampler['chains'])*sampler['burn_in']+m*sampler['thin']
        c=1 if l==1 else 2*arch.Q+1
        span=0 if arch.widths[l-1]==n else (cfg['recovery']['first_span_points']*arch.d*(arch.Q+1) if l==1 else cfg['recovery']['intermediate_span_points']*arch.widths[l-1]*(arch.Q+1))
        stages.append({'layer':l,'span_queries':span,'membership_queries':2*bis*transitions*c,
            'moment_queries':m*n*(arch.suffix_degree(l)+1)*c,'transitions':transitions,'bisections_per_ray':bis,
            'inner_radius':inner,'outer_radius':outer,'working_arrays_bytes':8*(2*m*n+4*n*n)})
    n=arch.widths[-2]
    probes = (cfg['final_layer']['hessian_directions']
              if cfg['final_layer'].get('method') == 'random_combination'
              else cfg['final_layer']['perturbation_candidates'])
    final=(1+probes)*n*(n+1)//2*(arch.k+1)*(2*arch.Q+1)
    total=sum(s['span_queries']+s['membership_queries']+s['moment_queries'] for s in stages)+final
    if cfg['kind']=='baselines':total=cfg['baseline']['label_budget']
    return {'architecture':cfg['architecture'],'Q':arch.Q,'suffix_degrees':[arch.suffix_degree(l) for l in range(1,arch.L)],
        'stages':stages,'final_hessian_generic_queries':final,'generic_learning_query_estimate':total,
        'budget':cfg['oracle']['max_real_learning_queries'],'within_base_estimate':total<=cfg['oracle']['max_real_learning_queries'],
        'degree_within_strict_limit':arch.Q<=cfg['oracle']['max_degree_strict'],
        'estimate_note':'Generic real-simulation branch, no caching or precision retries. Debug oracle costs differ; ledger is authoritative.'}
