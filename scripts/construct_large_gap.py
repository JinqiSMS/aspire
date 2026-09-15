"""Freeze gap-screened positive teachers BEFORE any recovery experiment.

construction.tex motivates unequal coupling; its local sufficient theorem is
not claimed for the stronger coupling here. All candidates and exclusions are
saved. Selection never sees a learner, prediction error, or recovery seed.
"""
from pathlib import Path
import sys
import os
import itertools
import json
import hashlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from aspire.teacher import Teacher
from aspire.instances import diagnose
from aspire.diagnostics.population import integrate_body_3d, value_gradient_batch
from aspire.recovery.moments import relative_gap
from aspire.io import write_json

OUT = ROOT/'instances'/'large_gap'
SEED = 20260915


def signatures(w, a, order, rotation_seed=8128):
    t = Teacher([np.eye(3), w], a, 4)
    sx, sg = integrate_body_3d(lambda x: value_gradient_batch(t, x), 16, order, rotation_seed)
    # Full matrices are integrated. Diagonal products use the known sign symmetry
    # only for the evaluator's signature definition, not for learner directions.
    theta = np.diag(sx)*np.diag(sg)
    return {'theta': theta, 'relative_gap': relative_gap(theta),
            'offdiagonal_position': np.linalg.norm(sx-np.diag(np.diag(sx))),
            'offdiagonal_gradient': np.linalg.norm(sg-np.diag(np.diag(sg)))}


def candidates():
    # Paper's ordered-diagonal family, including explicitly labelled extensions
    # outside the theorem's unspecified small-coupling constants.
    for t in [.0001, .001, .005, .02, .05, .1, .2, .3, .4, .5]:
        idx=np.arange(1,4); w=t*(idx[:,None]+idx[None,:])/9
        np.fill_diagonal(w,0); np.fill_diagonal(w,1-w.sum(axis=0))
        yield 'paper_ordered_extension', {'t':t}, w, np.ones(3)/3
    for edges in itertools.combinations([.005,.025,.075,.15,.25,.35,.45],3):
        p,q,r=edges;w=np.array([[1-p-q,p,q],[p,1-p-r,r],[q,r,1-q-r]])
        yield 'symmetric_unequal_edges', {'edges':edges}, w, np.ones(3)/3
    rng=np.random.default_rng(SEED)
    for alpha in [.15,.3,.7,1.,2.]:
        for index in range(1000):
            w=rng.dirichlet(np.full(3,alpha),size=3).T
            # A fixed positive floor avoids near-zero coefficients; stronger
            # row-balance/conditioning acceptance is checked separately.
            w=.97*w+.03/3
            a=.7*rng.dirichlet(np.ones(3))+.3/3
            yield 'asymmetric_dirichlet', {'alpha':alpha,'draw':index}, w, a


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,(family,parameters,w,a) in enumerate(candidates()):
        diag=diagnose([np.eye(3),w],a)
        valid=bool(w.min()>0 and diag['kappa_actual']<=4 and diag['mu_actual']>=.25)
        row={'candidate_id':index,'family':family,'parameters':parameters,
             'W2':w,'a':a,'algebra':diag,'algebra_accepted':valid}
        if valid:row['coarse']=signatures(w,a,32)
        rows.append(row)
    qualified=sorted([r for r in rows if r['algebra_accepted']],key=lambda r:r['coarse']['relative_gap'],reverse=True)
    finalists=qualified[:40]
    for r in finalists:
        r['refinement']=[{'order':n,**signatures(r['W2'],r['a'],n)} for n in [64,128,256]]
        r['rotation_check']=signatures(r['W2'],r['a'],256,1729)
        theta=r['refinement'][-1]['theta']
        r['theta_change_relative']=float(np.max(abs(theta-r['refinement'][-2]['theta']))/max(theta))
        r['rotation_theta_change_relative']=float(np.max(abs(theta-r['rotation_check']['theta']))/max(theta))
        r['verified_gap']=min(r['refinement'][-1]['relative_gap'],r['rotation_check']['relative_gap'])
    finalists.sort(key=lambda r:r['verified_gap'],reverse=True)
    chosen=[]
    for r in finalists:
        if r['verified_gap']<.04 or max(r['theta_change_relative'],r['rotation_theta_change_relative'])>1e-8:continue
        if any(np.linalg.norm(r['W2']-s['W2'])<.3 for s in chosen):continue
        chosen.append(r)
        if len(chosen)==3:break
    manifest={'seed':SEED,'architecture':{'d':8,'hidden_widths':[3,3],'k':4},
        'selection_rule':{'conditioning_max':4,'mu_min':.25,'coarse_order':32,'refine_top':40,
            'minimum_verified_gap':.04,'maximum_relative_theta_change':1e-8,
            'selection':'three largest verified gaps, matrix Frobenius distance >=0.3',
            'recovery_results_used':False},
        'candidate_count':len(rows),'algebra_accepted_count':len(qualified),
        'refined_count':len(finalists),'selected_ids':[r['candidate_id'] for r in chosen],
        'construction_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paper_sources':{name:hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()
            for name in ['construction.tex','problem_setup_and_preliminaries.tex','algo_and_main_results.tex']},
        'claim':'numerically_verified_population_gap; not an analytic or interval lower-bound certificate',
        'candidates':rows}
    write_json(OUT/'construction_manifest.json',manifest)
    for index,r in enumerate(chosen):
        write_json(OUT/f'instance_{index}.json',{'instance_id':f'large_gap_{index}',
            'candidate_id':r['candidate_id'],'k':4,'hidden_widths':[3,3],
            'higher_weights':[r['W2']],'output_weights':r['a'],
            'public_bounds':{'kappa0':4,'mu':.25,'gamma':None},
            'verified_population_gaps':[r['verified_gap']],
            'construction_manifest':'instances/large_gap/construction_manifest.json',
            'gap_certificate':'numerical_convergence_only'})
        print('SELECTED',index,'candidate',r['candidate_id'],'gap',r['verified_gap'],
              'cond',r['algebra']['kappa_actual'],'mu',r['algebra']['mu_actual'],
              'W2',r['W2'].tolist(),'a',r['a'].tolist(),flush=True)
    print('CANDIDATES',len(rows),'ALGEBRA_ACCEPTED',len(qualified),'SELECTED',len(chosen),flush=True)
    if len(chosen)!=3:raise RuntimeError('Predeclared construction target not met; inspect saved candidates')


if __name__=='__main__':main()
