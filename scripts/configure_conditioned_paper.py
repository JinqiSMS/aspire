"""Freeze previously identified, jointly conditioned candidates before HR runs."""
from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import yaml
from aspire.io import write_json
from aspire.instances import diagnose
from construct_large_gap import signatures


def main():
    source=ROOT/'instances/large_gap/construction_manifest.json'
    old=json.loads(source.read_text(encoding='utf-8'))
    out=ROOT/'instances/conditioned_paper';out.mkdir(parents=True,exist_ok=True)
    manifest={'selection':'candidates 15 and 1586 proposed before any new learner run; 15 is primary',
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'candidate_ids':[15,1586],'instances':[],
              'sampling_plan':{'pilot_steps':[32,128],'pilot_count':8192,
                'main_steps':128,'main_counts':[65536,262144,1048576],
                'seeds':[0,1],'final_candidates':1,'tau':.2},
              'claim':'empirical mixing and numerical population checks, not a certified theorem budget'}
    for index,candidate in enumerate(manifest['candidate_ids']):
        row=next(r for r in old['candidates'] if r['candidate_id']==candidate)
        w=np.array(row['W2']);a=np.array(row['a'])
        algebra=diagnose([np.eye(3),w],a)
        checks=[{'order':order,'rotation':rotation,**signatures(w,a,order,rotation)}
                for order,rotation in [(64,8128),(128,8128),(256,1729)]]
        h2=12*(w*a)@w.T
        assert min(c['relative_gap'] for c in checks)>.02
        assert np.linalg.cond(h2)<3 and algebra['kappa_actual']<2 and algebra['mu_actual']>=.25
        instance={'instance_id':f'conditioned_paper_{index}','candidate_id':candidate,
            'k':4,'hidden_widths':[3,3],'higher_weights':[w],'output_weights':a,
            'public_bounds':{'kappa0':2.,'mu':.25,'gamma':None},
            'verified_population_gaps':[float(checks[-1]['relative_gap'])],
            'gap_certificate':'numerical_convergence_only'}
        path=out/f'instance_{index}.json';write_json(path,instance)
        manifest['instances'].append({'index':index,'candidate':candidate,'algebra':algebra,
            'population_checks':checks,'hessian2_eigenvalues':np.linalg.eigvalsh(h2),
            'hessian2_condition':np.linalg.cond(h2),'instance_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        base={'experiment_name':f'conditioned_paper_{index}','kind':'recovery',
            'architecture':{'d':8,'hidden_widths':[3,3],'k':4},
            'teacher':{'family':'frozen','instance_path':str(path.relative_to(ROOT)),
                'instance_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kappa0_max':2.,'mu_min':.25},
            'seeds':[0,1],
            'oracle':{'mode':'strict_real','cache_size':0,'max_real_learning_queries':10000000000,'wall_seconds':14400},
            'sampler':{'mode':'independent_endpoints','batch_size':4096,'endpoint_steps':128,
                'bisection_absolute_tolerance':1e-7},
            'diagnostic':{'reference_directions':0},
            'final_layer':{'tau_fin':.2,'perturbation_candidates':1},
            'evaluation':{'parameter_delta':.1},
            'sweep':[{'sampler.endpoint_steps':t,'recovery.moment_samples_by_layer':[8192]} for t in [32,128]]+
                    [{'recovery.moment_samples_by_layer':[m]} for m in [65536,262144,1048576]]}
        (ROOT/f'configs/conditioned_paper_{index}.yaml').write_text(yaml.safe_dump(base,sort_keys=False),encoding='utf-8')
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'frozen_ids':manifest['candidate_ids'],'checks':[{k:r[k] for k in ('index','hessian2_condition')} for r in manifest['instances']]}))


if __name__=='__main__':main()
