"""Save real-oracle cubature controls separately from paper-algorithm runs."""
from pathlib import Path
import sys
import os
import json
import hashlib
import time
import argparse
from copy import deepcopy
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import yaml
from aspire.config import resolve,read_config,fingerprint
from aspire.architecture import Architecture
from aspire.instances import generate
from aspire.experiments.runner import seed_map,learner_settings,environment,implementation_snapshot
from aspire.evaluation.metrics import evaluate_recovery
from aspire.query_ledger import CountedRealOracle,QueryLedger
from aspire.io import write_json,save_npz
from cubature_control_core import recover_cubature_control


def main(index):
    source=read_config(ROOT/'configs'/f'large_gap_{index}.yaml')
    source.update(seeds=[],sweep=[],kind='recovery')
    base=resolve(source);arch=Architecture(**base['architecture'])
    core_hash,snapshot=implementation_snapshot()
    script_paths=[Path(__file__),Path(__file__).with_name('cubature_control_core.py')]
    driver_hash=hashlib.sha256(b''.join(p.read_bytes() for p in script_paths)).hexdigest()
    for order in [16,32,64]:
        for seed in [0,1]:
            started=time.perf_counter();cfg=deepcopy(base);cfg['seed']=seed
            cfg['experiment_name']=f'large_gap_cubature_{index}'
            seeds=seed_map(cfg);run_id=f'seed_{seed}_order_{order}_{fingerprint(cfg)}_{driver_hash[:8]}'
            folder=ROOT/'results'/cfg['experiment_name']/run_id
            if (folder/'metrics.json').exists():continue
            folder.mkdir(parents=True,exist_ok=True)
            for script in script_paths:(folder/script.name).write_bytes(script.read_bytes())
            (folder/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            write_json(folder/'control_protocol.json',{'kind':'cubature_control','sampler_mode':'deterministic_angular',
                'order':order,'direction_count':2*order**2,'derivative_source':'real_degree_16_interpolation',
                'teacher_information_available_to_learner':False,'paper_sampling_guarantee_claimed':False})
            write_json(folder/'seed_manifest.json',seeds)
            write_json(folder/'metadata.json',{'run_id':run_id,'implementation_hash':core_hash,
                'implementation_snapshot':str(snapshot),'control_driver_sha256':driver_hash,'environment':environment(),
                'oracle_mode':'strict_real','sampler_mode':'deterministic_angular','guarantee_status':'empirical_only_control'})
            teacher,diag=generate(arch,cfg['teacher'],np.random.default_rng(seeds['teacher']))
            write_json(folder/'instance_diagnostics.json',diag)
            save_npz(folder/'teacher_parameters.npz',W1=teacher.weights[0],W2=teacher.weights[1],a=teacher.a)
            ledger=QueryLedger(cfg['oracle']['max_real_learning_queries'],cfg['oracle']['cache_size'],cfg['oracle']['wall_seconds'])
            evaluation=QueryLedger(budget=10**18)
            result=recover_cubature_control(CountedRealOracle(teacher.forward,arch.d,ledger),architecture=arch,
                public_bounds={'kappa0':4,'mu':.25,'gamma':None},config=learner_settings(cfg),order=order,
                rng=np.random.default_rng(seeds['algorithm']))
            learning_time=time.perf_counter()-started
            metrics={'experiment_name':cfg['experiment_name'],'run_id':run_id,'seed':seed,
                'teacher_seed':seeds['teacher'],'algorithm_seed':seeds['algorithm'],'kind':'cubature_control',
                'architecture':cfg['architecture'],'Q':arch.Q,'oracle_mode':'strict_real',
                'sampler_mode':'deterministic_angular','teacher_family':'frozen','configuration':cfg,
                'implementation_hash':core_hash,'control_driver_sha256':driver_hash,
                'guarantee_status':'empirical_only_control','moment_samples':0,'angular_directions':2*order**2,
                'integration_order':order,'status':result['status'],'failure_reason':result['failure_reason']}
            metrics.update(evaluate_recovery(teacher,result,arch,cfg['evaluation'],np.random.default_rng(seeds['test']),evaluation))
            metrics.update(n_real_learning_queries=ledger.counts['n_real_calls_total'],
                n_real_evaluation_queries=evaluation.counts['n_real_calls_total'],query_counts=ledger.snapshot(),
                wall_time_learning=learning_time,wall_time_total=time.perf_counter()-started)
            save_npz(folder/'artifacts'/'arrays.npz',**result['artifacts'])
            recovered={f'W{i+1}':w for i,w in enumerate(result['hidden_weights'])}
            if result['output_weights_raw'] is not None:recovered.update(a_raw=result['output_weights_raw'],a_projected=result['output_weights_projected'])
            save_npz(folder/'recovered_parameters.npz',**recovered)
            write_json(folder/'metrics.json',metrics)
            write_json(folder/'query_counts.json',{'learning':ledger.snapshot(),'evaluation':evaluation.snapshot()})
            from aspire.io import clean
            (folder/'stage_diagnostics.jsonl').write_text(''.join(json.dumps(clean(s))+'\n' for s in result['stages']),encoding='utf-8')
            write_json(folder/'checksums.json',{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='checksums.json'})
            print(index,seed,order,result['status'],'W_errors',metrics['per_layer_errors'],'a_error',metrics['output_l1_error'],
                  'success',metrics['parameter_success'],'queries',metrics['n_real_learning_queries'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--instance',type=int,required=True,choices=[0,1,2])
    main(parser.parse_args().instance)
