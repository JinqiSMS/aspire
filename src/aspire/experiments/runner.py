"""One run is an auditable unit, including numerical failures and partial layers."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
import zipfile
import numpy as np
import yaml
from .. import ROOT,__version__
from ..architecture import Architecture
from ..config import resolve,fingerprint,set_dotted
from ..io import inside,write_json,save_npz,clean
from ..instances import generate
from ..query_ledger import QueryLedger,CountedRealOracle
from ..recovery.network import recover_network
from ..evaluation.metrics import evaluate_recovery,evaluate_predictor,parameter_metrics
from ..diagnostics.experiments import complex_debug_recovery,single_layer,prefix_error,oracle_accuracy,sampler_check,population_moments
from ..baselines.models import FITTERS
from ..status import NumericalFailure

def seed_map(cfg):
    names=['teacher','algorithm','test','baseline_data','baseline_initialization','diagnostics']
    mapping={name:int(np.random.SeedSequence([cfg['seed'],i+1]).generate_state(1)[0]) for i,name in enumerate(names)}
    for name in ('teacher','algorithm','test'):
        override=cfg['randomness'][name+'_seed']
        if override is not None:mapping[name]=override
    return mapping


def learner_settings(cfg):
    """Do not expose frozen teacher paths, selection records, or parameters."""
    return {key:deepcopy(cfg[key]) for key in ('oracle','recovery','sampler','final_layer')}

def environment():
    versions={}
    for name in ('numpy','scipy','mpmath','matplotlib','PyYAML','pytest','torch'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'aspire_version':__version__}

def implementation_snapshot():
    files=sorted((ROOT/'src').rglob('*.py'))
    digest=hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode()+
        p.read_bytes().replace(b'\r\n',b'\n') for p in files)).hexdigest()
    path=inside('implementations')/(digest[:16]+'.zip');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
            for p in files:archive.write(p,p.relative_to(ROOT).as_posix())
    return digest,path

def expanded(raw,seeds=None):
    base=resolve(raw)
    selected=seeds if seeds is not None else (base['seeds'] or [base['seed']])
    variants=base['sweep'] or [{}]
    for variant in variants:
        modified=deepcopy(base)
        for key,value in variant.items():set_dotted(modified,key,value)
        for seed in selected:
            cfg=deepcopy(modified);cfg['seed']=seed;cfg['seeds']=[];cfg['sweep']=[]
            if cfg['kind']=='baselines':
                for method in cfg['baseline']['methods']:
                    one=deepcopy(cfg);one['baseline']['methods']=[method];yield resolve(one)
            else:yield resolve(cfg)

def run_one(cfg,resume_path=None):
    started=time.perf_counter();arch=Architecture(**cfg['architecture']);seeds=seed_map(cfg)
    implementation_hash,snapshot=implementation_snapshot()
    run_id=f"seed_{cfg['seed']}_{fingerprint(cfg)}_{implementation_hash[:8]}"
    folder=inside(resume_path) if resume_path else inside(cfg['output']['directory'])/cfg['experiment_name']/run_id
    folder=inside(folder)
    # Validate resume before writing any files to an existing run.
    if resume_path:
        old_state=folder/'checkpoints'/'state.json'
        if not old_state.exists():raise ValueError('No completed-layer checkpoint to resume')
        resume_state=json.loads(old_state.read_text(encoding='utf-8'))
        if resume_state['config_hash']!=fingerprint(cfg) or resume_state.get('implementation_hash')!=implementation_hash:
            raise ValueError('Checkpoint configuration or implementation differs; start a new run')
        old_metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        run_id=old_metadata['run_id']
    if folder.exists() and (folder/'metrics.json').exists() and not resume_path:
        saved=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        print(f"Cached run: {cfg['experiment_name']} / {run_id}: {saved['status']}",flush=True)
        return folder,saved
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write_json(folder/'seed_manifest.json',seeds)
    write_json(folder/'metadata.json',{'run_id':run_id,'created_at':datetime.now(timezone.utc).isoformat(),
        'config_hash':fingerprint(cfg),'environment':environment(),'source_manifest':str(ROOT/'source_manifest.json'),
        'implementation_hash':implementation_hash,'implementation_snapshot':str(snapshot),
        'guarantee_status':'empirical_only','oracle_mode':cfg['oracle']['mode'],'sampler_mode':cfg['sampler']['mode']})
    rng=np.random.default_rng(seeds['algorithm']);eval_rng=np.random.default_rng(seeds['test'])
    restored=None;counts=None
    cp=folder/'checkpoints'/'state.json'
    if resume_path and not cp.exists():raise ValueError('No completed-layer checkpoint to resume')
    if resume_path and cp.exists():
        state=json.loads(cp.read_text(encoding='utf-8'))
        if state['config_hash']!=fingerprint(cfg):raise ValueError('Checkpoint configuration mismatch')
        if state.get('implementation_hash',implementation_hash)!=implementation_hash:raise ValueError('Checkpoint implementation differs; start a new run')
        arrays=np.load(folder/'checkpoints'/'prefix.npz')
        restored={'weights':[arrays[f'W{i+1}'] for i in range(state['completed_layers'])],'stages':state['stages']}
        rng.bit_generator.state=state['rng_state']
        # Includes all spending after the last successful layer, if failure occurred.
        counts=json.loads((folder/'query_counts.json').read_text(encoding='utf-8'))['learning'] if (folder/'query_counts.json').exists() else state['counts']
    ledger=QueryLedger(cfg['oracle']['max_real_learning_queries'],cfg['oracle']['cache_size'],cfg['oracle']['wall_seconds'],counts)
    evaluation_ledger=QueryLedger(budget=10**18)
    base={'experiment_name':cfg['experiment_name'],'run_id':run_id,'kind':cfg['kind'],'seed':cfg['seed'],
          'teacher_seed':seeds['teacher'],'algorithm_seed':seeds['algorithm'],'architecture':cfg['architecture'],'Q':arch.Q,
          'oracle_mode':cfg['oracle']['mode'],'sampler_mode':cfg['sampler']['mode'],'guarantee_status':'empirical_only',
          'teacher_family':cfg['teacher']['family'],'parameter_success':False,'status':'running',
          'failure_reason':None,'moment_samples':cfg['recovery']['moment_samples_by_layer'][0],
          'configuration':cfg,'implementation_hash':implementation_hash}
    artifacts={};stages=[];teacher=None
    def checkpoint(weights,records,rng_state,counts,arrays):
        save_npz(folder/'checkpoints'/'prefix.npz',**{f'W{i+1}':w for i,w in enumerate(weights)})
        write_json(cp,{'config_hash':fingerprint(cfg),'implementation_hash':implementation_hash,'completed_layers':len(weights),'stages':records,'rng_state':rng_state,'counts':counts})
        if cfg['output']['save_artifacts']:save_npz(folder/'checkpoints'/'stage_artifacts.npz',**arrays)
        print(f"  layer {len(weights)} recovered; real queries={counts['n_real_calls_total']}",flush=True)
    print(f"Run {cfg['experiment_name']} / {run_id} [{cfg['oracle']['mode']}, {cfg['sampler']['mode']}]",flush=True)
    try:
        teacher,instance_diag=generate(arch,cfg['teacher'],np.random.default_rng(seeds['teacher']))
        write_json(folder/'instance_diagnostics.json',instance_diag)
        save_npz(folder/'teacher_parameters.npz',**{f'W{i+1}':w for i,w in enumerate(teacher.weights)},a=teacher.a)
        real=CountedRealOracle(teacher.forward,arch.d,ledger,batch_value=teacher.forward)
        bounds={'kappa0':cfg['teacher']['kappa0_max'],'mu':cfg['teacher']['mu_min'],'gamma':cfg['teacher']['gamma_bound']}
        kind=cfg['kind']
        if kind=='recovery':
            learning_config=learner_settings(cfg)
            if cfg['oracle']['mode']=='strict_real':
                result=recover_network(real,architecture=arch,public_bounds=bounds,config=learning_config,rng=rng,checkpoint=checkpoint,restored=restored)
            elif cfg['oracle']['mode']=='complex_debug':
                result=complex_debug_recovery(real,teacher,arch,bounds,learning_config,rng,checkpoint,restored)
            else:raise ValueError('Truth suffix and exact prefix modes are isolated diagnostics, not end-to-end recovery')
            learning_finished=time.perf_counter();stages=result['stages'];artifacts=result['artifacts']
            if restored and (folder/'checkpoints'/'stage_artifacts.npz').exists():
                previous=np.load(folder/'checkpoints'/'stage_artifacts.npz');artifacts={**dict(previous),**artifacts}
            base.update(status=result['status'],failure_reason=result['failure_reason'],failure_details=result.get('failure_details'))
            base.update(evaluate_recovery(teacher,result,arch,cfg['evaluation'],eval_rng,evaluation_ledger))
            arrays={f'W{i+1}':w for i,w in enumerate(result['hidden_weights'])}
            if result['output_weights_raw'] is not None:arrays.update(a_raw=result['output_weights_raw'],a_projected=result['output_weights_projected'])
            save_npz(folder/'recovered_parameters.npz',**arrays)
            if base['status']=='complete' and not base['parameter_success']:base['accuracy_status']='parameter_error_above_target'
        elif kind=='single_layer':
            if cfg['oracle']['mode']!='true_suffix_debug':raise ValueError('Single-layer truth diagnostic must be labeled true_suffix_debug')
            metric,artifacts=single_layer(teacher,arch,bounds,cfg,rng,ledger);base.update(metric,status='complete')
            learning_finished=time.perf_counter()
        elif kind=='prefix_error':
            if cfg['oracle']['mode']!='exact_prefix_debug':raise ValueError('Prefix perturbation uses evaluator truth and must be labeled debug')
            metric,artifacts=prefix_error(real,teacher,arch,cfg,rng,ledger);base.update(metric,status='complete')
            learning_finished=time.perf_counter()
        elif kind=='oracle_accuracy':
            metric,artifacts=oracle_accuracy(cfg,rng);base.update(metric,status='complete')
            learning_finished=time.perf_counter()
        elif kind=='sampler_check':
            metric,artifacts=sampler_check(cfg,rng,ledger);base.update(metric,status='complete')
            learning_finished=time.perf_counter()
        elif kind=='baselines':
            method=cfg['baseline']['methods'][0];base['method']=method
            if method not in FITTERS:raise ValueError('Unknown baseline')
            count=cfg['baseline']['label_budget'];valid=max(1,int(count*cfg['baseline']['validation_fraction']));train=count-valid
            if train<2:raise ValueError('Too few training labels')
            data_rng=np.random.default_rng(seeds['baseline_data']);x=data_rng.normal(size=(count,arch.d))
            with ledger.scope('training'):y=real.batch(x[:train])
            with ledger.scope('validation'):yv=real.batch(x[train:])
            baseline_settings={**cfg['baseline'],'_deadline':started+cfg['oracle']['wall_seconds']}
            predictor,diag,artifacts,parameters=FITTERS[method](x[:train],y,x[train:],yv,arch,baseline_settings,np.random.default_rng(seeds['baseline_initialization']))
            learning_finished=time.perf_counter()
            base.update(diag,status='complete',label_budget=count,training_labels=train,validation_labels=valid)
            base.update(evaluate_predictor(teacher,predictor,arch,cfg['evaluation'],eval_rng,evaluation_ledger,parameters[0] if parameters else None))
            if parameters:
                base.update(parameter_metrics(teacher,{'hidden_weights':parameters[0],'output_weights_raw':parameters[1],'status':'complete'},cfg['evaluation']['parameter_delta']))
            else:base['parameter_metrics_unavailable_reason']='predictor_has_no_target_network_parameterization'
        else:raise ValueError(f'Unsupported kind {kind}')
        base['wall_time_learning']=learning_finished-started
        if cfg['evaluation']['population_gap_directions'] and kind=='recovery':
            base['population_gap_by_layer']=[population_moments(teacher,l,cfg['evaluation']['population_gap_directions'],cfg['evaluation']['population_gap_batches'],np.random.default_rng(seeds['diagnostics']+l)) for l in range(1,arch.L-1)]
    except NumericalFailure as error:
        base.update(status='failed',failure_reason=error.reason,failure_details=error.details)
    except Exception as error:
        base.update(status='error',failure_reason=type(error).__name__,failure_details=str(error))
        (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(traceback.format_exc(),flush=True)
    base.update(n_real_learning_queries=ledger.counts['n_real_calls_total'],
                n_real_evaluation_queries=evaluation_ledger.counts['n_real_calls_total'],
                n_kept_samples=ledger.counts['n_kept_samples'],n_hr_transitions=ledger.counts['n_hr_transitions'],
                wall_time_total=time.perf_counter()-started,wall_time_oracle=ledger.oracle_seconds)
    base['query_counts']=ledger.snapshot()
    write_json(folder/'query_counts.json',{'learning':ledger.snapshot(),'evaluation':evaluation_ledger.snapshot()})
    write_json(folder/'metrics.json',base)
    write_json(folder/'runtime.json',{k:v for k,v in base.items() if k.startswith('wall_time')})
    import csv
    scalar={k:v for k,v in clean(base).items() if not isinstance(v,(list,dict))}
    with (folder/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scalar));writer.writeheader();writer.writerow(scalar)
    (folder/'stage_diagnostics.jsonl').write_text(''.join(json.dumps(clean(s),ensure_ascii=False,allow_nan=False)+'\n' for s in stages),encoding='utf-8')
    if cfg['output']['save_artifacts'] and artifacts:save_npz(folder/'artifacts'/'arrays.npz',**artifacts)
    write_json(folder/'checksums.json',{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='checksums.json'})
    print(f"  {base['status']}; queries={base['n_real_learning_queries']}; parameter_error={base.get('max_weight_operator_error')}; reason={base['failure_reason']}",flush=True)
    return folder,base
