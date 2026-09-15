"""Teacher-cluster bootstrap, separate from descriptive individual-run curves."""
from collections import defaultdict
from copy import deepcopy
import json
import numpy as np

def cluster_summary(runs,replicates=1000,seed=4387):
    groups=defaultdict(list)
    for run in runs:
        if run['status']=='invalid':continue
        if run['kind'] not in ('recovery','single_layer','baselines'):continue
        cfg=deepcopy(run.get('configuration',{}))
        for k in ('seed','seeds','sweep','randomness','output'):cfg.pop(k,None)
        key=json.dumps({'experiment':run['experiment_name'],'method':run.get('method'),
                        'oracle_mode':run.get('oracle_mode'),'sampler_mode':run.get('sampler_mode'),
                        'architecture':run.get('architecture'),'configuration':cfg,
                        'implementation_hash':run.get('implementation_hash')},sort_keys=True)
        groups[key].append(run)
    rng=np.random.default_rng(seed);summaries=[]
    for key,group in groups.items():
        teachers=defaultdict(list)
        for r in group:teachers[r.get('teacher_seed',r['seed'])].append(r)
        row={'group':json.loads(key),'runs':len(group),'independent_teacher_count':len(teachers),
             'bootstrap_replicates':replicates,'weighting':'equal teacher weights, then equal algorithm-run weights within teacher',
             'confidence_interpretation':'descriptive_only_fewer_than_5_teachers' if len(teachers)<5 else '95_percent_teacher_cluster_bootstrap_interval',
             'run_ids':[r['run_id'] for r in group]}
        for field in ('parameter_success','max_weight_operator_error','single_layer_operator_error','empirical_gaussian_nmse'):
            if field=='parameter_success' and group[0].get('parameter_metrics_unavailable_reason'):continue
            values=[]
            for attempts in teachers.values():
                valid=[a[field] for a in attempts if a.get(field) is not None]
                if valid:values.append(float(np.mean(valid)))
            if not values:continue
            values=np.array(values);draws=np.mean(rng.choice(values,(replicates,len(values)),replace=True),axis=1)
            row[field]={'teacher_weighted_mean':float(values.mean()),'interval95':np.quantile(draws,[.025,.975]).tolist(),
                        'teachers_with_metric':len(values),'conditional_on_metric_available':field!='parameter_success'}
        summaries.append(row)
    return summaries
