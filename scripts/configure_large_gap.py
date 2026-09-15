"""Freeze the experiment grid and instance checksums before recovery starts."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml

for index in range(3):
    path=ROOT/'instances'/'large_gap'/f'instance_{index}.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    config={'experiment_name':f'large_gap_{index}','architecture':{'d':8,'hidden_widths':[3,3],'k':4},
        'teacher':{'family':'frozen','instance_path':str(path.relative_to(ROOT)),
            'instance_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kappa0_max':4,'mu_min':.25},
        'seeds':[0,1],'oracle':{'max_real_learning_queries':10000000,'wall_seconds':1800},
        'sampler':{'chains':2,'burn_in':256,'thin':2},
        'diagnostic':{'reference_directions':0},
        'evaluation':{'parameter_delta':.1},'sweep':[]}
    for kind,mode in [('single_layer','true_suffix_debug'),('recovery','strict_real')]:
        for count in [1024,4096,16384]:
            config['sweep'].append({'kind':kind,'oracle.mode':mode,
                                   'recovery.moment_samples_by_layer':[count]})
    (ROOT/'configs'/f'large_gap_{index}.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')

# Same new sampler, seeds, budgets and public radii, original unscreened family.
control={'experiment_name':'large_gap_original_control','seeds':[0,1],
    'teacher':{'kappa0_max':4,'mu_min':.25},
    'sampler':{'chains':2,'burn_in':256,'thin':2},
    'oracle':{'max_real_learning_queries':10000000,'wall_seconds':1800},
    'recovery':{'moment_samples_by_layer':[16384]},
    'diagnostic':{'reference_directions':0},
    'sweep':[{'kind':'single_layer','oracle.mode':'true_suffix_debug'},
             {'kind':'recovery','oracle.mode':'strict_real'}]}
(ROOT/'configs'/'large_gap_original_control.yaml').write_text(yaml.safe_dump(control,sort_keys=False),encoding='utf-8')
print('Frozen 36 gap-selected runs + 4 original-family matched controls; 2 seeds per latent instance.')
