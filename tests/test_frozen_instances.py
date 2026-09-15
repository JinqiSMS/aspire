import hashlib
import json
import numpy as np
import pytest
from aspire import ROOT
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.instances import generate
from aspire.experiments.runner import learner_settings


def frozen_config():
    path=ROOT/'tmp'/'frozen_instance_test.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    data={'k':4,'hidden_widths':[3,3],'instance_id':'test',
        'higher_weights':[[[.8,.1,.1],[.1,.7,.2],[.1,.2,.7]]],
        'output_weights':[.2,.3,.5],'verified_population_gaps':[.01],
        'gap_certificate':'test_only'}
    path.write_text(json.dumps(data),encoding='utf-8')
    return resolve({'teacher':{'family':'frozen','instance_path':str(path),
        'instance_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})


def test_frozen_latent_is_unchanged_across_embedding_seeds():
    cfg=frozen_config();arch=Architecture(**cfg['architecture'])
    t1,d1=generate(arch,cfg['teacher'],np.random.default_rng(10))
    t2,_=generate(arch,cfg['teacher'],np.random.default_rng(20))
    assert not np.allclose(t1.weights[0],t2.weights[0])
    assert np.array_equal(t1.weights[1],t2.weights[1]) and np.array_equal(t1.a,t2.a)
    assert d1['instance_id']=='test'
    cfg['teacher']['instance_sha256']='changed'
    with pytest.raises(ValueError,match='checksum'):generate(arch,cfg['teacher'],np.random.default_rng(0))


def test_frozen_rejects_architecture_mismatch():
    cfg=frozen_config()
    with pytest.raises(ValueError,match='architecture'):
        generate(Architecture(8,(4,3),4),cfg['teacher'],np.random.default_rng(0))


def test_learner_settings_exclude_teacher_selection_information():
    cfg=frozen_config();settings=learner_settings(cfg)
    assert set(settings)=={'oracle','recovery','sampler','final_layer'}
    assert 'frozen_instance_test' not in json.dumps(settings)
    settings['recovery']['gap_floor']=1
    assert cfg['recovery']['gap_floor']!=1
