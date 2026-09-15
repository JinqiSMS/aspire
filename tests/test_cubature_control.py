"""The additional control must preserve real-oracle isolation and budgets."""
import ast
import importlib.util
import numpy as np
from aspire import ROOT
from aspire.architecture import Architecture
from aspire.config import resolve
from aspire.experiments.runner import learner_settings
from aspire.query_ledger import QueryLedger,CountedRealOracle


def load_control():
    path=ROOT/'scripts'/'cubature_control_core.py'
    spec=importlib.util.spec_from_file_location('cubature_control_core_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def test_cubature_stops_at_real_query_budget():
    module,_=load_control()
    ledger=QueryLedger(budget=37,cache_size=0)
    def real_value(x):
        assert np.isrealobj(x)
        return np.sum(x**16)
    result=module.recover_cubature_control(CountedRealOracle(real_value,3,ledger),
        architecture=Architecture(3,(3,3),4),public_bounds={'kappa0':4,'mu':.25},
        config=learner_settings(resolve({})),order=16,rng=np.random.default_rng(0))
    assert result['status']=='failed' and result['failure_reason']=='budget_exhausted'
    assert ledger.counts['n_real_calls_total']==37
    assert not result['hidden_weights']


def test_cubature_core_does_not_import_truth_helpers():
    _,path=load_control()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node,ast.ImportFrom):
            assert not any(part in (node.module or '') for part in ['teacher','diagnostics','evaluation','instances'])
