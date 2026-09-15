"""CLI: experiments, dry run, stage-boundary resume, plots, report."""
import argparse
import json
import sys
from pathlib import Path
from . import ROOT
from .config import read_config,resolve
from .experiments.runner import expanded,run_one
from .experiments.budget import estimate
from .io import inside,clean

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=None)
    parser.add_argument('--suite',choices=['smoke','core','paper','stress'])
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',help='Run directory with checkpoints')
    parser.add_argument('--budget',type=int,help='Explicit override of real learning query cap')
    parser.add_argument('--plots',action='store_true',help='Rebuild figures without running experiments')
    parser.add_argument('--report',action='store_true',help='Rebuild offline report without running experiments')
    parser.add_argument('--no-plots',action='store_true',help='Save results now; rebuild all figures later')
    parser.add_argument('--results',default='results')
    args=parser.parse_args(argv)
    if args.plots or args.report:
        from .plot import build_all
        build_all(inside(args.results),inside('figures'),inside('reports'))
        return
    if args.resume:
        folder=inside(args.resume);cfg=resolve(read_config(folder/'resolved_config.yaml'))
        if args.budget:cfg['oracle']['max_real_learning_queries']=args.budget
        configs=[cfg]
    else:
        paths=[inside(args.config)] if args.config else [inside('configs/strict_smoke.yaml')]
        if args.suite:
            suite=read_config(inside('configs/suites')/(args.suite+'.yaml'))
            paths=[inside('configs')/name for name in suite['configs']]
        configs=[]
        for path in paths:configs.extend(expanded(read_config(path),args.seeds))
        if args.budget:
            for cfg in configs:cfg['oracle']['max_real_learning_queries']=args.budget
    if args.dry_run:
        print(json.dumps(clean([{'experiment':c['experiment_name'],'seed':c['seed'],**estimate(c)} for c in configs]),indent=2))
        return
    errors=0
    for cfg in configs:
        _,result=run_one(cfg,args.resume)
        errors+=result['status']=='error'
    if not args.no_plots and any(c['output']['make_figures'] for c in configs):
        from .plot import build_all
        build_all(inside('results'),inside('figures'),inside('reports'))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
