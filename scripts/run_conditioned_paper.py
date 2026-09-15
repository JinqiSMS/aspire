"""Run the frozen independent-endpoint protocol, with explicit stage selection."""
from pathlib import Path
import sys,os,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from aspire.config import read_config
from aspire.experiments.runner import expanded,run_one


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--instance',type=int,default=0,choices=[0,1])
    p.add_argument('--config',default=None,help='Optional config path relative to aspire')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--counts',type=int,nargs='+',default=[8192])
    p.add_argument('--steps',type=int,nargs='+',default=[32,128])
    args=p.parse_args()
    configs=expanded(read_config(ROOT/args.config if args.config else ROOT/f'configs/conditioned_paper_{args.instance}.yaml'),args.seeds)
    for cfg in configs:
        if cfg['recovery']['moment_samples_by_layer'][0] not in args.counts:continue
        if cfg['sampler']['endpoint_steps'] not in args.steps:continue
        run_one(cfg)


if __name__=='__main__':main()
