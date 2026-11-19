"""Portable public entry point. Run outputs are new directories, never overwritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')

def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/script)]+[str(x) for x in args],check=True,cwd=str(ROOT))

def train(method,seed,output,kind='formal',steps=1200000):
    args=['--name',method+'_seed'+str(seed),'--method',method,'--seed',seed,
          '--kind',kind,'--steps',steps,'--output',Path(output).resolve()]
    if kind=='formal':
        args+=['--protocol',ROOT/'config'/('control_protocol.json' if method=='control' else 'transfer_protocol.json')]
    if method=='refined':args+=['--teacher-manifest',ROOT/'config/teacher.json']
    call('src/train_single.py',*args)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    s.add_parser('build-teacher',help='Recreate the exact 420 MB selected teacher in cache/')
    e=s.add_parser('evaluate');e.add_argument('--run',choices=[f'{m}_seed{k}' for m in ['control','transfer'] for k in range(4)],default='transfer_seed0');e.add_argument('--output',required=True,type=Path)
    v=s.add_parser('verify');v.add_argument('--output',required=True,type=Path);v.add_argument('--replay',action='store_true',help='Also regenerate all eight final free/neutral trajectories')
    t=s.add_parser('train');t.add_argument('--method',choices=['control','refined'],required=True);t.add_argument('--seed',type=int,choices=range(4),required=True);t.add_argument('--output',required=True,type=Path)
    m=s.add_parser('train-matrix');m.add_argument('--output',required=True,type=Path)
    c=s.add_parser('smoke');c.add_argument('--method',choices=['control','refined'],default='refined');c.add_argument('--steps',type=int,default=80032);c.add_argument('--seed',type=int,default=0);c.add_argument('--output',required=True,type=Path)
    a=s.add_parser('audit-training');a.add_argument('--run',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    manifest=json.loads((ROOT/'package_manifest.json').read_text(encoding='utf8'))
    for name,entry in manifest['files'].items():
        path=ROOT/name
        assert path.is_file() and path.stat().st_size==entry['bytes'],name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],name
    if args.command=='build-teacher':call('src/build_teacher.py')
    elif args.command=='evaluate':call('src/evaluate_single.py','--model',ROOT/'models'/args.run/'final.pt','--output',args.output.resolve())
    elif args.command=='verify':
        extra=['--replay'] if args.replay else []
        call('audit/verify_delivery.py','--output',args.output.resolve(),*extra)
    elif args.command=='train':train(args.method,args.seed,args.output)
    elif args.command=='smoke':train(args.method,args.seed,args.output,'engineering',args.steps)
    elif args.command=='audit-training':call('audit/verify_training.py','--run',args.run.resolve(),'--output',args.output.resolve())
    elif args.command=='train-matrix':
        output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
        for method in ['control','refined']:
            for seed in range(4):train(method,seed,output/(method+'_seed'+str(seed)))

if __name__=='__main__':main()
