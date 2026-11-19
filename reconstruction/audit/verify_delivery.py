"""Check package hashes and independently recompute saved physics/observations/Q.

Optional replay invokes the production evaluator and compares every stored array.
It is saved-model inference, not a repetition of the archived full training.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import reference_core as ref
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    out=args.output.resolve();assert not out.exists(),'Use a new verification directory';out.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=ref.read(ROOT/'package_manifest.json')
    for name,entry in manifest['files'].items():
        path=ROOT/name;assert path.is_file() and path.stat().st_size==entry['bytes'] and ref.sha(path)==entry['sha256'],name
    expected=ref.read(ROOT/'reference/expected.json')
    assert {k:v['sha256'] for k,v in expected['inputs'].items()}==ref.EXPECTED_INPUTS
    paths={k:ROOT/v['path'] for k,v in expected['inputs'].items()}
    frames={k:pd.read_parquet(paths[k]) for k in ['price','load','ev']}
    assert all(len(x)==560640 and np.array_equal(x.index,frames['load'].index) for x in frames.values())
    assert all(len(x.columns)==1 for x in frames.values())
    data=(frames['price']['price'].to_numpy(np.float64),frames['load']['load_kw'].to_numpy(np.float64),frames['ev']['ev_demand_kw'].to_numpy(np.float64),ref.read(paths['soh']))
    rows=[];replays=[]
    for run in expected['runs']:
        weights={name:torch.load(ROOT/path,map_location='cpu',weights_only=True) for name,path in run['models'].items()}
        for filename,metric,neutral,model in [('initial_trace.npz','initial',False,'initial.pt'),('trace.npz','free_terminal',False,'final.pt'),('neutral_trace.npz','equal_terminal_evaluation',True,'final.pt')]:
            result=ref.verify_trace(ROOT/run['traces'][filename],data,np.asarray(run['environment']['action_grid']),run[metric],neutral,weights[model])
            rows.append(dict(run=run['name'],phase=metric,result=result))
        if args.replay:
            target=out/run['name']
            subprocess.run([sys.executable,str(ROOT/'src/evaluate_single.py'),'--model',str(ROOT/run['models']['final.pt']),'--output',str(target)],check=True,cwd=str(ROOT),stdout=subprocess.DEVNULL)
            compared=[]
            for name in ['trace.npz','neutral_trace.npz']:
                with np.load(target/name,allow_pickle=False) as actual,np.load(ROOT/run['traces'][name],allow_pickle=False) as old:
                    assert set(actual.files)==set(old.files)
                    for key in old.files:assert np.array_equal(actual[key],old[key]),(run['name'],name,key)
                    compared.append(dict(file=name,arrays=len(old.files),all_arrays_exactly_equal=True))
            replays.append(dict(run=run['name'],windows=compared))
    result=dict(status='all_pass',manifest_sha256=ref.sha(ROOT/'package_manifest.json'),verified_static_files=len(manifest['files']),
        independent_evaluation_windows=len(rows),independent_Q_rows=len(rows)*17520,results=rows,
        fresh_inference_windows=2*len(replays),fresh_inference_array_comparisons=replays,new_training_steps=0,
        scope='Saved-model evidence and optional inference replay. This does not rerun the historical 9.6 million training steps.')
    (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','verified_static_files','independent_evaluation_windows','independent_Q_rows','fresh_inference_windows','new_training_steps']}))

if __name__=='__main__':main()
