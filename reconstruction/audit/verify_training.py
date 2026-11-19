"""Independent transition, replay-link, teacher-label and saved-network checks.

No training/environment/teacher implementation is imported. Stage-1 teacher
labels are sampled by a fixed update-index rule; stage-2 labels are checked in
full. Historical soft teacher targets and every gradient are not reconstructed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import reference_core as ref

CORE_SHA = '535013dcf8eed171da2fee51417b8b9cbed048fc4838d34e481ed4f552a9a710'
ROOT=Path(__file__).resolve().parents[1]
N = 17520
CAP = 200000


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def binding(b):
    p=Path(b['path']);p=p.resolve() if p.is_absolute() else (ROOT/p).resolve();assert p.is_file() and sha(p)==b['sha256'],str(p)
    if 'bytes' in b:assert p.stat().st_size==b['bytes']
    return p


def load_data(receipt):
    assert set(receipt['inputs'])==set(ref.EXPECTED_INPUTS)
    paths={}
    for key,expected in ref.EXPECTED_INPUTS.items():
        assert receipt['inputs'][key]['sha256']==expected
        paths[key]=binding(receipt['inputs'][key])
    frames={k:pd.read_parquet(paths[k]) for k in ['price','load','ev']}
    assert all(len(v)==560640 and np.array_equal(v.index,frames['price'].index) for v in frames.values())
    return (frames['price']['price'].to_numpy(np.float64),frames['load']['load_kw'].to_numpy(np.float64),
            frames['ev']['ev_demand_kw'].to_numpy(np.float64),ref.read(paths['soh']))


def centered_advantage(obs,states):
    hidden=torch.from_numpy(np.asarray(obs,np.float32)).unsqueeze(0).expand(len(states),-1,-1)
    with torch.no_grad():
        for index,name in enumerate(['feature_layers.0','feature_layers.3','advantage_stream.0','advantage_stream.2']):
            weight=torch.stack([s[name+'.weight'] for s in states])
            bias=torch.stack([s[name+'.bias'] for s in states])
            hidden=torch.bmm(hidden,weight.transpose(1,2))+bias[:,None,:]
            if index<3:hidden=hidden.relu()
        return (hidden-hidden.mean(-1,keepdim=True)).mean(0)


class TeacherReference:
    def __init__(self,receipt,data):
        self.receipt=receipt;self.data=data;self.groups={};self.table=None;self.verified={}
        teacher=receipt['teacher']
        if receipt['method']=='control':
            assert teacher=={} and receipt['imitation_weight']==0
            return
        ep=binding(teacher['ensemble_receipt']);er=ref.read(ep)
        assert er['rule']=='objective-blend' and er['weight']==.1
        assert er['members']==teacher['ensemble_models']
        assert {k:v['sha256'] for k,v in er['inputs'].items()}==ref.EXPECTED_INPUTS
        self.groups={'base':[],'cost':[]}
        for member in teacher['ensemble_models']:
            p=binding(member);saved=torch.load(p,map_location='cpu',weights_only=True)
            state=saved['q_network_state_dict'] if member['kind']=='archived' else saved
            ref.weights_valid(state);self.groups[member['role']].append(state)
        assert len(self.groups['base'])==8 and len(self.groups['cost'])==2
        self.verified['ensemble_receipt']=dict(path=str(ep),sha256=sha(ep))
        self.verified['ensemble_models']=teacher['ensemble_models']
        if receipt['method'].startswith('refined'):
            manifest=binding(teacher['dp_manifest']);m=ref.read(manifest);folder=manifest.parent
            wrapper=binding(teacher['dp_source'])
            assert sha(folder/m['teacher_source_file'])==m['teacher_source_sha256']
            assert sha(folder/m['builder_source_file'])==m['builder_sha256']
            assert sha(folder/m['protocol_file'])==m['protocol_sha256']
            assert set(m['input_sha256'].values())==set(ref.EXPECTED_INPUTS.values())
            table=(folder/m['value_table_file']).resolve();offset=(folder/m['offsets_file']).resolve()
            assert sha(table)==m['value_table_sha256'] and sha(offset)==m['offsets_sha256']
            self.table=np.load(table,mmap_mode='r',allow_pickle=False)
            self.offset=np.load(offset,mmap_mode='r',allow_pickle=False)
            assert self.table.shape==(N+1,3001) and self.table.dtype==np.float64
            assert self.offset.shape==(N+1,) and self.offset.dtype==np.float64
            self.beta=float(m['beta'])
            self.verified['offline_DP']=dict(manifest=dict(path=str(manifest),sha256=sha(manifest)),
                wrapper=dict(path=str(wrapper),sha256=sha(wrapper)),teacher_source_sha256=m['teacher_source_sha256'],
                value_table_sha256=m['value_table_sha256'],offsets_sha256=m['offsets_sha256'],
                information='Full future offline load/price/availability; training supervision, not independent RL discovery')

    def labels(self,stage,obs,source,energy):
        price,load,ev,_=self.data
        if stage==1:
            q=[]
            for role in ['base','cost']:
                value=centered_advantage(obs,self.groups[role])
                q.append(value/value.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6))
            return (.9*q[0]+.1*q[1]).argmax(-1).numpy()
        assert stage==2 and self.table is not None
        energy=np.clip(np.asarray(energy,np.float64),0.,75.)
        command=np.arange(-10,11,dtype=np.float64)/10.
        p=np.clip(command[None,:],-energy[:,None]/.25,(75.-energy[:,None])/.25)
        p=np.where((ev[source]>.1)[:,None],p,0.)
        after=np.clip(energy[:,None]+.25*p,0.,75.)
        position=after/.025;nearest=np.rint(position)
        position=np.where(abs(position-nearest)<=1e-7,nearest,position)
        lo=np.floor(position).astype(np.int64);hi=np.minimum(lo+1,3000);fraction=position-lo
        future=(1-fraction)*self.table[source[:,None]+1,lo]+fraction*self.table[source[:,None]+1,hi]
        terminal=source==N-1
        if terminal.any():future[terminal]=-((after[terminal]-37.5)*4)**2/N-self.offset[N]
        coefficient=2*(load[source]-load[:N].mean())+self.beta*price[source]/4000.
        cost=p*p+coefficient[:,None]*p+future
        score=-(cost-cost.min(1,keepdims=True))/load[:N].var()
        score=(score-score.mean(1,keepdims=True)).astype(np.float32)
        return score.argmax(1)


def observations(ts,energy,soh,history,price,load):
    mean=load[:N].mean();std=load[:N].std()
    return np.column_stack([price[ts+j]/100. for j in range(6)]+[
        (load[ts]-mean)/std,(load[ts+1]-mean)/std,energy/75.,soh,
        (ts%96)/96.,ts/N]+[history[:,j]/20. for j in range(3)]).astype(np.float32)


def training_audit(run,receipt,data,teacher):
    price,load,ev,params=data;total=int(receipt['steps']);grid=np.asarray(receipt['environment']['action_grid'])
    assert receipt['replay_capacity']==CAP
    all_obs=np.empty((total,15),np.float32);all_energy=np.empty(total,np.float64)
    all_actions=np.empty(total,np.int8)
    stages={str(i):dict(batches=0,rows=0,labels=0) for i in (1,2)}
    checked_labels={str(i):0 for i in (1,2)}
    checks=dict(transition_rows=0,updates=0,replay_sample_links=0)
    maximum=dict(observation=0.,next_observation=0.,power=0.,energy=0.,reward=0.)
    expected_update=np.arange(32,total+1,4,dtype=np.int64)
    update_cursor=0;cursor=0;last_energy=37.5;last_soh=1.;progress=[]
    # Fixed sampling before seeing labels: each 1000th global update and each stage endpoint.
    stage1_count=len(expected_update) if receipt['method']=='ensemble' else int(np.count_nonzero(expected_update<=80000)) if receipt['method'].startswith('refined') else 0
    for entry in receipt['chunks']:
        path=binding(entry);assert path.parent==run/'training'
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        count=len(a['action']);begin=cursor;end=cursor+count
        assert entry['begin_step']==begin+1 and entry['end_step']==end and entry['transitions']==count
        assert count==min(40000,total-begin)
        idx=np.arange(begin,end);ts=idx%N
        assert np.array_equal(a['source_index'],ts)
        assert a['obs'].shape==a['next_obs'].shape==(count,15)
        assert a['obs'].dtype==a['next_obs'].dtype==np.float32
        assert a['reward'].dtype==np.float32 and a['done'].dtype==bool
        assert np.all((a['action']>=0)&(a['action']<21)) and np.issubdtype(a['action'].dtype,np.integer)
        assert np.array_equal(a['done'],ts==N-1)
        assert all(np.isfinite(v).all() for v in a.values())
        before=a['energy_before'];after=a['energy_after'];command=grid[a['action']]
        previous=np.r_[last_energy,after[:-1]];previous=np.where(ts==0,37.5,previous)
        ref.close(before,previous,atol=1e-9,label='energy continuity/reset')
        p=np.where(ev[ts]>.1,np.clip(command,-before/.25,(75.-before)/.25),0.)
        maximum['power']=max(maximum['power'],ref.close(a['power_kw'],p,atol=1e-9,label='training power'))
        maximum['energy']=max(maximum['energy'],ref.close(after,before+p*.25,atol=1e-9,label='training energy'))
        assert min(before.min(),after.min())>=-1e-9 and max(before.max(),after.max())<=75.+1e-9
        old_soh=np.empty(count);next_soh=np.empty(count)
        if 'calendar_aging' in params:
            calendar=params['calendar_aging']*.25;cycle=params.get('cycle_aging',.00005);depth=params.get('depth_factor',1.2);temp=params.get('temperature_factor',1.)
        else:
            calendar=params.get('a',1e-6)*.25;cycle=params.get('b',2e-6);depth=params.get('c',5e-7)*1000;temp=1.
        deg=calendar+np.where(abs(p)>.1,cycle*abs((after-before)/75.)*depth*temp,0.)
        first=0
        while first<count:
            stop=min(count,first+N-int(ts[first]));start_soh=1. if ts[first]==0 else last_soh
            sums=np.cumsum(deg[first:stop]);next_soh[first:stop]=start_soh-sums
            old_soh[first:stop]=np.r_[start_soh,next_soh[first:stop-1]]
            last_soh=float(next_soh[stop-1]);first=stop
        assert next_soh.min()>.7
        all_actions[begin:end]=a['action'];all_obs[begin:end]=a['obs'];all_energy[begin:end]=before
        history=np.empty((count,3),np.int64)
        for column,back in enumerate((3,2,1)):
            history[:,column]=np.where(ts>=back,all_actions[np.maximum(idx-back,0)],10)
        obs=observations(ts,before,old_soh,history,price,load)
        next_history=np.column_stack([history[:,1],history[:,2],a['action']])
        nxt=observations(ts+1,after,next_soh,next_history,price,load)
        maximum['observation']=max(maximum['observation'],ref.close(a['obs'],obs,atol=2e-7,label='training obs'))
        maximum['next_observation']=max(maximum['next_observation'],ref.close(a['next_obs'],nxt,atol=2e-7,label='training next obs'))
        target=load[:N].mean()+(37.5-before)/96.
        raw_reward=-(load[ts]+p-target)**2/load[:N].var()-.02*(p*.25*price[ts]/1000)/np.mean(load[:N]*.25*price[:N]/1000)
        expected_reward=raw_reward.astype(np.float32)
        assert np.allclose(a['reward'],expected_reward,rtol=2e-6,atol=2e-6),'B reward'
        maximum['reward']=max(maximum['reward'],float(np.max(abs(a['reward']-expected_reward))))
        update=a['update_steps'].astype(np.int64);m=len(update)
        assert entry['updates']==m and np.array_equal(update,expected_update[update_cursor:update_cursor+m])
        assert np.all((update>=begin+1)&(update<=end))
        assert a['replay_indices'].shape==a['is_weights'].shape==a['teacher_label'].shape==(m,32)
        assert all(a[k].shape==(m,) for k in ['td_loss','imitation_loss','teacher_stage'])
        ix=a['replay_indices'];assert np.all((ix>=0)&(ix<np.minimum(update[:,None],CAP)))
        gid=update[:,None]-1-((update[:,None]-1-ix)%CAP)
        assert np.all((gid>=0)&(gid<update[:,None])&(gid>=np.maximum(0,update[:,None]-CAP)))
        sampled_source=gid%N;valid=ev[sampled_source]>.1
        weights=a['is_weights'];assert np.all((weights>0)&(weights<=1.))
        assert np.all(weights.max(1)==1.)
        assert np.all(a['td_loss']>=0) and np.all(a['imitation_loss']>=-1e-6)
        expected_stage=np.zeros(m,np.int8) if receipt['method']=='control' else np.where(update>80000,2,1).astype(np.int8) if receipt['method'].startswith('refined') else np.ones(m,np.int8)
        assert np.array_equal(a['teacher_stage'],expected_stage)
        labels=a['teacher_label']
        if receipt['method']=='control':assert np.all(labels==-1) and np.all(a['imitation_loss']==0.)
        else:
            assert np.all(labels[~valid]==-1) and np.all((labels[valid]>=0)&(labels[valid]<21))
        for stage in (1,2):
            positions=np.flatnonzero(expected_stage==stage)
            stages[str(stage)]['batches']+=len(positions);stages[str(stage)]['rows']+=32*len(positions)
            stages[str(stage)]['labels']+=int(valid[positions].sum())
            if stage==1:
                global_updates=update_cursor+positions
                positions=positions[(global_updates%1000==0)|(global_updates==stage1_count-1)]
                # Keep the original32-row batch shape, including unavailable rows:
                # bmm FP32 ties must not be changed by regrouping selected rows.
                for selected in positions:
                    records=gid[selected]
                    expected=teacher.labels(1,all_obs[records],records%N,all_energy[records])
                    mask=valid[selected]
                    assert np.array_equal(expected[mask],labels[selected][mask]),('stage1 teacher argmax',int(update[selected]))
                    checked_labels['1']+=int(mask.sum())
                continue
            for first_batch in range(0,len(positions),128):
                chosen=positions[first_batch:first_batch+128]
                if not len(chosen):continue
                records=gid[chosen].reshape(-1);mask=valid[chosen].reshape(-1)
                if not mask.any():continue
                records=records[mask]
                expected=teacher.labels(stage,all_obs[records],records%N,all_energy[records])
                got=labels[chosen].reshape(-1)[mask]
                assert np.array_equal(expected,got),('teacher argmax mismatch',stage,int(update[chosen[0]]),int(np.sum(expected!=got)))
                checked_labels[str(stage)]+=len(records)
        update_cursor+=m;cursor=end;last_energy=float(after[-1])
        checks['transition_rows']+=count;checks['updates']+=m;checks['replay_sample_links']+=m*32
        progress.append(dict(steps=end,updates=update_cursor,teacher_counts=json.loads(json.dumps(stages))))
    assert cursor==total and update_cursor==len(expected_update)==receipt['updates']
    assert stages==receipt['teacher_counts']
    actual_progress=[json.loads(line) for line in (run/'progress.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(actual_progress)==len(progress)
    for actual,expected in zip(actual_progress,progress):
        assert all(actual[k]==v for k,v in expected.items()) and actual['seconds']>=0
    return dict(**checks,maximum_errors=maximum,teacher_stage_counts=stages,
                independently_recomputed_valid_teacher_labels=checked_labels,
                teacher_label_scope='Stage1: global update indices divisible by1000 plus stage last; Stage2: every valid stored argmax label',
                terminal_next_observations_checked=True,
                limits=['IS positivity and batch normalization checked; full historical priorities/probabilities are not stored',
                        'Every sampled slot is linked to its actual recorded transition; no claim of replaying every historical gradient',
                        'Only teacher argmax labels are stored; historical soft teacher logits and imitation-loss values cannot be fully independently reconstructed'])


def verify(run):
    receipt_path=run/'receipt.json';receipt=ref.read(receipt_path)
    assert receipt['status']=='complete' and receipt['method'] in ['control','ensemble','refined','refined_strong']
    assert receipt['updates']==(receipt['steps']-32)//4+1
    expected_weight={'control':0.,'ensemble':4.,'refined':4.,'refined_strong':10.}[receipt['method']]
    assert receipt['imitation_weight']==expected_weight
    assert receipt['ensemble_only_through_step']==(80000 if receipt['method'].startswith('refined') else None)
    sources={k:dict(path=str(binding(v)),sha256=v['sha256']) for k,v in receipt['source'].items()}
    assert sources['native_dddqn_per_sb3.py']['sha256']==ref.EXPECTED_NATIVE['native_dddqn_per_sb3.py']
    expected=json.loads((ROOT/'config/control_protocol.json').read_text(encoding='utf8'))['source_sha256']
    assert {k:v['sha256'] for k,v in sources.items()}==expected
    if receipt['kind']=='formal':
        protocol=binding(receipt['formal_protocol']);p=ref.read(protocol)
        assert p['status']=='frozen' and p['steps_per_seed']==receipt['steps']==1200000
        assert p['method']==receipt['method'] and receipt['seed'] in p['seeds']==[0,1,2,3]
        assert p['source_sha256']=={k:v['sha256'] for k,v in receipt['source'].items()}
        if receipt['method'].startswith('refined'):assert p['teacher_manifest_sha256']==receipt['teacher']['dp_manifest']['sha256']
    data=load_data(receipt);cfg=receipt['environment'];grid=np.asarray(cfg['action_grid'],np.float64)
    assert cfg['episode_length']==N and cfg['battery_capacity']==75. and cfg['initial_soc']==.5 and cfg['time_step']==.25
    assert cfg['mode']=='migration' and cfg['observation_mode']=='normalized15' and cfg['reward_mode']=='quadratic_inventory'
    assert cfg['smooth_weight']==1. and cfg['economic_weight']==.02 and cfg['soc_weight']==cfg['soh_weight']==0.
    assert np.array_equal(grid,np.arange(-10,11)/10.) and receipt['training']['dropout']==0.
    files={k:binding(v) for k,v in receipt['files'].items()};assert all(path==run/name for name,path in files.items())
    states={}
    for name in ['initial.pt','final.pt']:
        states[name]=torch.load(files[name],map_location='cpu',weights_only=True);ref.weights_valid(states[name])
    changed=[k for k in states['initial.pt'] if not torch.equal(states['initial.pt'][k],states['final.pt'][k])]
    assert changed
    snapshot=torch.load(files['optimizer_target.pt'],map_location='cpu',weights_only=True)
    ref.weights_valid(snapshot['target']);assert snapshot['updates']==receipt['updates']
    optimizer=snapshot['optimizer'];assert len(optimizer['state'])==12
    for state in optimizer['state'].values():
        assert int(state['step'].item())==receipt['updates']
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    for group in optimizer['param_groups']:
        assert group['lr']==.0003 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
    teacher=TeacherReference(receipt,data)
    training=training_audit(run,receipt,data,teacher)
    results=[]
    for filename,metric,neutral,weights in [('initial_trace.npz','initial',False,'initial.pt'),('trace.npz','free_terminal',False,'final.pt'),('neutral_trace.npz','equal_terminal_evaluation',True,'final.pt')]:
        row=ref.verify_trace(files[filename],data,grid,receipt[metric],neutral,states[weights])
        row['model_sha256']=sha(files[weights]);row['phase']=metric;results.append(row)
    return dict(status='all_pass',run=str(run),kind=receipt['kind'],method=receipt['method'],seed=receipt['seed'],
        receipt_sha256=sha(receipt_path),source_sha256=sha(__file__),reference_core_sha256=CORE_SHA,
        bound_experiment_sources=sources,real_training_steps=receipt['steps'],TD_updates=receipt['updates'],
        checked_training_transitions=training['transition_rows'],checked_replay_sample_links=training['replay_sample_links'],
        teacher_stage_counts=training['teacher_stage_counts'],teacher_provenance=teacher.verified,training=training,
        checked_evaluation_windows=3,checked_evaluation_rows=3*N,independent_Q_rows=3*N,results=results,
        optimizer_updates_each_parameter=receipt['updates'],changed_online_tensors=changed,
        single_network_parameters=36502,no_teacher_required_for_checked_inference=True,new_training_steps=0,
        limits=['Free evaluation actions are independently reproduced by one15-input/21-output neural state_dict; neutral adds a separately checked execution wrapper',
                'Teacher supervision is part of training for transfer methods; these are not teacher-free RL discoveries',
                'Optimizer/target snapshot lacks complete replay priorities and complete RNG metadata, so exact resumability is not certified',
                'Independent teacher-label coverage and missing soft-logit/gradient history limits are explicit in training audit',
                'Same fixed archived source prefix for training/evaluation; no held-out generalization claim'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run).resolve();output=Path(args.output).resolve()
    assert not output.exists(),'Refuse overwrite'
    assert sha(ref.__file__)==CORE_SHA
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=verify(run);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as file:json.dump(result,file,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','real_training_steps','TD_updates','checked_evaluation_rows','independent_Q_rows']}))


if __name__=='__main__':main()
