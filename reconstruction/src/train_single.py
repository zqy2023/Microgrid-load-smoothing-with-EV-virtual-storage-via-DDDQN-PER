"""Train one 15-input, 21-action DDDQN/PER per picked seed."""
import argparse
import json
import os
from pathlib import Path
import time

for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var,"1")

from evaluate_single import ROOT, OLD_CONFIG, VARIANTS, RetunedEnv, evaluate, load_data, native, torch
from migration_env import MigrationEnv
from priority_learner import Learner, Replay, verify_sampler
from teaching_targets import TeachingTargets
import evaluate_single
import numpy as np
import priority_learner
import teaching_targets

STEPS=1200000
SEEDS=(0,1,2,3)
PHASE_END=80000
METHODS={
    "ensemble":dict(weight=4., refining=False),
    "refined":dict(weight=4., refining=True),
    "refined_strong":dict(weight=10., refining=True),
    "control":dict(weight=0., refining=False),
}


def sources():
    paths=(Path(__file__),Path(priority_learner.__file__),Path(teaching_targets.__file__),
           Path(evaluate_single.__file__),OLD_CONFIG)
    return {**native.source_bindings(),**{p.name:native.file_binding(p) for p in paths}}


class Journal:
    def __init__(self,folder):
        self.folder=folder
        self.folder.mkdir()
        self.entries=[]
        self.clear()

    def clear(self):
        self.obs=[]
        self.next_obs=[]
        self.action=[]
        self.reward=[]
        self.done=[]
        self.source=[]
        self.power=[]
        self.before=[]
        self.after=[]
        self.steps=[]
        self.indices=[]
        self.weights=[]
        self.td=[]
        self.imitation=[]
        self.stage=[]
        self.labels=[]

    def transition(self,obs,nxt,action,reward,done,info):
        self.obs.append(obs)
        self.next_obs.append(nxt)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.source.append(info["source_index"])
        self.power.append(info["power_kw"])
        self.before.append(info["energy_before_kwh"])
        self.after.append(info["energy_after_kwh"])

    def update(self,step,stage,info):
        ix,w,td,aux,label=info
        self.steps.append(step)
        self.indices.append(ix)
        self.weights.append(w)
        self.td.append(td)
        self.imitation.append(aux)
        self.stage.append(stage)
        self.labels.append(label)

    def flush(self,total):
        path=self.folder/("chunk_%05d.npz"%len(self.entries))
        np.savez_compressed(path,obs=np.asarray(self.obs,dtype=np.float32),
            next_obs=np.asarray(self.next_obs,dtype=np.float32),
            action=np.asarray(self.action,dtype=np.int8),
            reward=np.asarray(self.reward,dtype=np.float32),done=np.asarray(self.done,dtype=bool),
            source_index=np.asarray(self.source,dtype=np.int16),power_kw=np.asarray(self.power),
            energy_before=np.asarray(self.before),energy_after=np.asarray(self.after),
            update_steps=np.asarray(self.steps,dtype=np.int32),
            replay_indices=np.asarray(self.indices,dtype=np.int32).reshape(-1,32),
            is_weights=np.asarray(self.weights,dtype=np.float32).reshape(-1,32),
            td_loss=np.asarray(self.td,dtype=np.float32),
            imitation_loss=np.asarray(self.imitation,dtype=np.float32),
            teacher_stage=np.asarray(self.stage,dtype=np.int8),
            teacher_label=np.asarray(self.labels,dtype=np.int8).reshape(-1,32))
        self.entries.append(dict(begin_step=total-len(self.obs)+1,end_step=total,
                                 transitions=len(self.obs),updates=len(self.steps),
                                 **native.file_binding(path)))
        self.clear()


def require_formal(args,source):
    doc=json.loads(args.protocol.read_text(encoding="utf8"))
    assert doc["status"]=="frozen" and doc["steps_per_seed"]==STEPS
    assert args.steps==STEPS and args.seed in doc["seeds"]==list(SEEDS)
    assert args.method==doc["method"]
    assert {k:v["sha256"] for k,v in source.items()} == doc["source_sha256"]
    if METHODS[args.method]["refining"]:
        assert native.sha(args.teacher_manifest)==doc["teacher_manifest_sha256"]
    return native.file_binding(args.protocol)


def run(args):
    tracked=sources()
    protocol=require_formal(args,tracked) if args.kind=="formal" else None
    out=Path(args.output).resolve() if args.output else ROOT/"runs"/args.name
    out.mkdir(parents=True,exist_ok=False)
    native.seed_all(args.seed)
    rng=np.random.RandomState(args.seed)
    data=load_data()
    config=native.normalized_config(json.loads(OLD_CONFIG.read_text(encoding="utf8")),data)
    agent=native.make_agent(config)
    env=MigrationEnv(data,config["environment"])
    env.neutral=False
    parameters=METHODS[args.method]
    teacher=TeachingTargets(data,use_ensemble=bool(parameters["weight"]))
    if parameters["refining"]:
        teacher.attach_dp(args.teacher_manifest)
    learner=Learner(agent,Replay(),rng)
    torch.save(agent.q_network.state_dict(),out/"initial.pt")
    init=evaluate(env,agent,out/"initial_trace.npz")
    obs,_=env.reset(seed=args.seed)
    journal=Journal(out/"training")
    t0=time.perf_counter()
    for step in range(1,args.steps+1):
        epsilon=1-.95*min((step-1)/120000,1)
        action=learner.action(obs,epsilon)
        nxt,reward,term,trunc,info=env.step(action)
        done=term or trunc
        learner.replay.add(obs,action,reward,nxt,done,info["source_index"],
                           info["energy_before_kwh"])
        journal.transition(obs,nxt,action,reward,done,info)
        if step>=32 and (step-32)%4==0:
            stage=0 if not parameters["weight"] else 1
            if parameters["refining"] and step>PHASE_END:
                stage=2
            update=learner.update(teacher,stage,parameters["weight"])
            journal.update(step,stage,update)
        obs=nxt
        if done:
            if info["source_index"]!=17519:
                raise RuntimeError("Unexpected early terminal")
            obs,_=env.reset()
        if step%40000==0 or step==args.steps:
            journal.flush(step)
            elapsed=time.perf_counter()-t0
            line=dict(steps=step,updates=learner.updates,seconds=elapsed,
                      teacher_counts=teacher.counts)
            with (out/"progress.jsonl").open("a",encoding="utf8") as f:
                f.write(json.dumps(line,ensure_ascii=False)+"\n")
            print(json.dumps(line),flush=True)
        if args.kind=="formal" and step in (240000,600000,960000):
            torch.save(agent.q_network.state_dict(),out/("step_%07d.pt"%step))
    duration=time.perf_counter()-t0
    assert learner.updates==native.expected_updates(args.steps)
    assert sum(j["transitions"] for j in journal.entries)==args.steps
    assert sum(j["updates"] for j in journal.entries)==learner.updates
    assert tracked==sources()
    torch.save(agent.q_network.state_dict(),out/"final.pt")
    torch.save(dict(target=agent.target_network.state_dict(),optimizer=agent.optimizer.state_dict(),
                    rng=learner.rng.get_state()[1].tolist(),updates=learner.updates),out/"optimizer_target.pt")
    free=evaluate_single.evaluator(agent.q_network.state_dict(),data,out/"trace.npz")
    equal=evaluate_single.evaluator(agent.q_network.state_dict(),data,out/"neutral_trace.npz",neutral=True)
    files=["initial.pt","final.pt","optimizer_target.pt","initial_trace.npz","trace.npz",
           "neutral_trace.npz","progress.jsonl"]
    files += [p.name for p in out.glob("step_*.pt")]
    result=dict(status="complete",kind=args.kind,method=args.method,
        seed=args.seed,steps=args.steps,updates=learner.updates,train_seconds=duration,
        source=tracked,inputs=data.manifest()["inputs"],environment=config["environment"],
        training=config["training"],native_parameters=native.NATIVE_PARAMETERS,
        teacher=teacher.bindings,teacher_counts=teacher.counts,
        imitation_weight=parameters["weight"],
        ensemble_only_through_step=PHASE_END if parameters["refining"] else None,
        formal_protocol=protocol,
        files={name:native.file_binding(out/name) for name in files},
        replay_capacity=learner.replay.capacity,chunks=journal.entries,
        initial=init,free_terminal=free,equal_terminal_evaluation=equal,
        updates_note="After adding the current transition; begin at step 32; every four; PER with replacement",
        inference="Single native 15D/21-output DuelingQNetwork state_dict; largest Q output; teacher used for auxiliary training loss only.",
    )
    native.write_json(out/"receipt.json",result)
    print(json.dumps(dict(name=args.name,free=free,equal=equal)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sampler-check",action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--kind",choices=("engineering","pilot","formal"),default="formal")
    parser.add_argument("--steps",type=int,default=STEPS)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--teacher-manifest",type=Path)
    args=parser.parse_args()
    if args.sampler_check:
        print(json.dumps(verify_sampler()),flush=True)
        return
    if not args.name or args.method is None or args.seed is None or args.steps<32:
        parser.error("Training requires a run name, method, seed and at least 32 interactions")
    if Path(args.name).name!=args.name or args.name in (".",".."):
        parser.error("Use a local run name")
    if args.kind=="formal" and not args.protocol:
        parser.error("Formal runs require the frozen protocol")
    if METHODS[args.method]["refining"] and args.teacher_manifest is None:
        parser.error("A refined method requires a teacher manifest")
    run(args)


if __name__=="__main__":
    main()

