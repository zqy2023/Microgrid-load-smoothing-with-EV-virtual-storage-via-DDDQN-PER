"""Evaluate one saved 21-action network on the archived environment."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
from portable_runtime import OLD_CONFIG
from evaluation import RetunedEnv,VARIANTS,evaluate,load_data,torch
import portable_runtime as native
from migration_env import MigrationEnv


def evaluator(weights, data, trace, neutral=False):
    config = native.normalized_config(json.loads(OLD_CONFIG.read_text(encoding="utf8")),data)
    network = native.make_agent(config)
    network.q_network.load_state_dict(weights)
    if neutral:
        env=RetunedEnv(data,config["environment"],VARIANTS["C"],neutral=True)
    else:
        env=MigrationEnv(data,config["environment"])
        env.neutral=False
    return evaluate(env,network,trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    native.seed_all(0)
    data=load_data()
    model=torch.load(args.model,map_location="cpu",weights_only=True)
    baseline=evaluator(model,data,args.output/"trace.npz")
    equal=evaluator(model,data,args.output/"neutral_trace.npz",neutral=True)
    native.write_json(args.output/"receipt.json",dict(status="complete",
        source=native.file_binding(__file__), model=native.file_binding(args.model),
        free_terminal=baseline, equal_terminal_evaluation=equal,
        traces=[native.file_binding(args.output/name) for name in ("trace.npz","neutral_trace.npz")],
        implementation="One plain DuelingQNetwork state_dict. Greedy over the 21 model scores.",
        inputs=data.manifest()["inputs"],
    ))
    print(json.dumps(dict(free=baseline,neutral=equal)),flush=True)


if __name__=="__main__":
    main()

