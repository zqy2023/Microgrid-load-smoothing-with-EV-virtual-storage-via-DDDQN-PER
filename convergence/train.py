"""Train the paired N13 controllers while preserving dense checkpoints and CPU clocks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns, process_time_ns

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEALED = ROOT / "reconstruction"
sys.path.insert(0, str(SEALED / "src"))
sys.path.insert(0, str(ROOT / "ablation"))

import numpy as np
import torch

import portable_runtime as native
from migration_env import MigrationEnv, load_data
from priority_learner import Learner, Replay
from teaching_targets import TeachingTargets
from train_single import Journal, PHASE_END, sources as sealed_sources
from variants import UniformReplay

PROTOCOL = HERE / "protocol.json"
PUBLISHED = SEALED / "config" / "transfer_protocol.json"
TEACHER = SEALED / "config" / "teacher.json"


def source_bindings():
    own = (Path(__file__), PROTOCOL, ROOT / "ablation" / "variants.py")
    return {**sealed_sources(), **{path.name: native.file_binding(path) for path in own}}


def approved_configuration(bindings):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    original = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    if any(bindings[name]["sha256"] != expected for name, expected in original["source_sha256"].items()):
        raise ValueError("Sealed training code or configuration changed")
    if (native.sha(TEACHER) != original["teacher_manifest_sha256"]
            or protocol["seeds"] != original["seeds"]
            or protocol["interactions_per_seed"] != original["steps_per_seed"]):
        raise ValueError("Published two-stage targets or training budget changed")
    scheduled = protocol["evaluation"]["sample_steps"]
    if len(scheduled) != 84 or sorted(set(scheduled)) != scheduled or scheduled[-1] != 1200000:
        raise ValueError("Invalid fixed evaluation schedule")
    return protocol


def same_weights(actual, reference_path):
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if actual.keys() != reference.keys():
        raise ValueError(f"Weight architecture mismatch: {reference_path}")
    maximum_error = max(float((value - reference[name]).abs().max().item()) for name, value in actual.items())
    return maximum_error == 0.0, maximum_error


def run(args):
    sources = source_bindings()
    protocol = approved_configuration(sources)
    steps = protocol["interactions_per_seed"] if args.mode == "train" else 96
    save_points = set(protocol["evaluation"]["sample_steps"] if args.mode == "train" else (0, 32, 96))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir()
    native.seed_all(args.seed)
    torch.set_num_interop_threads(1)
    data = load_data()
    config = native.normalized_config(json.loads(native.OLD_CONFIG.read_text(encoding="utf-8")), data)
    agent = native.make_agent(config)
    rng = np.random.RandomState(args.seed)
    learner = Learner(agent, Replay() if args.arm == "per" else UniformReplay(), rng)
    env = MigrationEnv(data, config["environment"])
    env.neutral = False
    teacher = TeachingTargets(data, use_ensemble=True)
    teacher.attach_dp(TEACHER)
    base_weights = SEALED / "models" / f"transfer_seed{args.seed}"
    initial_match, _ = same_weights(agent.q_network.state_dict(), base_weights / "initial.pt")
    if not initial_match:
        raise ValueError("Paired initial network differs from published seed")

    checkpoints = []
    cpu_ns = 0
    wall_ns = 0

    def save(step):
        path = checkpoint_dir / f"step_{step:07d}.pt"
        torch.save(agent.q_network.state_dict(), path)
        checkpoints.append(dict(step=step, path=path.relative_to(out).as_posix(), sha256=native.sha(path),
                                bytes=path.stat().st_size, cpu_seconds=cpu_ns / 1e9, wall_seconds=wall_ns / 1e9))

    save(0)
    obs, _ = env.reset(seed=args.seed)
    journal = Journal(out / "training")
    cpu_start = process_time_ns()
    wall_start = perf_counter_ns()
    for step in range(1, steps + 1):
        epsilon = 1 - 0.95 * min((step - 1) / 120000, 1)
        action = learner.action(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        learner.replay.add(obs, action, reward, next_obs, done, info["source_index"], info["energy_before_kwh"])
        journal.transition(obs, next_obs, action, reward, done, info)
        if step >= 32 and (step - 32) % 4 == 0:
            stage = 1 if step <= PHASE_END else 2
            journal.update(step, stage, learner.update(teacher, stage, 4.0))
        obs = next_obs
        if done:
            if info["source_index"] != 17519:
                raise RuntimeError("Unexpected early episode boundary")
            obs, _ = env.reset()

        snapshot = step in save_points
        flush = step % 40000 == 0 or step == steps
        if snapshot or flush:
            cpu_ns += process_time_ns() - cpu_start
            wall_ns += perf_counter_ns() - wall_start
            if snapshot:
                save(step)
            if flush:
                journal.flush(step)
                record = dict(steps=step, updates=learner.updates, seconds=wall_ns / 1e9,
                              cpu_seconds=cpu_ns / 1e9, teacher_counts=teacher.counts)
                with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            cpu_start = process_time_ns()
            wall_start = perf_counter_ns()

    assert learner.updates == native.expected_updates(steps)
    assert sum(chunk["transitions"] for chunk in journal.entries) == steps
    assert sum(chunk["updates"] for chunk in journal.entries) == learner.updates
    assert [entry["step"] for entry in checkpoints] == sorted(save_points)
    assert teacher.counts["1"]["batches"] == native.expected_updates(min(PHASE_END, steps))
    assert teacher.counts["2"]["batches"] == learner.updates - teacher.counts["1"]["batches"]
    assert source_bindings() == sources
    endpoint = None
    error = None
    reference_binding = None
    if args.mode == "train":
        reference = (base_weights / "final.pt" if args.arm == "per" else
                     ROOT / "ablation" / "models" / f"no_per_seed{args.seed}" / "final.pt")
        endpoint, error = same_weights(agent.q_network.state_dict(), reference)
        reference_binding = native.file_binding(reference)
    torch.save(dict(target=agent.target_network.state_dict(), optimizer=agent.optimizer.state_dict(),
                    updates=learner.updates), out / "optimizer_target.pt")
    receipt = dict(
        schema="n13_convergence_train_v1", status="complete",
        kind="formal" if args.mode == "train" else "engineering",
        method="refined", arm=args.arm, seed=args.seed, steps=steps, updates=learner.updates,
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sources["protocol.json"]["sha256"], source=sources,
        checkpoints=checkpoints, endpoint_weight_match=endpoint, endpoint_max_abs_error=error,
        endpoint_reference=reference_binding, initial_weight_match=initial_match, thread_count=torch.get_num_threads(),
        torch_version=str(torch.__version__), training_process_cpu_seconds=cpu_ns / 1e9,
        training_chunks_wall_seconds=wall_ns / 1e9,
        inputs=data.manifest()["inputs"], environment=config["environment"], training=config["training"],
        teacher=teacher.bindings, teacher_counts=teacher.counts, imitation_weight=4.0,
        ensemble_only_through_step=PHASE_END, replay_capacity=learner.replay.capacity, chunks=journal.entries,
        optimizer_target=native.file_binding(out / "optimizer_target.pt"),
        progress=native.file_binding(out / "progress.jsonl"),
    )
    native.write_json(out / "receipt.json", receipt)
    print(json.dumps(dict(arm=args.arm, seed=args.seed, steps=steps, updates=learner.updates,
                          checkpoints=len(checkpoints), endpoint_weight_match=endpoint,
                          process_cpu_seconds=cpu_ns / 1e9)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "smoke"))
    parser.add_argument("--arm", choices=("per", "uniform"), required=True)
    parser.add_argument("--seed", type=int, choices=range(4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

