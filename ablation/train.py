"""Four-seed, full-budget component ablations for the published DDDQN/PER recipe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parent.parent
SEALED = ROOT / "reconstruction"
sys.path.insert(0, str(SEALED / "src"))

import numpy as np
import torch

from evaluate_single import evaluate, RetunedEnv, VARIANTS as TERMINAL_SETTINGS
from migration_env import load_data, MigrationEnv
import portable_runtime as native
from teaching_targets import TeachingTargets
from train_single import Journal, sources as common_sources, PHASE_END, STEPS, SEEDS

from variants import build_learner, VARIANTS

TEACHER_MANIFEST = SEALED / "config" / "teacher.json"
REFERENCE_PROTOCOL = SEALED / "config" / "transfer_protocol.json"
BASE_COMMIT = "e190821886143f70af46437d52db749bc1afc06a"


def source_hashes():
    own = (Path(__file__), Path(__file__).with_name("variants.py"))
    return {**common_sources(), **{p.name: native.file_binding(p) for p in own}}


def validate_shared_protocol(sources):
    protocol = json.loads(REFERENCE_PROTOCOL.read_text(encoding="utf-8"))
    for name, digest in protocol["source_sha256"].items():
        if sources[name]["sha256"] != digest:
            raise ValueError(f"Published training source differs: {name}")
    if native.sha(TEACHER_MANIFEST) != protocol["teacher_manifest_sha256"]:
        raise ValueError("Published two-stage training targets differ")
    if protocol["steps_per_seed"] != STEPS or protocol["seeds"] != list(SEEDS):
        raise ValueError("Published seed list or budget differs")


def validate_initial_weights(agent, variant, seed):
    initial = SEALED / "models" / f"transfer_seed{seed}" / "initial.pt"
    expected = torch.load(initial, map_location="cpu", weights_only=True)
    actual = agent.q_network.state_dict()
    keys = (key for key in expected if key.startswith("feature_layers.")) if variant == "no_dueling" else expected
    for key in keys:
        if not torch.equal(actual[key], expected[key]):
            raise ValueError(f"Paired initialization differs: {variant}/{seed}/{key}")
    return native.file_binding(initial)


def run(args):
    tracked = source_hashes()
    validate_shared_protocol(tracked)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    native.seed_all(args.seed)
    torch.set_num_interop_threads(1)

    data = load_data()
    cfg = native.normalized_config(json.loads(native.OLD_CONFIG.read_text(encoding="utf-8")), data)
    agent, learner = build_learner(cfg, args.variant, np.random.RandomState(args.seed))
    initial_binding = validate_initial_weights(agent, args.variant, args.seed)
    env = MigrationEnv(data, cfg["environment"])
    env.neutral = False
    teacher = TeachingTargets(data, use_ensemble=True)
    teacher.attach_dp(TEACHER_MANIFEST)

    if args.mode == "smoke":
        # Exercise both target sources on a valid replay row before the short run.
        # The measured journal below always uses the published 80,000-step boundary.
        obs, _ = env.reset(seed=args.seed)
        for stage in (1, 2):
            scores, available = teacher.scores(stage, torch.from_numpy(obs.reshape(1, -1)),
                                               np.array([0]), np.array([37.5]))
            assert scores.shape == (1, 21) and torch.isfinite(scores).all()
            assert available.shape == (1,)
            teacher.counts[str(stage)] = dict(batches=0, rows=0, labels=0)

    torch.save(agent.q_network.state_dict(), out / "initial.pt")
    initial = evaluate(env, agent, out / "initial_trace.npz")
    obs, _ = env.reset(seed=args.seed)
    journal = Journal(out / "training")
    before_training = perf_counter()
    for step in range(1, args.steps + 1):
        epsilon = 1 - 0.95 * min((step - 1) / 120000, 1)
        action = learner.action(obs, epsilon)
        nxt, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        learner.replay.add(obs, action, reward, nxt, done, info["source_index"], info["energy_before_kwh"])
        journal.transition(obs, nxt, action, reward, done, info)
        if step >= 32 and (step - 32) % 4 == 0:
            stage = 1 if step <= PHASE_END else 2
            update = learner.update(teacher, stage, 4.0)
            journal.update(step, stage, update)
        obs = nxt
        if done:
            if info["source_index"] != 17519:
                raise RuntimeError("Unexpected early terminal")
            obs, _ = env.reset()
        if step % 40000 == 0 or step == args.steps:
            journal.flush(step)
            row = dict(steps=step, updates=learner.updates, seconds=perf_counter() - before_training,
                       teacher_counts=teacher.counts)
            with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if args.mode == "train" and step in (240000, 600000, 960000):
            torch.save(agent.q_network.state_dict(), out / f"step_{step:07d}.pt")

    train_seconds = perf_counter() - before_training
    assert learner.updates == native.expected_updates(args.steps)
    assert sum(chunk["transitions"] for chunk in journal.entries) == args.steps
    assert sum(chunk["updates"] for chunk in journal.entries) == learner.updates
    assert teacher.counts["1"]["batches"] == native.expected_updates(min(PHASE_END, args.steps))
    assert teacher.counts["2"]["batches"] == learner.updates - teacher.counts["1"]["batches"]
    assert tracked == source_hashes()

    torch.save(agent.q_network.state_dict(), out / "final.pt")
    torch.save(dict(target=agent.target_network.state_dict(), optimizer=agent.optimizer.state_dict(),
                    rng=learner.rng.get_state()[1].tolist(), updates=learner.updates),
               out / "optimizer_target.pt")
    free = evaluate(env, agent, out / "trace.npz")
    neutral_env = RetunedEnv(data, cfg["environment"], TERMINAL_SETTINGS["C"], neutral=True)
    equal = evaluate(neutral_env, agent, out / "neutral_trace.npz")
    files = ["initial.pt", "final.pt", "optimizer_target.pt", "initial_trace.npz", "trace.npz",
             "neutral_trace.npz", "progress.jsonl"]
    files += [p.name for p in sorted(out.glob("step_*.pt"))]
    note = "Uniform replay with replacement; unit importance weights" if args.variant == "no_per" else "PER with replacement"
    result = dict(
        status="complete", kind="formal" if args.mode == "train" else "engineering",
        method="refined", ablation=args.variant, paired_reference_commit=BASE_COMMIT,
        seed=args.seed, steps=args.steps, updates=learner.updates, train_seconds=train_seconds,
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        source=tracked, inputs=data.manifest()["inputs"], environment=cfg["environment"],
        training=cfg["training"], native_parameters=native.NATIVE_PARAMETERS,
        teacher=teacher.bindings, teacher_counts=teacher.counts, imitation_weight=4.0,
        ensemble_only_through_step=PHASE_END, formal_protocol=native.file_binding(REFERENCE_PROTOCOL),
        paired_initial_weights=initial_binding,
        network_parameters=sum(p.numel() for p in agent.q_network.parameters()),
        files={name: native.file_binding(out / name) for name in files},
        replay_capacity=learner.replay.capacity, chunks=journal.entries, initial=initial,
        free_terminal=free, equal_terminal_evaluation=equal,
        updates_note=f"After adding the current transition; begin at step 32; every four; {note}",
        inference="One 15-input Q network; greedy over 21 action scores; no auxiliary training targets at inference.",
    )
    native.write_json(out / "receipt.json", result)
    print(json.dumps(dict(variant=args.variant, seed=args.seed, steps=args.steps, equal_terminal=equal)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "smoke"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.steps = STEPS if args.mode == "train" else 96
    run(args)


if __name__ == "__main__":
    main()

