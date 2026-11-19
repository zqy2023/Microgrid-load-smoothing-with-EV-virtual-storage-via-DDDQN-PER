"""Train and evaluate a DDPG agent with the archived 21-command plant."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import gymnasium as gym
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[2] / "00_归档重建可复现版"
N = 17520
GRID = np.arange(-10, 11, dtype=np.float64) / 10
MIDPOINTS = (GRID[:-1] + GRID[1:]) * .5
DDPG_SPEC = {
    "learning_rate": .0003, "buffer_size": 200000, "learning_starts": 31,
    "batch_size": 32, "tau": .005, "gamma": .99, "train_freq": 4, "gradient_steps": 1,
    "hidden_actor": [128, 128], "hidden_critic": [128, 128], "critic_count": 1,
    "exploration_noise": {"distribution": "Gaussian", "initial_std": .50,
                          "final_std": .10, "linear_decay_steps": 120000},
    "training_replay": "uniform with replacement",
    "target_updates": "Polyak update of actor and critic after every gradient update",
    "teaching_targets": False,
}
CHECKPOINT_STEPS = (240000, 600000, 960000)


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_file(path):
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def command_index(power):
    return int(np.searchsorted(MIDPOINTS, power, side="left"))


class GridAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert np.array_equal(env.action_grid, GRID)
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)

    def action(self, action):
        return command_index(float(action[0]))


class EvidenceCallback(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.checkpoints = []
        self.reward_sum = 0.
        self.elapsed = 0.
        self.last_logged = 0

    def _on_training_start(self):
        self.start = time.perf_counter()

    def _on_step(self):
        step = self.num_timesteps
        self.reward_sum += float(self.locals["rewards"][0])
        if step in CHECKPOINT_STEPS:
            stem = f"checkpoint_{step}"
            self.model.save(self.directory / f"{stem}.zip")
            self.checkpoints.append({
                "environment_interactions": int(step),
                "gradient_updates_at_save": int(self.model._n_updates),
                "model": f"{stem}.zip",
            })
        sigma = .50 - .40 * min(step / 120000., 1.)
        self.model.action_noise._sigma[:] = sigma
        if step % 40000 == 0:
            self._log(step)
        return True

    def _log(self, step):
        elapsed = time.perf_counter() - self.start
        logged_values = self.model.logger.name_to_value
        row = {
            "interactions": int(step),
            "gradient_updates_before_next_update": int(self.model._n_updates),
            "elapsed_seconds": round(elapsed, 3),
            "mean_training_reward": self.reward_sum / (step - self.last_logged),
            "last_actor_loss": logged_values.get("train/actor_loss", ""),
            "last_critic_loss": logged_values.get("train/critic_loss", ""),
        }
        path = self.directory / "learning.csv"
        needs_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, row)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        self.reward_sum = 0.
        self.last_logged = step

    def _on_training_end(self):
        if self.num_timesteps > self.last_logged:
            self._log(self.num_timesteps)
        self.elapsed = time.perf_counter() - self.start


def build_model(env, seed):
    return DDPG(
        "MlpPolicy", GridAction(env),
        learning_rate=DDPG_SPEC["learning_rate"], buffer_size=DDPG_SPEC["buffer_size"],
        learning_starts=DDPG_SPEC["learning_starts"], batch_size=DDPG_SPEC["batch_size"],
        tau=DDPG_SPEC["tau"], gamma=DDPG_SPEC["gamma"], train_freq=DDPG_SPEC["train_freq"],
        gradient_steps=DDPG_SPEC["gradient_steps"],
        action_noise=NormalActionNoise(np.zeros(1, np.float32), np.full(1, .50, np.float32)),
        policy_kwargs=dict(net_arch=dict(pi=DDPG_SPEC["hidden_actor"], qf=DDPG_SPEC["hidden_critic"]),
                           n_critics=DDPG_SPEC["critic_count"]),
        verbose=0, seed=seed, device="cpu",
    )


def evaluate_actor(actor, environment, metrics, destination, neutral):
    obs, _ = environment.reset()
    observations = np.empty((N, 15), np.float32)
    continuous = np.empty(N, np.float32)
    actions = np.empty(N, np.int16)
    energy = np.empty(N + 1)
    energy[0] = 37.5
    power = np.empty(N)
    command = np.empty(N)
    applied = np.empty(N)
    actor.eval()
    with torch.inference_mode():
        for row in range(N):
            observations[row] = obs
            continuous[row] = actor(torch.from_numpy(obs[None, :])).item()
            action = command_index(continuous[row])
            actions[row] = action
            obs, _, end, truncate, info = environment.step(action)
            if truncate or end != (row == N - 1):
                raise RuntimeError(f"Unexpected episode boundary at {row}")
            power[row] = info["power_kw"]
            command[row] = info["command_power_kw"]
            applied[row] = info.get("viability_adjusted_command_kw", command[row])
            energy[row + 1] = info["energy_after_kwh"]
    load = environment.load[:N]
    price = environment.price[:N]
    available = environment.ev[:N] > .1
    np.testing.assert_array_equal(command, GRID[actions])
    np.testing.assert_allclose(np.diff(energy), .25 * power, atol=1e-9, rtol=0)
    assert np.isfinite(continuous).all() and np.max(np.abs(continuous)) <= 1.
    assert np.max(np.abs(power)) <= environment.max_power + 1e-9
    assert not power[~available].any() and energy.min() >= -1e-9 and energy.max() <= 75. + 1e-9
    if neutral:
        assert abs(energy[-1] - energy[0]) < 1e-8
    result = metrics(load, price, power, energy)
    result.update(unavailable_slots=int((~available).sum()),
                  viability_adjusted_steps=int(np.count_nonzero(np.abs(command-applied)>1e-9)),
                  availability_evaluation="archived EV demand > 0.1")
    np.savez_compressed(destination, observations=observations, actions=actions,
                        continuous_actor_kw=continuous, energy_kwh=energy, power_kw=power,
                        command_kw=command, applied_command_kw=applied, load_kw=load, price=price,
                        available=available, net_kw=load+power, step_bill=(load+power)*price/4000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--seed", type=int, choices=range(4), required=True)
    parser.add_argument("--steps", type=int, default=1200000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-model", type=Path,
                        help="Evaluate an already-trained DDPG model on both terminal protocols.")
    args = parser.parse_args()
    if args.steps < 32 or args.steps % 4 != 0:
        parser.error("--steps must be a multiple of 4 and at least 32")
    archive = args.archive.resolve()
    sys.path.insert(0, str(archive / "src"))
    import portable_runtime as runtime
    from migration_env import MigrationEnv, load_data
    from evaluation import RetunedEnv, VARIANTS, metrics

    torch.set_num_interop_threads(1)
    runtime.seed_all(args.seed)
    source_config = archive / "config" / "full.json"
    data = load_data()
    config = runtime.normalized_config(json.loads(source_config.read_text(encoding="utf-8")), data)
    config = config["environment"]
    env = MigrationEnv(data, config)
    assert np.array_equal(env.action_grid, GRID) and env.max_steps == N
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    if args.evaluate_model:
        model_path = args.evaluate_model.resolve(strict=True)
        model = DDPG.load(model_path, device="cpu")
        train = None
    else:
        model = build_model(env, args.seed)
        callback = EvidenceCallback(destination)
        plan = {
            "status": "training", "seed": args.seed, "planned_interactions": args.steps,
            "archived_input_manifest": data.manifest()["inputs"], "environment": config,
            "ddpg_parameters": DDPG_SPEC, "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__, "script": bind_file(Path(__file__)),
            "manifest_sha256": sha256(archive / "package_manifest.json"),
            "policy_parameter_count": {
                "actor": sum(p.numel() for p in model.actor.parameters()),
                "critic": sum(p.numel() for p in model.critic.parameters()),
            },
        }
        runtime.write_json(destination / "training_plan.json", plan)
        model.learn(total_timesteps=args.steps, callback=callback)
        assert model.num_timesteps == args.steps
        assert model._n_updates == (args.steps - 32)//4 + 1
        model_path = destination / "model.zip"
        model.save(model_path)
        train = {
            "steps": int(model.num_timesteps), "updates": int(model._n_updates),
            "elapsed_seconds": callback.elapsed, "checkpoints": callback.checkpoints,
            "full_budget": model.num_timesteps == 1200000,
        }
    trace = destination / "trace.npz"
    neutral_trace = destination / "neutral_trace.npz"
    env.neutral = False
    free = evaluate_actor(model.actor, env, metrics, trace, neutral=False)
    neutral_env = RetunedEnv(data, config, VARIANTS["C"], neutral=True)
    equal = evaluate_actor(model.actor, neutral_env, metrics, neutral_trace, neutral=True)
    record = {
        "status": "complete", "baseline": "DDPG with archived discrete execution grid",
        "seed": args.seed, "training": train, "free_terminal": free, "equal_terminal_evaluation": equal,
        "inputs": data.manifest()["inputs"], "traces": [bind_file(path) for path in (trace, neutral_trace)],
        "archive_manifest_sha256": sha256(archive / "package_manifest.json"),
        "model": bind_file(model_path), "source": bind_file(Path(__file__)),
        "path_root": "Artifact paths are relative to this receipt. With --evaluate-model, model lives in its input directory.",
        "scope": "Trains on the same fixed archived evaluation window. For evaluation the actor's "
                 "continuous proposal is rounded to the nearest of the 21 fixed commands. "
                 "An exact midpoint uses the lower command. The target Q sees the continuous proposal. "
                 "The equal-terminal case uses the shared wrapper with future availability.",
    }
    runtime.write_json(destination / "receipt.json", record)
    print(json.dumps({"seed": args.seed, "training": train, "free": free, "equal": equal}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()

