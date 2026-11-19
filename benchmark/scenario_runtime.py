"""Evaluate frozen controllers on archived dates and EV-availability scenarios."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEALED = ROOT.parent / "reconstruction"
sys.path.insert(0, str(SEALED / "src"))
from evaluation import RetunedEnv, VARIANTS
from migration_env import MigrationEnv, load_data
from portable_runtime import make_agent


CASES = {
    "reference": ("2023-01-01", "2023-07-02 12:00", "MEDIUM", "training window"),
    "archive_autumn": ("2023-03-01", "2023-06-01", "MEDIUM", "in-training dates"),
    "jul_dec": ("2023-07-03", "2024-01-01", "MEDIUM", "unseen date labels; repeated load"),
    "late_winter": ("2023-07-03", "2023-09-01", "MEDIUM", "unseen date labels; repeated load"),
    "spring": ("2023-09-01", "2023-12-01", "MEDIUM", "unseen date labels; repeated load"),
    "early_summer": ("2023-12-01", "2024-01-01", "MEDIUM", "unseen date labels; repeated load"),
    "jul_dec_projected_low": ("2023-07-03", "2024-01-01", "LOW", "projected EV proxy"),
    "jul_dec_projected_high": ("2023-07-03", "2024-01-01", "HIGH", "projected EV proxy"),
    "jul_dec_ev_half": ("2023-07-03", "2024-01-01", "HALF", "one-factor stress test"),
    "jul_dec_ev_1_5x": ("2023-07-03", "2024-01-01", "ONE_AND_HALF", "one-factor stress test"),
}
PHASES = ("free_terminal", "equal_terminal_evaluation")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def projected_multipliers():
    data = json.loads((ROOT / "reference" / "ev_scenarios.json").read_text(encoding="utf-8"))
    return data["ev_multipliers"]


def select_window(data, start, end, multiplier):
    bounds = np.array([np.datetime64(start), np.datetime64(end)], dtype=data.timestamps.dtype)
    start_row, end_row = map(int, np.searchsorted(data.timestamps, bounds))
    assert np.array_equal(data.timestamps[[start_row, end_row]], bounds), bounds
    scenario_data = copy.copy(data)
    # Preserve five extra timestamped rows for the existing observations at the boundary.
    # Evaluation and terminal energy use exactly [start_row, end_row).
    indices = slice(start_row, end_row + 5)
    scenario_data.timestamps = data.timestamps[indices]
    scenario_data.load = data.load[indices]
    scenario_data.price = data.price[indices]
    scenario_data.ev = data.ev[indices] * multiplier
    return scenario_data, start_row, end_row


def make_env(data, size, base_config, neutral):
    config = dict(base_config, episode_length=size)
    if neutral:
        env = RetunedEnv(data, config, VARIANTS["C"], neutral=True)
        # The archived wrapper's original remaining-availability horizon is 17,520.
        # Bind it to the scenario window, leaving its execution logic intact.
        env.available = data.ev[:size] > 0.1
        env.remaining_availability = np.r_[np.cumsum(env.available[::-1])[::-1], 0]
    else:
        env = MigrationEnv(data, config)
        env.neutral = False
    assert env.max_steps == size
    return env


def summarise(load, price, power, energy):
    net_load = load + power
    full_day_rows = (len(load) // 96) * 96
    day_var = load[:full_day_rows].reshape(-1, 96).var(axis=1).mean()
    new_day_var = net_load[:full_day_rows].reshape(-1, 96).var(axis=1).mean()
    base_bill = float(np.dot(load, price) / 4000)
    bill = float(np.dot(net_load, price) / 4000)
    return {
        "pooled_pct": float(100 * (1 - net_load.var() / load.var())),
        "daily_pct": float(100 * (1 - new_day_var / day_var)),
        "saving_pct": float(-100 * np.dot(power, price) / 4000 / base_bill),
        "peak_to_valley_reduction_pct": float(100 * (1 - np.ptp(net_load) / np.ptp(load))),
        "baseline_bill": base_bill,
        "bill": bill,
        "final_energy_kwh": float(energy[-1]),
        "energy_change_kwh": float(energy[-1] - energy[0]),
        "energy_min_kwh": float(energy.min()),
        "energy_max_kwh": float(energy.max()),
        "power_min_kw": float(power.min()),
        "power_max_kw": float(power.max()),
        "full_days": full_day_rows // 96,
    }


@torch.inference_mode()
def run_controller(env, decide):
    count = env.max_steps
    power = np.empty(count)
    command = np.empty(count)
    applied = np.empty(count)
    energy = np.empty(count + 1)
    actions = np.empty(count, dtype=np.int16)
    observations, _ = env.reset()
    energy[0] = env.initial_soc * env.battery_capacity
    for t in range(count):
        action = decide(env, observations)
        observations, _, done, truncated, info = env.step(action)
        if truncated or done != (t == count - 1):
            raise RuntimeError(f"Unexpected episode boundary at step {t}")
        actions[t] = action
        power[t] = info["power_kw"]
        command[t] = info["command_power_kw"]
        applied[t] = info.get("viability_adjusted_command_kw", command[t])
        energy[t + 1] = info["energy_after_kwh"]
    available = env.ev[:count] > 0.1
    load, price = env.load[:count], env.price[:count]
    np.testing.assert_allclose(command, env.action_grid[actions], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.diff(energy), env.dt * power, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(
        power, np.where(available,
                        np.clip(applied, -energy[:-1] / env.dt, (env.battery_capacity - energy[:-1]) / env.dt),
                        0.0), atol=1e-9, rtol=0.0)
    assert np.max(np.abs(power)) <= env.max_power + 1e-9
    assert np.max(np.abs(applied)) <= env.max_power + 1e-9
    assert np.all((energy >= -1e-9) & (energy <= env.battery_capacity + 1e-9))
    if env.neutral:
        assert abs(energy[-1] - energy[0]) <= 1e-8
    metrics = summarise(load, price, power, energy)
    assert abs(metrics["bill"] - metrics["baseline_bill"] - np.dot(power, price) / 4000) < 1e-9
    metrics["viability_adjusted_steps"] = int(np.count_nonzero(np.abs(applied - command) > 1e-9))
    metrics["available_slots"] = int(np.count_nonzero(available))
    metrics["available_percent"] = float(100 * available.mean())
    return metrics, {
        "actions": actions, "power_kw": power, "command_kw": command,
        "applied_command_kw": applied, "energy_kwh": energy, "load_kw": load,
        "price": price, "available": available, "net_kw": load + power,
        "step_bill": (load + power) * price / 4000, "timestamps": env.timestamps[:count],
    }


def saved_policies(seed, config, methods):
    for method in methods:
        model_name = f"{method}_seed{seed}"
        file = SEALED / "models" / model_name / "final.pt"
        agent = make_agent(config)
        agent.q_network.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        network = agent.q_network.eval()

        def decide(_env, observation):
            scores = network(torch.from_numpy(observation).reshape(1, -1))
            return int(scores.argmax())

        yield model_name, file, decide


def check_reference(model_name, phase, trace):
    filename = "trace.npz" if phase == "free_terminal" else "neutral_trace.npz"
    with np.load(SEALED / "reference" / model_name / filename, allow_pickle=False) as original:
        for key, values in trace.items():
            if key != "timestamps":
                np.testing.assert_array_equal(values, original[key], err_msg=f"{model_name}: {key}")
