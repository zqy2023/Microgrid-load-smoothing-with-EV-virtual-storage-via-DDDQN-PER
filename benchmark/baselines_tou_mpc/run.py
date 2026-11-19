"""Run the task-book ToU and MPC controls against the sealed DDDQN experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_PACKAGE = HERE.parents[1] / "00_归档重建可复现版"
PHASES = ("free_terminal", "equal_terminal_evaluation")
BASELINES = ("tou", "mpc")
PERFORMANCE = ("pooled_pct", "daily_pct", "saving_pct")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path, relative_to):
    return {
        "path": Path(os.path.relpath(path, relative_to)).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class TimeOfUse:
    """Fixed repeating price bands and full-power inventory requests."""

    def __init__(self, price, grid, steps_per_day, calibration_days, band_slots):
        calibration = price[: calibration_days * steps_per_day]
        profile = np.median(calibration.reshape(-1, steps_per_day), axis=0)
        windows = (
            np.arange(steps_per_day)[:, None] + np.arange(band_slots)
        ) % steps_per_day
        averages = profile[windows].mean(axis=1)
        low = windows[np.argmin(averages)]
        remaining = ~np.isin(windows, low).any(axis=1)
        peak = windows[np.argmax(np.where(remaining, averages, -np.inf))]
        self.daily_actions = np.full(steps_per_day, np.argmin(np.abs(grid)), np.int16)
        self.daily_actions[low] = np.argmax(grid)
        self.daily_actions[peak] = np.argmin(grid)
        self.description = {
            "method": "Fixed daily hour bands calibrated on the first 28 complete archive days",
            "calibration_days": calibration_days,
            "band_slots": band_slots,
            "step_index_charge": low.tolist(),
            "step_index_discharge": peak.tolist(),
            "profile": "Slotwise price median, lowest fixed circular band and highest nonoverlapping circular band",
            "evaluation_billing": "Same signed archived interval prices as all other controls",
        }

    def action(self, t, _energy):
        return self.daily_actions[t % len(self.daily_actions)]


class PredictiveControl:
    """A finite-horizon, receding dynamic program over the existing command grid."""

    def __init__(self, data, config, rows, horizon):
        grid = np.asarray(config["action_grid"], dtype=np.float64)
        order = np.argsort(np.abs(grid), kind="stable")
        self.dt = config["time_step"]
        self.power_step = grid[len(grid) // 2 + 1]
        self.energy_step = self.dt * self.power_step
        self.deltas = np.rint(grid[order] / self.power_step).astype(int)
        assert np.allclose(self.deltas * self.power_step, grid[order], rtol=0, atol=1e-12)
        self.action_indices = order
        self.max_delta = int(np.abs(self.deltas).max())
        self.max_state = round(config["battery_capacity"] / self.energy_step)
        assert abs(self.max_state * self.energy_step - config["battery_capacity"]) < 1e-10
        self.initial_energy = config["battery_capacity"] * config["initial_soc"]
        assert config["reward_mode"] == "quadratic_inventory"
        states = np.arange(self.max_state + 1)
        self.transitions = np.clip(states[:, None] + self.deltas, 0, self.max_state)
        self.actual_power = (self.transitions - states[:, None]) * self.power_step
        self.targets = data.train_stats["mean_load"] + (
            self.initial_energy - states * self.energy_step
        ) / (self.dt * 384)
        self.variance_scale = config["smooth_weight"] / data.train_stats["var_load"]
        self.price_scale = (
            config["economic_weight"] * self.dt
            / (1000 * data.train_stats["mean_base_step_bill"])
        )
        self.data = data
        self.rows = rows
        self.available = data.ev[:rows] > 0.1
        assert 1 <= horizon <= 6
        self.horizon = horizon
        self.description = {
            "method": "Finite-horizon discrete dynamic programming, first action only, replan at every feasible slot",
            "horizon_steps": horizon,
            "objective": "Same quadratic inventory smoothing and signed-bill economic stage objective as sealed training environment",
            "price_preview": "t through t+5 recorded prices, same forecast slots exposed to archived DDDQN",
            "load_preview": "Recorded current and next loads as in archived DDDQN; next-load persistence for later steps",
            "availability_preview": "Current feasible gate is known to planner, held constant over horizon; the DDDQN network has no availability feature",
            "terminal": "No extra optimization terminal reward; equal-energy run uses the shared archived environment wrapper outside the planner",
            "tie_break": "Smallest absolute command, then lower grid index",
        }

    def action(self, t, energy):
        if not self.available[t]:
            return self.action_indices[0]
        horizon = min(self.horizon, self.rows - t)
        load = self.data.load
        price = self.data.price
        center = round(energy / self.energy_step)
        assert abs(center * self.energy_step - energy) < 1e-7
        first_state = max(0, center - horizon * self.max_delta)
        last_state = min(self.max_state, center + horizon * self.max_delta)
        cost_to_go = np.zeros(last_state - first_state + 1)
        for j in reversed(range(horizon)):
            start = max(0, center - j * self.max_delta)
            stop = min(self.max_state, center + j * self.max_delta) + 1
            power = self.actual_power[start:stop]
            stage_cost = (
                self.variance_scale * (load[t + min(j, 1)] + power - self.targets[start:stop, None]) ** 2
                + self.price_scale * price[t + j] * power
            )
            objective = stage_cost + cost_to_go[self.transitions[start:stop] - first_state]
            if j == 0:
                return self.action_indices[objective[0].argmin()]
            cost_to_go = objective.min(axis=1)
            first_state = start


def evaluate(env, policy, metrics, phases, path):
    rows, neutral = phases
    env.reset()
    actions = np.empty(rows, np.int16)
    power = np.empty(rows)
    command = np.empty(rows)
    applied = np.empty(rows)
    energy = np.empty(rows + 1)
    energy[0] = env.battery_capacity * env.initial_soc
    for t in range(rows):
        actions[t] = policy.action(t, energy[t])
        _, _, done, truncated, info = env.step(actions[t])
        assert not truncated and done == (t == rows - 1)
        power[t] = info["power_kw"]
        command[t] = info["command_power_kw"]
        applied[t] = info.get("viability_adjusted_command_kw", command[t])
        energy[t + 1] = info["energy_after_kwh"]
    load = env.load[:rows]
    price = env.price[:rows]
    available = env.ev[:rows] > 0.1
    net = load + power
    result = metrics(load, price, power, energy)
    result.update(
        unavailable_slots=int((~available).sum()),
        viability_adjusted_steps=int((np.abs(command - applied) > 1e-9).sum()),
        availability_evaluation="archived EV demand > 0.1",
    )
    assert not neutral or abs(result["energy_change_kwh"]) < 1e-8
    np.savez_compressed(
        path,
        actions=actions,
        power_kw=power,
        command_kw=command,
        applied_command_kw=applied,
        energy_kwh=energy,
        load_kw=load,
        price=price,
        available=available,
        net_kw=net,
        step_bill=net * price / 4000,
    )
    return result


def references(package, config):
    expected = read_json(package / "reference" / "expected.json")
    output = {}
    for family in ("control", "transfer"):
        runs = [run for run in expected["runs"] if run["family"] == family]
        assert len(runs) == 4 and all(
            run["environment"]["action_grid"] == config["action_grid"] for run in runs
        )
        output[family] = {}
        for phase in PHASES:
            output[family][phase] = {
                key: float(np.mean([run[phase][key] for run in runs]))
                for key in PERFORMANCE + ("energy_change_kwh", "final_energy_kwh", "bill")
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    output = args.output.resolve()
    if output == package or package in output.parents:
        parser.error("Choose a new output directory outside the sealed package.")
    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    sys.path.insert(0, str(package / "src"))
    from evaluation import N, RetunedEnv, VARIANTS, metrics
    from migration_env import MigrationEnv, load_data
    from portable_runtime import normalized_config

    data = load_data()
    config = normalized_config(read_json(package / "config" / "full.json"), data)["environment"]
    rows = protocol["archive_rows"]
    assert rows == N == config["episode_length"]
    grid = np.array(config["action_grid"])
    policies = {
        "tou": TimeOfUse(data.price, grid, round(24 / config["time_step"]), **{
            "calibration_days": protocol["tou"]["calibration_complete_days"],
            "band_slots": protocol["tou"]["band_slots"],
        }),
        "mpc": PredictiveControl(data, config, rows, protocol["mpc"]["horizon_steps"]),
    }
    comparisons = references(package, config)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "capstone_tou_mpc_baselines_v1",
        "protocol": protocol,
        "environment": config,
        "input_manifest": data.manifest(),
        "archive_window": {
            "rows": rows,
            "first_timestamp": str(data.timestamps[0]),
            "last_timestamp": str(data.timestamps[rows - 1]),
            "daily_complete_days": rows // 96,
            "daily_excluded_tail_rows": rows % 96,
        },
        "reference_dddqn": comparisons,
        "method_details": {name: policy.description for name, policy in policies.items()},
        "methods": {},
        "traces": {},
        "bindings": {
            "runner": binding(Path(__file__).resolve(), output),
            "protocol": binding(protocol_path, output),
            "archive_manifest": binding(package / "package_manifest.json", output),
            "archive_reference": binding(package / "reference" / "expected.json", output),
        },
    }
    for name, policy in policies.items():
        summary["methods"][name] = {}
        summary["traces"][name] = {}
        (output / name).mkdir()
        for phase in PHASES:
            neutral = phase == "equal_terminal_evaluation"
            env = (
                RetunedEnv(data, config, VARIANTS["C"], neutral=True)
                if neutral else MigrationEnv(data, config)
            )
            path = output / name / (phase + ".npz")
            start = time.perf_counter()
            result = evaluate(env, policy, metrics, (rows, neutral), path)
            summary["methods"][name][phase] = result
            summary["traces"][name][phase] = binding(path, output)
            print(name, phase, {key: round(result[key], 6) for key in PERFORMANCE},
                  round(time.perf_counter() - start, 2), "seconds", flush=True)
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
