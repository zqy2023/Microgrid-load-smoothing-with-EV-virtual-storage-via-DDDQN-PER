"""Reproduce the thesis benchmark from the public archive and frozen controllers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.dont_write_bytecode = True

import numpy as np
import torch

from baselines_tou_mpc.run import PredictiveControl, TimeOfUse
from external_runtime import prepared_cases, trace_binding
from scenario_runtime import (CASES, PHASES, ROOT, SEALED, check_reference, checksum,
                              load_data, make_env, projected_multipliers, run_controller,
                              saved_policies, select_window, write_json)


REFERENCE = ROOT / "reference"
METHODS = ("dddqn_per", "tou", "mpc", "ddpg")
SEEDS = range(4)
PUBLISHED_METRICS = ("pooled_pct", "daily_pct", "saving_pct", "peak_to_valley_reduction_pct",
                     "energy_change_kwh", "available_percent")
METRIC_TOLERANCE = 1e-8


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files():
    for folder, filename in ((ROOT, "benchmark_manifest.json"), (SEALED, "package_manifest.json")):
        manifest = read_json(folder / filename)
        for name, binding in manifest["files"].items():
            file = folder / name
            if not file.is_file() or file.stat().st_size != binding["bytes"] or checksum(file) != binding["sha256"]:
                raise ValueError(f"Release file does not match its manifest: {file.relative_to(ROOT.parent)}")


def compare_metrics(actual, desired, context):
    if actual.keys() != desired.keys():
        raise ValueError(f"Metric schema differs in {context}.")
    for key, value in desired.items():
        if not math.isclose(value, actual[key], rel_tol=0.0, abs_tol=METRIC_TOLERANCE):
            raise ValueError(f"{context}, {key}: expected {value!r}, got {actual[key]!r}")


def action_hash(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<i2").tobytes()).hexdigest()


def run_labels(row):
    return row["scenario"], row["phase"]


def make_policy(method, seed, data, config):
    environment = config["environment"]
    grid = np.asarray(environment["action_grid"], dtype=np.float64)
    if method == "dddqn_per":
        _, _, decide = next(saved_policies(seed, config, ("transfer",)))
        return lambda _data, _rows: decide
    if method == "ddpg":
        from stable_baselines3 import DDPG

        archive = ROOT / "models" / "ddpg" / f"seed{seed}.zip"
        actor = DDPG.load(archive, device="cpu").actor.eval()
        cutoffs = (grid[:-1] + grid[1:]) * 0.5

        def decide(_env, observation):
            power = actor(torch.from_numpy(observation[None, :])).item()
            return int(np.searchsorted(cutoffs, np.clip(power, grid[0], grid[-1]), side="left"))

        return lambda _data, _rows: decide

    def action(policy, env, _observation):
        return int(policy.action(env.current_step, env.current_soc * env.battery_capacity))

    protocol = read_json(ROOT / "baselines_tou_mpc" / "protocol.json")
    if method == "tou":
        timing = protocol["tou"]
        policy = TimeOfUse(data.price, grid, round(24 / environment["time_step"]),
                           calibration_days=timing["calibration_complete_days"],
                           band_slots=timing["band_slots"])
        return lambda _data, _rows: lambda env, obs: action(policy, env, obs)

    horizon = protocol["mpc"]["horizon_steps"]

    def build(window_data, rows):
        policy = PredictiveControl(window_data, environment, rows, horizon)
        return lambda env, obs: action(policy, env, obs)

    return build


def archive_job(job):
    expected, trace_directory = job
    method, seed = expected[0]["method"], expected[0]["seed"]
    expected = {run_labels(row): row for row in expected}
    data = load_data()
    source_config = read_json(SEALED / "config" / "full.json")
    config = dict(source_config, environment=make_env(data, 17520, source_config["environment"], False).config)
    build_policy = make_policy(method, seed, data, config)
    multipliers = projected_multipliers()
    results = []
    for scenario in dict.fromkeys(case for case, _ in expected):
        start, end, ev_scenario, _ = CASES[scenario]
        multiplier = multipliers[ev_scenario]
        window, first, stop = select_window(data, start, end, multiplier)
        rows = stop - first
        decide = build_policy(window, rows)
        for phase in PHASES:
            reference = expected[(scenario, phase)]
            context = (method, seed, scenario, phase)
            if (reference["rows"], reference["source_rows"], reference["ev_scenario"], reference["ev_multiplier"]) != (
                rows, [first, stop], ev_scenario, multiplier
            ):
                raise ValueError(f"Scenario coordinates differ in {context}.")
            env = make_env(window, rows, config["environment"], phase == PHASES[1])
            actual, trace = run_controller(env, decide)
            compare_metrics(actual, reference["metrics"], context)
            actions = action_hash(trace["actions"])
            if actions != reference["actions_sha256"]:
                raise ValueError(f"Executed command indices differ in {context}.")
            if method == "dddqn_per" and scenario == "reference":
                check_reference(f"transfer_seed{seed}", phase, trace)
            if trace_directory is not None:
                label = method if seed is None else f"{method}_seed{seed}"
                np.savez_compressed(trace_directory / f"{label}_{scenario}_{phase}.npz", **trace)
            results.append(dict(reference, metrics=actual))
        print(f"Verified {method} seed={seed} {scenario}, {rows:,} steps per terminal mode.", flush=True)
    return results


def external_job(job):
    seed, path, expected = job
    expected = {(row["season"], row["phase"]): row for row in expected}
    archive = load_data()
    config = read_json(SEALED / "config" / "full.json")
    _, weights, decide = next(saved_policies(seed, config, ("transfer",)))
    weights_sha = checksum(weights)
    results = []
    for season, (data, rows, _) in prepared_cases(archive, path).items():
        for phase in PHASES:
            reference = expected[(season, phase)]
            context = (seed, season, phase)
            if rows != reference["rows"] or weights_sha != reference["model_sha256"]:
                raise ValueError(f"External window coordinates or weights differ in {context}.")
            actual, trace = run_controller(make_env(data, rows, config["environment"], phase == PHASES[1]), decide)
            compare_metrics(actual, reference["metrics"], context)
            if trace_binding(trace) != reference["trace_bytes_sha256"]:
                raise ValueError(f"External trace differs in {context}.")
            results.append(dict(reference, metrics=actual, method="dddqn_per"))
        print(f"Verified external {season} seed={seed}, {rows:,} steps per terminal mode.", flush=True)
    return results


def configure_torch():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def execute(worker, jobs, workers):
    if workers == 1:
        configure_torch()
        groups = map(worker, jobs)
        return [row for group in groups for row in group]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_torch) as executor:
        return [row for group in executor.map(worker, jobs) for row in group]


def check_reference_tables(results, scope):
    name = "primary_comparison.tsv" if scope == "primary" else "archive_scenarios.tsv"
    grouped = {}
    for row in results:
        key = (row["method"], row["scenario"], row["phase"])
        grouped.setdefault(key, []).append(row)
    with (REFERENCE / name).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if row["method"] == "zero_storage":
                continue
            key = (row["method"], row["case"], row["phase"])
            windows = grouped.pop(key)
            if len(windows) != int(row["runs"]) or any(w["rows"] != int(row["rows_per_window"]) for w in windows):
                raise ValueError(f"Summary counts differ in {key}.")
            means = {metric: mean(w["metrics"][metric] for w in windows) for metric in PUBLISHED_METRICS}
            compare_metrics(means, {metric: float(row[f"{metric}_mean"]) for metric in means}, key)
    if grouped:
        raise ValueError(f"Summary table omits these replay groups: {sorted(grouped)}")


def write_results_table(path, results):
    groups = {}
    for row in results:
        groups.setdefault((row["scenario"], row["phase"], row["method"]), []).append(row)
    fields = ("scenario", "phase", "method", "runs", "rows_per_window", *PUBLISHED_METRICS,
              "baseline_bill", "bill", "final_energy_kwh")
    with path.open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        table.writeheader()
        for scenario in dict.fromkeys(key[0] for key in groups):
            for phase in PHASES:
                rows = groups[(scenario, phase, "dddqn_per")]
                zero = {metric: 0.0 for metric in PUBLISHED_METRICS if metric != "available_percent"}
                base_bill = mean(row["metrics"]["baseline_bill"] for row in rows)
                table.writerow(dict(scenario=scenario, phase=phase, method="zero_storage", runs=0,
                                    rows_per_window=rows[0]["rows"],
                                    baseline_bill=base_bill, bill=base_bill, **zero))
                for method in METHODS:
                    rows = groups[(scenario, phase, method)]
                    metrics = {metric: f"{mean(row['metrics'][metric] for row in rows):.9f}"
                               for metric in fields[5:]}
                    table.writerow(dict(scenario=scenario, phase=phase, method=method,
                                        runs=len(rows), rows_per_window=rows[0]["rows"], **metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    verify = subs.add_parser("verify", help="Run the primary windows, or all in-archive sensitivity windows.")
    verify.add_argument("--scope", choices=("primary", "archive"), default="primary")
    verify.add_argument("--save-traces", action="store_true", help="Also save regenerated per-window NPZ files.")
    external = subs.add_parser("external", help="Replay the separate four-season sensitivity with your aligned SGSC/AEMO table.")
    external.add_argument("--input", type=Path, required=True)
    for sub in (verify, external):
        sub.add_argument("--output", type=Path, required=True, help="A fresh directory outside the two release bundles.")
        sub.add_argument("--jobs", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == package or package in output.parents for package in (ROOT, SEALED)):
        parser.error("Choose an output directory outside benchmark/ and reconstruction/.")
    verify_files()
    if args.command == "external":
        expected = read_json(REFERENCE / "external_windows.json")
        source = args.input.resolve(strict=True)
        if checksum(source) != expected["source_csv_sha256"]:
            parser.error("The aligned external table does not match the study's SHA-256.")
        output.mkdir(parents=True, exist_ok=False)
        jobs = [(seed, source, [row for row in expected["rows"] if row["seed"] == seed]) for seed in SEEDS]
        results = execute(external_job, jobs, args.jobs)
        expected_dimensions = (32, 279_552)
        scope = "external"
    else:
        scope = args.scope
        expected = read_json(REFERENCE / "archive_windows.json")["rows"]
        if scope == "primary":
            expected = [row for row in expected if row["scenario"] == "reference"]
        output.mkdir(parents=True, exist_ok=False)
        traces = output / "traces" if args.save_traces else None
        if traces is not None:
            traces.mkdir()
        jobs = [([row for row in expected if row["method"] == method and row["seed"] == seed], traces)
                for method in METHODS for seed in (SEEDS if method in ("dddqn_per", "ddpg") else (None,))]
        results = execute(archive_job, jobs, args.jobs)
        expected_dimensions = (20, 350_400) if scope == "primary" else (200, 2_623_680)
        check_reference_tables(results, scope)
        write_results_table(output / "comparison.tsv", results)
    dimensions = len(results), sum(row["rows"] for row in results)
    if dimensions != expected_dimensions:
        raise ValueError(f"Expected replay totals {expected_dimensions}, got {dimensions}.")
    write_json(output / "verification.json", {
        "schema": "comparison_replay_verification_v1",
        "scope": scope,
        "status": "pass",
        "windows": dimensions[0],
        "simulated_steps": dimensions[1],
        "training_interactions": 0,
        "metrics_absolute_tolerance": METRIC_TOLERANCE,
        "benchmark_manifest_sha256": checksum(ROOT / "benchmark_manifest.json"),
        "reconstruction_manifest_sha256": checksum(SEALED / "package_manifest.json"),
        "results": results,
    })
    print(f"Verified {dimensions[0]} windows / {dimensions[1]:,} simulated steps. No training performed.")


if __name__ == "__main__":
    main()

