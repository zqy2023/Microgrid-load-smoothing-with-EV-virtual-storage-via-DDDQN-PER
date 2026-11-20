"""Replay the published ablation endpoints from archived data and fixed models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "benchmark"))

import numpy as np
import torch

from run import compare_metrics, configure_torch, read_json, verify_files
import scenario_runtime as scenario
from variants import VARIANTS, build_agent

SEEDS = (0, 1, 2, 3)
TRACE_KEYS = {
    "actions": "<i2",
    "command_kw": "<f8",
    "applied_command_kw": "<f8",
    "power_kw": "<f8",
    "energy_kwh": "<f8",
    "load_kw": "<f8",
    "price": "<f8",
    "available": "?",
    "net_kw": "<f8",
    "step_bill": "<f8",
}
METRICS = (
    "pooled_pct", "daily_pct", "saving_pct", "baseline_bill", "bill", "final_energy_kwh",
    "energy_change_kwh", "energy_min_kwh", "energy_max_kwh", "power_min_kw", "power_max_kw",
    "viability_adjusted_steps",
)


def trace_hashes(trace):
    return {name: hashlib.sha256(np.asarray(trace[name], dtype=dtype).tobytes(order="C")).hexdigest()
            for name, dtype in TRACE_KEYS.items()}


def verify_release():
    verify_files()
    manifest = read_json(HERE / "manifest.json")
    for name, record in manifest["files"].items():
        path = HERE / name
        if path.stat().st_size != record["bytes"] or scenario.checksum(path) != record["sha256"]:
            raise ValueError(f"Component ablation file differs from its manifest: {name}")


def model_path(variant, seed):
    if variant == "full":
        return scenario.SEALED / "models" / f"transfer_seed{seed}" / "final.pt"
    return HERE / "models" / f"{variant}_seed{seed}" / "final.pt"


@torch.inference_mode()
def replay(output):
    configure_torch()
    verify_release()
    expected = read_json(HERE / "reference" / "replay_expected.json")
    case_keys = {(v, s, p) for v in ("full", *VARIANTS) for s in SEEDS for p in scenario.PHASES}
    recorded = {(r["variant"], r["seed"], r["phase"]): r for r in expected["cases"]}
    if recorded.keys() != case_keys or len(expected["cases"]) != len(recorded):
        raise ValueError("Ablation replay matrix is incomplete or has duplicates.")
    config = read_json(scenario.SEALED / "config" / "full.json")
    start, stop, label, _ = scenario.CASES["reference"]
    data, first, end = scenario.select_window(scenario.load_data(), start, stop, scenario.projected_multipliers()[label])
    steps = end - first
    results = []
    for variant in ("full", *VARIANTS):
        for seed in SEEDS:
            path = model_path(variant, seed)
            digest = scenario.checksum(path)
            agent = scenario.make_agent(config) if variant == "full" else build_agent(config, variant)
            network = agent.q_network.eval()
            network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

            def decide(_env, observation):
                return int(network(torch.from_numpy(observation).reshape(1, -1)).argmax())

            for phase in scenario.PHASES:
                key = (variant, seed, phase)
                reference = recorded[key]
                if reference["steps"] != steps or reference["model_sha256"] != digest:
                    raise ValueError(f"Reference window or weights differ: {key}.")
                environment = scenario.make_env(data, steps, config["environment"], phase == scenario.PHASES[1])
                metrics, trace = scenario.run_controller(environment, decide)
                actual = {name: metrics[name] for name in METRICS}
                compare_metrics(actual, reference["metrics"], key)
                signature = trace_hashes(trace)
                if signature != reference["trace_sha256"]:
                    raise ValueError(f"Actions or physical trace differs: {key}.")
                if variant == "full":
                    scenario.check_reference(f"transfer_seed{seed}", phase, trace)
                results.append(dict(variant=variant, seed=seed, phase=phase, metrics=actual,
                                    model_sha256=digest, trace_sha256=signature))
            print(f"{variant} seed {seed}: both terminal modes verified", flush=True)

    receipt = dict(schema="component_ablation_replay_v1", status="pass", windows=len(results),
                   simulated_steps=steps * len(results), training_interactions=0, results=results,
                   ablation_manifest_sha256=scenario.checksum(HERE / "manifest.json"))
    scenario.write_json(output, receipt)
    print(f"{len(results)} verified windows / {steps * len(results):,} simulated steps / zero new training.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON verification receipt path")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    replay(output)


if __name__ == "__main__":
    main()

