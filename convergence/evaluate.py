"""Re-evaluate N13 checkpoints in the published equal-terminal scenario."""
import argparse
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ablation"))

import torch

from replay import METRICS, compare_metrics, configure_torch, model_path, read_json
from replay import scenario, trace_hashes, verify_release

PROTOCOL = HERE / "protocol.json"
PHASE = "equal_terminal_evaluation"


def validate_receipt(receipt, protocol, protocol_sha256, smoke):
    kind = "engineering" if smoke else "formal"
    if (receipt["schema"], receipt["status"], receipt["kind"]) != (
            "n13_convergence_train_v1", "complete", kind):
        raise ValueError(f"Expected a complete {kind} training receipt.")
    if receipt["protocol_sha256"] != protocol_sha256:
        raise ValueError("The training protocol digest does not match this evaluator's.")
    if receipt["arm"] not in protocol["arms"] or receipt["seed"] not in protocol["seeds"]:
        raise ValueError("The run is outside the fixed arm and seed list.")
    expected = [0, 32, 96] if smoke else protocol["evaluation"]["sample_steps"]
    rows = receipt["checkpoints"]
    if receipt["steps"] != expected[-1] or [row["step"] for row in rows] != expected:
        raise ValueError("Checkpoint schedule or interaction count is incomplete.")
    if receipt["updates"] != (receipt["steps"] - 32) // 4 + 1:
        raise ValueError("Optimizer update count differs from the published cadence.")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Checkpoint paths must be distinct.")
    for label in ("cpu_seconds", "wall_seconds"):
        elapsed = [row[label] for row in rows]
        if (not all(math.isfinite(t) and t >= 0 for t in elapsed)
                or any(current < prior for prior, current in zip(elapsed, elapsed[1:]))):
            raise ValueError(f"Invalid cumulative {label}.")


def load_checkpoint(run, record):
    relative = Path(record["path"])
    if relative.is_absolute():
        raise ValueError(f"Absolute checkpoint path in receipt: {relative}")
    path = (run / relative).resolve(strict=True)
    if run not in path.parents or not path.is_file():
        raise ValueError(f"Checkpoint is outside the run directory: {relative}")
    if path.stat().st_size != record["bytes"] or scenario.checksum(path) != record["sha256"]:
        raise ValueError(f"Checkpoint differs from its receipt: {relative}")
    return torch.load(path, map_location="cpu", weights_only=True)


def endpoint_reference(receipt, state, metrics, signature, trace):
    variant = "full" if receipt["arm"] == "per" else "no_per"
    seed = receipt["seed"]
    reference_path = model_path(variant, seed)
    published = torch.load(reference_path, map_location="cpu", weights_only=True)
    matched = state.keys() == published.keys() and all(torch.equal(state[name], value)
                                                       for name, value in published.items())
    if "endpoint_weight_match" in receipt and receipt["endpoint_weight_match"] != matched:
        raise ValueError("Endpoint weight matching disagrees with the training receipt.")
    if matched:
        reference = read_json(HERE.parent / "ablation" / "reference" / "replay_expected.json")
        record, = (case for case in reference["cases"]
                   if (case["variant"], case["seed"], case["phase"]) == (variant, seed, PHASE))
        if scenario.checksum(reference_path) != record["model_sha256"]:
            raise ValueError("Published endpoint weight digest mismatch.")
        compare_metrics({name: metrics[name] for name in METRICS}, record["metrics"],
                        (variant, seed, PHASE))
        if signature != record["trace_sha256"]:
            raise ValueError("Published endpoint action or physical trace digest mismatch.")
        if variant == "full":
            scenario.check_reference(f"transfer_seed{seed}", PHASE, trace)
    return matched


def evaluate(run, output, smoke):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    receipt_path = run / "receipt.json"
    receipt_sha256 = scenario.checksum(receipt_path)
    protocol_sha256 = scenario.checksum(PROTOCOL)
    receipt = read_json(receipt_path)
    protocol = read_json(PROTOCOL)
    validate_receipt(receipt, protocol, protocol_sha256, smoke)
    configure_torch()
    verify_release()

    evaluation = protocol["evaluation"]
    if (evaluation["window"], evaluation["terminal_rule"], evaluation["terminal_kwh"]) != (
            "reference", PHASE, 37.5):
        raise ValueError("This evaluator is bound to the published neutral reference window.")
    config = read_json(scenario.SEALED / "config" / "full.json")
    start, stop, label, _ = scenario.CASES["reference"]
    window, first, end = scenario.select_window(scenario.load_data(), start, stop,
                                                 scenario.projected_multipliers()[label])
    count = end - first
    if count != evaluation["rows_per_window"]:
        raise ValueError(f"Reference window has {count} rows.")
    network = scenario.make_agent(config).q_network.eval()

    def decide(_env, observation):
        return int(network(torch.from_numpy(observation).reshape(1, -1)).argmax())

    rows = []
    matched_endpoint = False
    for record in receipt["checkpoints"]:
        state = load_checkpoint(run, record)
        network.load_state_dict(state)
        env = scenario.make_env(window, count, config["environment"], neutral=True)
        metrics, trace = scenario.run_controller(env, decide)
        signature = trace_hashes(trace)
        if not smoke and record["step"] == receipt["steps"]:
            matched_endpoint = endpoint_reference(receipt, state, metrics, signature, trace)
        rows.append(dict(record, metrics=metrics, actions_sha256=signature["actions"],
                         trace_sha256=signature))
        print(f'{receipt["arm"]} seed {receipt["seed"]}: step {record["step"]}: '
              f'{metrics["pooled_pct"]:.5f}%', flush=True)

    if scenario.checksum(receipt_path) != receipt_sha256 or scenario.checksum(PROTOCOL) != protocol_sha256:
        raise ValueError("Training receipt or protocol changed during evaluation.")
    scenario.write_json(output, dict(
        schema="n13_convergence_evaluation_v1", status="pass", kind=receipt["kind"],
        arm=receipt["arm"], seed=receipt["seed"], protocol_sha256=protocol_sha256,
        receipt_sha256=receipt_sha256, window=evaluation["window"], rows_per_window=count,
        terminal_rule=evaluation["terminal_rule"], terminal_kwh=evaluation["terminal_kwh"],
        endpoint_reference_matched=matched_endpoint, rows=rows,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Accept a 96-step engineering run only")
    args = parser.parse_args()
    evaluate(args.run.resolve(strict=True), args.output.resolve(), args.smoke)


if __name__ == "__main__":
    main()

