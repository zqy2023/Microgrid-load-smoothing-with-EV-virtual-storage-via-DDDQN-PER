"""Independently audit full component-ablation runs and report paired outcomes."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SEALED = HERE.parent / "reconstruction"
sys.path.insert(0, str(SEALED / "audit"))
import reference_core as reference
import verify_training as training_audit

SEEDS = (0, 1, 2, 3)
VARIANTS = ("no_double", "no_dueling", "no_per")
STEPS, UPDATES = 1200000, 299993
BASE_COMMIT = "e190821886143f70af46437d52db749bc1afc06a"
PHASES = {"free_terminal": "trace.npz", "equal_terminal_evaluation": "neutral_trace.npz"}
METRICS = {"pooled_pct": "pooled", "daily_pct": "daily", "saving_pct": "saving"}
DIRECT_LAYERS = {
    "feature_layers.0": (128, 15),
    "feature_layers.3": (128, 128),
    "action_head.0": (119, 128),
    "action_head.2": (21, 119),
}
CHECKPOINTS = ("initial.pt", "final.pt", "optimizer_target.pt", "initial_trace.npz",
               "trace.npz", "neutral_trace.npz", "progress.jsonl")


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def binding_path(path, entry):
    """Check known local files without following paths in received receipts."""
    require(path.is_file() and training_audit.sha(path) == entry["sha256"]
            and path.stat().st_size == entry["bytes"], f"Binding mismatch: {path}")
    return path


def snapshot(path, variant):
    return validate_weights(torch.load(path, map_location="cpu", weights_only=True), variant)


def validate_weights(state, variant):
    if variant != "no_dueling":
        reference.weights_valid(state)
    else:
        expected_keys = {f"{name}.{part}" for name in DIRECT_LAYERS for part in ("weight", "bias")}
        require(set(state) == expected_keys, "Unexpected parameter names for the direct Q network")
        for name, shape in DIRECT_LAYERS.items():
            for part, expected in (("weight", shape), ("bias", (shape[0],))):
                tensor = state[f"{name}.{part}"]
                require(tuple(tensor.shape) == expected and tensor.dtype == torch.float32
                        and torch.isfinite(tensor).all().item(), f"Invalid parameter {name}.{part}")
    return state


def direct_q(observation, state):
    value = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        for index, name in enumerate(DIRECT_LAYERS):
            value = F.linear(value, state[f"{name}.weight"], state[f"{name}.bias"])
            if index < len(DIRECT_LAYERS) - 1:
                value = value.relu()
    return value.numpy()


def verify_direct_q(path, state):
    with np.load(path, allow_pickle=False) as trace:
        observations = trace["observations"]
        saved = trace["q_values"]
        actions = trace["actions"]
    rechecks = 0
    maximum = 0.0
    for begin in range(0, reference.N, 256):
        end = min(begin + 256, reference.N)
        calculated = direct_q(observations[begin:end], state)
        maximum = max(maximum, float(np.max(np.abs(calculated - saved[begin:end]))))
        flagged = np.flatnonzero(
            (~np.isclose(calculated, saved[begin:end], rtol=2e-6, atol=2e-5)).any(axis=1)
            | (calculated.argmax(axis=1) != actions[begin:end])
        )
        for offset in flagged:
            index = begin + int(offset)
            exact = direct_q(observations[index:index + 1], state)[0]
            require(np.allclose(exact, saved[index], rtol=2e-7, atol=2e-6)
                    and int(exact.argmax()) == int(actions[index]), f"Direct Q mismatch at row {index}")
            rechecks += 1
    return {"rows": reference.N, "q_mismatch_rows": 0, "action_mismatch_rows": 0,
            "maximum_batch256_difference": maximum, "batch1_rechecks": rechecks,
            "method": "Independent named linear/ReLU layers with a direct action head"}


def verify_trace(path, data, grid, recorded, phase, weights, variant):
    independent = None if variant == "no_dueling" else weights
    result = reference.verify_trace(path, data, grid, recorded,
                                    phase == "equal_terminal_evaluation", independent)
    if variant == "no_dueling":
        result["independent_Q"] = verify_direct_q(path, weights)
        result["independent_weights_available_for_this_trace"] = True
    result.pop("trace_path")
    result["metrics"]["initial_energy_kwh"] = 37.5
    return result


def verify_optimizer(path, variant, parameter_count):
    snapshot_data = torch.load(path, map_location="cpu", weights_only=True)
    target = validate_weights(snapshot_data["target"], variant)
    require(snapshot_data["updates"] == UPDATES and len(target) == parameter_count,
            "Target checkpoint update count or tensor count differs")
    optimizer = snapshot_data["optimizer"]
    require(len(optimizer["state"]) == parameter_count, "Missing optimizer parameters")
    for state in optimizer["state"].values():
        require(int(state["step"].item()) == UPDATES, "Incomplete optimizer updates")
        require(all(torch.isfinite(value).all().item() for value in state.values()
                    if torch.is_tensor(value)), "Nonfinite optimizer state")
    for group in optimizer["param_groups"]:
        require(group["lr"] == 3e-4 and tuple(group["betas"]) == (.9, .999)
                and group["eps"] == 1e-8, "Optimizer settings differ from the control")


def portable_journal(run, receipt):
    """Accept journals copied into another directory using their exact chunk names."""
    entries = receipt["chunks"]
    require(len(entries) == 30, "Expected all 30 journal chunks")
    normalized = []
    for index, entry in enumerate(entries):
        path = run / "training" / f"chunk_{index:05d}.npz"
        require(Path(entry["path"]).name == path.name, f"Unexpected journal sequence at {path}")
        # training_audit checks file hashes and sizes itself.
        normalized.append(dict(entry, path=str(path)))
    require(sorted(path.name for path in (run / "training").glob("chunk_*.npz"))
            == [Path(entry["path"]).name for entry in normalized], f"Unexpected chunks in {run}")
    return dict(receipt, chunks=normalized)


def verify_replay(receipt, variant, seed):
    """Verify uniform sampling exactly in the no-PER arm; audit IS evidence elsewhere."""
    unit = variant == "no_per"
    rng = np.random.RandomState(seed) if unit else None
    nonunit_count = 0
    for entry in receipt["chunks"]:
        with np.load(entry["path"], allow_pickle=False) as arrays:
            weights = arrays["is_weights"]
            nonunit_count += int(np.count_nonzero(weights != 1.0))
            if unit:
                indices = arrays["replay_indices"]
                actions = arrays["action"]
                update = 0
                for index, step in enumerate(range(entry["begin_step"], entry["end_step"] + 1)):
                    epsilon = 1 - .95 * min((step - 1) / 120000, 1)
                    if rng.random() < epsilon:
                        require(rng.randint(21) == actions[index],
                                f"Exploration RNG mismatch at transition {step}")
                    if step >= 32 and (step - 32) % 4 == 0:
                        expected = (rng.random(32) * min(step, 200000)).astype(np.int64)
                        require(np.array_equal(expected, indices[update]),
                                f"Uniform replay mismatch at transition {step}")
                        update += 1
                require(update == len(indices), f"Wrong replay count in {entry['path']}")
    require((nonunit_count == 0) == unit, f"Importance-weight policy did not match {variant}")
    return {"nonunit_importance_weights": nonunit_count,
            "exact_uniform_draws_recomputed": receipt["updates"] * 32 if unit else 0}


def verify_sealed():
    manifest = reference.read(SEALED / "package_manifest.json")
    for relative, entry in manifest["files"].items():
        binding_path(SEALED / relative, entry)
    expected = reference.read(SEALED / "reference" / "expected.json")
    runs = {int(run["seed"]): run for run in expected["runs"] if run["family"] == "transfer"}
    require(set(runs) == set(SEEDS), "A same-seed published control is missing")
    for seed, run in runs.items():
        require(run["steps"] == STEPS and run["updates"] == UPDATES,
                f"Published comparison budget differs at seed {seed}")
    return expected["inputs"], runs


def baseline_row(run, data):
    state_path = SEALED / run["models"]["final.pt"]
    state = snapshot(state_path, "full")
    grid = np.asarray(run["environment"]["action_grid"])
    phases = {phase: verify_trace(SEALED / run["traces"][name], data, grid, run[phase], phase, state,
                                  "full") for phase, name in PHASES.items()}
    row = {"variant": "full", "seed": run["seed"], "steps": STEPS, "updates": UPDATES,
           "training_source": "published archived full controller", "network_parameters": 36502,
           "weights_sha256": training_audit.sha(state_path), "evaluation_audit": phases}
    row.update({phase: evidence["metrics"] for phase, evidence in phases.items()})
    return row


def no_path_fields(value):
    if isinstance(value, dict):
        return {key: no_path_fields(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [no_path_fields(item) for item in value]
    return value


def trainer_bindings():
    protocol = reference.read(SEALED / "config" / "transfer_protocol.json")
    paths = {name: SEALED / "src" / name for name in protocol["source_sha256"]}
    paths["full.json"] = SEALED / "config" / "full.json"
    paths.update({name: HERE / name for name in ("train.py", "variants.py")})
    bindings = {
        name: {"bytes": path.stat().st_size, "sha256": training_audit.sha(path)}
        for name, path in paths.items()
    }
    require(all(bindings[name]["sha256"] == digest
                for name, digest in protocol["source_sha256"].items()),
            "A shared trainer source differs from the published control")
    return bindings


def ablation_row(run, variant, seed, archived, data, input_hashes, teacher, sources):
    receipt_path = run / "receipt.json"
    receipt = reference.read(receipt_path)
    require((receipt["status"], receipt["kind"], receipt["method"], receipt["ablation"],
             receipt["steps"], receipt["updates"], receipt["seed"])
            == ("complete", "formal", "refined", variant, STEPS, UPDATES, seed),
            f"Missing full-budget formal training: {run}")
    require(receipt["paired_reference_commit"] == BASE_COMMIT, f"Wrong parent commit: {run}")
    require(no_path_fields(receipt["source"]) == sources, f"Training source mismatch: {run}")
    binding_path(SEALED / "config" / "transfer_protocol.json", receipt["formal_protocol"])
    binding_path(SEALED / archived["models"]["initial.pt"], receipt["paired_initial_weights"])
    require(receipt["environment"] == archived["environment"]
            and {key: entry["sha256"] for key, entry in receipt["inputs"].items()} == input_hashes
            and receipt["training"] == reference.read(SEALED / "config" / "full.json")["training"]
            and receipt["imitation_weight"] == 4.0 and receipt["ensemble_only_through_step"] == 80000,
            f"Baseline conditions differ: {run}")
    for name in CHECKPOINTS:
        require(Path(receipt["files"][name]["path"]).name == name, f"Invalid file binding: {name}")
        binding_path(run / name, receipt["files"][name])
    for name in ("step_0240000.pt", "step_0600000.pt", "step_0960000.pt"):
        if name in receipt["files"]:
            binding_path(run / name, receipt["files"][name])
    for key, value in teacher.receipt["teacher"].items():
        require(no_path_fields(receipt["teacher"][key]) == no_path_fields(value),
                f"Different teaching artifact in {run}: {key}")

    initial = snapshot(run / "initial.pt", variant)
    final = snapshot(run / "final.pt", variant)
    archive_initial = snapshot(SEALED / archived["models"]["initial.pt"], "full")
    shared = [name for name in initial if variant != "no_dueling" or name.startswith("feature_layers.")]
    require(all(torch.equal(initial[name], archive_initial[name]) for name in shared),
            f"Shared initial weights differ at seed {seed}")
    require(any(not torch.equal(initial[name], final[name]) for name in initial), "Weights never changed")
    parameter_count = sum(tensor.numel() for tensor in final.values())
    require(parameter_count == receipt["network_parameters"], "Incorrect parameter count")
    verify_optimizer(run / "optimizer_target.pt", variant, len(final))
    normalized = portable_journal(run, receipt)
    audited = training_audit.training_audit(run, normalized, data, teacher)
    replay = verify_replay(normalized, variant, seed)
    grid = np.asarray(receipt["environment"]["action_grid"])
    phases = {phase: verify_trace(run / name, data, grid, receipt[phase], phase, final, variant)
              for phase, name in PHASES.items()}
    phases["initial"] = verify_trace(run / "initial_trace.npz", data, grid, receipt["initial"],
                                     "initial", initial, variant)

    row = {"variant": variant, "seed": seed, "steps": STEPS, "updates": UPDATES,
           "training_source": "new full-budget training", "network_parameters": parameter_count,
           "receipt_sha256": training_audit.sha(receipt_path),
           "weights_sha256": receipt["files"]["final.pt"]["sha256"],
           "training_audit": audited, "sampler_audit": replay, "evaluation_audit": phases}
    row.update({phase: phases[phase]["metrics"] for phase in PHASES})
    return row


def build_results(runs_path):
    inputs, published = verify_sealed()
    input_hashes = {key: value["sha256"] for key, value in inputs.items()}
    data = training_audit.load_data({"inputs": inputs})
    paths = {(variant, seed): runs_path / f"{variant}_seed{seed}" for variant in VARIANTS for seed in SEEDS}
    missing = [str(path) for path in paths.values() if not (path / "receipt.json").is_file()]
    require(not missing, "Formal runs missing receipts: " + ", ".join(missing))
    first = reference.read(paths[(VARIANTS[0], SEEDS[0])] / "receipt.json")
    teacher = training_audit.TeacherReference(first, data)
    teacher_manifest = training_audit.binding(first["teacher"]["dp_manifest"])
    canonical = reference.read(SEALED / "config" / "teacher.json")
    actual = reference.read(teacher_manifest)
    for key in ("beta", "value_table_sha256", "offsets_sha256", "input_sha256",
                "teacher_source_sha256", "builder_sha256", "protocol_sha256"):
        require(actual[key] == canonical[key], f"Offline teacher differs: {key}")
    require(first["teacher"]["ensemble_receipt"]["sha256"]
            == training_audit.sha(SEALED / "config" / "ensemble_teacher.json"),
            "Ensemble teacher differs from the published controller")

    sources = trainer_bindings()
    baseline = {seed: baseline_row(published[seed], data) for seed in SEEDS}
    rows = list(baseline.values())
    for (variant, seed), path in paths.items():
        print(f"Auditing {variant} seed {seed}...", flush=True)
        rows.append(ablation_row(path, variant, seed, published[seed], data, input_hashes, teacher, sources))
    for row in rows:
        control = baseline[row["seed"]]
        row["paired_drop_pp"] = {
            phase: {metric: control[phase][metric] - row[phase][metric] for metric in METRICS}
            for phase in PHASES
        }
    summary = []
    for variant in ("full",) + VARIANTS:
        family = [row for row in rows if row["variant"] == variant]
        for phase in PHASES:
            aggregate = {}
            for metric in METRICS:
                values = [row[phase][metric] for row in family]
                drops = [row["paired_drop_pp"][phase][metric] for row in family]
                aggregate[metric] = {
                    "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                    "min": min(values), "max": max(values), "mean_paired_drop_pp": statistics.mean(drops),
                    "min_paired_drop_pp": min(drops), "max_paired_drop_pp": max(drops),
                }
            summary.append({"variant": variant, "phase": phase, "n": len(family), "metrics": aggregate})
    return {"schema": "dddqn_component_ablation_v1", "status": "all_pass", "seeds": list(SEEDS),
            "interactions_per_seed": STEPS, "gradient_updates_per_seed": UPDATES,
            "new_training_interactions": len(paths) * STEPS,
            "archived_full_controller_interactions": len(SEEDS) * STEPS,
            "paired_delta_definition": "same-seed full controller minus ablated controller, percentage points",
            "archived_inputs_sha256": input_hashes, "paired_reference_commit": BASE_COMMIT,
            "trainer_source_bindings": sources,
            "sealed_manifest_sha256": training_audit.sha(SEALED / "package_manifest.json"),
            "teacher_provenance": no_path_fields(teacher.verified),
            "limits": ["Main 17,520-row archived window was used for training and evaluation.",
                       "Equal-terminal evaluation applies a common future-availability wrapper.",
                       "TD targets and historical gradient steps were not reconstructed.",
                       "Teacher labels were audited on preset stage-1 batches and every valid stage-2 row."],
            "runs": rows, "aggregate": summary}


def per_seed_rows(results):
    for row in results["runs"]:
        for phase in PHASES:
            values = row[phase]
            yield {"variant": row["variant"], "seed": row["seed"], "phase": phase,
                   "interactions": row["steps"], "gradient_updates": row["updates"],
                   **{metric: values[metric] for metric in METRICS},
                   **{key: values[key] for key in ("initial_energy_kwh", "final_energy_kwh",
                                                   "energy_change_kwh")},
                   **{f"full_minus_{short}_pp": row["paired_drop_pp"][phase][metric]
                      for metric, short in METRICS.items()}}


def summary_rows(results):
    for row in results["aggregate"]:
        yield {"variant": row["variant"], "phase": row["phase"], "n": row["n"],
               **{f"{metric}_{field}": value for metric, metrics in row["metrics"].items()
                  for field, value in metrics.items()}}


def write_tsv(path, rows):
    rows = list(rows)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"Use a new output directory: {output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = build_results(args.runs.resolve())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "ablation_results.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    write_tsv(output / "per_seed.tsv", per_seed_rows(results))
    write_tsv(output / "summary.tsv", summary_rows(results))
    print(f"Verified {len(results['runs'])} runs including the archived full controller.", flush=True)


if __name__ == "__main__":
    main()

