"""Independently audit the paired N13 training journals."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEALED = ROOT / "reconstruction"
sys.path.insert(0, str(ROOT / "ablation"))

import torch
import audit_runs as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = HERE / "protocol.json"
    protocol = audit.reference.read(protocol_path)
    protocol_hash = audit.training_audit.sha(protocol_path)
    inputs, archived = audit.verify_sealed()
    expected_inputs = {name: item["sha256"] for name, item in inputs.items()}
    data = audit.training_audit.load_data({"inputs": inputs})
    shared = audit.reference.read(SEALED / "config" / "transfer_protocol.json")
    audited = []
    teacher = None
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            run = (args.runs / f"{arm}_seed{seed}").resolve(strict=True)
            raw = run / "receipt.json"
            receipt = audit.reference.read(raw)
            actual = (receipt["status"], receipt["kind"], receipt["steps"], receipt["updates"],
                      receipt["arm"], receipt["seed"], receipt["protocol_sha256"], receipt["thread_count"])
            audit.require(actual == ("complete", "formal", 1200000, 299993, arm, seed, protocol_hash, 1),
                          f"Incomplete paired training: {run.name}")
            audit.require(receipt["environment"] == archived[seed]["environment"]
                          and {key: item["sha256"] for key, item in receipt["inputs"].items()} == expected_inputs,
                          f"Archived inputs or physical settings differ: {run.name}")
            audit.require(receipt["method"] == "refined" and receipt["imitation_weight"] == 4.0
                          and receipt["ensemble_only_through_step"] == 80000,
                          f"Unexpected auxiliary training schedule: {run.name}")
            for name, expected in shared["source_sha256"].items():
                audit.require(receipt["source"][name]["sha256"] == expected, f"Common source differs: {name}")
            audit.binding_path(HERE / "train.py", receipt["source"]["train.py"])
            audit.binding_path(protocol_path, receipt["source"]["protocol.json"])
            audit.binding_path(ROOT / "ablation" / "variants.py", receipt["source"]["variants.py"])
            audit.binding_path(run / "progress.jsonl", receipt["progress"])
            checkpoint = audit.binding_path(run / "optimizer_target.pt", receipt["optimizer_target"])
            audit.verify_optimizer(checkpoint, "full", 12)
            if teacher is None:
                teacher = audit.training_audit.TeacherReference(receipt, data)
                audit.require(receipt["teacher"]["dp_manifest"]["sha256"] == shared["teacher_manifest_sha256"],
                              "Offline targets differ from the published protocol")
            for key, common in teacher.receipt["teacher"].items():
                audit.require(audit.no_path_fields(receipt["teacher"][key]) == audit.no_path_fields(common),
                              f"Different teaching artifact: {run.name}/{key}")
            journal = audit.portable_journal(run, receipt)
            proof = audit.training_audit.training_audit(run, journal, data, teacher)
            sampler = audit.verify_replay(journal, "no_per" if arm == "uniform" else "full", seed)
            audited.append(dict(arm=arm, seed=seed, receipt_sha256=audit.training_audit.sha(raw),
                                steps=receipt["steps"], updates=receipt["updates"], verification=proof,
                                sampler=sampler))
            print(f"{run.name}: {proof['transition_rows']:,} training interactions audited", flush=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"schema": "n13_convergence_journal_audit_v1", "status": "pass",
                   "protocol_sha256": protocol_hash, "training_runs": audited,
                   "total_interactions": sum(row["steps"] for row in audited)}, stream, indent=2,
                  ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

