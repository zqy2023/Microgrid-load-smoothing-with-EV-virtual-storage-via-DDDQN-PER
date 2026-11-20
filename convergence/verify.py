"""Replay every published convergence checkpoint and verify its output."""
import argparse
import json
from pathlib import Path

from run_cohort import GROUPS, SEEDS, cohort
from summarise import PROTOCOL_FILE, digest, read, report, summarise


HERE = Path(__file__).resolve().parent


def verify(output, workers):
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    evaluations = output / "evaluations"
    logs.mkdir()
    evaluations.mkdir()
    data = HERE / "reference"

    def record(phase, event, **fields):
        item = json.dumps(dict(phase=phase, event=event, **fields), sort_keys=True)
        with (logs / "cohort.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(item + "\n")
        print(item, flush=True)

    if not cohort("evaluate", HERE.parent, data / "runs", evaluations, workers, record, logs):
        raise RuntimeError(f"Checkpoint replay failed. See {logs}.")

    results = HERE / "results"
    reference_summary = read(results / "convergence_summary.json")
    reference_digests = {row["file"]: row["evaluation_sha256"]
                         for row in reference_summary["source_evaluations"]}
    for arm in GROUPS:
        for seed in SEEDS:
            name = f"{arm}_seed{seed}.json"
            expected = data / "evaluations" / name
            if digest(expected) != reference_digests[name]:
                raise ValueError(f"Published evaluation digest differs: {name}")
            if read(expected) != read(evaluations / name):
                raise ValueError(f"Replayed evaluation differs: {name}")

    # A replay can encode an identical JSON document with different newline
    # bytes. Keep the verified publication digests when comparing summaries.
    summary, curves = summarise(evaluations, read(PROTOCOL_FILE), digest(PROTOCOL_FILE))
    for source in summary["source_evaluations"]:
        source["evaluation_sha256"] = reference_digests[source["file"]]
    if summary != reference_summary:
        raise ValueError("Recomputed summary differs from the published result.")
    audit = read(results / "training_journal_audit.json")
    audited_receipts = {(row["arm"], row["seed"]): row["receipt_sha256"]
                        for row in audit["training_runs"]}
    if (audit["status"] != "pass" or audit["protocol_sha256"] != digest(PROTOCOL_FILE)
            or audit["total_interactions"] != 9600000
            or len(audited_receipts) != summary["formal_runs"]):
        raise ValueError("Training-journal audit reference is incomplete.")
    for row in summary["source_evaluations"]:
        if audited_receipts[row["arm"], row["seed"]] != row["training_receipt_sha256"]:
            raise ValueError(f"Audit receipt binding differs: {row['file']}")
    (output / "recomputed_report.md").write_text(report(summary), encoding="utf-8")
    proof = {
        "status": "pass",
        "mode": "saved_checkpoint_replay",
        "checkpoints": len(curves),
        "runs": summary["formal_runs"],
        "evaluation_simulated_steps": summary["evaluation_simulated_steps"],
        "new_training_interactions": 0,
        "historical_clock_records_checked": True,
        "protocol_sha256": digest(PROTOCOL_FILE),
    }
    with (output / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(f"Verified {len(curves)} checkpoints and the paired summary.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="A fresh directory for replay results and logs.")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    verify(args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
