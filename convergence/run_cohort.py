"""Run the fixed N13 matrix as single-threaded jobs in a balanced cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GROUPS = ("per", "uniform")
SEEDS = (0, 1, 2, 3)


def complete(phase, output, arm, seed):
    receipt = output / "receipt.json" if phase == "train" else output
    if not receipt.is_file():
        return False
    data = json.loads(receipt.read_text(encoding="utf-8-sig"))
    common = data.get("arm") == arm and data.get("seed") == seed
    if phase == "train":
        return (common and data.get("status") == "complete" and data.get("kind") == "formal"
                and data.get("steps") == 1200000 and data.get("updates") == 299993
                and len(data.get("checkpoints", ())) == 84)
    return (common and data.get("status") == "pass" and len(data.get("rows", ())) == 84)


def cohort(phase, repo, runs, evaluations, workers, record, logs):
    pending = []
    failed = []
    for seed in SEEDS:
        for arm in GROUPS:
            name = f"{arm}_seed{seed}"
            raw = runs / name
            output = raw if phase == "train" else evaluations / f"{name}.json"
            if complete(phase, output, arm, seed):
                record(phase, "already_complete", job=name)
            elif output.exists():
                failed.append(name)
                record(phase, "existing_nonfinal_output", job=name)
            else:
                pending.append((arm, seed, name, raw, output))

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    windows = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    active = {}
    while pending or active:
        while pending and len(active) < workers:
            arm, seed, name, raw, output = pending.pop(0)
            executable = repo / "convergence" / f"{phase}.py"
            command = [sys.executable, "-X", "utf8", str(executable)]
            if phase == "train":
                command += ["train", "--arm", arm, "--seed", str(seed)]
            else:
                command += ["--run", str(raw)]
            command += ["--output", str(output)]
            logfile = logs / f"{phase}_{name}.log"
            with logfile.open("ab") as stream:
                try:
                    proc = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=subprocess.STDOUT, creationflags=windows)
                except OSError as error:
                    failed.append(name)
                    record(phase, "launch_failed", job=name, error=str(error))
                    continue
            active[name] = proc, arm, seed, output
            record(phase, "started", job=name, pid=proc.pid)
        if active:
            time.sleep(3)
        for name, (proc, arm, seed, output) in tuple(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[name]
            if code == 0 and complete(phase, output, arm, seed):
                record(phase, "complete", job=name, returncode=code)
            else:
                record(phase, "failed", job=name, returncode=code)
                failed.append(name)
    record(phase, "cohort_complete", failed=failed)
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--phase", choices=("both", "train", "evaluate"), default="both")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    runs = args.runs.resolve(strict=True)
    evaluations = args.evaluations.resolve(strict=True)
    logs = runs / "logs"
    logs.mkdir(exist_ok=True)
    events = logs / "cohort.jsonl"

    def record(phase, event, **fields):
        item = json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                               event=event, **fields))
        with events.open("a", encoding="utf-8") as stream:
            stream.write(item + "\n")
        print(item, flush=True)

    phases = ("train", "evaluate") if args.phase == "both" else (args.phase,)
    if phases == ("evaluate",) and not all(complete("train", runs / f"{arm}_seed{seed}", arm, seed)
                                           for arm in GROUPS for seed in SEEDS):
        raise ValueError("Evaluation requires all completed training receipts")
    for phase in phases:
        if not cohort(phase, repo, runs, evaluations, args.workers, record, logs):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

