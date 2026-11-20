"""Summarise the predeclared N13 sampled performance and training timing comparison."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
PROTOCOL_FILE = HERE / "protocol.json"
ARMS = ("per", "uniform")
SEEDS = (0, 1, 2, 3)
METRICS = ("pooled_pct", "daily_pct", "saving_pct")
EARLY_STEPS = (18000, 40000)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def check_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"Nonfinite or invalid number in {label}.")


def checked_run(path, arm, seed, protocol, protocol_sha):
    doc = read(path)
    key = f"{arm}/seed{seed}"
    require((doc["schema"], doc["status"], doc["arm"], doc["seed"], doc["protocol_sha256"])
            == ("n13_convergence_evaluation_v1", "pass", arm, seed, protocol_sha),
            f"Evaluation schema, identity, or protocol differs for {key}.")
    require(valid_sha(doc["receipt_sha256"]) and isinstance(doc["endpoint_reference_matched"], bool),
            f"Missing formal training evidence for {key}.")
    evaluation = protocol["evaluation"]
    require(all(doc[field] == evaluation[field]
                for field in ("window", "rows_per_window", "terminal_rule", "terminal_kwh")),
            f"Different evaluation conditions for {key}.")
    rows = doc["rows"]
    require([row["step"] for row in rows] == evaluation["sample_steps"],
            f"Missing, duplicated, or unordered checkpoints for {key}.")
    previous = dict(cpu_seconds=0.0, wall_seconds=0.0)
    for row in rows:
        context = f"{key}/{row['step']}"
        require(valid_sha(row["sha256"]) and valid_sha(row["actions_sha256"])
                and isinstance(row["bytes"], int) and row["bytes"] > 0
                and isinstance(row["path"], str) and bool(row["path"]),
                f"Incomplete checkpoint binding for {context}.")
        for name, timing in previous.items():
            check_number(row[name], f"{context}/{name}")
            require(row[name] >= timing, f"Backwards cumulative timing for {context}/{name}.")
        previous = {name: row[name] for name in previous}
        for name in METRICS:
            check_number(row["metrics"][name], f"{context}/{name}")
        require(math.isclose(row["metrics"]["final_energy_kwh"], evaluation["terminal_kwh"],
                             rel_tol=0, abs_tol=1e-8),
                f"Terminal energy differs for {context}.")
    require(rows[-1]["cpu_seconds"] > 0 and rows[-1]["wall_seconds"] > 0,
            f"Missing measured training time for {key}.")
    return doc


def crossing(rows, index):
    """Localise the sampled threshold entry by its preceding checkpoint."""
    sample = rows[index]
    return {
        "step": sample["step"],
        "interval_lower_step": rows[index - 1]["step"] if index else 0,
        "interval_upper_step": sample["step"],
        "interval_lower_open": index > 0,
        "cpu_seconds_at_checkpoint": sample["cpu_seconds"],
        "wall_seconds_at_checkpoint": sample["wall_seconds"],
    }


def attainment(rows, threshold, minimum_tail, metric):
    meets = [row["metrics"][metric] >= threshold for row in rows]
    end = rows[-1]["step"]
    last_eligible = max(row["step"] for row in rows if end - row["step"] >= minimum_tail)
    hit = next((i for i, truth in enumerate(meets) if truth), None)
    last_failure = max((i for i, truth in enumerate(meets) if not truth), default=-1)
    entry = last_failure + 1
    first = crossing(rows, hit) if hit is not None else None
    result = {
        "first": first,
        "first_status": "observed" if first else "not_observed",
        "first_right_censored_after_step": None if first else end,
        "sustained": None,
        "sustained_candidate": None,
        "sustained_right_censored_after_step": last_eligible,
    }
    if entry >= len(rows):
        result["sustained_status"] = "fell_below_by_endpoint" if first else "not_observed"
        return result
    candidate = crossing(rows, entry)
    candidate["observed_tail_steps"] = end - candidate["step"]
    if candidate["observed_tail_steps"] < minimum_tail:
        result["sustained_status"] = "insufficient_follow_up"
        result["sustained_candidate"] = candidate
        return result
    result["sustained"] = candidate
    result["sustained_status"] = "observed"
    result["sustained_right_censored_after_step"] = None
    return result


def paired_crossing(per, uniform):
    if per is None or uniform is None:
        return None
    return {
        "uniform_minus_per_checkpoint_steps": uniform["step"] - per["step"],
        "uniform_minus_per_cpu_seconds": uniform["cpu_seconds_at_checkpoint"] - per["cpu_seconds_at_checkpoint"],
        "uniform_minus_per_wall_seconds": uniform["wall_seconds_at_checkpoint"] - per["wall_seconds_at_checkpoint"],
    }


def paired_budget(per, uniform):
    return {
        "per_cpu_seconds": per["cpu_seconds"],
        "uniform_cpu_seconds": uniform["cpu_seconds"],
        "uniform_minus_per_cpu_seconds": uniform["cpu_seconds"] - per["cpu_seconds"],
        "per_cpu_saving_pct_relative_to_uniform": 100 * (uniform["cpu_seconds"] - per["cpu_seconds"]) / uniform["cpu_seconds"],
        "per_wall_seconds": per["wall_seconds"],
        "uniform_wall_seconds": uniform["wall_seconds"],
        "uniform_minus_per_wall_seconds": uniform["wall_seconds"] - per["wall_seconds"],
    }


def number(value, digits=2):
    return f"{value:,.{digits}f}"


def signed(value, digits=2):
    return f"{value:+,.{digits}f}"


def format_crossing(record):
    if record is None:
        return "未观测到"
    if not record["interval_lower_open"]:
        return "初始（0）"
    return f"({record['interval_lower_step']:,}, {record['interval_upper_step']:,}]"


def format_sustained(record):
    if record["sustained"] is not None:
        return format_crossing(record["sustained"])
    if record["sustained_status"] == "insufficient_follow_up":
        entry = record["sustained_candidate"]["step"]
        return f"尾段观测不足（{entry:,}）"
    if record["sustained_status"] == "fell_below_by_endpoint":
        return "终点低于门槛"
    return "未达标"


def arm_stats(values):
    return {"count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}


def paired_stats(matches):
    observed = [p for p in matches if p is not None]
    if not observed:
        return {"pairs": 0}
    steps = [p["uniform_minus_per_checkpoint_steps"] for p in observed]
    return {
        "pairs": len(observed),
        "per_earlier": sum(step > 0 for step in steps),
        "uniform_earlier": sum(step < 0 for step in steps),
        "same_sampled_step": steps.count(0),
        "uniform_minus_per_steps": arm_stats(steps),
        "uniform_minus_per_cpu_seconds": arm_stats([item["uniform_minus_per_cpu_seconds"] for item in observed]),
    }


def summarise(directory, protocol, protocol_sha):
    evaluation = protocol["evaluation"]
    thresholds = protocol["thresholds"]
    target = thresholds["primary_pct"]
    sensitivity = thresholds["sensitivity_pct"]
    tail = thresholds["minimum_sustained_tail_steps"]
    checkpoint_count = len(evaluation["sample_steps"])
    comparisons = []
    curves = []
    source = []
    for seed in SEEDS:
        arms = {}
        for arm in ARMS:
            path = directory / f"{arm}_seed{seed}.json"
            run = checked_run(path, arm, seed, protocol, protocol_sha)
            arms[arm] = run
            source.append({
                "arm": arm, "seed": seed, "file": path.name, "evaluation_sha256": digest(path),
                "training_receipt_sha256": run["receipt_sha256"],
                "endpoint_reference_matched": run["endpoint_reference_matched"],
            })
            for record in run["rows"]:
                curves.append(dict(arm=arm, seed=seed, step=record["step"], cpu_seconds=record["cpu_seconds"],
                                   wall_seconds=record["wall_seconds"],
                                   **{name: record["metrics"][name] for name in METRICS},
                                   actions_sha256=record["actions_sha256"], checkpoint_sha256=record["sha256"]))
        per, uniform = (arms[arm]["rows"] for arm in ARMS)
        require(per[0]["actions_sha256"] == uniform[0]["actions_sha256"], f"Paired start differs at seed {seed}.")
        indexed = {arm: {row["step"]: row for row in arms[arm]["rows"]} for arm in ARMS}
        comparison = {
            "seed": seed,
            "early_pooled_pct": {str(step): {arm: indexed[arm][step]["metrics"]["pooled_pct"] for arm in ARMS}
                                 for step in EARLY_STEPS},
            "terminal_pooled_pct": {arm: arms[arm]["rows"][-1]["metrics"]["pooled_pct"] for arm in ARMS},
            "timing_at_fixed_budget": paired_budget(per[-1], uniform[-1]),
            "thresholds": [],
        }
        for limit in [target, *sensitivity]:
            per_attainment = attainment(per, limit, tail, evaluation["primary_metric"])
            uniform_attainment = attainment(uniform, limit, tail, evaluation["primary_metric"])
            comparison["thresholds"].append({
                "threshold_pct": limit, "per": per_attainment, "uniform": uniform_attainment,
                "paired_first": paired_crossing(per_attainment["first"], uniform_attainment["first"]),
                "paired_sustained": paired_crossing(per_attainment["sustained"], uniform_attainment["sustained"]),
            })
        comparisons.append(comparison)

    aggregate = []
    for index, threshold in enumerate([target, *sensitivity]):
        entries = [pair["thresholds"][index] for pair in comparisons]
        aggregate.append({
            "threshold_pct": threshold,
            "first_observed": {arm: sum(entry[arm]["first"] is not None for entry in entries) for arm in ARMS},
            "sustained_observed": {arm: sum(entry[arm]["sustained"] is not None for entry in entries) for arm in ARMS},
            "paired_first": paired_stats([entry["paired_first"] for entry in entries]),
            "paired_sustained": paired_stats([entry["paired_sustained"] for entry in entries]),
        })

    timing = [pair["timing_at_fixed_budget"] for pair in comparisons]
    totals = {arm: sum(pair[f"{arm}_cpu_seconds"] for pair in timing) for arm in ARMS}
    summary = {
        "schema": "n13_convergence_summary_v1",
        "status": "pass",
        "protocol_sha256": protocol_sha,
        "base_commit": protocol["base_commit"],
        "seeds": list(SEEDS),
        "formal_runs": len(source),
        "new_training_interactions": len(source) * protocol["interactions_per_seed"],
        "checkpoints_per_run": checkpoint_count,
        "evaluated_windows": len(curves),
        "evaluation_simulated_steps": len(curves) * evaluation["rows_per_window"],
        "evaluation": {name: value for name, value in evaluation.items() if name != "sample_steps"},
        "thresholds": thresholds,
        "source_evaluations": source,
        "early_pooled_pct_mean": {str(step): {
            arm: mean(pair["early_pooled_pct"][str(step)][arm] for pair in comparisons)
            for arm in ARMS} for step in EARLY_STEPS},
        "fixed_budget_timing": {
            "per_cpu_seconds": arm_stats([pair["per_cpu_seconds"] for pair in timing]),
            "uniform_cpu_seconds": arm_stats([pair["uniform_cpu_seconds"] for pair in timing]),
            "paired_uniform_minus_per_cpu_seconds": arm_stats([pair["uniform_minus_per_cpu_seconds"] for pair in timing]),
            "paired_per_cpu_saving_pct_relative_to_uniform": arm_stats(
                [pair["per_cpu_saving_pct_relative_to_uniform"] for pair in timing]),
            "total_cpu_seconds_by_arm": totals,
            "total_cpu_seconds": sum(totals.values()),
        },
        "threshold_summary": aggregate,
        "pairs": comparisons,
        "scope": protocol["scope"],
        "timing_scope": protocol["timing"],
    }
    return summary, curves


def report(summary):
    target = summary["thresholds"]["primary_pct"]
    tail = summary["thresholds"]["minimum_sustained_tail_steps"]
    result = summary["threshold_summary"][0]
    timing = summary["fixed_budget_timing"]
    source = summary["source_evaluations"]
    verified_endpoints = sum(row["endpoint_reference_matched"] for row in source)
    text = [
        "# N13 采样达标与训练速度",
        "",
        f"用四个配对随机种子比较 PER 和均匀有放回回放。每个配置按相同的动作参考、奖励和约束训练 120 万步。",
        f"共完成 {summary['formal_runs']} 次训练、{summary['new_training_interactions']:,} 次仿真交互。",
        f"每次训练保留 {summary['checkpoints_per_run']} 个检查点。固定权重重放了 {summary['evaluated_windows']} 个主窗口，"
        f"共 {summary['evaluation_simulated_steps']:,} 个评估时步。"
        f"评估采用相同的 17,520 点归档主窗口和起止电量 37.5 kWh。",
        "",
        f"主门槛为全期方差下降 {target}%。持续达标要求从某个检查点开始，后续所有采样点均达标，"
        f"并至少有 {tail:,} 步后续观察。区间以该点前一检查点和该点为边界，仅反映采样所见。"
        "记录中未能确认持续达标的情况保留为删失。",
        "",
        "## 第 18,000 和 40,000 步",
        "",
        "| 种子 | PER 18,000 | 均匀 18,000 | PER 40,000 | 均匀 40,000 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["pairs"]:
        a, b = (row["early_pooled_pct"][str(step)] for step in EARLY_STEPS)
        cells = [number(series[arm]) + "%" for series in (a,b) for arm in ARMS]
        text.append(f"| {row['seed']} | " + " | ".join(cells) + " |")
    text += [
        "",
        f"## {target}% 主门槛的逐种子结果",
        "",
        "| 种子 | PER 首次 | 均匀首次 | PER 持续 | 均匀持续 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary["pairs"]:
        pair = row["thresholds"][0]
        cells = [format_crossing(pair[arm]["first"]) for arm in ARMS]
        cells.extend(format_sustained(pair[arm]) for arm in ARMS)
        text.append(f"| {row['seed']} | " + " | ".join(cells) + " |")

    text += [
        "",
        f"PER 有 {result['sustained_observed']['per']}/4 个种子持续达标，"
        f"均匀回放有 {result['sustained_observed']['uniform']}/4 个。",
    ]
    for label, key in (("首次", "paired_first"), ("持续", "paired_sustained")):
        paired = result[key]
        if paired["pairs"]:
            delta = paired["uniform_minus_per_steps"]["mean"]
            text.append(f"{paired['pairs']} 对种子均可比较{label}达标步数。PER 较早 {paired['per_earlier']} 对，"
                        f"均匀回放较早 {paired['uniform_earlier']} 对，同一检查点 {paired['same_sampled_step']} 对。"
                        f"均匀减 PER 的平均检查点差为 {signed(delta, 0)} 步，"
                        f"累计进程 CPU 时间差均值为 {signed(paired['uniform_minus_per_cpu_seconds']['mean'])} 秒。")
        else:
            text.append(f"{label}达标步数没有两侧均可观察的配对。")
    text += [
        "",
        "## 门槛敏感性",
        "",
        "| 全期方差门槛 | PER 持续达标 | 均匀持续达标 | 两侧持续达标配对 | 均匀减 PER 平均步数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["threshold_summary"]:
        paired = entry["paired_sustained"]
        delta = signed(paired["uniform_minus_per_steps"]["mean"], 0) if paired["pairs"] else "未观测到"
        sustained = entry["sustained_observed"]
        text.append(f"| {entry['threshold_pct']}% | {sustained['per']}/4 | {sustained['uniform']}/4"
                    f" | {paired['pairs']} | {delta} |")
    text += [
        "",
        "## 固定训练预算的进程 CPU 时间",
        "",
        "| 种子 | PER 秒 | 均匀回放秒 | 均匀减 PER 秒 | PER 相对均匀节省率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["pairs"]:
        paired = row["timing_at_fixed_budget"]
        text.append(f"| {row['seed']} | {number(paired['per_cpu_seconds'])} | {number(paired['uniform_cpu_seconds'])}"
                    f" | {signed(paired['uniform_minus_per_cpu_seconds'])}"
                    f" | {signed(paired['per_cpu_saving_pct_relative_to_uniform'])}% |")
    text += [
        "",
        f"PER 和均匀回放的平均进程 CPU 时间分别为 {number(timing['per_cpu_seconds']['mean'])} 秒和"
        f" {number(timing['uniform_cpu_seconds']['mean'])} 秒。"
        f"配对节省率的均值为 {signed(timing['paired_per_cpu_saving_pct_relative_to_uniform']['mean'])}%。"
        "正值表示 PER 使用较少 CPU 时间。"
        f"八次训练累计使用 {number(timing['total_cpu_seconds'])} 秒进程 CPU 时间。",
        "",
        "CPU 计时覆盖训练计算与内存记账，暂停于检查点写入和训练日志落盘。主机并发影响墙钟耗时，"
        "逐种子与逐检查点的墙钟值保留在表格中。均匀回放沿用树结构的插入路径和公共的更新后残差计算，"
        "所以耗时比较适用于这两份具体实现。",
        "",
        f"固定权重终点有 {verified_endpoints}/{len(source)} 个与既有归档端点复核一致。"
        "本次训练与评估共享当前归档主窗口。结果适用于这一数据和配置。"
        "历史 Base DQN 属于另一训练配置。"
        "旧稿的 18,000 步、40,000 步和约 35% 训练时间节省仍需相应的历史证据。",
        "",
    ]
    return "\n".join(text)


def per_seed_rows(summary):
    for pair in summary["pairs"]:
        timing = pair["timing_at_fixed_budget"]
        for entry in pair["thresholds"]:
            short = dict(seed=pair["seed"], threshold_pct=entry["threshold_pct"])
            for arm in ARMS:
                item = entry[arm]
                for kind in ("first", "sustained"):
                    observed = item[kind] or {}
                    short[f"{arm}_{kind}_status"] = item[f"{kind}_status"]
                    for name in ("step", "interval_lower_step", "interval_upper_step", "cpu_seconds_at_checkpoint",
                                 "wall_seconds_at_checkpoint"):
                        short[f"{arm}_{kind}_{name}"] = observed.get(name)
                    short[f"{arm}_{kind}_right_censored_after_step"] = item[f"{kind}_right_censored_after_step"]
                candidate = item["sustained_candidate"] or {}
                short[f"{arm}_unconfirmed_candidate_step"] = candidate.get("step")
                short[f"{arm}_unconfirmed_candidate_tail_steps"] = candidate.get("observed_tail_steps")
                short[f"{arm}_pooled_pct_18000"] = pair["early_pooled_pct"]["18000"][arm]
                short[f"{arm}_pooled_pct_40000"] = pair["early_pooled_pct"]["40000"][arm]
                short[f"{arm}_pooled_pct_1200000"] = pair["terminal_pooled_pct"][arm]
            for label in ("first", "sustained"):
                observed_pair = entry[f"paired_{label}"] or {}
                for name in ("uniform_minus_per_checkpoint_steps", "uniform_minus_per_cpu_seconds",
                             "uniform_minus_per_wall_seconds"):
                    short[f"{label}_{name}"] = observed_pair.get(name)
            yield dict(short, **timing)


def tsv(path, rows):
    rows = list(rows)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL_FILE)
    steps = protocol["evaluation"]["sample_steps"]
    require(protocol["schema"] == "n13_sampled_convergence_protocol_v1"
            and tuple(protocol["arms"]) == ARMS and tuple(protocol["seeds"]) == SEEDS
            and len(steps) == 84 and sorted(set(steps)) == steps and steps[0] == 0
            and all(step in steps for step in EARLY_STEPS) and steps[-1] == protocol["interactions_per_seed"] == 1200000,
            "Unexpected sampled convergence protocol.")
    summary, curves = summarise(args.evaluations.resolve(strict=True), protocol, digest(PROTOCOL_FILE))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "convergence_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output / "N13收敛速度核验报告.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(summary))
    tsv(args.output / "per_seed.tsv", per_seed_rows(summary))
    tsv(args.output / "checkpoint_curve.tsv", curves)
    print(f"{summary['formal_runs']} audited runs, {len(curves)} checkpoint evaluations.")


if __name__ == "__main__":
    main()

