#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练过程监控脚本
================
替代旧 `fig4_1_learning_monitoring_panel.py` 功能。

特点：
1. 扫描多个训练输出目录（`--run_dirs`），自动识别 `progress.csv` 或 `training_progress.csv`。
2. 支持绘制多指标曲线（默认 episode_reward、loss）。
3. 图表按 PNG + PDF 双格式输出到 `figures/`；所有曲线共用同一配色方案，方便比较。
4. 若目录内包含 `training_summary.json`，则汇总保存为单个 JSON。
5. 纯中文注释，无 emoji。

使用示例：
    python learning_process_monitor.py \
        --run_dirs results/train_seed_0_20250625_235132 results/train_seed_1_20250626_003424 \
        --labels 种子0 种子1 \
        --metrics episode_reward loss epsilon
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---- 全局绘图风格 -----------------------------------------------------------
sns.set_theme(style="ticks", font="Times New Roman", palette="tab10")
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# ---- 常量 -------------------------------------------------------------------
_PROG_CSV_CANDIDATES = ["progress.csv", "training_progress.csv"]
_SUMMARY_JSON_NAME = "training_summary.json"

# ---- 辅助函数 ----------------------------------------------------------------

def _find_progress_file(run_dir: Path) -> Optional[Path]:
    """在 `run_dir` 下寻找进度 CSV。"""
    for name in _PROG_CSV_CANDIDATES:
        fp = run_dir / name
        if fp.is_file():
            return fp
    return None


def _load_progress(csv_path: Path) -> pd.DataFrame:
    """读取进度 CSV 并统一列名。"""
    df = pd.read_csv(csv_path)

    # 统一 timesteps 列
    if "timesteps" not in df.columns:
        if "step" in df.columns:
            df.rename(columns={"step": "timesteps"}, inplace=True)
        elif "global_step" in df.columns:
            df.rename(columns={"global_step": "timesteps"}, inplace=True)
        else:
            raise ValueError(f"{csv_path} 缺少 timesteps/step 列")

    return df


def _collect_runs(run_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    """收集多个 run 的进度 DataFrame。"""
    data: Dict[str, pd.DataFrame] = {}
    for d in run_dirs:
        prog_fp = _find_progress_file(d)
        if not prog_fp:
            print(f"[警告] {d} 未找到 progress CSV，已跳过。", file=sys.stderr)
            continue
        try:
            df = _load_progress(prog_fp)
            data[d.name] = df
        except Exception as e:
            print(f"[警告] 读取 {prog_fp} 失败: {e}", file=sys.stderr)
    return data


def _plot_metric(runs: Dict[str, pd.DataFrame], metric: str, labels: List[str] | None, out_prefix: Path):
    """绘制单个指标曲线。"""
    plt.figure(figsize=(10, 5))

    palette = sns.color_palette("tab10", n_colors=len(runs))
    for i, (run_name, df) in enumerate(runs.items()):
        if metric not in df.columns:
            print(f"[提示] {run_name} 不包含列 {metric}，跳过。", file=sys.stderr)
            continue
        label = labels[i] if labels and i < len(labels) else run_name
        sns.lineplot(x=df["timesteps"], y=df[metric], label=label, color=palette[i])

    plt.xlabel("训练步数")
    plt.ylabel(metric)
    plt.title(f"{metric} 曲线对比")
    plt.legend()
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(f"_{metric}.png"), dpi=300)
    plt.savefig(out_prefix.with_suffix(f"_{metric}.pdf"))
    plt.close()

# ---- 主函数 ------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="训练过程监控脚本")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="训练输出目录列表（含 progress.csv）")
    parser.add_argument("--labels", nargs="*", help="曲线标签，可选，与 run_dirs 对应")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["episode_reward"],
        help="需要绘制的指标列名，默认只绘制 episode_reward",
    )
    parser.add_argument("--output_prefix", help="输出文件前缀，默认 figures/learning_monitor_<timestamp>")

    args = parser.parse_args(argv)

    run_dirs = [Path(p) for p in args.run_dirs]
    runs = _collect_runs(run_dirs)
    if not runs:
        parser.error("未能找到任何有效 progress CSV")

    # 输出前缀
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = Path("figures") / f"learning_monitor_{ts}"

    # 绘制指标
    for metric in args.metrics:
        _plot_metric(runs, metric, args.labels, prefix)

    # 汇总 training_summary.json（如有）
    summaries: Dict[str, dict] = {}
    for d in run_dirs:
        summary_fp = d / _SUMMARY_JSON_NAME
        if summary_fp.is_file():
            try:
                summaries[d.name] = json.loads(summary_fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[警告] 无法解析 {summary_fp}", file=sys.stderr)
    if summaries:
        summary_out = prefix.with_suffix("_summary.json")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print("已输出汇总 JSON:", summary_out)

    print("图表输出目录:", prefix.parent.resolve())


if __name__ == "__main__":
    main() 