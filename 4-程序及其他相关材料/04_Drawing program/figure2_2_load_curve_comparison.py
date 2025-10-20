#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负荷曲线对比脚本 (替代 create_figure_5_1.py)
------------------------------------------------
功能：
1. 读取基线与多条策略负荷曲线（CSV/Parquet/JSON）。
2. 自动按时间索引对齐并插值补缺。
3. 计算 MSE/MAE/MAPE 指标。
4. 生成对比图并保存 PNG/PDF。
5. 输出指标 JSON 方便论文引用。

使用示例：
    python load_curve_comparison.py \
        --baseline data_processed/load_baseline.csv \
        --strategy results/dddqn_load.csv \
        --label_baseline "传统策略" \
        --label_strategy "DDDQN-PER" \
        --output_prefix figures/fig5_1_load_curve
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")  # 统一绘图风格


def _load_series(path: Path) -> pd.Series:
    """根据文件后缀读取为 pd.Series（索引为时间）。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if df.index.dtype != "datetime64[ns]":
            df.set_index(df.columns[0], inplace=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
        if df.index.dtype != "datetime64[ns]":
            df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    series = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    return series.sort_index()


def _align_series(reference: pd.Series, series_list: List[pd.Series]) -> List[pd.Series]:
    """将 series_list 时间对齐到 reference 索引并插值补缺。"""
    aligned = []
    for s in series_list:
        s_aligned = s.reindex(reference.index)
        s_aligned = s_aligned.interpolate(method="time")
        aligned.append(s_aligned)
    return aligned


def _metrics(baseline: pd.Series, target: pd.Series) -> dict:
    """计算误差指标。"""
    diff = target - baseline
    mse = float((diff ** 2).mean())
    mae = float(diff.abs().mean())
    safe_baseline = baseline.replace(0, pd.NA)
    mape = float((diff.abs() / safe_baseline).mean(skipna=True) * 100)
    return {"MSE": mse, "MAE": mae, "MAPE": mape}


def _plot(baseline: pd.Series, strategies: List[pd.Series], labels: List[str], out_prefix: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(baseline.index, baseline.values, label=labels[0], linewidth=2)
    for s, lab in zip(strategies, labels[1:]):
        plt.plot(s.index, s.values, label=lab, linewidth=2)

    plt.xlabel("时间")
    plt.ylabel("负荷 (kW)")
    plt.title("负荷曲线对比")
    plt.legend()
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.savefig(out_prefix.with_suffix(".pdf"))
    plt.close()


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="负荷曲线对比")
    parser.add_argument("--baseline", required=True, help="基线负荷曲线文件")
    parser.add_argument("--strategy", required=True, nargs="+", help="策略负荷曲线文件列表")
    parser.add_argument("--label_baseline", default="Baseline", help="基线标签")
    parser.add_argument("--label_strategy", nargs="+", help="策略标签，与 --strategy 数量一致")
    parser.add_argument("--output_prefix", help="输出前缀（不含扩展名），默认 figures/load_curve_<timestamp>")

    args = parser.parse_args(argv)

    baseline = _load_series(Path(args.baseline))
    strategies = [_load_series(Path(p)) for p in args.strategy]
    strategies = _align_series(baseline, strategies)

    # 处理标签
    labels = [args.label_baseline]
    if args.label_strategy:
        if len(args.label_strategy) != len(strategies):
            parser.error("--label_strategy 数量需与 --strategy 相同")
        labels.extend(args.label_strategy)
    else:
        labels.extend([f"Strategy_{i}" for i in range(len(strategies))])

    # 输出前缀
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = Path("figures") / f"load_curve_{ts}"

    # 计算并保存指标
    metrics = {lab: _metrics(baseline, s) for lab, s in zip(labels[1:], strategies)}
    metrics_file = prefix.with_suffix("_metrics.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    # 绘图
    _plot(baseline, strategies, labels, prefix)

    print("图表保存至:", prefix.with_suffix(".png"))
    print("指标保存至:", metrics_file)


if __name__ == "__main__":
    main() 