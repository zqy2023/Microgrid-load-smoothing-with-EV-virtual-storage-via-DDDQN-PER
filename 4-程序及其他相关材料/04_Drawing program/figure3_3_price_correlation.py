#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负荷-价格相关性分析脚本
========================
替代旧 `fig3_3_load_price_correlation_heatmap.py`。

功能：
1. 读取负荷曲线与电价曲线（支持 CSV/Parquet/JSON），确保时间索引对齐。
2. 可按天、周、整周期等窗口重采样后计算皮尔逊相关系数。
3. 输出相关系数矩阵 (CSV) 与热图 (PNG/PDF)。
4. 简洁中文注释，无 emoji。

使用示例：
    python load_price_correlation.py \
        --load data_processed/load_15min_fullyear.parquet \
        --price data_processed/price_15min_fullyear.parquet \
        --resample 1D \
        --output_prefix figures/fig3_3_load_price_corr
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

# -----------------------------------------------------------------------------

def _load_series(path: Path, name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")

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

    if isinstance(df, pd.DataFrame):
        # 若有多列，默认取第一列
        series = df.iloc[:, 0]
    else:
        series = df

    series = series.sort_index()
    series.name = name
    return series


def _align_and_resample(load: pd.Series, price: pd.Series, rule: Optional[str]) -> tuple[pd.Series, pd.Series]:
    # 对齐索引
    idx = load.index.intersection(price.index)
    load_aligned = load.reindex(idx)
    price_aligned = price.reindex(idx)

    if rule:
        load_aligned = load_aligned.resample(rule).mean()
        price_aligned = price_aligned.resample(rule).mean()

    # 再次删除任意缺失
    aligned_idx = load_aligned.dropna().index.intersection(price_aligned.dropna().index)
    return load_aligned.reindex(aligned_idx), price_aligned.reindex(aligned_idx)


def _compute_correlations(load: pd.Series, price: pd.Series) -> pd.DataFrame:
    df = pd.concat([load, price], axis=1)
    corr = df.corr(method="pearson")
    return corr


def _plot_heatmap(corr: pd.DataFrame, prefix: Path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.title("负荷-电价相关性")
    plt.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(prefix.with_suffix(".png"), dpi=300)
    plt.savefig(prefix.with_suffix(".pdf"))
    plt.close()


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="负荷-价格相关性分析")
    parser.add_argument("--load", required=True, help="负荷曲线文件")
    parser.add_argument("--price", required=True, help="电价曲线文件")
    parser.add_argument("--resample", help="重采样规则，如 1H、1D")
    parser.add_argument("--output_prefix", help="输出前缀，默认 figures/load_price_corr_<timestamp>")

    args = parser.parse_args(argv)

    load_series = _load_series(Path(args.load), "load")
    price_series = _load_series(Path(args.price), "price")

    load_aligned, price_aligned = _align_and_resample(load_series, price_series, args.resample)

    corr_df = _compute_correlations(load_aligned, price_aligned)

    # 输出前缀
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = Path("figures") / f"load_price_corr_{ts}"

    corr_csv = prefix.with_suffix("_matrix.csv")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(corr_csv)

    _plot_heatmap(corr_df, prefix)

    print("相关性矩阵已保存:", corr_csv)
    print("热图已保存至:", prefix.parent.resolve())


if __name__ == "__main__":
    main() 