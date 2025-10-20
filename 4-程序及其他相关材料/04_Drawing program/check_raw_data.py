#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查raw_data中的parquet文件数据结构
"""

import pandas as pd
from pathlib import Path

def check_raw_data():
    """检查raw_data中的文件结构"""
    raw_data_dir = Path("E:/Mgrid/final-project/raw_data")
    
    parquet_files = [
        "load_15min_fullyear.parquet",
        "price_15min_fullyear.parquet", 
        "ev_demand_15min_fullyear.parquet"
    ]
    
    for file_name in parquet_files:
        file_path = raw_data_dir / file_name
        if file_path.exists():
            print(f"检查文件: {file_name}")
            try:
                df = pd.read_parquet(file_path)
                print(f"列名: {list(df.columns)}")
                print(f"数据形状: {df.shape}")
                print(f"数据类型:")
                for col in df.columns:
                    print(f"{col}: {df[col].dtype}")
                print(f"前5行数据:")
                print(df.head())
            except Exception as e:
                print(f"读取失败: {e}")
        else:
            print(f"文件不存在: {file_name}")
    
    json_files = ["soh_params_fullyear.json"]
    for file_name in json_files:
        file_path = raw_data_dir / file_name
        if file_path.exists():
            print(f"检查文件: {file_name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                print(f"JSON内容: {data}")
            except Exception as e:
                print(f"读取失败: {e}")
        else:
            print(f"文件不存在: {file_name}")

if __name__ == "__main__":
    check_raw_data() 