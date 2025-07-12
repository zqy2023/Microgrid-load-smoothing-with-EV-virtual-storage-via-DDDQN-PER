#!/usr/bin/env python3
"""测试训练脚本的基本功能"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """创建测试数据"""
    # 创建测试数据目录
    data_dir = Path("../../02_datasets/data_processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试负荷数据
    n_steps = 1000
    load_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='15min'),
        'load_kw': np.random.normal(100, 20, n_steps)
    })
    load_data.to_parquet(data_dir / "load.parquet")
    
    # 创建测试电价数据
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='15min'),
        'price': np.random.uniform(50, 150, n_steps)
    })
    price_data.to_parquet(data_dir / "price.parquet")
    
    # 创建测试EV数据
    ev_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_steps, freq='15min'),
        'available': np.random.choice([True, False], n_steps, p=[0.8, 0.2])
    })
    ev_data.to_parquet(data_dir / "ev.parquet")
    
    # 创建SOH参数
    soh_params = {
        'calendar_aging': 0.0001,
        'cycle_aging': 0.00005,
        'temperature_factor': 1.0,
        'depth_factor': 1.2
    }
    
    import json
    with open(data_dir / "soh_params.json", 'w') as f:
        json.dump(soh_params, f, indent=2)
    
    print("测试数据创建完成")

def test_imports():
    """测试导入"""
    try:
        from evves_env import EVVESEnv
        print("✓ EVVESEnv 导入成功")
        
        from dddqn_per_sb3 import DDDQN_PER_Agent
        print("✓ DDDQN_PER_Agent 导入成功")
        
        from train_dddqn import TrainingDataCollector
        print("✓ TrainingDataCollector 导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_environment():
    """测试环境"""
    try:
        from evves_env import EVVESEnv
        
        # 创建测试数据
        n_steps = 100
        price_data = pd.DataFrame({
            'price': np.random.uniform(50, 150, n_steps)
        })
        load_data = pd.DataFrame({
            'load_kw': np.random.normal(100, 20, n_steps)
        })
        ev_data = pd.DataFrame({
            'available': np.random.choice([True, False], n_steps, p=[0.8, 0.2])
        })
        
        # 传入完整环境配置
        config = {
            'battery_capacity': 75.0,
            'max_power': 50.0,
            'initial_soc': 0.5,
            'episode_length': 50,
            'soh_params': {
                'calendar_aging': 0.0001,
                'cycle_aging': 0.00005,
                'temperature_factor': 1.0,
                'depth_factor': 1.2
            },
            'reward_weights': {
                'economic': 1.0,
                'degradation': 0.5,
                'grid_stability': 0.3
            },
            'time_step': 0.25
        }
        env = EVVESEnv(price_data, load_data, ev_data, config)
        print("✓ 环境创建成功")
        
        # 测试reset和step
        obs = env.reset()
        print(f"✓ 环境reset成功，观察空间: {obs.shape}")
        
        action = 10  # 中间动作
        obs, reward, done, info = env.step(action)
        print(f"✓ 环境step成功，reward: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试训练脚本...")
    
    # 创建测试数据
    create_test_data()
    
    # 测试导入
    if not test_imports():
        print("导入测试失败，退出")
        return
    
    # 测试环境
    if not test_environment():
        print("环境测试失败，退出")
        return
    
    print("所有测试通过！训练脚本应该可以正常运行。")

if __name__ == "__main__":
    main() 