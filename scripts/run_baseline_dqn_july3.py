#!/usr/bin/env python3
"""
基准DQN训练脚本
使用7月3日DDDQN-PER相同配置运行基准DQN
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_baseline_dqn_training():
    """运行基准DQN训练"""
    
    logger.info("开始基准DQN训练")
    logger.info("使用7月3日DDDQN-PER配置")
    logger.info("="*80)
    
    # 检查数据文件
    data_files = {
        "load": "data_processed/load_15min_fullyear.parquet",
        "price": "data_processed/price_15min_fullyear.parquet", 
        "ev": "data_processed/ev_demand_15min_fullyear.parquet",
        "soh": "data_processed/soh_params_fullyear.json"
    }
    
    logger.info("检查数据文件...")
    for name, path in data_files.items():
        if Path(path).exists():
            logger.info(f"已找到: {name}: {path}")
        else:
            logger.error(f"未找到: {name}: {path}")
            return False
    
    # 训练配置
    training_config = {
        "training_steps": 1200000,  # 与DDDQN-PER相同
        "episode_length": 96,       # 与DDDQN-PER相同
        "buffer_size": 150000,      # 基准DQN使用较小缓冲区
        "network_width": 128,       # 基准DQN使用较小网络
        "learning_rate": 3e-4,      # 与DDDQN-PER相同
        "seeds": [0, 1, 2, 3]       # 4个种子
    }
    
    logger.info("训练配置:")
    logger.info(f"   训练步数: {training_config['training_steps']:,}")
    logger.info(f"   Episode长度: {training_config['episode_length']}")
    logger.info(f"   缓冲区大小: {training_config['buffer_size']:,}")
    logger.info(f"   网络宽度: {training_config['network_width']}")
    logger.info(f"   学习率: {training_config['learning_rate']}")
    logger.info(f"   随机种子: {training_config['seeds']}")
    
    # 运行训练
    for seed in training_config['seeds']:
        logger.info(f"\n开始种子 {seed} 的训练...")
        
        try:
            # 构建命令
        cmd = [
                "python", "-u",  # 无缓冲输出
                "01_source_code/code/baseline_dqn_train.py",
                "--load", data_files['load'],
                "--price", data_files['price'],
                "--ev", data_files['ev'],
                "--soh", data_files['soh'],
            "--seed", str(seed),
            "--steps", str(training_config['training_steps']),
            "--episode_length", str(training_config['episode_length']),
                "--buffer_size", str(training_config['buffer_size']),
                "--network_width", str(training_config['network_width']),
                "--learning_rate", str(training_config['learning_rate'])
            ]
            
            # 执行训练
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时输出日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())
            
            # 检查训练结果
            if process.returncode != 0:
                logger.error(f"种子 {seed} 训练失败")
                return False
            
            logger.info(f"种子 {seed} 训练完成")
            
        except Exception as e:
            logger.error(f"种子 {seed} 训练出错: {e}")
            return False
    
    logger.info("\n所有训练完成")
        return True


if __name__ == "__main__":
    run_baseline_dqn_training() 