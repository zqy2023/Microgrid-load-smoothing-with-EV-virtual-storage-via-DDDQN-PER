#!/usr/bin/env python3
"""
结果可视化工具
处理和分析DDDQN实验结果
"""

import argparse
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultsProcessor:
    """实验结果处理器"""
    
    def __init__(self):
        self.metrics = ['reward', 'cost', 'soc', 'grid_impact']
        
    def load_training_data(self, log_dir: str) -> pd.DataFrame:
        """加载训练日志数据"""
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        data = {}
        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            data[tag] = pd.DataFrame(events)
        
        return data
    
    def process_episode_data(self, data_dir: str) -> pd.DataFrame:
        """处理每轮训练数据"""
        episode_files = Path(data_dir).glob("episode_*.csv")
        dfs = []
        
        for file in episode_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """计算基础统计指标"""
        stats = {}
        for metric in self.metrics:
            if metric in df.columns:
                stats[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'median': df[metric].median()
                }
        return stats


def main():
    """Main results processing pipeline."""
    parser = argparse.ArgumentParser(description="Process EVVES-DDDQN results")
    
    parser.add_argument("--data_dir", required=True, help="Directory containing experiment data")
    parser.add_argument("--out", required=True, help="Output directory for processed data")
    
    args = parser.parse_args()
    
    try:
        processor = ResultsProcessor()
        
        # Load and process data
        training_data = processor.load_training_data(args.data_dir)
        episode_data = processor.process_episode_data(args.data_dir)
        
        # Calculate statistics
        stats = processor.calculate_statistics(episode_data)
        
        # Save processed data
        Path(args.out).mkdir(parents=True, exist_ok=True)
        
        with open(Path(args.out) / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        for metric, df in training_data.items():
            df.to_csv(Path(args.out) / f"{metric}_data.csv", index=False)
        
        logger.info("Results processing completed")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main() 