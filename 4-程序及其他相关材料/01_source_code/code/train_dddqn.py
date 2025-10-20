#!/usr/bin/env python3
"""
DDDQN训练脚本
用于电动汽车充放电控制的双重深度Q网络训练
"""

import argparse
import pandas as pd
import numpy as np
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from dddqn_per_sb3 import PrioritizedReplayBuffer
from stable_baselines3.dqn.policies import DQNPolicy
from evves_env import EVVESEnv
import warnings
warnings.filterwarnings("ignore")

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_env(load_data: pd.DataFrame, price_data: pd.DataFrame, 
            ev_data: pd.DataFrame, soh_params: dict, seed: int = 0):
    """创建带随机种子的环境"""
    def _init():
        # Convert soh_params to config format expected by EVVESEnv
        config = {'soh_params': soh_params} if soh_params else None
        env = EVVESEnv(price_data, load_data, ev_data, config)
        return env
    return _init


class TrainingDataCollector:
    """收集训练reward和step，保存为json。"""
    def __init__(self, save_path):
        self.save_path = Path(save_path) / "training_data.json"
        self.episode_rewards = []
        self.episode_steps = []
        self.current_reward = 0
        self.current_steps = 0

    def __call__(self, locals_, globals_):
        # 兼容SB3回调接口
        if 'reward' in locals_:
            self.current_reward += locals_['reward']
            self.current_steps += 1
        if locals_.get('done', False):
            self.episode_rewards.append(self.current_reward)
            self.episode_steps.append(self.current_steps)
            self.current_reward = 0
            self.current_steps = 0
        return True

    def save_training_data(self):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps
        }
        with open(self.save_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(data, f, indent=2, ensure_ascii=False)


def train_dddqn(args):
    """训练DDDQN模型"""
    # 加载数据
    load_data = pd.read_parquet(args.load)
    price_data = pd.read_parquet(args.price)
    ev_data = pd.read_parquet(args.ev)
    with open(args.soh, 'r') as f:
        soh_params = json.load(f)
    
    # 创建向量化环境
    env_fns = [make_env(load_data, price_data, ev_data, soh_params, seed=args.seed)]
    env = SubprocVecEnv(env_fns)
    
    # 模型配置
    model_config = {
        'policy': 'MlpPolicy',
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'learning_starts': 1000,
        'batch_size': 64,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05
    }
    
    # 初始化模型
    model = DQN(env=env, **model_config)
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="dddqn"
    )
    
    # 训练模型
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_callback,
        log_interval=4
    )
    
    # 保存最终模型
    model.save(f"{args.save_dir}/dddqn_final.zip")
    logger.info("训练完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练DDDQN模型")
    
    parser.add_argument("--load", required=True, help="负荷数据文件")
    parser.add_argument("--price", required=True, help="电价数据文件")
    parser.add_argument("--ev", required=True, help="EV需求数据文件")
    parser.add_argument("--soh", required=True, help="SOH参数文件")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--steps", type=int, default=1200000, help="训练步数")
    parser.add_argument("--buffer_size", type=int, default=150000, help="回放缓冲区大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="检查点频率")
    parser.add_argument("--save_dir", default="models", help="模型保存目录")
    
    args = parser.parse_args()
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        train_dddqn(args)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    main() 