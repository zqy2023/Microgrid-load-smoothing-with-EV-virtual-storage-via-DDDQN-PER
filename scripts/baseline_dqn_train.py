#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline-DQN training script with 6-month episode (17,520 steps). Uses real data only."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from evves_env import EVVESEnv  # 绝对导入
from train_dddqn import TrainingDataCollector


def _make_env(price_df, load_df, ev_df, soh_dict, episode_len):
    """Create environment factory for DummyVecEnv."""
    def _init():
        # Create complete config with default values
        config = {
            'episode_length': episode_len,
            'battery_capacity': 75.0,  # Paper config: 75 kWh
            'max_power': 50.0,  # Paper config: ±50kW
            'initial_soc': 0.5,  # 50%
            'soh_params': soh_dict,
            'reward_weights': {
                'economic': 1.0,
                'degradation': 0.5,
                'grid_stability': 0.3
            },
            'time_step': 0.25  # 15 minutes = 0.25 hours
        }
        env = EVVESEnv(price_data=price_df,
                       load_data=load_df,
                       ev_data=ev_df,
                       config=config)
        return Monitor(env)
    return _init


def train(ns: argparse.Namespace) -> bool:
    """Train Baseline-DQN model."""
    logger.info("Training Baseline-DQN | seed=%d", ns.seed)

    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f"results/train_seed_{ns.seed}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    price_df = pd.read_parquet(ns.price)
    load_df = pd.read_parquet(ns.load)
    ev_df = pd.read_parquet(ns.ev)
    soh_dict = json.loads(Path(ns.soh).read_text(encoding='utf-8'))

    env = DummyVecEnv([
        _make_env(price_df, load_df, ev_df, soh_dict, ns.episode_length)
    ])

    policy_kwargs = {
        'net_arch': [ns.net_width, ns.net_width],
        'activation_fn': torch.nn.ReLU,
    }

    model = DQN(
        "MlpPolicy", env,
        learning_rate=3e-4,  # Paper config: 0.0003
        buffer_size=ns.buffer,
        learning_starts=1000,
        batch_size=32,  # Paper config: 32
        tau=1.0,
        gamma=0.99,  # Paper config: 0.99
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,  # Paper config: 1000
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        device=ns.device,
        verbose=1,
        tensorboard_log=f"runs/baseline-dqn-{ts}-seed-{ns.seed}"
    )

    data_cb = TrainingDataCollector(save_path=out_dir)
    ckpt_cb = CheckpointCallback(save_freq=100_000, save_path=out_dir / "checkpoints", name_prefix="baseline_dqn_ckpt")

    logger.info("Training for %s steps (episode_len=%d)", f"{ns.steps:,}", ns.episode_length)
    model.learn(total_timesteps=ns.steps, callback=[data_cb, ckpt_cb])

    if ns.save:
        model.save(ns.save)
        logger.info("Model saved to %s", ns.save)

    data_cb.save_training_data()
    logger.info("Training data saved to %s", out_dir)
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline-DQN training (6-month episode)")
    p.add_argument('--load', required=True, help='Real load parquet')
    p.add_argument('--price', required=True, help='Real price parquet')
    p.add_argument('--ev', required=True, help='Real EV demand parquet')
    p.add_argument('--soh', required=True, help='SOH json')

    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--steps', type=int, default=1_200_000, help='Training steps')
    p.add_argument('--episode_length', type=int, default=17_520, help='Episode length (6 months, 96*183)')
    p.add_argument('--buffer', type=int, default=150_000, help='Replay buffer size (paper config)')
    p.add_argument('--net_width', type=int, default=128, help='Hidden layer width (paper config)')
    p.add_argument('--device', default='auto', help='cpu/cuda/auto')
    p.add_argument('--save', help='Path to save model .zip')
    return p


def main():
    parser = build_parser()
    ns = parser.parse_args()
    if ns.device == 'auto':
        ns.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    success = train(ns)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 