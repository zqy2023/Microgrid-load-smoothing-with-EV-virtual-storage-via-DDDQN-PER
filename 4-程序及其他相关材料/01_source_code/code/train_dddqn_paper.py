#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DDDQN-PER training script with paper configuration."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from evves_env import EVVESEnv  # 绝对导入
from dddqn_per_sb3 import DDDQN_PER_Agent  # 绝对导入


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
    """Train DDDQN-PER model."""
    logger.info("Training DDDQN-PER | seed=%d", ns.seed)

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

    # Create DDDQN-PER agent
    agent = DDDQN_PER_Agent(
        state_size=15,
        action_size=21,
        learning_rate=3e-4,  # Paper config: 0.0003
        gamma=0.99,  # Paper config: 0.99
        buffer_size=200000,  # Paper config: 200,000
        batch_size=32,  # Paper config: 32
        target_update_frequency=1000,  # Paper config: 1000
        device=ns.device
    )

    # Training loop
    obs = env.reset()
    total_reward = 0
    episode_count = 0
    
    for step in range(ns.steps):
        action = agent.act(obs[0], training=True)
        next_obs, reward, done, info = env.step([action])
        agent.remember(obs[0], action, reward, next_obs[0], done)
        agent.train_step()
        total_reward += reward
        if done:
            episode_count += 1
            logger.info("Episode %d completed, total reward: %.2f", episode_count, total_reward)
            obs = env.reset()
            total_reward = 0
        else:
            obs = next_obs
        if step % 10000 == 0:
            logger.info("Step %d/%d, episodes: %d", step, ns.steps, episode_count)

    if ns.save:
        agent.save(ns.save)
        logger.info("Model saved to %s", ns.save)

    logger.info("Training completed")
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DDDQN-PER training (paper config)")
    p.add_argument('--load', required=True, help='Load data parquet')
    p.add_argument('--price', required=True, help='Price data parquet')
    p.add_argument('--ev', required=True, help='EV demand data parquet')
    p.add_argument('--soh', required=True, help='SOH parameters json')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--steps', type=int, default=1_200_000, help='Training steps')
    p.add_argument('--episode_length', type=int, default=17_520, help='Episode length')
    p.add_argument('--device', default='auto', help='cpu/cuda/auto')
    p.add_argument('--save', help='Path to save model')
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