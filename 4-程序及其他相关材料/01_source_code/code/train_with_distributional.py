#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_distributional.py

Training wrapper that supports either the core DDDQN-PER agent (existing) or the new
QR-DQN distributional agent. CLI flag --use_distributional toggles using the QR-DQN agent.

This script is intentionally lightweight and keeps integration points minimal:
- Loads existing data (price, load, ev, soh) as parquet or csv as in original train_dddqn_paper.py
- Constructs environment via the repo's EVVESEnv
- If --use_distributional is set, uses QRDQNAgent; otherwise attempts to construct existing DDDQN_PER_Agent
- Training loop structure mirrors original script to ease comparison
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Attempt to import environment and existing agent
try:
    from evves_env import EVVESEnv  # repo env
except Exception:
    EVVESEnv = None

try:
    from dddqn_per_sb3 import DDDQN_PER_Agent
except Exception:
    DDDQN_PER_Agent = None

from distributional_agent import QRDQNAgent  # newly added file in same folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _make_env(price_df, load_df, ev_df, soh_dict, episode_len, market_args=None):
    def _init():
        config = {
            'episode_length': episode_len,
            'battery_capacity': 75.0,
            'max_power': 50.0,
            'initial_soc': 0.5,
            'soh_params': soh_dict,
            'reward_weights': {
                'economic': 1.0,
                'degradation': 0.5,
                'grid_stability': 0.3
            },
            'time_step': 0.25
        }
        env = EVVESEnv(price_data=price_df, load_data=load_df, ev_data=ev_df, config=config)
        return Monitor(env)
    return _init


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out) / f"train_seed_{args.seed}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    price_df = pd.read_parquet(args.price) if args.price.endswith('.parquet') else pd.read_csv(args.price)
    load_df = pd.read_parquet(args.load) if args.load.endswith('.parquet') else pd.read_csv(args.load)
    ev_df = pd.read_parquet(args.ev) if args.ev.endswith('.parquet') else pd.read_csv(args.ev)
    soh_dict = json.loads(Path(args.soh).read_text(encoding='utf-8'))

    env = DummyVecEnv([_make_env(price_df, load_df, ev_df, soh_dict, args.episode_length)])

    # Agent selection
    if args.use_distributional:
        logger.info("Using QR-DQN distributional agent")
        agent = QRDQNAgent(
            state_size=args.state_size,
            action_size=args.action_size,
            n_quantiles=args.n_quantiles,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            tau=args.tau,
            dueling=not args.no_dueling,
            device=torch.device(args.device)
        )
        # Note: integration with vectorized envs and Monitor wrapper is simplified here.
    else:
        if DDDQN_PER_Agent is None:
            raise RuntimeError("DDDQN_PER_Agent is not importable in this environment.")
        logger.info("Using existing DDDQN-PER agent")
        agent = DDDQN_PER_Agent(
            state_size=args.state_size,
            action_size=args.action_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update_frequency=args.target_update_frequency,
            device=args.device
        )

    # simplified training loop (keeps parity with original train_dddqn_paper)
    obs = env.reset()
    total_reward = 0.0
    episode_count = 0

    for step in range(args.steps):
        state = obs[0][0] if isinstance(obs, (list, tuple)) else obs[0]
        action = agent.act(state, training=True, epsilon=max(0.01, 0.1*(1 - step / args.steps)))
        next_obs, reward, done, info = env.step([action])
        total_reward += reward[0] if isinstance(reward, (list, tuple)) else reward

        # store transition for non-PER agent
        try:
            agent.store_transition(state, action, reward[0], next_obs[0][0], done[0])
        except Exception:
            # some agents handle storage internally
            pass

        # sample learn step for agents that expose learn()
        try:
            # If using PER, the learn signature may differ; this wrapper expects the agent.learn to handle that
            agent.learn()
        except TypeError:
            # maybe agent.learn expects experiences as arguments; skip if not available
            pass

        obs = next_obs

        if done[0]:
            episode_count += 1
            obs = env.reset()

    # Save final model if agent supports save()
    try:
        agent.save(str(Path(out_dir) / "final_model.pth"))
    except Exception:
        logger.warning("Agent save failed or not implemented.")

    logger.info("Training finished. Results saved to %s", out_dir)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train with optional distributional QR-DQN")
    parser.add_argument("--price", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--ev", required=True)
    parser.add_argument("--soh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--episode_length", type=int, default=96)
    parser.add_argument("--state_size", type=int, default=15)
    parser.add_argument("--action_size", type=int, default=21)
    parser.add_argument("--use_distributional", action="store_true")
    parser.add_argument("--n_quantiles", type=int, default=31)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--target_update_frequency", type=int, default=1000)
    parser.add_argument("--no_dueling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
