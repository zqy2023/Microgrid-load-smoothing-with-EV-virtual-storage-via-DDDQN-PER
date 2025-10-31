#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distributional_agent.py
QR-DQN (Quantile Regression DQN) agent that is compatible with the existing
DDDQN_PER_Agent interface in the repository.

Design goals (minimal invasive changes):
- Provide a QRDQNAgent class implementing similar public methods as DDDQN_PER_Agent:
  - act(state, training=True, epsilon=0.0)
  - step(...) / learn(...) compatible with PER memory used previously
  - save/load utilities
- Use a flexible network that can operate in non-distributional mode (returns expectations)
  and distributional mode (returns quantiles).
- Default n_quantiles=31 (reduced complexity vs 51)
- Keep hyperparameters minimal and configurable
- Keep PER compatibility: accept experiences with importance-sampling weights and indices
"""

import math
import random
from typing import Tuple, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Try to import existing PER memory and utilities if present; fallback to simple replay
try:
    from dddqn_per_sb3 import DDDQN_PER_Agent  # type: ignore
except Exception:
    DDDQN_PER_Agent = None  # for typing only

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QRNetwork(nn.Module):
    """QR-DQN network: outputs (action_size x n_quantiles) quantile values."""
    def __init__(self, state_size: int, action_size: int, n_quantiles: int = 31, hidden_sizes=(128, 128), dueling: bool = True):
        super(QRNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_quantiles = n_quantiles
        self.dueling = dueling

        # feature extractor
        layers = []
        in_dim = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature = nn.Sequential(*layers)

        if dueling:
            self.value_layer = nn.Linear(in_dim, n_quantiles)
            self.adv_layer = nn.Linear(in_dim, action_size * n_quantiles)
        else:
            self.quantile_layer = nn.Linear(in_dim, action_size * n_quantiles)

        # register tau (quantile fractions) for QR loss computation
        tau = (2 * torch.arange(n_quantiles).float() + 1) / (2.0 * n_quantiles)
        self.register_buffer("tau", tau)  # shape (n_quantiles,)

    def forward(self, x: torch.Tensor, return_distribution: bool = True) -> torch.Tensor:
        """
        If return_distribution: returns shape (batch, action_size, n_quantiles)
        Else: returns expected Q-values (batch, action_size)
        """
        features = self.feature(x)
        batch = features.size(0)

        if self.dueling:
            v = self.value_layer(features).view(batch, 1, self.n_quantiles)           # (b,1,n)
            a = self.adv_layer(features).view(batch, self.action_size, self.n_quantiles)  # (b,A,n)
            q_dist = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q_dist = self.quantile_layer(features).view(batch, self.action_size, self.n_quantiles)

        if return_distribution:
            # return raw quantile values (not probabilities) shape (b, A, n)
            return q_dist
        else:
            # return expectation
            return q_dist.mean(dim=2)  # (b, A)


class QRDQNAgent:
    """
    QR-DQN agent with PER compatibility.

    Public methods:
    - act(state, training=True, epsilon=0.0)
    - learn(experiences, gamma)
    - store_transition(...)
    - save(path), load(path)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        n_quantiles: int = 31,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        tau: float = 1e-3,
        dueling: bool = True,
        device: torch.device = DEVICE,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.n_quantiles = n_quantiles
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.qnetwork_local = QRNetwork(state_size, action_size, n_quantiles, dueling=dueling).to(self.device)
        self.qnetwork_target = QRNetwork(state_size, action_size, n_quantiles, dueling=dueling).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Simple replay fallback (if no PER memory provided)
        self.memory = deque(maxlen=200000)
        self.use_per = False  # set True if integrating repository PER memory externally

        # quantile fractions (tau) used in loss are stored in network buffer
        self.n_step = 0

    # ---------- public API ----------
    def act(self, state: np.ndarray, training: bool = True, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection. state: numpy array (state_size,)"""
        if training and random.random() < epsilon:
            return random.randrange(self.action_size)

        self.qnetwork_local.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_size)
            q_values = self.qnetwork_local(s, return_distribution=False)  # (1, A)
            action = q_values.argmax(dim=1).item()
        self.qnetwork_local.train()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Fallback experience storage (if external PER not used)"""
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones).astype(np.uint8)).to(self.device),
        )

    def learn_from_per_batch(self, experiences: Tuple[torch.Tensor, ...], weights: torch.Tensor, indices=None):
        """
        experiences: (states, actions, rewards, next_states, dones)
        weights: importance sampling weights (batch,)
        indices: indices in PER to update priorities (optional)
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.size(0)
        N = self.n_quantiles
        device = self.device

        # --- compute target quantiles ---
        with torch.no_grad():
            # next-state distributions from local for action selection (Double DQN style)
            next_dist_local = self.qnetwork_local(next_states, return_distribution=True)  # (b, A, n)
            next_q_values = next_dist_local.mean(dim=2)  # (b, A)
            next_actions = next_q_values.argmax(dim=1)  # (b,)

            # gather distributions for chosen actions from target network
            next_dist_target = self.qnetwork_target(next_states, return_distribution=True)  # (b, A, n)
            next_actions_idx = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, N)
            next_dist = next_dist_target.gather(1, next_actions_idx).squeeze(1)  # (b, n)

            # compute Bellman target quantiles
            rewards_expanded = rewards.unsqueeze(1).expand(-1, N)
            dones_expanded = dones.unsqueeze(1).expand(-1, N)
            support = rewards_expanded + (1.0 - dones_expanded) * (self.gamma ** 1) * next_dist  # (b, n)
            # note: in QR-DQN we regress to these targets directly (no projection as in C51)

        # --- current quantiles for selected actions ---
        actions_idx = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, N)
        curr_dist = self.qnetwork_local(states, return_distribution=True).gather(1, actions_idx).squeeze(1)  # (b, n)

        # compute quantile regression Huber loss
        # expand dims for pairwise differences
        td_error = support.unsqueeze(2) - curr_dist.unsqueeze(1)  # (b, n_target, n_pred)
        huber = F.smooth_l1_loss(curr_dist.unsqueeze(1), support.unsqueeze(2), reduction='none')  # shape (b, n_target, n_pred)

        tau = self.qnetwork_local.tau.to(device)  # (n,)
        tau = tau.view(1, N).expand(batch_size, N)  # (b, n)
        # compute quantile regression loss per QR-DQN paper
        # Note: simplified implementation—computationally heavy pairwise ops; acceptable for moderate batch sizes
        error_sign = (td_error.detach() < 0).float()
        quantile_loss = (torch.abs(tau.unsqueeze(2) - error_sign) * huber).mean(dim=(1, 2))  # (b,)

        # apply IS weights
        if weights is None:
            loss = quantile_loss.mean()
        else:
            loss = (weights * quantile_loss).mean()

        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()

        # soft update target
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # return per-sample td_error for PER priority update (use mean abs error as priority)
        per_td = quantile_loss.detach().cpu().numpy()
        return loss.item(), per_td

    def learn(self, experiences, gamma: float = None):
        """Wrapper that accepts either PER-supplied experiences or uses internal buffer."""
        if gamma is None:
            gamma = self.gamma

        # If experiences are provided as (states, actions, rewards, next_states, dones, weights, indices)
        if isinstance(experiences, tuple) and len(experiences) == 7:
            states, actions, rewards, next_states, dones, weights, indices = experiences
            loss, priorities = self.learn_from_per_batch((states, actions, rewards, next_states, dones), weights, indices)
            return loss, priorities
        else:
            # fallback: sample from internal replay
            if len(self.memory) < self.batch_size:
                return None, None
            batch = self.sample_batch()
            loss, _ = self.learn_from_per_batch(batch, None)
            return loss, None

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            'local_state': self.qnetwork_local.state_dict(),
            'target_state': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(ckpt['local_state'])
        self.qnetwork_target.load_state_dict(ckpt['target_state'])
        self.optimizer.load_state_dict(ckpt.get('optimizer', {}))
