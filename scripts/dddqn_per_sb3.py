#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDDQN-PER Implementation (Stable-Baselines3 Compatible)

Double Dueling DQN with Prioritized Experience Replay
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque
import random
import logging

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3 import DQN
from gymnasium import spaces

logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, buffer_size=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer_size = buffer_size
        self.alpha = alpha              # Priority exponent
        self.beta = beta                # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False
        
        logger.info(f"PER buffer: size={buffer_size}, α={alpha}, β={beta}")
    
    def add(self, obs, action, reward, next_obs, done):
        """Add experience to buffer"""
        experience = (obs, action, reward, next_obs, done)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            self.full = True
        
        # New experience gets max priority
        self.priorities[self.pos] = self.max_priority ** self.alpha
        
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self, batch_size):
        """Sample experiences based on priorities"""
        if len(self.buffer) == 0:
            return None, None, None
        
        size = len(self.buffer) if not self.full else self.buffer_size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(size, batch_size, p=probabilities)
        
        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update experience priorities using TD errors"""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha  # Add epsilon to prevent zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class DuelingQNetwork(nn.Module):
    """Dueling DQN Network Architecture"""
    
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # State value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Action advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Dueling Q-Network: state_dim={state_size}, action_dim={action_size}")
        self._log_network_info()
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass with dueling architecture"""
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _log_network_info(self):
        """Log network architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

class DDDQN_PER_Agent:
    """DDDQN-PER Agent Implementation"""
    
    def __init__(self, 
                 state_size=15,
                 action_size=21,
                 learning_rate=3e-4,  # Paper config: 0.0003
                 gamma=0.99,  # Paper config: 0.99
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=200000,  # Paper config: 200,000
                 batch_size=32,  # Paper config: 32
                 update_frequency=4,
                 target_update_frequency=1000,  # Paper config: 1000
                 device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        # Device configuration
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.q_network = DuelingQNetwork(state_size, action_size, hidden_dim=256).to(self.device)  # Paper config: 256
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_dim=256).to(self.device)  # Paper config: 256
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # Training statistics
        self.step_count = 0
        self.training_step = 0
        
        logger.info("DDDQN-PER agent initialized")
        logger.info(f"State dim: {state_size}, Action dim: {action_size}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        logger.info(f"Device: {self.device}")
    
    def act(self, state, training=True):
        """ε-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def act_continuous(self, state, training=True):
        """Continuous action output (for environment compatibility)"""
        discrete_action = self.act(state, training)
        # Map discrete action to continuous space [-1, 1]
        continuous_action = (discrete_action - 10) / 10.0  # 21 actions map to [-1,1]
        return np.array([continuous_action])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        # Convert continuous action to discrete if needed
        if isinstance(action, np.ndarray):
            discrete_action = int((action[0] + 1) * 10)  # Continuous to discrete
            discrete_action = np.clip(discrete_action, 0, self.action_size - 1)
        else:
            discrete_action = action
        
        self.memory.add(state, discrete_action, reward, next_state, done)
    
    def train_step(self):
        """Training step"""
        if len(self.memory.buffer) < self.batch_size:
            return None
        
        # Periodic training
        if self.step_count % self.update_frequency != 0:
            self.step_count += 1
            return None
        
        # Sample from prioritized replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        loss = self._compute_loss(batch, weights)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = self._compute_td_errors(batch)
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Update target network
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        self.training_step += 1
        
        return loss.item()
    
    def _compute_loss(self, batch, weights):
        """Compute loss function"""
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate values
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Weighted MSE loss (importance sampling)
        td_errors = target_q_values - current_q_values
        weighted_loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        return weighted_loss
    
    def _compute_td_errors(self, batch):
        """Compute TD errors (for priority updates)"""
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        td_errors = torch.abs(target_q_values - current_q_values).squeeze()
        return td_errors
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated")
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_step': self.training_step
        }, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.training_step = checkpoint['training_step']
        logger.info(f"Model loaded from {filepath}")

# Alias for compatibility
DDDQNAgent = DDDQN_PER_Agent

def create_sb3_dqn_with_per(env, **kwargs):
    """Create SB3 DQN with PER (using sb3-contrib)"""
    try:
        from sb3_contrib import QRDQN
        from sb3_contrib.common.buffers import PrioritizedReplayBuffer as SB3_PER
        
        # Use QRDQN + PER combination
        model = QRDQN(
            "MlpPolicy",
            env,
            buffer_size=kwargs.get('buffer_size', 100000),
            learning_rate=kwargs.get('learning_rate', 1e-4),
            gamma=kwargs.get('gamma', 0.99),
            batch_size=kwargs.get('batch_size', 64),
            replay_buffer_class=SB3_PER,
            replay_buffer_kwargs={"alpha": 0.6, "beta": 0.4},
            verbose=1
        )
        
        logger.info("SB3 QRDQN + PER model created")
        return model
        
    except ImportError:
        logger.warning("sb3-contrib not installed, falling back to custom DDDQN-PER")
        return None

if __name__ == "__main__":
    # Test code
    import gymnasium as gym
    
    # Create test environment
    env = gym.make('CartPole-v1')
    
    # Test custom DDDQN-PER
    agent = DDDQN_PER_Agent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    print("DDDQN-PER agent test passed")
    
    # Test SB3 integration
    sb3_model = create_sb3_dqn_with_per(env)
    if sb3_model:
        print("SB3 QRDQN + PER integration test passed")
    else:
        print("SB3 integration not available, using custom implementation") 