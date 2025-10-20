#!/usr/bin/env python3
"""
Reward Function Utilities

Purpose:
    Modular reward calculation functions for multi-objective V2G optimization.
    Supports economic, battery health, and grid stability objectives.
    
Usage:
    from code.utils.reward_functions import RewardCalculator
    calculator = RewardCalculator(config)
    reward = calculator.calculate(state, action, next_state, info)

Dependencies:
    - numpy>=1.24.0

Authors: [Author Names]
License: MIT
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Multi-objective reward calculator for V2G optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward calculator.
        
        Args:
            config: Configuration dictionary with reward weights and parameters
        """
        self.config = config
        self.weights = config.get('reward_weights', {
            'economic': 1.0,
            'degradation': 0.5,
            'grid_stability': 0.3
        })
        
        # Dynamic lambda parameters
        self.price_max = config.get('price_max', 200.0)  # Maximum price for normalization
        self.alpha = config.get('alpha', 1.0)  # Lambda scaling factor
        
        logger.info(f"RewardCalculator initialized with weights: {self.weights}")
        logger.info(f"Dynamic lambda enabled: price_max={self.price_max}, alpha={self.alpha}")
    
    def calculate(self, state: np.ndarray, action: int, next_state: np.ndarray, 
                  info: Dict[str, Any]) -> float:
        """
        Calculate total reward for V2G action with dynamic lambda.
        
        Args:
            state: Current environment state
            action: Action taken (0-20 mapping to -50 to +50 kW)
            next_state: Resulting environment state
            info: Additional step information
            
        Returns:
            Total weighted reward (scaled to ±100)
        """
        # Calculate dynamic lambda based on current price
        current_price = info.get('price', 50.0)  # Default price if not available
        lam = (current_price / self.price_max) ** self.alpha
        
        # Extract components
        economic_reward = self._economic_reward(info)
        degradation_penalty = self._degradation_penalty(info)
        grid_reward = self._grid_stability_reward(state, action, info)
        
        # Combine with dynamic lambda and weights
        total_reward = (
            economic_reward * self.weights['economic'] / 100.0 +  # Scale economic
            degradation_penalty * self.weights['degradation'] * 50.0 +  # Scale degradation
            grid_reward * lam * self.weights['grid_stability'] * 30.0  # Dynamic grid reward
        )
        
        # Ensure reward is in ±100 range
        total_reward = np.clip(total_reward, -100.0, 100.0)
        
        return total_reward
    
    def _economic_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate economic reward based on electricity cost/revenue.
        
        Positive for revenue (discharging at high prices)
        Negative for cost (charging at high prices)
        """
        step_cost = info.get('step_cost', 0.0)
        return -step_cost  # Negative cost becomes positive reward
    
    def _degradation_penalty(self, info: Dict[str, Any]) -> float:
        """
        Calculate battery degradation penalty.
        
        Penalizes actions that cause significant battery wear.
        """
        degradation = info.get('degradation', 0.0)
        return -degradation * 1000  # Scale up and make penalty
    
    def _grid_stability_reward(self, state: np.ndarray, action: int, 
                               info: Dict[str, Any]) -> float:
        """
        Calculate grid stability reward with enhanced peak shaving logic.
        
        Rewards peak shaving (discharge during high load)
        and valley filling (charge during low load).
        """
        power_output = info.get('power_output', 0.0)
        
        # Extract load information from state (assuming normalized)
        if len(state) >= 6:
            current_load = state[5] if len(state) > 5 else state[2]  # Flexible indexing
        else:
            return 0.0
        
        # Define dynamic peak/off-peak thresholds
        peak_threshold = 0.75  # High load threshold
        off_peak_threshold = 0.35  # Low load threshold
        
        # Normalize load (assuming it's already normalized in state)
        load_level = current_load
        
        if load_level > peak_threshold and power_output < 0:
            # Discharging during peak hours - strong positive reward
            return abs(power_output) * 0.2
        elif load_level < off_peak_threshold and power_output > 0:
            # Charging during off-peak hours - moderate positive reward
            return power_output * 0.1
        elif load_level > peak_threshold and power_output > 0:
            # Charging during peak hours - penalty
            return -power_output * 0.15
        else:
            # Neutral conditions
            return 0.0
    
    def get_reward_breakdown(self, state: np.ndarray, action: int, 
                            next_state: np.ndarray, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Useful for analysis and debugging.
        """
        economic = self._economic_reward(info)
        degradation = self._degradation_penalty(info)
        grid = self._grid_stability_reward(state, action, info)
        
        return {
            'economic': economic,
            'degradation': degradation,
            'grid_stability': grid,
            'total': (economic * self.weights['economic'] +
                     degradation * self.weights['degradation'] +
                     grid * self.weights['grid_stability'])
        } 