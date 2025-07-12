#!/usr/bin/env python3
"""
V2G Energy Scheduling Environment

EV V2G energy scheduling RL environment.
Includes price optimization, battery health, and grid stability objectives.

Environment config:
- State space: Price(5d), Load(3d), Battery(2d), Time(1d)
- Action space: 21 discrete charge/discharge levels (-50kW to +50kW, 5kW steps)
- Reward function: Cost reduction, battery protection, grid support

Dependencies:
- gym>=0.21.0
- numpy>=1.24.0
- pandas>=2.0.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EVVESEnv(gym.Env):
    """
    V2G Energy Scheduling Environment
    
    State:
    - Price: Current and trend (5 dims)
    - Load: Current and forecast (3 dims)
    - Battery: SOC, availability (2 dims)
    - Time: Period index (1 dim)
    
    Actions:
    - 21 discrete levels (-50kW to +50kW, step 5kW)
    - Negative: Discharge (V2G)
    - Zero: Idle
    - Positive: Charge
    
    Reward:
    - Cost reduction
    - Battery preservation
    - Grid support
    """
    
    def __init__(self, price_data: pd.DataFrame, load_data: pd.DataFrame, 
                 ev_data: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize EVVES environment.
        
        Args:
            price_data: Electricity price time series
            load_data: Load demand time series  
            ev_data: EV availability and parameters
            config: Environment configuration parameters
        """
        super().__init__()
        
        # Store data
        self.price_data = price_data.copy()
        self.load_data = load_data.copy()
        self.ev_data = ev_data.copy()
        
        # Environment configuration
        self.config = config or self._default_config()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(21)  # 0..20 → -50..+50 kW
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        # Support paper config: 17,520 steps (6 months data)
        if 'episode_length' in self.config:
            self.max_steps = min(self.config['episode_length'], len(self.price_data) - 5)
        else:
            self.max_steps = len(self.price_data) - 5  # Reserve for lookahead
        self.done = False
        
        # EV parameters
        self.battery_capacity = self.config['battery_capacity']  # kWh
        self.max_power = self.config['max_power']  # kW
        self.initial_soc = self.config['initial_soc']  # 0-1
        self.soh_params = self.config['soh_params']
        
        # State variables
        self.current_soc = self.initial_soc
        self.current_soh = 1.0
        self.total_cost = 0.0
        self.total_degradation = 0.0
        
        # Action oscillation tracking
        self.last_action = 0
        self.action_changes = []
        
        logger.info(f"Environment initialized with {self.max_steps} steps")
    
    def _idx_to_power(self, idx):
        """Map action index 0..20 to power -50..+50 kW, step 5 kW."""
        return (idx - 10) * 5.0
    
    def _default_config(self) -> Dict:
        """Default environment configuration."""
        return {
            'battery_capacity': 75.0,  # Paper config: 75 kWh
            'max_power': 50.0,  # Paper config: ±50kW
            'initial_soc': 0.5,  # 50%
            'episode_length': 17520,  # Paper config: 17,520 steps (6 months data)
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
            'time_step': 0.25  # 15 minutes = 0.25 hours
        }
    
    def reset(self, seed=None, options=None) -> tuple:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.current_soh = 1.0
        self.total_cost = 0.0
        self.total_degradation = 0.0
        self.last_action = 0
        self.action_changes = []
        self.done = False
        
        logger.debug("Reset environment state")
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (0-20 mapping to -50 to +50 kW)
            
        Returns:
            observation: Next state
            reward: Step reward
            done: Episode termination flag
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")
        
        # Convert action to power output
        power_output = self._idx_to_power(action)
        
        # Track action changes
        action_change = abs(power_output - self.last_action)
        self.action_changes.append(action_change)
        self.last_action = power_output
        
        # Calculate current oscillation rate
        current_rosc = np.mean(self.action_changes) if self.action_changes else 0.0
        
        # Check EV availability
        ev_available = self._get_ev_availability()
        if not ev_available:
            power_output = 0.0  # Force idle if EV not available
        
        # Update battery state
        energy_delta = power_output * self.config['time_step']  # kWh
        new_soc = np.clip(
            self.current_soc + energy_delta / self.battery_capacity,
            0.0, 1.0
        )
        
        # Calculate battery degradation
        degradation = self._calculate_degradation(power_output, self.current_soc, new_soc)
        self.current_soh -= degradation
        self.total_degradation += degradation
        
        # Calculate step cost/revenue
        current_price = self.price_data.iloc[self.current_step]['price']
        step_cost = -power_output * self.config['time_step'] * current_price / 1000  # Convert to $
        self.total_cost += step_cost
        
        # Dynamic price weight adjustment
        # Rolling 7-day max price (672 * 15-min slots = 7 days)
        left = max(0, self.current_step - 672)
        rolling_max_price = self.price_data.iloc[left:self.current_step+1]['price'].max()
        lam = max(current_price / (rolling_max_price + 1e-9), 0.0)
        
        # Calculate reward with dynamic pricing
        reward = self._calculate_reward(power_output, step_cost, degradation, lam)
        
        # Update state
        self.current_soc = new_soc
        self.current_step += 1
        
        # Check termination
        self.done = (self.current_step >= self.max_steps) or (self.current_soh < 0.7)
        
        # Prepare info
        info = {
            'soc': self.current_soc,
            'soh': self.current_soh,
            'power_output': power_output,
            'step_cost': step_cost,
            'total_cost': self.total_cost,
            'degradation': degradation,
            'ev_available': ev_available,
            'price': current_price,
            'action_change': action_change,
            'current_rosc': current_rosc
        }
        
        # Return 5 values for gymnasium compatibility: obs, reward, terminated, truncated, info
        terminated = self.done
        truncated = False  # We don't use truncation in this environment
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (15 dimensions)."""
        if self.current_step >= len(self.price_data):
            # Handle edge case
            return np.zeros(15, dtype=np.float32)
        
        # Price features (6 dimensions: current + 5 future)
        current_price = self.price_data.iloc[self.current_step]['price']
        price_trends = self._get_price_trend(5)  # Get 5 future prices
        price_features = np.array([
            current_price / 100.0,  # Normalized current price
            *price_trends  # 5 future prices
        ])
        
        # Load features (2 dimensions: current + 1 future)
        current_load = self.load_data.iloc[self.current_step]['load_kw']
        load_forecasts = self._get_load_forecast(1)  # Get 1 future load
        load_features = np.array([
            current_load / 1000.0,  # Normalized current load
            load_forecasts[0]  # Next period load
        ])
        
        # Battery features (4 dimensions)
        battery_features = np.array([
            self.current_soc,
            self.current_soh,
            self.current_step % 96 / 96.0,  # Time of day
            float(self.current_step) / 1000.0  # Normalized step count
        ])
        
        # Recent action features (3 dimensions)
        # Initialize action history if not exists
        if not hasattr(self, 'action_history'):
            self.action_history = [10, 10, 10]  # Default to idle actions
        
        recent_actions = self.action_history[-3:] if len(self.action_history) >= 3 else [10] * 3
        while len(recent_actions) < 3:
            recent_actions = [10] + recent_actions  # Pad with idle actions
        action_features = np.array(recent_actions[-3:]) / 20.0  # Normalize to 0-1
        
        # Combine all features (15 dimensions total)
        obs = np.concatenate([
            price_features,    # 6 dims
            load_features,     # 2 dims  
            battery_features,  # 4 dims
            action_features    # 3 dims
        ])
        
        # Ensure exactly 15 dimensions
        assert len(obs) == 15, f"Expected 15 dimensions, got {len(obs)}"
        
        return obs.astype(np.float32)
    
    def _get_price_trend(self, n_periods):
        """Get price trend for next n periods."""
        trends = []
        for i in range(1, n_periods + 1):
            if self.current_step + i < len(self.price_data):
                future_price = self.price_data.iloc[self.current_step + i]['price']
                trends.append(future_price / 100.0)
            else:
                trends.append(0.0)
        return trends
    
    def _get_load_forecast(self, n_periods):
        """Get load forecast for next n periods."""
        forecasts = []
        for i in range(1, n_periods + 1):
            if self.current_step + i < len(self.load_data):
                future_load = self.load_data.iloc[self.current_step + i]['load_kw']
                forecasts.append(future_load / 1000.0)
            else:
                forecasts.append(0.0)
        return forecasts
    
    def _get_ev_availability(self) -> bool:
        """Check if EV is available at current time step."""
        if self.current_step < len(self.ev_data):
            # For now, assume EV is always available if demand > 0
            # In a real scenario, this would check actual availability
            ev_demand = self.ev_data.iloc[self.current_step]['ev_demand_kw']
            return ev_demand > 0.1  # Available if demand > 0.1 kW
        return True  # Default to available
    
    def _calculate_degradation(self, power: float, soc_old: float, soc_new: float) -> float:
        """
        Calculate battery degradation based on usage pattern.
        
        Uses empirical model from Applied Energy 2023 standards.
        """
        params = self.soh_params
        
        # Handle both detailed and simplified parameter formats
        if 'calendar_aging' in params:
            # Detailed format
            calendar_deg = params['calendar_aging'] * self.config['time_step']
            cycle_aging = params.get('cycle_aging', 0.00005)
            depth_factor = params.get('depth_factor', 1.2)
            temp_factor = params.get('temperature_factor', 1.0)
        else:
            # Simplified format (a, b, c)
            calendar_deg = params.get('a', 1e-6) * self.config['time_step']
            cycle_aging = params.get('b', 2e-6)
            depth_factor = params.get('c', 5e-7) * 1000  # Scale up for reasonable values
            temp_factor = 1.0
        
        # Cycle aging (usage-based)
        if abs(power) > 0.1:  # Only if significant power flow
            depth_of_discharge = abs(soc_new - soc_old)
            cycle_deg = (cycle_aging * 
                        depth_of_discharge * 
                        depth_factor * 
                        temp_factor)
        else:
            cycle_deg = 0.0
        
        return calendar_deg + cycle_deg
    
    def _calculate_reward(self, power: float, cost: float, degradation: float, lam: float = 1.0) -> float:
        """
        Calculate multi-objective reward with dynamic pricing weight.
        
        Uses scaled reward components and dynamic price weight λ(t)
        """
        # Get current load for grid stability calculation
        current_load = self.load_data.iloc[self.current_step]['load_kw']
        
        # Assume EV demand (simplified as fraction of load)
        ev_demand = current_load * 0.1  # 10% of load as EV demand
        grid_net_kw = current_load + power - ev_demand
        
        # Reward components scaled to [-1,1] range
        cost_reward = -cost / 100.0                    # Economic component ≈ [-1,1]
        peak_reward = -abs(grid_net_kw) / 10.0         # Grid stability component
        soh_penalty = -(1.0 - self.current_soh) * 5.0 # Battery health penalty
        soc_penalty = -abs(self.current_soc - 0.5) * 2.0  # SOC balance penalty
        
        # Weighted combination with dynamic price weight
        total_reward = lam * cost_reward + peak_reward + soh_penalty + soc_penalty
        
        return total_reward
    
    def render(self, mode='human'):
        """Render environment state (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}, SOC: {self.current_soc:.3f}, "
                  f"SOH: {self.current_soh:.3f}, Cost: ${self.total_cost:.2f}")
    
    def get_metrics(self) -> Dict:
        """Get environment performance metrics."""
        return {
            'total_cost': self.total_cost,
            'total_degradation': self.total_degradation,
            'final_soc': self.current_soc,
            'final_soh': self.current_soh,
            'steps_completed': self.current_step,
            'avg_ctotal_per_step': self.total_cost / max(1, self.current_step),
            'rosc': float(np.mean(self.action_changes)) if self.action_changes else 0.0  # Action oscillation rate
        } 