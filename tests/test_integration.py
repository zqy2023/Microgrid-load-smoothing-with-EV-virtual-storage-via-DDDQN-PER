#!/usr/bin/env python3
"""
Integration test with dummy data to verify environment functionality.
Tests that the environment can be created and used with sample data.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add source code directory to path
source_dir = Path(__file__).parent.parent / "4-程序及其他相关材料" / "01_source_code" / "code"
sys.path.insert(0, str(source_dir))


def create_dummy_data():
    """Create minimal dummy data for testing."""
    # Create 100 timesteps of dummy data
    n_steps = 100
    
    # Price data - use 'price' column as expected by environment
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_steps, freq='15min'),
        'price': np.random.uniform(10, 30, n_steps)  # Price in cents/kWh
    })
    
    # Load data - use 'load_kw' column as expected by environment
    load_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_steps, freq='15min'),
        'load_kw': np.random.uniform(100, 300, n_steps)  # Load in kW
    })
    
    # EV data - use expected column names (ev_demand_kw indicates availability)
    ev_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_steps, freq='15min'),
        'ev_demand_kw': np.random.uniform(1, 50, n_steps),  # EV demand > 0 means available
    })
    
    return price_data, load_data, ev_data


def test_environment_creation():
    """Test that environment can be created with dummy data."""
    from evves_env import EVVESEnv
    
    price_data, load_data, ev_data = create_dummy_data()
    
    # Create environment
    env = EVVESEnv(price_data, load_data, ev_data)
    
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None
    
    print("✓ Environment created successfully")


def test_environment_reset():
    """Test that environment can be reset."""
    from evves_env import EVVESEnv
    
    price_data, load_data, ev_data = create_dummy_data()
    env = EVVESEnv(price_data, load_data, ev_data)
    
    obs, info = env.reset()
    
    assert obs is not None
    assert obs.shape == (15,)  # 15-dimensional observation space
    assert info is not None
    
    print("✓ Environment reset successful")


def test_environment_step():
    """Test that environment can execute steps."""
    from evves_env import EVVESEnv
    
    price_data, load_data, ev_data = create_dummy_data()
    env = EVVESEnv(price_data, load_data, ev_data)
    
    obs, info = env.reset()
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert obs.shape == (15,)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    
    print("✓ Environment step execution successful")


def test_action_space():
    """Test action space boundaries."""
    from evves_env import EVVESEnv
    
    price_data, load_data, ev_data = create_dummy_data()
    env = EVVESEnv(price_data, load_data, ev_data)
    
    # Test action space
    assert env.action_space.n == 21  # 21 discrete actions
    
    # Test action mapping
    assert env._idx_to_power(0) == -50.0   # Min power
    assert env._idx_to_power(10) == 0.0    # Idle
    assert env._idx_to_power(20) == 50.0   # Max power
    
    print("✓ Action space validation successful")


def test_config_override():
    """Test environment configuration override."""
    from evves_env import EVVESEnv
    
    price_data, load_data, ev_data = create_dummy_data()
    
    custom_config = {
        'battery_capacity': 100.0,
        'max_power': 60.0,
        'initial_soc': 0.8,
        'soh_params': {
            'calendar_aging': 0.0002,
            'cycle_aging': 0.0001
        }
    }
    
    env = EVVESEnv(price_data, load_data, ev_data, config=custom_config)
    
    assert env.battery_capacity == 100.0
    assert env.max_power == 60.0
    assert env.initial_soc == 0.8
    
    print("✓ Configuration override successful")


if __name__ == "__main__":
    tests = [
        test_environment_creation,
        test_environment_reset,
        test_environment_step,
        test_action_space,
        test_config_override,
    ]
    
    print("Running integration tests with dummy data...\n")
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print(f"\n✅ All {len(tests)} integration tests passed!")
    print("The environment is fully functional and ready for training.")
