#!/usr/bin/env python3
"""
Smoke tests for Microgrid Load Smoothing project.
Tests basic imports and module availability.
"""
import sys
from pathlib import Path

# Add source code directory to path
source_dir = Path(__file__).parent.parent / "4-程序及其他相关材料" / "01_source_code" / "code"
sys.path.insert(0, str(source_dir))


def test_standard_library_imports():
    """Test that all required standard libraries can be imported."""
    import numpy
    import pandas
    import torch
    import gymnasium
    import stable_baselines3
    import matplotlib
    import seaborn
    import pyarrow
    assert True


def test_project_imports():
    """Test that project modules can be imported."""
    import evves_env
    import dddqn_per_sb3
    import train_dddqn
    assert True


def test_evves_env_class():
    """Test that EVVESEnv class is accessible."""
    from evves_env import EVVESEnv
    assert EVVESEnv is not None


def test_prioritized_replay_buffer():
    """Test that PrioritizedReplayBuffer class is accessible."""
    from dddqn_per_sb3 import PrioritizedReplayBuffer
    assert PrioritizedReplayBuffer is not None


def test_gymnasium_compatibility():
    """Test gymnasium environment interface."""
    import gymnasium as gym
    from gymnasium import spaces
    assert gym is not None
    assert spaces is not None


def test_stable_baselines3_dqn():
    """Test that DQN is available from stable-baselines3."""
    from stable_baselines3 import DQN
    assert DQN is not None


def test_torch_available():
    """Test that PyTorch is properly installed."""
    import torch
    assert torch.__version__ is not None
    # Test basic tensor creation
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)


if __name__ == "__main__":
    # Run all tests manually if not using pytest
    tests = [
        test_standard_library_imports,
        test_project_imports,
        test_evves_env_class,
        test_prioritized_replay_buffer,
        test_gymnasium_compatibility,
        test_stable_baselines3_dqn,
        test_torch_available,
    ]
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            sys.exit(1)
    
    print(f"\nAll {len(tests)} tests passed!")
