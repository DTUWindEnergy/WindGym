"""
Simple test script for RewardCalculator to verify it works correctly.
"""

import sys
import numpy as np
from collections import deque

# Import the reward calculator directly
sys.path.insert(0, "/home/user/WindGym")
from WindGym.core.reward_calculator import RewardCalculator


def test_reward_calculator_init():
    """Test RewardCalculator initialization."""
    print("Testing RewardCalculator initialization...")

    # Test Baseline reward
    rc = RewardCalculator(
        power_reward_type="Baseline",
        power_scaling=1.0,
        action_penalty=0.01,
        action_penalty_type="change",
    )
    assert rc.power_reward_type == "Baseline"
    print("✓ Baseline reward initialization works")

    # Test Power_avg reward
    rc = RewardCalculator(
        power_reward_type="Power_avg",
        power_scaling=2.0,
        action_penalty=0.05,
        action_penalty_type="total",
    )
    assert rc.power_scaling == 2.0
    print("✓ Power_avg reward initialization works")

    # Test Power_diff reward
    rc = RewardCalculator(
        power_reward_type="Power_diff", power_scaling=1.0, power_window_size=50
    )
    assert rc._power_window_size == 50
    print("✓ Power_diff reward initialization works")

    print()


def test_baseline_reward():
    """Test baseline reward calculation."""
    print("Testing baseline reward calculation...")

    rc = RewardCalculator(power_reward_type="Baseline", power_scaling=1.0)

    # Create mock deques
    farm_deque = deque([100.0, 110.0, 105.0])
    baseline_deque = deque([100.0, 100.0, 100.0])

    reward = rc.calculate_power_reward(
        farm_power_deque=farm_deque, baseline_power_deque=baseline_deque
    )

    # Agent avg = 105, baseline avg = 100
    # Expected reward = 105/100 - 1 = 0.05
    expected = 0.05
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print(f"✓ Baseline reward calculated correctly: {reward}")
    print()


def test_power_avg_reward():
    """Test power_avg reward calculation."""
    print("Testing power_avg reward calculation...")

    rc = RewardCalculator(power_reward_type="Power_avg", power_scaling=1.0)

    farm_deque = deque([1000.0, 1200.0, 1100.0])
    rated_power = 1000.0
    n_turbines = 3

    reward = rc.calculate_power_reward(
        farm_power_deque=farm_deque, rated_power=rated_power, n_turbines=n_turbines
    )

    # Agent avg = 1100
    # Expected reward = 1100 / (3 * 1000) = 0.3667
    expected = 1100.0 / 3000.0
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print(f"✓ Power_avg reward calculated correctly: {reward}")
    print()


def test_action_penalty():
    """Test action penalty calculation."""
    print("Testing action penalty calculation...")

    # Test "change" penalty
    rc = RewardCalculator(
        power_reward_type="None", action_penalty=0.1, action_penalty_type="change"
    )

    old_yaws = np.array([0.0, 5.0, -5.0])
    new_yaws = np.array([2.0, 7.0, -3.0])
    yaw_max = 30.0

    penalty = rc.calculate_action_penalty(old_yaws, new_yaws, yaw_max)

    # Average change = (2 + 2 + 2) / 3 = 2.0
    # Penalty = 0.1 * 2.0 = 0.2
    expected = 0.2
    assert abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    print(f"✓ Change penalty calculated correctly: {penalty}")

    # Test "total" penalty
    rc = RewardCalculator(
        power_reward_type="None", action_penalty=0.1, action_penalty_type="total"
    )

    penalty = rc.calculate_action_penalty(old_yaws, new_yaws, yaw_max)

    # Average yaw = (2 + 7 + 3) / 3 = 4.0
    # Normalized = 4.0 / 30.0 = 0.1333
    # Penalty = 0.1 * 0.1333 = 0.01333
    expected = 0.1 * (12.0 / 3.0) / 30.0
    assert abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    print(f"✓ Total penalty calculated correctly: {penalty}")
    print()


def test_total_reward():
    """Test total reward calculation."""
    print("Testing total reward calculation...")

    rc = RewardCalculator(
        power_reward_type="Baseline",
        power_scaling=2.0,
        action_penalty=0.1,
        action_penalty_type="change",
    )

    farm_deque = deque([100.0, 110.0, 105.0])
    baseline_deque = deque([100.0, 100.0, 100.0])
    old_yaws = np.array([0.0, 5.0])
    new_yaws = np.array([2.0, 7.0])
    yaw_max = 30.0

    total_reward, breakdown = rc.calculate_total_reward(
        farm_power_deque=farm_deque,
        baseline_power_deque=baseline_deque,
        old_yaws=old_yaws,
        new_yaws=new_yaws,
        yaw_max=yaw_max,
        n_turbines=2,
    )

    # Power reward = 0.05 (as calculated before)
    # Scaled = 0.05 * 2.0 = 0.1
    # Action penalty = 0.1 * 2.0 = 0.2
    # Total = 0.1 - 0.2 = -0.1

    print(f"  Power reward: {breakdown['power_reward']}")
    print(f"  Scaled power reward: {breakdown['scaled_power_reward']}")
    print(f"  Action penalty: {breakdown['action_penalty']}")
    print(f"  Total reward: {breakdown['total_reward']}")

    expected_total = 0.1 - 0.2
    assert abs(total_reward - expected_total) < 1e-6
    print(f"✓ Total reward calculated correctly: {total_reward}")
    print()


def test_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    # Test invalid power_reward_type
    try:
        rc = RewardCalculator(power_reward_type="Invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for invalid power_reward_type: {e}")

    # Test Power_diff without window size
    try:
        rc = RewardCalculator(power_reward_type="Power_diff")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for missing window_size: {e}")

    # Test Power_diff with small window size
    try:
        rc = RewardCalculator(power_reward_type="Power_diff", power_window_size=30)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for small window_size: {e}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RewardCalculator Module")
    print("=" * 60)
    print()

    test_reward_calculator_init()
    test_baseline_reward()
    test_power_avg_reward()
    test_action_penalty()
    test_total_reward()
    test_validation()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
