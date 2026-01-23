"""
Tests for RewardCalculator to verify it works correctly.
"""

import pytest
import numpy as np
from collections import deque

from WindGym.core.reward_calculator import RewardCalculator


def test_reward_calculator_init():
    """Test RewardCalculator initialization."""
    # Test Baseline reward
    rc = RewardCalculator(
        power_reward_type="Baseline",
        power_scaling=1.0,
        action_penalty=0.01,
        action_penalty_type="change",
    )
    assert rc.power_reward_type == "Baseline"

    # Test Power_avg reward
    rc = RewardCalculator(
        power_reward_type="Power_avg",
        power_scaling=2.0,
        action_penalty=0.05,
        action_penalty_type="total",
    )
    assert rc.power_scaling == 2.0

    # Test Power_diff reward
    rc = RewardCalculator(
        power_reward_type="Power_diff", power_scaling=1.0, power_window_size=50
    )
    assert rc._power_window_size == 50


def test_baseline_reward():
    """Test baseline reward calculation."""
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


def test_power_avg_reward():
    """Test power_avg reward calculation."""
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


def test_action_penalty():
    """Test action penalty calculation."""
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


def test_total_reward():
    """Test total reward calculation."""
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
    expected_total = 0.1 - 0.2
    assert abs(total_reward - expected_total) < 1e-6


def test_validation():
    """Test configuration validation."""
    # Test invalid power_reward_type
    with pytest.raises(ValueError):
        RewardCalculator(power_reward_type="Invalid")

    # Test Power_diff without window size
    with pytest.raises(ValueError):
        RewardCalculator(power_reward_type="Power_diff")

    # Test Power_diff with small window size
    with pytest.raises(ValueError):
        RewardCalculator(power_reward_type="Power_diff", power_window_size=30)


def test_invalid_action_penalty_type():
    """Test validation of invalid action_penalty_type."""
    with pytest.raises(ValueError, match="action_penalty_type must be one of"):
        RewardCalculator(
            power_reward_type="None",
            action_penalty=0.1,
            action_penalty_type="invalid_type",
        )


def test_baseline_reward_without_baseline_deque():
    """Test that baseline reward raises error without baseline_power_deque."""
    rc = RewardCalculator(power_reward_type="Baseline", power_scaling=1.0)

    farm_deque = deque([100.0, 110.0, 105.0])

    with pytest.raises(ValueError, match="baseline_power_deque required"):
        rc.calculate_power_reward(
            farm_power_deque=farm_deque,
            baseline_power_deque=None,  # Missing!
        )


def test_power_avg_reward_without_rated_power():
    """Test that Power_avg reward raises error without rated_power."""
    rc = RewardCalculator(power_reward_type="Power_avg", power_scaling=1.0)

    farm_deque = deque([1000.0, 1200.0, 1100.0])

    with pytest.raises(ValueError, match="rated_power required"):
        rc.calculate_power_reward(
            farm_power_deque=farm_deque,
            rated_power=None,  # Missing!
        )


def test_baseline_reward_zero_baseline_power():
    """Test that zero baseline power raises ValueError."""
    rc = RewardCalculator(power_reward_type="Baseline", power_scaling=1.0)

    farm_deque = deque([100.0, 110.0, 105.0])
    baseline_deque = deque([0.0, 0.0, 0.0])  # Zero baseline power

    with pytest.raises(ValueError, match="Baseline power is zero"):
        rc.calculate_power_reward(
            farm_power_deque=farm_deque, baseline_power_deque=baseline_deque
        )


def test_none_reward_type():
    """Test that 'None' reward type returns 0."""
    rc = RewardCalculator(power_reward_type="None", power_scaling=1.0)

    farm_deque = deque([100.0, 110.0, 105.0])

    reward = rc.calculate_power_reward(farm_power_deque=farm_deque)
    assert reward == 0.0


def test_action_penalty_negligible():
    """Test that negligible action penalty (< 0.001) returns 0."""
    rc = RewardCalculator(
        power_reward_type="None",
        action_penalty=0.0001,  # Very small
        action_penalty_type="change",
    )

    old_yaws = np.array([0.0, 5.0, -5.0])
    new_yaws = np.array([10.0, 15.0, 5.0])

    penalty = rc.calculate_action_penalty(old_yaws, new_yaws, yaw_max=30.0)
    assert penalty == 0.0


def test_action_penalty_none_type():
    """Test that None action_penalty_type returns 0."""
    rc = RewardCalculator(
        power_reward_type="None", action_penalty=1.0, action_penalty_type=None
    )

    old_yaws = np.array([0.0, 5.0])
    new_yaws = np.array([10.0, 15.0])

    penalty = rc.calculate_action_penalty(old_yaws, new_yaws, yaw_max=30.0)
    assert penalty == 0.0


def test_power_diff_reward():
    """Test Power_diff reward calculation."""
    rc = RewardCalculator(
        power_reward_type="Power_diff", power_scaling=1.0, power_window_size=50
    )

    # Create a deque with increasing power (improvement over time)
    # First values low, last values high
    power_values = list(range(100, 200))  # 100 values from 100 to 199
    farm_deque = deque(power_values)

    reward = rc.calculate_power_reward(farm_power_deque=farm_deque, n_turbines=2)

    # Power should have increased, so reward should be positive
    assert reward > 0, "Power_diff reward should be positive for increasing power"


def test_total_reward_with_all_components():
    """Test total reward calculation with power and penalty."""
    rc = RewardCalculator(
        power_reward_type="Baseline",
        power_scaling=1.5,
        action_penalty=0.2,
        action_penalty_type="total",
    )

    farm_deque = deque([120.0, 120.0, 120.0])
    baseline_deque = deque([100.0, 100.0, 100.0])
    old_yaws = np.array([0.0, 0.0])
    new_yaws = np.array([15.0, 15.0])  # 50% of yaw_max
    yaw_max = 30.0

    total_reward, breakdown = rc.calculate_total_reward(
        farm_power_deque=farm_deque,
        baseline_power_deque=baseline_deque,
        old_yaws=old_yaws,
        new_yaws=new_yaws,
        yaw_max=yaw_max,
        n_turbines=2,
    )

    # Verify breakdown contains expected keys
    assert "power_reward" in breakdown
    assert "scaled_power_reward" in breakdown
    assert "action_penalty" in breakdown
    assert "total_reward" in breakdown

    # Power reward = 120/100 - 1 = 0.2
    assert abs(breakdown["power_reward"] - 0.2) < 1e-6

    # Scaled = 0.2 * 1.5 = 0.3
    assert abs(breakdown["scaled_power_reward"] - 0.3) < 1e-6

    # Action penalty: mean(abs([15, 15])) / 30 * 0.2 = 0.5 * 0.2 = 0.1
    assert abs(breakdown["action_penalty"] - 0.1) < 1e-6

    # Total = 0.3 - 0.1 = 0.2
    assert abs(total_reward - 0.2) < 1e-6


def test_track_power_not_implemented():
    """Test that track_power=True raises NotImplementedError."""
    with pytest.raises(
        NotImplementedError, match="Power tracking reward is not yet implemented"
    ):
        RewardCalculator(power_reward_type="Baseline", track_power=True)
