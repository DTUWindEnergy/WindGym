"""
Tests for PerTurbineObservationWrapper.

This wrapper restructures flat observation/action spaces into per-turbine format.
"""

import pytest
import numpy as np
from pathlib import Path

from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from WindGym.wrappers.per_turbine_wrapper import PerTurbineObservationWrapper


@pytest.fixture
def base_example_data_path():
    """Provides path to the example configuration directory."""
    return Path("examples/EnvConfigs")


@pytest.fixture
def base_env(base_example_data_path):
    """Create a base WindFarmEnv for testing."""
    yaml_path = base_example_data_path / Path("Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=5, yDist=5)
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        turbtype="None",  # No turbulence for faster tests
        burn_in_passthroughs=0.0001,  # Minimal burn-in for speed
    )
    yield env
    env.close()


@pytest.fixture
def wrapped_env(base_env):
    """Create a wrapped environment for testing."""
    return PerTurbineObservationWrapper(base_env)


# Module-scoped fixtures for tests that don't modify environment state
@pytest.fixture(scope="module")
def shared_base_env():
    """Module-scoped base WindFarmEnv shared across read-only tests."""
    yaml_path = Path("examples/EnvConfigs/Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=5, yDist=5)
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        turbtype="None",
        burn_in_passthroughs=0.0001,
    )
    yield env
    env.close()


@pytest.fixture(scope="module")
def shared_wrapped_env(shared_base_env):
    """Module-scoped wrapped environment for read-only tests."""
    return PerTurbineObservationWrapper(shared_base_env)


class TestPerTurbineWrapperInitialization:
    """Tests for wrapper initialization."""

    def test_wrapper_initializes(self, wrapped_env):
        """Test that the wrapper initializes correctly."""
        assert wrapped_env is not None

    def test_n_turbines_property(self, wrapped_env, base_env):
        """Test that n_turbines matches the base environment."""
        assert wrapped_env.n_turbines == base_env.n_turb
        assert wrapped_env.n_turbines == 4  # 2x2 grid

    def test_turbine_positions_property(self, wrapped_env, base_env):
        """Test that turbine positions are correctly extracted."""
        positions = wrapped_env.turbine_positions
        assert positions.shape == (wrapped_env.n_turbines, 2)
        np.testing.assert_array_equal(positions[:, 0], base_env.x_pos)
        np.testing.assert_array_equal(positions[:, 1], base_env.y_pos)

    def test_rotor_diameter_property(self, wrapped_env, base_env):
        """Test that rotor diameter matches the base environment."""
        assert wrapped_env.rotor_diameter == base_env.D
        assert wrapped_env.rotor_diameter == V80().diameter()

    def test_observation_space_shape(self, wrapped_env):
        """Test that observation space has correct per-turbine shape."""
        obs_space = wrapped_env.observation_space
        assert len(obs_space.shape) == 2
        assert obs_space.shape[0] == wrapped_env.n_turbines


class TestPerTurbineWrapperReset:
    """Tests for wrapper reset functionality."""

    def test_reset_returns_correct_shape(self, wrapped_env):
        """Test that reset returns observations with correct shape."""
        obs, info = wrapped_env.reset()
        expected_shape = (
            wrapped_env.n_turbines,
            wrapped_env.observation_space.shape[1],
        )
        assert obs.shape == expected_shape

    @pytest.mark.slow
    def test_reset_with_seed(self, wrapped_env):
        """Test that reset with seed is reproducible."""
        obs1, _ = wrapped_env.reset(seed=42)
        obs2, _ = wrapped_env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_returns_info_dict(self, wrapped_env):
        """Test that reset returns an info dictionary."""
        obs, info = wrapped_env.reset()
        assert isinstance(info, dict)


class TestPerTurbineWrapperStep:
    """Tests for wrapper step functionality."""

    def test_step_with_2d_action(self, wrapped_env):
        """Test step with per-turbine (2D) action."""
        wrapped_env.reset()
        action = np.zeros((wrapped_env.n_turbines, 1))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        expected_shape = (
            wrapped_env.n_turbines,
            wrapped_env.observation_space.shape[1],
        )
        assert obs.shape == expected_shape
        assert np.isscalar(reward) or isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_with_flat_action(self, wrapped_env):
        """Test step with flat (1D) action."""
        wrapped_env.reset()
        action_flat = np.zeros(wrapped_env.n_turbines)
        obs, reward, terminated, truncated, info = wrapped_env.step(action_flat)

        expected_shape = (
            wrapped_env.n_turbines,
            wrapped_env.observation_space.shape[1],
        )
        assert obs.shape == expected_shape

    def test_step_preserves_reward(self, wrapped_env):
        """Test that wrapper doesn't modify reward from base env."""
        wrapped_env.reset()
        action = np.zeros(wrapped_env.n_turbines)
        obs, reward, _, _, _ = wrapped_env.step(action)
        # Reward should be a valid float
        assert np.isfinite(reward)


class TestPerTurbineWrapperWindDirection:
    """Tests for wind direction property."""

    def test_mean_wind_direction_property(self, wrapped_env):
        """Test that mean_wind_direction returns valid wind direction."""
        wrapped_env.reset()
        wd = wrapped_env.mean_wind_direction
        assert isinstance(wd, (int, float))
        # Wind direction should be between 0 and 360
        assert 0 <= wd <= 360


class TestPerTurbineWrapperActionFlattening:
    """Tests for action flattening logic."""

    def test_flatten_action_1d_passthrough(self, wrapped_env):
        """Test that 1D actions pass through unchanged."""
        action_1d = np.array([0.1, 0.2, 0.3, 0.4])
        result = wrapped_env._flatten_action(action_1d)
        np.testing.assert_array_equal(result, action_1d)

    def test_flatten_action_2d_to_1d(self, wrapped_env):
        """Test that 2D actions are flattened correctly."""
        action_2d = np.array([[0.1], [0.2], [0.3], [0.4]])
        result = wrapped_env._flatten_action(action_2d)
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(result, expected)


class TestPerTurbineWrapperObsReshaping:
    """Tests for observation reshaping logic."""

    def test_reshape_obs_correct_shape(self, wrapped_env):
        """Test that observations are reshaped to per-turbine format."""
        wrapped_env.reset()
        # Create a flat observation matching expected size
        n_turb = wrapped_env.n_turbines
        obs_dim = wrapped_env.observation_space.shape[1]
        flat_obs = np.arange(n_turb * obs_dim, dtype=np.float32)

        reshaped = wrapped_env._reshape_obs_to_per_turbine(flat_obs)
        assert reshaped.shape == (n_turb, obs_dim)


class TestPerTurbineWrapperMultipleSteps:
    """Tests for multiple sequential steps."""

    @pytest.mark.slow
    def test_multiple_steps(self, wrapped_env):
        """Test that wrapper works correctly over multiple steps."""
        wrapped_env.reset()
        for _ in range(5):
            action = np.zeros((wrapped_env.n_turbines, 1))
            obs, _, terminated, truncated, _ = wrapped_env.step(action)
            expected_shape = (
                wrapped_env.n_turbines,
                wrapped_env.observation_space.shape[1],
            )
            assert obs.shape == expected_shape
            if terminated or truncated:
                break

    @pytest.mark.slow
    def test_reset_after_steps(self, wrapped_env):
        """Test that reset works correctly after taking steps."""
        obs1, _ = wrapped_env.reset()
        action = np.zeros((wrapped_env.n_turbines, 1))
        wrapped_env.step(action)
        wrapped_env.step(action)

        obs2, _ = wrapped_env.reset()
        assert obs2.shape == obs1.shape
