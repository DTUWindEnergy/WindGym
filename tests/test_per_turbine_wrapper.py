"""
Tests for PerTurbineObservationWrapper.

This wrapper restructures flat observation/action spaces into per-turbine format.
"""

import pytest
import numpy as np
from pathlib import Path

from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular

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


# =============================================================================
# Derating support (yaw+derate and derate-only envs)
# =============================================================================


@pytest.fixture(scope="module")
def derating_turbine():
    """Synthetic turbine whose powerCtFunction accepts a 'derate' input
    (same shape as the fixture in test_derating.py)."""
    ws = np.linspace(0.0, 30.0, 61)
    derate = np.linspace(0.0, 0.8, 17)

    p_avail = 2.0e6 * np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 3
    power = p_avail[:, None] * (1.0 - derate[None, :])
    ct = 0.8 * (1.0 - derate[None, :]) ** 1.5 * np.ones_like(ws)[:, None]

    pctf = PowerCtNDTabular(
        input_keys=["ws", "derate"],
        value_lst=[ws, derate],
        power_arr=power,
        power_unit="W",
        ct_arr=ct,
        default_value_dict={"derate": 0.0},
    )
    for gi in pctf.interp:
        gi.bounds = "limit"

    return WindTurbine(
        name="SyntheticDerating",
        diameter=80.0,
        hub_height=70.0,
        powerCtFunction=pctf,
    )


def make_derating_config(**overrides):
    """Minimal config for a fast pywake-backend derating env."""
    config = {
        "derate_action": True,
        "derate_min": 0.0,
        "derate_max": 0.8,
        "derate_method": "absolute",
        "ActionMethod": "wind",
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 9.0,
            "ws_max": 9.0,
            "TI_min": 0.06,
            "TI_max": 0.06,
            "wd_min": 270.0,
            "wd_max": 270.0,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "change"},
        "power_def": {
            "Power_reward": "Power_avg",
            "Power_avg": 5,
            "Power_scaling": 1.0,
        },
        "mes_level": {
            "turb_ws": True,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": False,
            "ws_history_N": 1,
            "ws_history_length": 10,
            "ws_window_length": 10,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": False,
            "wd_history_N": 1,
            "wd_history_length": 10,
            "wd_window_length": 10,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 1,
            "yaw_history_length": 10,
            "yaw_window_length": 10,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 10,
            "power_window_length": 10,
        },
    }
    config.update(overrides)
    return config


def make_derating_env(turbine, n_turb=3, config=None, **kwargs):
    d = turbine.diameter()
    return WindFarmEnv(
        turbine=turbine,
        x_pos=np.arange(n_turb) * 5.0 * d,
        y_pos=np.zeros(n_turb),
        config=config if config is not None else make_derating_config(),
        backend="pywake",
        reset_init=False,
        **kwargs,
    )


class TestPerTurbineWrapperDerating:
    """Tests for yaw+derate and derate-only envs."""

    def test_yaw_derate_action_space_shape(self, derating_turbine):
        env = make_derating_env(derating_turbine)
        wrapped = PerTurbineObservationWrapper(env)
        assert wrapped.action_dim_per_turbine == 2
        assert wrapped.action_space.shape == (3, 2)
        env.close()

    def test_yaw_derate_action_mapping_not_scrambled(self, derating_turbine):
        """Derate only turbine 0 via [yaw_i, derate_i] rows: a wrong
        (turbine-grouped) flattening would derate turbine 1 instead."""
        env = make_derating_env(derating_turbine)
        wrapped = PerTurbineObservationWrapper(env)
        wrapped.reset(seed=0)

        # derate action -1 maps to derate_min (=0), +1 to derate_max (=0.8)
        action = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(action)

        derate = np.asarray(env.unwrapped.current_derate)
        assert derate[0] > 0.0
        np.testing.assert_allclose(derate[1:], 0.0, atol=1e-12)
        assert obs.shape == (
            wrapped.n_turbines,
            wrapped.observation_space.shape[1],
        )
        env.close()

    def test_derate_only_env(self, derating_turbine):
        """yaw_action=False gives 1 action per turbine (derate)."""
        env = make_derating_env(
            derating_turbine, config=make_derating_config(yaw_action=False)
        )
        wrapped = PerTurbineObservationWrapper(env)
        assert wrapped.action_dim_per_turbine == 1
        assert wrapped.action_space.shape == (3, 1)

        wrapped.reset(seed=0)
        action = np.array([[1.0], [-1.0], [-1.0]], dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(action)

        derate = np.asarray(env.unwrapped.current_derate)
        assert derate[0] > 0.0
        np.testing.assert_allclose(derate[1:], 0.0, atol=1e-12)
        assert obs.shape == (
            wrapped.n_turbines,
            wrapped.observation_space.shape[1],
        )
        env.close()

    def test_wrong_2d_shape_raises(self, derating_turbine):
        env = make_derating_env(derating_turbine)
        wrapped = PerTurbineObservationWrapper(env)
        with pytest.raises(ValueError, match="shape"):
            wrapped._flatten_action(np.zeros((2, 2)))  # wrong n_turbines
        with pytest.raises(ValueError, match="shape"):
            wrapped._flatten_action(np.zeros((3, 1)))  # wrong act dim
        env.close()

    def test_wrong_flat_length_raises(self, derating_turbine):
        env = make_derating_env(derating_turbine)
        wrapped = PerTurbineObservationWrapper(env)
        with pytest.raises(ValueError, match="length"):
            wrapped._flatten_action(np.zeros(3))  # needs n_turb * 2 = 6
        env.close()

    def test_farm_level_obs_rejected(self, derating_turbine):
        """The obs reshape assumes purely per-turbine obs; farm-level
        measurements must be rejected at construction."""
        config = make_derating_config()
        config["mes_level"]["farm_ws"] = True
        env = make_derating_env(derating_turbine, config=config)
        with pytest.raises(ValueError, match="obs_var"):
            PerTurbineObservationWrapper(env)
        env.close()


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
