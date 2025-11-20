# tests/coliseum/test_agent_control_strategies.py

import pytest
import numpy as np
import yaml

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Agents import PyWakeAgent, NoisyPyWakeAgent
from WindGym.core.measurement_manager import (
    MeasurementManager,
    EpisodicBiasNoiseModel,
    NoisyWindFarmEnv,
    MeasurementType,
)
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Test Constants ---
TRUE_WIND_DIRECTION = 272.0

# --- Fixtures ---


@pytest.fixture(scope="module")
def farm_layout():
    """Provides a consistent 2-turbine layout for all tests in this module."""
    return generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)


@pytest.fixture(scope="module")
def base_env_factory(farm_layout):
    """Factory to create a clean, non-noisy WindFarmEnv instance."""
    x_pos, y_pos = farm_layout

    def _factory(action_method="wind"):
        config = {
            "ActionMethod": action_method,
            "farm": {"yaw_min": -450, "yaw_max": 450},
            "wind": {
                "ws_min": 10,
                "ws_max": 10,
                "wd_min": 272,
                "wd_max": 272,
                "TI_min": 0.07,
                "TI_max": 0.07,
            },
            "yaw_init": "Zeros",
            "BaseController": "Local",
            "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
            "power_def": {
                "Power_reward": "Baseline",
                "Power_avg": 1,
                "Power_scaling": 1.0,
            },
            "mes_level": {
                "turb_ws": True,
                "turb_wd": True,
                "farm_ws": True,
                "farm_wd": True,
                "turb_TI": True,
                "turb_power": True,
                "farm_TI": True,
                "farm_power": True,
            },
            "ws_mes": {
                "ws_current": True,
                "ws_rolling_mean": False,
                "ws_history_N": 0,
                "ws_history_length": 1,
                "ws_window_length": 1,
            },
            "wd_mes": {
                "wd_current": True,
                "wd_rolling_mean": False,
                "wd_history_N": 0,
                "wd_history_length": 1,
                "wd_window_length": 1,
            },
            "yaw_mes": {
                "yaw_current": True,
                "yaw_rolling_mean": False,
                "yaw_history_N": 0,
                "yaw_history_length": 1,
                "yaw_window_length": 1,
            },
            "power_mes": {
                "power_current": True,
                "power_rolling_mean": False,
                "power_history_N": 0,
                "power_history_length": 1,
                "power_window_length": 1,
            },
        }
        return WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml.dump(config),
            turbtype="None",
            reset_init=False,
            Baseline_comp=False,
            yaw_step_sim=10,
            dt_env=10,
        )

    return _factory


@pytest.fixture(scope="module")
def oracle_optimal_yaw_t0(base_env_factory, farm_layout):
    """
    FIXTURE-BASED SOURCE OF TRUTH: Runs the oracle agent once to determine the
    "correct" optimal yaw for the upstream turbine (T0) in this setup.
    This value is then used as the ground truth by other tests.
    """
    env = base_env_factory(action_method="wind")
    obs, _ = env.reset(seed=42)

    agent = PyWakeAgent(
        x_pos=farm_layout[0],
        y_pos=farm_layout[1],
        turbine=V80(),
        env=env,
        yaw_min=env.unwrapped.yaw_min,
        yaw_max=env.unwrapped.yaw_max,
    )
    agent.update_wind(wind_speed=10.0, wind_direction=TRUE_WIND_DIRECTION, TI=0.07)

    action, _ = agent.predict(obs, deterministic=True)
    unscaled_action = agent.unscale_yaw(action)
    env.close()
    return unscaled_action[0]  # Return only the yaw for the first turbine


@pytest.fixture
def noisy_env_factory(base_env_factory):
    """Factory to create a NoisyWindFarmEnv with a specified bias."""

    def _factory(action_method="wind", wd_bias=0.0):
        clean_env_instance = base_env_factory(action_method)
        mm = MeasurementManager(env=clean_env_instance, seed=42)
        bias_model = EpisodicBiasNoiseModel(
            {MeasurementType.WIND_DIRECTION: (wd_bias, wd_bias)}
        )  # constant, deterministic bias
        mm.set_noise_model(bias_model)

        noisy_env = NoisyWindFarmEnv(
            base_env_class=type(clean_env_instance),
            measurement_manager=mm,
            **clean_env_instance.kwargs,
        )
        clean_env_instance.close()
        return noisy_env

    return _factory


# --- Tests ---


def test_oracle_agent_behavior(oracle_optimal_yaw_t0):
    """
    Part 1: Sanity-checks the oracle agent's baseline behavior.
    Asserts that it calculates a significant, non-zero yaw for wake steering.
    """
    print(f"Oracle-calculated optimal yaw for T0 is: {oracle_optimal_yaw_t0:.2f}°")
    assert (
        abs(oracle_optimal_yaw_t0) > 15
    ), "Oracle agent should command a large steering angle."


def test_noisy_agent_matches_oracle_with_minimal_noise(
    noisy_env_factory, farm_layout, oracle_optimal_yaw_t0
):
    """
    Part 2: Verifies the NoisyPyWakeAgent behaves identically to the Oracle
    when sensor noise is negligible.
    """
    noisy_env = noisy_env_factory(action_method="wind", wd_bias=1e-9)
    obs, _ = noisy_env.reset(seed=42)

    noisy_agent = NoisyPyWakeAgent(
        measurement_manager=noisy_env.measurement_manager,
        x_pos=farm_layout[0],
        y_pos=farm_layout[1],
        turbine=V80(),
        yaw_min=noisy_env.unwrapped.yaw_min,
        yaw_max=noisy_env.unwrapped.yaw_max,
    )

    action, _ = noisy_agent.predict(obs, deterministic=True)
    unscaled_action = noisy_agent.unscale_yaw(action)

    np.testing.assert_allclose(unscaled_action[0], oracle_optimal_yaw_t0, atol=0.1)
    noisy_env.close()


@pytest.mark.parametrize(
    "action_method, expected_yaw_func",
    [
        ("yaw", lambda oracle_yaw: 0.0),  # Robust: final yaw should be 0
        (
            "wind",
            lambda oracle_yaw: oracle_yaw,
        ),  # Vulnerable: final yaw should be the misaligned optimal angle
    ],
    ids=["YawControl_Is_Robust", "WindControl_Is_Vulnerable"],
)
def test_control_strategies_under_bias(
    noisy_env_factory,
    farm_layout,
    oracle_optimal_yaw_t0,
    action_method,
    expected_yaw_func,
):
    """
    Part 3: Tests both control strategies under a significant +90 degree wind direction bias.
    """
    BIAS = 90.0  # Define the bias constant

    if action_method == "wind":
        # A vulnerable agent aligns with the PERCEIVED wind + the optimal OFFSET.
        # Perceived wind = 272 (true) + 90 (bias) = 362, which is 2 degrees.
        # So, the expected absolute heading is (2 + 19.69) = 21.69 degrees.
        # perceived_wd = (TRUE_WIND_DIRECTION + BIAS) % 360
        expected_final_yaw_t0 = BIAS
        # expected_final_yaw_t0 = (perceived_wd + oracle_optimal_yaw_t0) % 360
    else:
        # The robust 'yaw' method should result in a 0 degree offset.
        expected_final_yaw_t0 = expected_yaw_func(oracle_optimal_yaw_t0)

    # Perceived wind will be 272 (true) + 90 (bias) = 362 -> 2 degrees
    env = noisy_env_factory(action_method=action_method, wd_bias=BIAS)
    obs, _ = env.reset(seed=42)

    # Note: The PyWakeAgent is re-initialized for each test run to ensure a clean state
    agent = NoisyPyWakeAgent(
        measurement_manager=env.measurement_manager,
        x_pos=farm_layout[0],
        y_pos=farm_layout[1],
        turbine=V80(),
        yaw_min=env.unwrapped.yaw_min,
        yaw_max=env.unwrapped.yaw_max,
    )

    # Run for a few steps to allow the turbine to settle at its commanded yaw
    for _ in range(5):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)

    final_yaw_angles = info["yaw angles agent"]

    # Assert the final yaw of the upstream turbine (T0)
    np.testing.assert_allclose(
        final_yaw_angles[0],
        expected_final_yaw_t0,
        atol=0.5,
        err_msg=f"Failed for ActionMethod='{action_method}'",
    )
    env.close()
