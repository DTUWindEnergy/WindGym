# tests/test_determinism.py

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
import yaml  # For reading YAML in the test

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from gymnasium.utils.env_checker import data_equivalence  # Helper from Gymnasium
from copy import deepcopy  # For debugging RNG state if needed
from WindGym.utils.generate_layouts import generate_square_grid


# A controlled YAML configuration string for this specific determinism test.
DETERMINISM_TEST_YAML_STRING = """
yaw_init: "Zeros"
noise: "None" # Start with no noise to test base determinism
BaseController: "Local"
ActionMethod: "yaw"
Track_power: False
farm:
  yaw_min: -30
  yaw_max: 30
  xDist: 5
  yDist: 3
  nx: 2
  ny: 1
wind:
  ws_min: 9 
  ws_max: 9
  TI_min: 0.07
  TI_max: 0.07
  wd_min: 270
  wd_max: 270
act_pen:
  action_penalty: 0.0 
  action_penalty_type: "Change"
power_def:
  Power_reward: "Baseline"
  Power_avg: 5 
  Power_scaling: 1.0
mes_level:
  turb_ws: True
  turb_wd: True
  turb_TI: True
  turb_power: True
  farm_ws: True
  farm_wd: True
  farm_TI: True
  farm_power: True
ws_mes: {ws_current: True, ws_rolling_mean: True, ws_history_N: 1, ws_history_length: 5, ws_window_length: 2}
wd_mes: {wd_current: True, wd_rolling_mean: True, wd_history_N: 1, wd_history_length: 5, wd_window_length: 2}
yaw_mes: {yaw_current: True, yaw_rolling_mean: True, yaw_history_N: 1, yaw_history_length: 5, yaw_window_length: 2}
power_mes: {power_current: True, power_rolling_mean: True, power_history_N: 1, power_history_length: 5, power_window_length: 2}
"""


@pytest.fixture
def deterministic_env_config_path():
    """Creates a temporary YAML file with a fixed config for determinism testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(DETERMINISM_TEST_YAML_STRING)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)


@pytest.mark.slow
def test_step_determinism_basic(deterministic_env_config_path):
    """
    A minimal test to check if resetting with the same seed and applying
    the same action yields identical observations and rewards.
    This mimics the core of gymnasium's check_env for step determinism.
    """
    common_seed = 123
    env = None
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    try:
        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=deterministic_env_config_path,
            seed=common_seed,
            dt_sim=1,
            dt_env=10,
            yaw_step_sim=1.0,
            turbtype="None",
            Baseline_comp=True,
            fill_window=5,
            burn_in_passthroughs=0.0001,  # Minimal burn-in for faster tests
            reset_init=True,  # Ensure reset is called in __init__ which will call super().reset()
        )

        env.action_space.seed(common_seed)
        action = env.action_space.sample()

        # --- First run ---
        # env.reset() was already called by __init__ due to reset_init=True
        # For this test, we want to control the seed explicitly for the two trials
        obs1, info1 = env.reset(seed=common_seed)
        next_obs1, reward1, terminated1, truncated1, info_step1 = env.step(action)

        # --- Second run ---
        obs2, info2 = env.reset(seed=common_seed)
        next_obs2, reward2, terminated2, truncated2, info_step2 = env.step(action)

        # --- Assertions ---
        assert data_equivalence(
            next_obs1, next_obs2
        ), f"Observations are not deterministic.\nObs1: {next_obs1}\nObs2: {next_obs2}"

        assert np.isclose(
            reward1, reward2, atol=1e-8
        ), f"Rewards are not deterministic. Rew1: {reward1}, Rew2: {reward2}"

        assert (
            terminated1 == terminated2
        ), f"Termination flags are not deterministic. Term1: {terminated1}, Term2: {terminated2}"
        assert (
            truncated1 == truncated2
        ), f"Truncation flags are not deterministic. Trunc1: {truncated1}, Trunc2: {truncated2}"

        assert np.isclose(
            info_step1["Power agent"], info_step2["Power agent"], atol=1e-8
        ), f"Agent Power in info not deterministic. P1: {info_step1['Power agent']}, P2: {info_step2['Power agent']}"
        if env.Baseline_comp:
            assert np.isclose(
                info_step1["Power baseline"], info_step2["Power baseline"], atol=1e-8
            ), f"Baseline Power in info not deterministic. P1_base: {info_step1['Power baseline']}, P2_base: {info_step2['Power baseline']}"

    finally:
        if env:
            env.close()


@pytest.mark.slow
def test_step_determinism_with_noise(
    deterministic_env_config_path,
):  # Still uses the base YAML initially
    """
    Tests determinism specifically when 'Normal' noise is enabled.
    """
    common_seed = 456
    env = None

    # Modify the YAML string content for this test
    noisy_yaml_string = DETERMINISM_TEST_YAML_STRING.replace(
        'noise: "None"', 'noise: "Normal"'
    )

    # Create a temporary YAML file with the modified noisy configuration
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(noisy_yaml_string)
        noisy_yaml_filepath = tmp_file.name
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    try:
        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=noisy_yaml_filepath,  # Use the modified YAML path
            seed=common_seed,
            dt_sim=1,
            dt_env=10,
            yaw_step_sim=1.0,
            turbtype="None",
            Baseline_comp=True,
            fill_window=5,
            burn_in_passthroughs=0.0001,  # Minimal burn-in for faster tests
            reset_init=True,
        )

        env.action_space.seed(common_seed)
        action = env.action_space.sample()

        obs1, _ = env.reset(seed=common_seed)
        next_obs1, reward1, _, _, info_step1 = env.step(action)

        obs2, _ = env.reset(seed=common_seed)
        next_obs2, reward2, _, _, info_step2 = env.step(action)

        assert data_equivalence(
            next_obs1, next_obs2
        ), f"Observations with 'Normal' noise are not deterministic.\nObs1: {next_obs1}\nObs2: {next_obs2}"

        assert np.isclose(
            reward1, reward2, atol=1e-8
        ), f"Rewards with 'Normal' noise are not deterministic. Rew1: {reward1}, Rew2: {reward2}"

    finally:
        if env:
            env.close()
        os.remove(noisy_yaml_filepath)  # Clean up the temporary noisy YAML
