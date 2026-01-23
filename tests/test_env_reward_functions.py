# tests/test_env_reward_functions.py

import pytest
import yaml
from pathlib import Path
import os
import numpy as np

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid

from py_wake.examples.data.hornsrev1 import V80


# Base YAML configuration parts
YAML_HEADER_MINIMAL = """
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
Track_power: False
"""
YAML_FARM_MINIMAL = """
farm:
  yaw_min: -30
  yaw_max: 30
  xDist: 5
  yDist: 3
  nx: 2
  ny: 1
"""
YAML_WIND_FIXED = """ # Fixed wind for predictability
wind:
  ws_min: 9
  ws_max: 9
  TI_min: 0.07
  TI_max: 0.07
  wd_min: 270
  wd_max: 270
"""
YAML_ACTION_PENALTY_ZERO = """
act_pen:
  action_penalty: 0.0
  action_penalty_type: "Change"
"""
YAML_MES_LEVEL_MINIMAL = """
mes_level:
  turb_ws: True # Minimal necessary for some internal calcs if TI is on
  turb_wd: False
  turb_TI: False
  turb_power: True # Need this for farm_measurements to populate power
  farm_ws: False
  farm_wd: False
  farm_TI: False
  farm_power: True # Need this for farm_measurements to populate power
"""
YAML_MEASUREMENT_SETTINGS_MINIMAL = """ # Minimal for power_avg=1
ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""


def assemble_reward_test_yaml(power_reward_type, power_avg=1, power_scaling=1.0):
    power_def_yaml = f"""
power_def:
  Power_reward: "{power_reward_type}"
  Power_avg: {power_avg}
  Power_scaling: {power_scaling}
"""
    return "\n".join(
        [
            YAML_HEADER_MINIMAL,
            YAML_FARM_MINIMAL,
            YAML_WIND_FIXED,
            YAML_ACTION_PENALTY_ZERO,
            power_def_yaml,
            YAML_MES_LEVEL_MINIMAL,
            YAML_MEASUREMENT_SETTINGS_MINIMAL,
        ]
    )


def run_env_and_get_reward(
    yaml_content, temp_yaml_file_factory, constructor_overrides=None
):
    yaml_filepath = temp_yaml_file_factory(yaml_content, "reward_test")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env_params = {
        "turbine": V80(),
        "x_pos": x_pos,
        "y_pos": y_pos,
        "config": yaml_filepath,
        "seed": 123,
        "dt_sim": 1,
        "dt_env": 10,  # Gives some steps for DWM to run
        "yaw_step_sim": 1.0,
        "turbtype": "None",  # Simplifies flow, focuses on power output based on yaw
        "fill_window": 1,  # Ensures deques are populated minimally and predictably
        "reset_init": True,
    }
    if constructor_overrides:
        env_params.update(constructor_overrides)

    env = None
    try:
        env = WindFarmEnv(**env_params)
        obs, info_reset = env.reset(seed=123)

        # Use a zero action for predictability if action penalty is zero
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_obs, reward, terminated, truncated, info_step = env.step(action)

        return reward, info_step, env  # Return env for further inspection if needed
    finally:
        if env:
            env.close()
        if os.path.exists(yaml_filepath):
            os.remove(yaml_filepath)


def test_power_reward_baseline(temp_yaml_file_factory):
    # Power_avg = 1, Power_scaling = 1.0, action_penalty = 0.0
    yaml_content = assemble_reward_test_yaml(power_reward_type="Baseline", power_avg=1)

    # Baseline_comp should be True for this reward type
    reward, info, env = run_env_and_get_reward(
        yaml_content, temp_yaml_file_factory, {"Baseline_comp": True}
    )

    agent_power = info["Power agent"]
    baseline_power = info["Power baseline"]

    assert env.power_avg == 1, "Test assumes power_avg is 1 for simple deque check"
    # With power_avg=1, farm_pow_deq and base_pow_deq means are the current values
    # after the internal steps of env.step()

    expected_reward = 0.0
    if baseline_power != 0:
        expected_reward = (agent_power / baseline_power) - 1.0
    else:
        # Handle case where baseline_power is 0, PPO reward would likely be 0 or a penalty
        # Depending on WindFarmEnv's internal handling, this might be an edge case.
        # For this test, if agent_power is also 0, reward is 0. If agent_power > 0, reward is large.
        # The current env logic has a `0/0` which would raise ZeroDivisionError.
        # This test should ideally mock or control powers to avoid actual ZeroDivisionError.
        # For now, assuming baseline_power > 0 due to fixed wind speed.
        pytest.skip(
            "Baseline power is zero, reward calculation is undefined or error in env."
        )

    assert np.isclose(
        reward, expected_reward, atol=1e-6
    ), f"Baseline reward mismatch. Got {reward}, expected {expected_reward}. AgentP: {agent_power}, BaseP: {baseline_power}"


def test_power_reward_power_avg(temp_yaml_file_factory):
    # Power_avg = 1, Power_scaling = 1.0, action_penalty = 0.0
    yaml_content = assemble_reward_test_yaml(power_reward_type="Power_avg", power_avg=1)
    reward, info, env = run_env_and_get_reward(
        yaml_content,
        temp_yaml_file_factory,
        {"Baseline_comp": False},  # Not strictly needed but good for isolation
    )

    agent_power_sum = info[
        "Power agent"
    ]  # This is sum of turbine powers for current step

    assert env.power_avg == 1, "Test assumes power_avg is 1 for simple deque check"
    # With power_avg=1, np.mean(env.farm_pow_deq) is agent_power_sum

    expected_reward = 0.0
    if env.rated_power != 0 and env.n_turb != 0:
        expected_reward = agent_power_sum / env.n_turb / env.rated_power
    else:
        pytest.skip("Rated power or n_turb is zero, reward calculation issue.")

    assert np.isclose(
        reward, expected_reward, atol=1e-4
    ), f"Power_avg reward mismatch. Got {reward}, expected {expected_reward}. AgentP sum: {agent_power_sum}"


def test_power_reward_none(temp_yaml_file_factory):
    # Power_scaling = 1.0, action_penalty = 0.0
    yaml_content = assemble_reward_test_yaml(power_reward_type="None")
    reward, info, env = run_env_and_get_reward(
        yaml_content, temp_yaml_file_factory, {"Baseline_comp": False}
    )

    # Expected reward from power component is 0, action penalty is 0.
    expected_reward = 0.0
    assert np.isclose(
        reward, expected_reward, atol=1e-8
    ), f"None reward mismatch. Got {reward}, expected {expected_reward}"


def test_power_reward_power_diff(temp_yaml_file_factory):
    power_avg_val = 40  # Must be >= 40 for Power_diff
    yaml_content = assemble_reward_test_yaml(
        power_reward_type="Power_diff", power_avg=power_avg_val
    )
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    yaml_filepath = temp_yaml_file_factory(yaml_content, "reward_power_diff")
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        seed=123,
        dt_sim=1,
        dt_env=10,
        turbtype="None",
        fill_window=power_avg_val,  # Fill the entire deque for predictability
        reset_init=True,
    )

    # Run enough steps to have distinct "oldest" and "latest" windows after reset's filling
    # The reset already fills `power_avg_val` times if fill_window is set so.
    obs, info_reset = env.reset(seed=123)

    # farm_pow_deq is now full from reset.
    # Take one more step to define a new "latest" window based on this step's power.
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    next_obs, reward, terminated, truncated, info_step = env.step(action)

    # After the step, farm_pow_deq has shifted.
    # The reward calculation uses the state of farm_pow_deq *after* the current step's power is added.

    # To verify, we need to reconstruct what power_latest and power_oldest would have been *inside* power_rew_diff
    # This is tricky as the deque state is internal. A simpler check for Power_diff:
    # 1. Ensure error for power_avg < 40
    # 2. Check that reward is non-zero if power changes significantly over the deque.

    # For now, let's check the ValueError
    yaml_content_invalid = assemble_reward_test_yaml(
        power_reward_type="Power_diff", power_avg=10
    )
    yaml_filepath_invalid = temp_yaml_file_factory(
        yaml_content_invalid, "power_diff_invalid"
    )
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
    with pytest.raises(
        ValueError,
        match="The Power_avg must be larger then 40 for the Power_diff reward",
    ):
        WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath_invalid,
            reset_init=True,
        )

    # A more qualitative test for Power_diff:
    # If power consistently increases, reward should be positive.
    # This requires more intricate step-by-step control of power values, possibly by mocking parts of DWMFlowSimulation,
    # or by running multiple steps and checking the trend.

    # For a simple check here with the current setup:
    # The reward should be a float value.
    assert isinstance(reward, (np.float32, float))
    # We can't easily predict the exact value without knowing the exact sequence of powers that filled the deque.
    # However, if all powers in the deque were identical, the reward should be ~0.
    # If the power from the latest step (info_step["Power agent"]) caused a significant change
    # compared to what was pushed out of the deque, the reward will be non-zero.

    env.close()
    if os.path.exists(yaml_filepath):
        os.remove(yaml_filepath)
    if os.path.exists(yaml_filepath_invalid):
        os.remove(yaml_filepath_invalid)


# --- Test for Action Penalty Type ---
@pytest.mark.parametrize(
    "action_type, penalty_value, initial_yaw_offset, step_action, expected_penalty_factor",
    [
        (
            "Change",
            0.1,
            0.0,
            np.array([0.5, 0.5]),
            0.5 * 1.0,
        ),  # action * yaw_step = 0.5 change
        (
            "Change",
            0.1,
            10.0,
            np.array([-0.5, -0.5]),
            0.5 * 1.0,
        ),  # action * yaw_step = -0.5 change
        (
            "Total",
            0.1,
            0.0,
            np.array([1.0, 1.0]),
            (1.0 * 1.0) / 30.0,
        ),  # new yaw = 1.0. Max_yaw is 30 (from YAML)
        (
            "Total",
            0.1,
            20.0,
            np.array([-1.0, -1.0]),
            0.1 * (20.0 - 10.0) / 30,
        ),  # new yaw = 9.0
    ],
)
def test_action_penalty_logic(
    temp_yaml_file_factory,
    action_type,
    penalty_value,
    initial_yaw_offset,
    step_action,
    expected_penalty_factor,
):
    power_def_yaml = """
power_def:
  Power_reward: "None" # Isolate penalty
  Power_avg: 1
  Power_scaling: 1.0 
"""
    action_pen_yaml = f"""
act_pen:
  action_penalty: {penalty_value}
  action_penalty_type: "{action_type}"
"""
    yaml_content = "\n".join(
        [
            YAML_HEADER_MINIMAL,
            YAML_FARM_MINIMAL,
            YAML_WIND_FIXED,
            action_pen_yaml,
            power_def_yaml,
            YAML_MES_LEVEL_MINIMAL,
            YAML_MEASUREMENT_SETTINGS_MINIMAL,
        ]
    )
    yaml_filepath = temp_yaml_file_factory(yaml_content, f"action_pen_{action_type}")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        seed=123,
        dt_sim=1,
        dt_env=10,
        yaw_step_sim=1.0,  # yaw_step = 1.0 for direct action mapping
        turbtype="None",
        fill_window=1,
        reset_init=True,
    )

    # Manually set initial yaw for "Change" penalty predictability
    obs, info = env.reset(seed=123)
    env.fs.windTurbines.yaw = np.array(
        [initial_yaw_offset] * env.n_turb, dtype=np.float32
    )
    if env.Baseline_comp:  # Should be false if Power_reward is "None"
        env.fs_baseline.windTurbines.yaw = np.array(
            [initial_yaw_offset] * env.n_turb, dtype=np.float32
        )

    # The step action is scaled by yaw_step (1.0 here) when ActionMethod is "yaw"
    # For "Change", old_yaws will be initial_yaw_offset. New yaws will be initial_yaw_offset + step_action * yaw_step.
    # Change = abs(new_yaws - old_yaws) = abs(step_action * yaw_step)
    # For "Total", new_yaws = initial_yaw_offset + step_action * yaw_step. Penalty based on mean(abs(new_yaws))/yaw_max.

    next_obs, reward, _, _, info_step = env.step(step_action)

    # Reward = power_reward (0) + track_reward (0) - action_penalty_val
    # So, reward = -action_penalty_val
    # action_penalty_val = env.action_penalty * pen_factor
    # Here, env.action_penalty is `penalty_value` from parametrize.

    actual_penalty_component = -reward
    expected_calculated_penalty = (
        penalty_value * expected_penalty_factor * env.sim_steps_per_env_step
    )

    assert np.isclose(
        actual_penalty_component, expected_calculated_penalty, atol=1e-5
    ), f"Action penalty mismatch for type '{action_type}'. Got penalty {-reward}, expected {expected_calculated_penalty}"

    env.close()
    if os.path.exists(yaml_filepath):
        os.remove(yaml_filepath)
