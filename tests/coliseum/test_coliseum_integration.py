# tests/test_colesium_integration.py

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.farm_eval import FarmEval
from WindGym.Agents import PyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

# --- Fixtures ---


@pytest.fixture
def temp_yaml_file_integration():
    """Provides a path to a temporary, valid YAML file for integration tests."""
    # This YAML now correctly uses ActionMethod: "wind" for agents that output absolute targets.
    yaml_config = """
    yaw_init: "Zeros"
    noise: "None"
    BaseController: "Local"
    ActionMethod: "wind"
    farm: {yaw_min: -30, yaw_max: 30}
    wind: {ws_min: 10, ws_max: 10, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
    act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
    power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
    mes_level: {turb_ws: True, turb_wd: True, turb_TI: True, turb_power: True, farm_ws: True, farm_wd: True, farm_TI: True, farm_power: True}
    ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
    wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
    yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
    power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_file:
        tmp_file.write(yaml_config)
        filepath = tmp_file.name

    yield filepath

    os.remove(filepath)


class TestColiseumAndAgentIntegration:
    """
    This test suite validates the integration between the Coliseum evaluator,
    the FarmEval environment, and specific agents like PyWakeAgent.
    """

    def test_coliseum_runs_end_to_end_with_pywake_agent(
        self, temp_yaml_file_integration
    ):
        """
        Verifies that the Coliseum can run a full time-series evaluation with a
        PyWakeAgent without errors. This is a high-level test to ensure all
        components are wired together correctly.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

        with open(temp_yaml_file_integration, "r") as f:
            config = yaml.safe_load(f)
        yaw_min = config["farm"]["yaw_min"]
        yaw_max = config["farm"]["yaw_max"]

        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                Baseline_comp=False,
                n_passthrough=0.01,
                burn_in_passthroughs=0.01,
            )

        pywake_agent = PyWakeAgent(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=V80(),
            yaw_min=yaw_min,
            yaw_max=yaw_max,
            env=env_factory(),
        )

        coliseum = Coliseum(
            env_factory, agents={"PyWake": pywake_agent}, n_passthrough=1.0
        )

        # The main assertion is that this call completes without raising an exception.
        summary_df = coliseum.run_time_series_evaluation(num_episodes=1, seed=42)

        # We can add a simple, robust check on the result's structure.
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert "PyWake" in summary_df.columns
        assert not summary_df["PyWake"].isnull().any()

    def test_environment_interprets_pywake_action_correctly(
        self, temp_yaml_file_integration
    ):
        """
        Verifies that the environment correctly interprets the PyWakeAgent's action
        and moves the turbine by the expected amount in a single step. This is a
        more precise and robust check than the previous end-of-episode assertion.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

        with open(temp_yaml_file_integration, "r") as f:
            config = yaml.safe_load(f)
        yaw_min = config["farm"]["yaw_min"]
        yaw_max = config["farm"]["yaw_max"]

        # 1. Setup the environment with a specific yaw_step for a predictable outcome
        YAW_STEP = 2.0
        env = FarmEval(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=temp_yaml_file_integration,
            turbtype="None",
            reset_init=True,
            Baseline_comp=False,
            yaw_step_sim=YAW_STEP,
            n_passthrough=0.5,
            burn_in_passthroughs=0.01,
        )
        obs, info = env.reset(seed=42)  # Initial yaw starts at 0.0

        # 2. Initialize the agent and find its optimal target
        pywake_agent = PyWakeAgent(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=V80(),
            yaw_min=yaw_min,
            yaw_max=yaw_max,
            env=env,
        )
        pywake_agent.update_wind(wind_speed=10, wind_direction=270, TI=0.07)

        # 3. Get the agent's action and take one step in the environment
        action, _ = pywake_agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # 4. Assert the result after one step
        final_yaw_t1 = info["yaw angles agent"][0]

        # The yaw should have moved from 0.0 exactly by YAW_STEP towards the positive target.
        assert (
            final_yaw_t1 == YAW_STEP
        ), "Turbine did not yaw by the correct amount in one step."

        env.close()
