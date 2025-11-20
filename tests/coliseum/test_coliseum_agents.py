# test_coliseum_agents.py

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os
from unittest.mock import patch, MagicMock

from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.farm_eval import FarmEval
from WindGym.Agents import PyWakeAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site

# --- Test Class for Advanced Coliseum Features ---


class TestColiseumAdvanced:
    """Tests advanced and specific logic within the Coliseum evaluator."""

    def test_create_env_factory_with_site(self, temp_yaml_file):
        """Covers the static helper method for creating a site-based factory."""
        mock_site = Hornsrev1Site()
        mock_kwargs = {
            "x_pos": [0],
            "y_pos": [0],
            "turbine": V80(),
            "config": temp_yaml_file,
            "n_passthrough": 0.05,
            "burn_in_passthroughs": 0.01,
            "turbtype": "None",
        }

        factory = Coliseum.create_env_factory_with_site(
            env_class=FarmEval, site=mock_site, **mock_kwargs
        )

        env = factory()

        assert isinstance(env, FarmEval)
        assert env.sample_site is mock_site
        env.close()

    def test_pywake_agent_update_wind_is_called(self, temp_yaml_file):
        """
        Tests that for agents with an `update_wind` method (like PyWakeAgent),
        the Coliseum calls it with the correct wind conditions during grid evaluation.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

        # 1. Create a real PyWakeAgent instance
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        # 2. Create the Coliseum instance with this agent
        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=temp_yaml_file,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.001,
            )

        coliseum = Coliseum(
            env_factory,
            agents={"PyWake": pywake_agent},
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        # 3. "Spy" on the agent's update_wind method
        with patch.object(
            pywake_agent, "update_wind", wraps=pywake_agent.update_wind
        ) as mock_update_method:
            # 4. Run a single-point grid evaluation
            ws_test, wd_test, ti_test = 10, 270, 0.08

            coliseum.run_wind_grid_evaluation(
                ws_min=ws_test,
                ws_max=ws_test,
                ws_step=1,
                wd_min=wd_test,
                wd_max=wd_test,
                wd_step=1,
                ti_min=ti_test,
                ti_max=ti_test,
                ti_points=1,
            )

            # 5. Assert that the spy was called correctly
            mock_update_method.assert_called_once()

            # FIX: Use keyword arguments in the assertion to match the actual call
            mock_update_method.assert_called_with(
                wind_speed=ws_test, wind_direction=wd_test, TI=ti_test
            )


# --- Integration Tests for PyWakeAgent ---


@pytest.fixture
def temp_yaml_file_integration():
    """Provides a path to a temporary, valid YAML file for the duration of a test."""
    yaml_config = """
    yaw_init: "Zeros"
    noise: "None"
    BaseController: "Local"
    ActionMethod: "yaw"
    farm: {yaw_min: -30, yaw_max: 30}
    wind: {ws_min: 10, ws_max: 10, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
    act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
    power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
    mes_level: {turb_ws: True, turb_wd: True, turb_TI: True, turb_power: False, farm_ws: False, farm_wd: False, farm_TI: False, farm_power: False}
    ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 50, ws_window_length: 1}
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


class TestColiseumIntegration:
    """
    Tests the Coliseum framework with real agents and environments,
    focusing on end-to-end functionality without mocks.
    """

    def test_create_env_factory_with_site(self, temp_yaml_file_integration):
        """Covers the static helper method for creating a site-based factory."""
        mock_site = Hornsrev1Site()
        mock_kwargs = {
            "x_pos": [0],
            "y_pos": [0],
            "turbine": V80(),
            "config": temp_yaml_file_integration,
            "n_passthrough": 0.05,
            "burn_in_passthroughs": 0.01,
            "turbtype": "None",
        }

        factory = Coliseum.create_env_factory_with_site(
            env_class=FarmEval, site=mock_site, **mock_kwargs
        )

        env = factory()

        assert isinstance(env, FarmEval)
        assert env.sample_site is mock_site
        env.close()

    def test_pywake_agent_in_time_series_evaluation(self, temp_yaml_file_integration):
        """
        Tests that a PyWakeAgent can run a full time-series evaluation and that its
        internal state is correctly updated by the Coliseum framework.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        # The env_factory will create an env with ws=10, wd=270 from the YAML
        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.001,
            )

        coliseum = Coliseum(
            env_factory,
            agents={"PyWake": pywake_agent},
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        # Run a single episode. This will trigger the agent's update_wind method.
        coliseum.run_time_series_evaluation(num_episodes=1, seed=42)

        # After the run, the agent's internal state should match the environment's.
        # This is a direct test that the synchronization happened.
        assert pywake_agent.wsp == [10]
        assert pywake_agent.wdir == [270]
        assert pywake_agent.TI == 0.07

        # Also, check that the agent has calculated a valid optimization result.
        assert pywake_agent.optimized is True
        assert pywake_agent.optimized_yaws is not None
        # For this specific case (2 turbines at 270 deg), the optimal yaw for the upstream
        # turbine should be non-zero to steer the wake.
        assert pywake_agent.optimized_yaws[0] != 0

    def test_pywake_agent_synchronization(self, temp_yaml_file_integration):
        """
        Tests that a PyWakeAgent's internal state is correctly updated by the Coliseum
        framework before a time-series episode run.
        """
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.001,
            )

        coliseum = Coliseum(
            env_factory,
            agents={"PyWake": pywake_agent},
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        coliseum.run_time_series_evaluation(num_episodes=1, seed=42)

        assert pywake_agent.wsp == [10]
        assert pywake_agent.wdir == [270]
        assert pywake_agent.TI == 0.07
        assert pywake_agent.optimized is True

    def test_pywake_agent_in_grid_evaluation(self, temp_yaml_file_integration):
        """
        Tests that the PyWakeAgent's state is correctly updated for each point
        in a grid evaluation.
        """

        class SpyPyWakeAgent(PyWakeAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.update_calls = []

            def update_wind(self, wind_speed, wind_direction, TI):
                self.update_calls.append(
                    {"ws": wind_speed, "wd": wind_direction, "ti": TI}
                )
                super().update_wind(wind_speed, wind_direction, TI)

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)
        spy_agent = SpyPyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

        def env_factory():
            return FarmEval(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=temp_yaml_file_integration,
                turbtype="None",
                reset_init=True,
                finite_episode=True,
                n_passthrough=0.1,
                burn_in_passthroughs=0.01,
            )

        coliseum = Coliseum(
            env_factory,
            agents={"SpyWake": spy_agent},
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        ws_grid = [9, 11]
        wd_grid = [275]
        ti_val = 0.07

        coliseum.run_wind_grid_evaluation(
            ws_min=min(ws_grid),
            ws_max=max(ws_grid),
            ws_step=2,
            wd_min=min(wd_grid),
            wd_max=max(wd_grid),
            wd_step=10,
            ti_min=ti_val,
            ti_max=ti_val,
            ti_points=1,
        )

        assert len(spy_agent.update_calls) == 2

        expected_calls = [
            {"ws": 9.0, "wd": 275.0, "ti": ti_val},
            {"ws": 11.0, "wd": 275.0, "ti": ti_val},
        ]
        assert spy_agent.update_calls == expected_calls

    def test_pywake_agent_wrapper_logic(self, temp_yaml_file, monkeypatch):
        """
        Tests the internal logic of WindFarmEnv's PyWakeAgentWrapper.
        Verifies correct scaling and clipping of yaw commands from PyWakeAgent.
        """
        # Create a YAML config with BaseController set to "PyWake"
        pywake_controller_yaml = """
        yaw_init: "Zeros"
        noise: "None"
        BaseController: "PyWake" # This is what we are testing
        ActionMethod: "wind" # PyWakeAgent returns 'wind' type actions
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

        # Write this config to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as tmp_file:
            tmp_file.write(pywake_controller_yaml)
            yaml_path = tmp_file.name

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

        # Create a FarmEval instance with the PyWake baseline controller
        env = FarmEval(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_path,
            turbtype="None",
            Baseline_comp=True,  # Essential for _base_controller to be set
            reset_init=True,
            finite_episode=True,
            dt_sim=1,
            dt_env=1,
            yaw_step_sim=5.0,  # Define a specific yaw_step_sim for clipping test
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        # Mock the internal pywake_agent.predict method to control its output
        # Case 1: Agent suggests a large positive action (should be clipped)
        mock_pywake_action = np.array([0.8, 0.2])  # Action in [-1, 1] range

        def mock_predict_agent(*args, **kwargs):
            return mock_pywake_action, None

        monkeypatch.setattr(env.pywake_agent, "predict", mock_predict_agent)

        # Mock a simplified fs (FlowSimulation) object needed by _base_controller
        # Specifically, it needs fs.windTurbines.yaw to apply clipping based on current yaw.
        mock_fs = MagicMock()
        mock_fs.windTurbines.yaw = np.array(
            [0.0, 0.0]
        )  # Initial yaw angles for baseline farm

        # Call the PyWakeAgentWrapper via _base_controller
        # This will simulate one step of the baseline controller
        calculated_yaws = env._base_controller(fs=mock_fs, yaw_step=env.yaw_step_sim)

        # Assertions
        # Action is 0.8, scaled to yaw_range. Default yaw_min=-30, yaw_max=30.
        # Scaled yaw goal = (0.8 + 1.0) / 2.0 * (30 - (-30)) + (-30)
        # = 1.8 / 2.0 * 60 - 30 = 0.9 * 60 - 30 = 54 - 30 = 24 degrees
        # Initial yaw is 0.0. yaw_step_sim is 5.0.
        # Since calculated_yaw_goal (24) > yaw_step_sim (5), it should be clipped to initial + yaw_step_sim.
        # Expected yaw = initial_yaw + yaw_step_sim = 0.0 + 5.0 = 5.0

        # For the second turbine: action is 0.2
        # Scaled yaw goal = (0.2 + 1.0) / 2.0 * 60 - 30 = 0.6 * 60 - 30 = 36 - 30 = 6 degrees
        # Expected yaw = initial_yaw + yaw_step_sim = 0.0 + 5.0 = 5.0 (since 6 > 5, it's clipped to 5)

        expected_yaws = np.array([5.0, 5.0])
        np.testing.assert_allclose(calculated_yaws, expected_yaws, atol=1e-5)

        # Test another scenario: action leads to a yaw within yaw_step_sim limits
        mock_fs.windTurbines.yaw = np.array([10.0, 10.0])
        mock_pywake_action = np.array(
            [-0.5, 0.0]
        )  # This action would lead to -15 degrees and 0 degrees if unclipped

        # New scaled yaw goals:
        # -0.5 -> (-0.5 + 1.0) / 2.0 * 60 - 30 = 0.25 * 60 - 30 = 15 - 30 = -15 degrees
        # 0.0 -> (0.0 + 1.0) / 2.0 * 60 - 30 = 0.5 * 60 - 30 = 30 - 30 = 0 degrees

        # Target: [-15, 0]
        # Current: [10, 10]
        # Yaw_step_sim: 5.0
        # For first turbine: target -15, current 10. Change needed: -25. Clipped change: -5. New yaw: 10 - 5 = 5.0
        # For second turbine: target 0, current 10. Change needed: -10. Clipped change: -5. New yaw: 10 - 5 = 5.0

        # NOTE: The _base_controller function calculates the `new_yaws` based on the target, then applies `yaw_step_sim` limits.
        # The line `np.clip(new_yaws, yaw_min, yaw_max)` where `yaw_min = fs.windTurbines.yaw - self.yaw_step_sim`
        # and `yaw_max = fs.windTurbines.yaw + self.yaw_step_sim` is the key.

        calculated_yaws_2 = env._base_controller(fs=mock_fs, yaw_step=env.yaw_step_sim)

        # Expected result based on logic:
        # Original new_yaws = [-15, 0]
        # yaw_min_clip = [10-5, 10-5] = [5, 5]
        # yaw_max_clip = [10+5, 10+5] = [15, 15]
        # new_yaws after first clip: np.clip([-15, 0], [5, 5], [15, 15]) = [5, 5]
        # This result is then clipped by global yaw_min=-30 and yaw_max=30, which doesn't change it.
        expected_yaws_2 = np.array([5.0, 5.0])
        np.testing.assert_allclose(calculated_yaws_2, expected_yaws_2, atol=1e-5)

        env.close()
        os.remove(yaml_path)
