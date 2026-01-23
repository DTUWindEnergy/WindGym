import pytest
import yaml
import numpy as np
from unittest.mock import MagicMock
import copy

from WindGym import WindFarmEnv
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid


# A minimal, but COMPLETE, YAML string for testing.
YAML_CONTENT_STRING = """
yaw_init: "Zeros"
noise: "None"
BaseController: "PyWake"
ActionMethod: "wind"
farm: {yaw_min: -25, yaw_max: 25}
wind: {ws_min: 9, ws_max: 9, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
act_pen: {action_penalty: 0.01, action_penalty_type: "Total"}
power_def: {Power_reward: "Baseline", Power_avg: 1, Power_scaling: 1.0}
mes_level: {turb_ws: True, turb_wd: True, turb_TI: False, turb_power: True, farm_ws: True, farm_wd: True, farm_TI: False, farm_power: True}
ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 0, ws_history_length: 1, ws_window_length: 1}
wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 0, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 0, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 0, power_history_length: 1, power_window_length: 1}
"""


class TestConfigLoading:
    """Tests for the new `config` configuration loading feature."""

    def test_load_config_from_string(self):
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=YAML_CONTENT_STRING,
            reset_init=False,
        )
        assert env.yaw_min == -25
        assert env.BaseController == "PyWake"
        env.close()

    # def test_yaml_content_precedence(self, temp_yaml_file_factory):
    #     yaml_filepath = temp_yaml_file_factory(
    #         YAML_CONTENT_STRING, "precedence_test_file"
    #     )
    #     config_dict = yaml.safe_load(YAML_CONTENT_STRING)
    #     config_dict["farm"]["yaw_min"] = -15
    #     yaml_content_modified = yaml.dump(config_dict)

    #     x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
    #     env = WindFarmEnv(
    #         turbine=V80(),
    #         x_pos=x_pos,
    #         y_pos=y_pos,
    #         yaml_path=yaml_filepath,
    #         config=yaml_content_modified,
    #         reset_init=False,
    #     )
    #     assert env.yaw_min == -15
    #     env.close()

    def test_error_on_no_config_provided(self):
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
        with pytest.raises(
            ValueError,
            match="A configuration must be provided via the `config` argument.",
        ):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=None,
            )
