# tests/test_baseline_controllers.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid
import yaml


@pytest.fixture
def pywake_local_env(temp_yaml_file_factory):
    """Creates an environment configured to use the PyWake_local baseline controller."""
    config_dict = {
        "yaw_init": "Zeros",
        "noise": "None",
        "ActionMethod": "yaw",
        "BaseController": "PyWake_local",
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10,
            "ws_max": 10,
            "TI_min": 0.07,
            "TI_max": 0.07,
            "wd_min": 270,
            "wd_max": 270,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": True,
            "turb_power": True,
            "farm_ws": True,
            "farm_wd": True,
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
    yaml_path = temp_yaml_file_factory(config_dict, "pywake_local_config")

    x_pos, y_pos = generate_square_grid(V80(), nx=2, ny=1, xDist=7, yDist=7)
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        Baseline_comp=True,
        reset_init=True,
        turbtype="None",
        seed=123,
    )
    yield env
    env.close()


def test_pywake_local_baseline_controller(pywake_local_env):
    """
    Tests the `if self.py_agent_mode == "local":` code path for the PyWake_local baseline controller.
    """
    env = pywake_local_env

    mock_local_windspeed = np.array(
        [
            [10.0, 9.0],  # u component (x-dir)
            [1.0, 0.0],  # v component (y-dir)
            [0.0, 0.0],  # w component (z-dir)
        ]
    )

    env.fs_baseline.windTurbines.get_rotor_avg_windspeed = MagicMock(
        return_value=mock_local_windspeed
    )
    env.pywake_agent.update_wind = MagicMock(wraps=env.pywake_agent.update_wind)

    env.step(env.action_space.sample())

    env.pywake_agent.update_wind.assert_called()

    # Use floating point comparison with tolerance instead of int() conversion
    # which would lose precision (e.g., 270.7 would incorrectly pass as 270)
    WIND_DIR_TOLERANCE = 1.0  # degrees
    WIND_SPEED_TOLERANCE = 0.5  # m/s

    actual_wdir = float(env.pywake_agent.wdir[0])
    actual_wsp = float(env.pywake_agent.wsp[0])
    expected_wdir = 270.0
    expected_wsp = 10.0

    assert abs(actual_wdir - expected_wdir) < WIND_DIR_TOLERANCE, (
        f"Wind direction {actual_wdir} differs from expected {expected_wdir} "
        f"by more than {WIND_DIR_TOLERANCE} degrees"
    )
    assert abs(actual_wsp - expected_wsp) < WIND_SPEED_TOLERANCE, (
        f"Wind speed {actual_wsp} differs from expected {expected_wsp} "
        f"by more than {WIND_SPEED_TOLERANCE} m/s"
    )
