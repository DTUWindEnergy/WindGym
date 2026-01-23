import pytest
import yaml
from pathlib import Path
import numpy as np

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid


# --- Helper to get a basic, mostly complete YAML string ---
def get_base_yaml_dict():
    return {
        "yaw_init": "Zeros",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
            # "xDist": 5,
            # "yDist": 3,
            # "nx": 2,
            # "ny": 1,
        },
        "wind": {
            "ws_min": 8,
            "ws_max": 10,
            "TI_min": 0.05,
            "TI_max": 0.10,
            "wd_min": 268,
            "wd_max": 272,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 5, "Power_scaling": 1.0},
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
            "ws_rolling_mean": True,
            "ws_history_N": 1,
            "ws_history_length": 5,
            "ws_window_length": 2,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": 1,
            "wd_history_length": 5,
            "wd_window_length": 2,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": 1,
            "yaw_history_length": 5,
            "yaw_window_length": 2,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": 1,
            "power_history_length": 5,
            "power_window_length": 2,
        },
    }


# --- Tests for Basic Settings ---
@pytest.mark.parametrize(
    "setting, value",
    [
        ("yaw_init", "Random"),
        ("yaw_init", "Zeros"),
        ("BaseController", "Global"),
        ("BaseController", "Local"),
        ("ActionMethod", "wind"),
        ("ActionMethod", "yaw"),
    ],
)
def test_initial_settings_loading(temp_yaml_file_factory, setting, value):
    config_dict = get_base_yaml_dict()
    config_dict[setting] = value
    yaml_content = yaml.dump(config_dict)
    yaml_filepath = temp_yaml_file_factory(yaml_content, f"initial_{setting}_{value}")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        reset_init=False,
    )  # reset_init=False to speed up, only check config loading

    assert getattr(env, setting) == value
    env.close()  # Ensure resources are released if any were partially acquired


# --- Tests for Farm Parameter Loading ---
def test_farm_params_loading(temp_yaml_file_factory):
    config_dict = get_base_yaml_dict()
    config_dict["farm"] = {
        "yaw_min": -25,
        "yaw_max": 25,
        "xDist": 4,
        "yDist": 4,
        "nx": 3,
        "ny": 2,
    }
    yaml_content = yaml.dump(config_dict)
    yaml_filepath = temp_yaml_file_factory(yaml_content, "farm_params")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=3, ny=2, xDist=4, yDist=4)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        reset_init=False,
    )

    assert env.yaw_min == -25
    assert env.yaw_max == 25
    assert env.n_turb == 6  # 3 * 2
    env.close()


# --- Tests for Wind Parameter Loading ---
def test_wind_params_loading(temp_yaml_file_factory):
    config_dict = get_base_yaml_dict()
    config_dict["wind"] = {
        "ws_min": 7,
        "ws_max": 12,
        "TI_min": 0.03,
        "TI_max": 0.12,
        "wd_min": 200,
        "wd_max": 300,
    }
    yaml_content = yaml.dump(config_dict)
    yaml_filepath = temp_yaml_file_factory(yaml_content, "wind_params")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        reset_init=False,
    )

    assert env.ws_inflow_min == 7
    assert env.ws_inflow_max == 12
    assert env.TI_inflow_min == 0.03
    assert env.TI_inflow_max == 0.12
    assert env.wd_inflow_min == 200
    assert env.wd_inflow_max == 300
    env.close()


# --- Tests for Measurement Level Configuration ---
@pytest.mark.parametrize(
    "mes_config_key, mes_config_value, expected_attr_dict",
    [
        ("turb_wd", False, {"farm_measurements.turb_wd": False}),
        ("farm_TI", True, {"farm_measurements.farm_TI": True}),
        # Add more specific mes_level checks
    ],
)
def test_measurement_level_loading(
    temp_yaml_file_factory, mes_config_key, mes_config_value, expected_attr_dict
):
    config_dict = get_base_yaml_dict()
    config_dict["mes_level"][mes_config_key] = mes_config_value
    yaml_content = yaml.dump(config_dict)
    yaml_filepath = temp_yaml_file_factory(
        yaml_content, f"mes_level_{mes_config_key}_{mes_config_value}"
    )
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        reset_init=False,
    )

    for attr_path, expected_val in expected_attr_dict.items():
        # Helper to navigate nested attributes if needed, e.g., "farm_measurements.turb_wd"
        obj = env
        parts = attr_path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        assert (
            getattr(obj, parts[-1]) == expected_val
        ), f"Attribute {attr_path} mismatch"
    env.close()


# --- Tests for Specific Measurement Settings (e.g., ws_mes) ---
def test_ws_mes_settings_loading(temp_yaml_file_factory):
    config_dict = get_base_yaml_dict()
    config_dict["ws_mes"] = {
        "ws_current": False,
        "ws_rolling_mean": True,
        "ws_history_N": 2,
        "ws_history_length": 10,
        "ws_window_length": 3,
    }
    yaml_content = yaml.dump(config_dict)
    yaml_filepath = temp_yaml_file_factory(yaml_content, "ws_mes_custom")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        reset_init=False,
    )  # Calls _init_farm_mes

    # Check attributes of the Mes object for ws in farm_measurements.farm_mes (and potentially turb_mes if farm_ws is True)
    # This assumes farm_ws is true in base_yaml_dict
    assert env.farm_measurements.farm_mes.ws.current is False
    assert env.farm_measurements.farm_mes.ws.rolling_mean is True
    assert env.farm_measurements.farm_mes.ws.history_N == 2
    assert env.farm_measurements.farm_mes.ws.history_length == 10
    assert env.farm_measurements.farm_mes.ws.window_length == 3

    # Also check one of the turbine's ws_mes if turb_ws is True
    if env.farm_measurements.turb_ws:
        assert env.farm_measurements.turb_mes[0].ws.current is False
        assert env.farm_measurements.turb_mes[0].ws.rolling_mean is True
    env.close()
