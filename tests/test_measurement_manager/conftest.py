# ./tests/conftest.py

import pytest
import numpy as np
import yaml
import tempfile
import os

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from WindGym.utils.generate_layouts import generate_square_grid
from WindGym.core.measurement_manager import MeasurementSpec, MeasurementType


@pytest.fixture
def basic_specs():
    """Provides a basic list of MeasurementSpec for testing noise models directly."""
    return [
        MeasurementSpec(
            name="turb_0/ws_current",
            measurement_type=MeasurementType.WIND_SPEED,
            index_range=(0, 1),
            min_val=0.0,
            max_val=20.0,
            turbine_id=0,
        ),
        MeasurementSpec(
            name="turb_0/wd_current",
            measurement_type=MeasurementType.WIND_DIRECTION,
            index_range=(1, 2),
            min_val=260.0,
            max_val=280.0,
            turbine_id=0,
            noise_sensitivity=0.5,
        ),
    ]


@pytest.fixture(scope="module")
def smoke_test_env():
    """
    Creates a real, minimal WindFarmEnv instance for integration/smoke testing.
    This is scoped to the module to be created only once per test run.
    """
    # FIXED: Ensure scaling ranges in YAML match the test's expectations or vice-versa.
    # Here, we align the YAML config to match the ranges hardcoded in the tests.
    yaml_config = {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
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
        "power_def": {"Power_reward": "None", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": True,
            "farm_wd": False,  # Keep farm_wd False as in original smoke_test_env for consistency
            "farm_TI": True,
            "farm_power": False,
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
            "power_current": False,
            "power_rolling_mean": False,
            "power_history_N": 0,
            "power_history_length": 1,
            "power_window_length": 1,
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_file:
        yaml.dump(yaml_config, tmp_file)
        yaml_filepath = tmp_file.name

    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_filepath,
        turbtype="None",
        reset_init=True,
        seed=42,
        # Explicitly pass the scaling ranges that tests expect for consistency
        ws_scaling_min=2.0,  # Aligned with test expectations
        ws_scaling_max=25.0,  # Aligned with test expectations
        wd_scaling_min=265.0,  # Aligned with test expectations
        wd_scaling_max=275.0,  # Aligned with test expectations
    )

    yield env

    env.close()
    os.remove(yaml_filepath)


@pytest.fixture
def env_factory(tmp_path):
    """
    A factory fixture to create a real WindFarmEnv from a given YAML dictionary.
    This allows tests to easily create envs with specific configurations.
    """
    created_envs = []

    def _create_env(
        config_dict: dict, ws_min_s=2.0, ws_max_s=25.0, wd_min_s=265.0, wd_max_s=275.0
    ):  # Add scaling args
        yaml_path = tmp_path / f"config_{hash(str(config_dict))}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_path,
            turbtype="None",
            reset_init=True,
            seed=42,
            # Pass scaling explicitly
            ws_scaling_min=ws_min_s,
            ws_scaling_max=ws_max_s,
            wd_scaling_min=wd_min_s,
            wd_scaling_max=wd_max_s,
        )
        created_envs.append(env)
        return env

    yield _create_env

    # Cleanup all created environments
    for env in created_envs:
        env.close()
