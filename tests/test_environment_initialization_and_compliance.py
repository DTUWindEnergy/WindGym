# tests/test_specific_features.py
from unittest.mock import patch

import pytest
import yaml
import numpy as np
import os
import tempfile
from pathlib import Path
import gymnasium as gym

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from gymnasium.utils.env_checker import check_env
from WindGym.utils.generate_layouts import (
    generate_square_grid,
)  # Import the layout generator
from tkinter import TclError


# Helper to get a base, mostly complete YAML dictionary for tests
def get_base_yaml_dict(nx=2, ny=1, history_length=5, window_length=2, history_n=1):
    """
    Provides a base dictionary for YAML configuration.
    Default nx=2 to prevent time_max=0 issues with single turbine farms.
    """
    return {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
            "xDist": 5,
            "yDist": 3,
            "nx": nx,
            "ny": ny,
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
        "power_def": {
            "Power_reward": "Baseline",
            "Power_avg": 1,
            "Power_scaling": 1.0,
        },
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
            "ws_history_N": history_n,
            "ws_history_length": history_length,
            "ws_window_length": window_length,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": history_n,
            "wd_history_length": history_length,
            "wd_window_length": window_length,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": history_n,
            "yaw_history_length": history_length,
            "yaw_window_length": window_length,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": history_n,
            "power_history_length": history_length,
            "power_window_length": window_length,
        },
    }


@pytest.fixture
def temp_yaml_filepath_factory():
    """Factory for creating temporary YAML files for tests."""
    created_files = []

    def _create_temp_yaml(config_dict, name_suffix=""):
        content_str = yaml.dump(config_dict)
        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f"_{name_suffix}.yaml", encoding="utf-8"
        )
        tf.write(content_str)
        filepath = tf.name
        tf.close()
        created_files.append(filepath)
        return filepath

    yield _create_temp_yaml
    for f_path in created_files:
        if os.path.exists(f_path):
            os.remove(f_path)


@pytest.fixture
def mock_mann_methods(monkeypatch):
    """Mocks turbulence generation/loading to make tests faster."""
    from dynamiks.sites.turbulence_fields import MannTurbulenceField

    def mock_generate(*args, **kwargs):
        field_data = np.zeros((1, 1, 1, 3))
        coords = (np.array([0.0]), np.array([0.0]), np.array([90.0]))
        return MannTurbulenceField(field_data, coords)

    def mock_from_netcdf(filename):
        field_data = np.zeros((1, 1, 1, 3))
        coords = (np.array([0.0]), np.array([0.0]), np.array([90.0]))
        mocked_tf = MannTurbulenceField(field_data, coords)
        setattr(mocked_tf, "mocked_filename", filename)
        return mocked_tf

    monkeypatch.setattr(MannTurbulenceField, "generate", mock_generate)
    monkeypatch.setattr(MannTurbulenceField, "from_netcdf", mock_from_netcdf)


def calculate_expected_obs_dim(config_dict, n_turbines):
    """Calculates the expected observation dimension based on the YAML configuration."""
    mes_level = config_dict.get("mes_level", {})
    ws_mes_cfg = config_dict.get("ws_mes", {})
    wd_mes_cfg = config_dict.get("wd_mes", {})
    yaw_mes_cfg = config_dict.get("yaw_mes", {})
    power_mes_cfg = config_dict.get("power_mes", {})

    one_turb_mes_obs_actual = 0
    if mes_level.get("turb_ws", False):
        one_turb_mes_obs_actual += ws_mes_cfg.get("ws_current", False) + (
            ws_mes_cfg.get("ws_rolling_mean", False) * ws_mes_cfg.get("ws_history_N", 0)
        )
    if mes_level.get("turb_wd", False):
        one_turb_mes_obs_actual += wd_mes_cfg.get("wd_current", False) + (
            wd_mes_cfg.get("wd_rolling_mean", False) * wd_mes_cfg.get("wd_history_N", 0)
        )

    one_turb_mes_obs_actual += yaw_mes_cfg.get("yaw_current", False) + (
        yaw_mes_cfg.get("yaw_rolling_mean", False) * yaw_mes_cfg.get("yaw_history_N", 0)
    )

    if mes_level.get("turb_TI", False):
        one_turb_mes_obs_actual += 1

    if mes_level.get("turb_power", False):
        one_turb_mes_obs_actual += power_mes_cfg.get("power_current", False) + (
            power_mes_cfg.get("power_rolling_mean", False)
            * power_mes_cfg.get("power_history_N", 0)
        )

    total_turbine_level_obs = one_turb_mes_obs_actual * n_turbines

    farm_level_obs = 0
    if mes_level.get("farm_ws", False):
        farm_level_obs += ws_mes_cfg.get("ws_current", False) + (
            ws_mes_cfg.get("ws_rolling_mean", False) * ws_mes_cfg.get("ws_history_N", 0)
        )
    if mes_level.get("farm_wd", False):
        farm_level_obs += wd_mes_cfg.get("wd_current", False) + (
            wd_mes_cfg.get("wd_rolling_mean", False) * wd_mes_cfg.get("wd_history_N", 0)
        )
    if mes_level.get("farm_TI", False):
        farm_level_obs += 1
    if mes_level.get("farm_power", False):
        farm_level_obs += power_mes_cfg.get("power_current", False) + (
            power_mes_cfg.get("power_rolling_mean", False)
            * power_mes_cfg.get("power_history_N", 0)
        )

    return total_turbine_level_obs + farm_level_obs


class TestSpecificFeatures:
    @pytest.mark.parametrize(
        "fill_window_config_val, expected_steps_on_reset_factor_str",
        [(True, "hist_max"), (3, "val_3"), (False, "val_1"), (10, "hist_max_capped")],
    )
    def test_fill_window_logic(
        self,
        temp_yaml_filepath_factory,
        fill_window_config_val,
        expected_steps_on_reset_factor_str,
        mock_mann_methods,
    ):
        base_hist_len = 5
        config_dict = get_base_yaml_dict(history_length=base_hist_len, history_n=1)
        # Use a deterministic reward (0.0) to pass check_env
        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict, f"fill_window_{str(fill_window_config_val).lower()}"
        )

        farm_params = config_dict["farm"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=farm_params["nx"],
            ny=farm_params["ny"],
            xDist=farm_params["xDist"],
            yDist=farm_params["yDist"],
        )

        hist_max_calculated = base_hist_len

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            fill_window=fill_window_config_val,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_steps = 0
        if expected_steps_on_reset_factor_str == "hist_max":
            expected_steps = hist_max_calculated
        elif expected_steps_on_reset_factor_str == "val_3":
            expected_steps = 3
        elif expected_steps_on_reset_factor_str == "val_1":
            expected_steps = 1
        elif expected_steps_on_reset_factor_str == "hist_max_capped":
            expected_steps = min(fill_window_config_val, hist_max_calculated)

        assert (
            env.steps_on_reset == expected_steps
        ), f"Calculated hist_max: {hist_max_calculated}"
        check_env(env, skip_render_check=True)
        env.close()

    @pytest.mark.parametrize(
        "render_mode_val",
        [
            None,
            "human",
        ],
    )
    def test_render_mode_behavior(
        self, temp_yaml_filepath_factory, render_mode_val, mock_mann_methods
    ):
        """
        Tests supported render modes. 'rgb_array' is removed as requested.
        """
        config_dict = get_base_yaml_dict()
        yaml_filepath = temp_yaml_filepath_factory(
            config_dict, f"render_{render_mode_val}"
        )

        farm_params = config_dict["farm"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=farm_params["nx"],
            ny=farm_params["ny"],
            xDist=farm_params["xDist"],
            yDist=farm_params["yDist"],
        )

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            render_mode=render_mode_val,
            reset_init=True,
            seed=42,
            turbtype="None",
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        assert env.render_mode == render_mode_val

        try:
            # Test render call after reset
            env.init_render()
            frame = env.render()
            if render_mode_val is None:
                assert frame is None, "None render mode should return None"

            # Test render call after a step
            env.step(env.action_space.sample())
            frame_after_step = env.render()
            if render_mode_val is None:
                assert frame_after_step is None

        except Exception as e:
            if "TclError" in str(e) and render_mode_val == "human":  # pragma: no cover
                pytest.skip(
                    "Skipping human render mode test in headless environment due to TclError."
                )
            else:  # pragma: no cover
                raise e
        finally:
            env.close()

    def test_render_with_probes(self, temp_yaml_filepath_factory, mock_mann_methods):
        """
        Tests that the render method correctly visualizes probes if they are defined
        in the configuration, covering the `if hasattr(self, "probes")` block.
        """
        config_dict = get_base_yaml_dict()
        config_dict["probes"] = [
            {
                "name": "test_probe",
                "position": [100, 50, 90],
                "probe_type": "WS",
                "turbine_position": [0, 0, 90],
            },
            {
                "name": "test_ti_probe",
                "relative_position": [-100, 50, 0],
                "turbine_index": 0,
                "probe_type": "TI",
            },
            {
                "name": "test_unknown_probe",
                "position": [200, 50, 90],
                "probe_type": "UNKNOWN",
                "turbine_position": [0, 0, 90],
            },
        ]
        yaml_filepath = temp_yaml_filepath_factory(config_dict, "render_with_probes")

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
        env = None
        try:
            env = WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                render_mode="rgb_array",
                reset_init=True,
                turbtype="None",
            )

            with (
                patch("matplotlib.pyplot.show"),
                patch("matplotlib.pyplot.imshow"),
                patch("matplotlib.pyplot.pause"),
            ):
                frame = env.render()  # Should run without error and return a frame
                assert isinstance(frame, np.ndarray)
                assert frame.shape[2] == 3  # Check for RGB channels

        except TclError:
            pytest.skip("Skipping human render mode test in headless environment.")
        finally:
            if env:
                env.close()

    @pytest.mark.parametrize(
        "turb_wd_enabled, farm_wd_enabled, wd_current, wd_rolling, wd_hist_n",
        [
            (True, True, True, True, 2),
            (False, False, True, False, 0),
            (True, False, True, False, 0),
            (False, True, False, True, 1),
            (True, True, False, False, 0),
        ],
    )
    def test_wind_direction_features_in_observations(
        self,
        temp_yaml_filepath_factory,
        mock_mann_methods,
        turb_wd_enabled,
        farm_wd_enabled,
        wd_current,
        wd_rolling,
        wd_hist_n,
    ):
        config_dict = get_base_yaml_dict(
            nx=2, ny=1, history_length=5, window_length=2, history_n=max(1, wd_hist_n)
        )

        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        config_dict["mes_level"]["turb_wd"] = turb_wd_enabled
        config_dict["mes_level"]["farm_wd"] = farm_wd_enabled

        config_dict["wd_mes"] = {
            "wd_current": wd_current,
            "wd_rolling_mean": wd_rolling,
            "wd_history_N": wd_hist_n,
            "wd_history_length": 5,
            "wd_window_length": 2,
        }

        # Isolate by turning off other sensor types
        config_dict["mes_level"]["turb_ws"] = False
        config_dict["mes_level"]["turb_TI"] = False
        config_dict["mes_level"]["turb_power"] = False
        config_dict["mes_level"]["farm_ws"] = False
        config_dict["mes_level"]["farm_TI"] = False
        config_dict["mes_level"]["farm_power"] = False

        config_dict["ws_mes"]["ws_current"] = False
        config_dict["ws_mes"]["ws_rolling_mean"] = False
        config_dict["power_mes"]["power_current"] = False
        config_dict["power_mes"]["power_rolling_mean"] = False
        config_dict["yaw_mes"] = {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 0,
            "yaw_history_length": 5,
            "yaw_window_length": 2,
        }

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict,
            f"obs_wd_{turb_wd_enabled}_{farm_wd_enabled}_{wd_current}_{wd_rolling}_{wd_hist_n}",
        )

        farm_params = config_dict["farm"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=farm_params["nx"],
            ny=farm_params["ny"],
            xDist=farm_params["xDist"],
            yDist=farm_params["yDist"],
        )

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_dim = calculate_expected_obs_dim(config_dict, env.n_turb)
        actual_dim = env.observation_space.shape[0]

        assert (
            actual_dim == expected_dim
        ), f"Observation space dimension mismatch. Expected {expected_dim}, got {actual_dim}."

        check_env(env, skip_render_check=True)
        env.close()

    @pytest.mark.parametrize(
        "mes_type_key, current_flag, rolling_flag, history_n_val",
        [
            ("ws_mes", True, False, 0),
            ("ws_mes", False, True, 3),
            ("ws_mes", True, True, 1),
            ("power_mes", True, False, 0),
            ("power_mes", False, True, 2),
        ],
    )
    def test_specific_measurement_configurations(
        self,
        temp_yaml_filepath_factory,
        mock_mann_methods,
        mes_type_key,
        current_flag,
        rolling_flag,
        history_n_val,
    ):
        n_turb_nx = 2
        config_dict = get_base_yaml_dict(
            nx=n_turb_nx,
            ny=1,
            history_length=10,
            window_length=3,
            history_n=max(1, history_n_val),
        )

        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        config_dict["mes_level"] = {
            "turb_ws": False,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        }

        prefix_under_test = mes_type_key.split("_")[0]
        config_dict["mes_level"][f"turb_{prefix_under_test}"] = True
        config_dict["mes_level"][f"farm_{prefix_under_test}"] = True

        config_dict[mes_type_key] = {
            f"{prefix_under_test}_current": current_flag,
            f"{prefix_under_test}_rolling_mean": rolling_flag,
            f"{prefix_under_test}_history_N": history_n_val,
            f"{prefix_under_test}_history_length": 10,
            f"{prefix_under_test}_window_length": 3,
        }

        for other_mes_key_loop in ["ws_mes", "wd_mes", "power_mes"]:
            if other_mes_key_loop != mes_type_key:
                prefix = other_mes_key_loop.split("_")[0]
                config_dict[other_mes_key_loop][f"{prefix}_current"] = False
                config_dict[other_mes_key_loop][f"{prefix}_rolling_mean"] = False
                config_dict[other_mes_key_loop][f"{prefix}_history_N"] = 0

        config_dict["yaw_mes"] = {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 0,
            "yaw_history_length": 10,
            "yaw_window_length": 3,
        }

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict,
            f"obs_cfg_{mes_type_key}_{current_flag}_{rolling_flag}_{history_n_val}",
        )

        farm_params = config_dict["farm"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=farm_params["nx"],
            ny=farm_params["ny"],
            xDist=farm_params["xDist"],
            yDist=farm_params["yDist"],
        )

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_dim = calculate_expected_obs_dim(config_dict, n_turb_nx)
        actual_dim = env.observation_space.shape[0]

        assert (
            actual_dim == expected_dim
        ), f"Obs dim mismatch for {mes_type_key}. Expected {expected_dim}, got {actual_dim}."

        check_env(env, skip_render_check=True)
        env.close()

    def test_track_power_not_implemented(
        self, temp_yaml_filepath_factory, mock_mann_methods
    ):
        """Test that Track_power=True raises NotImplementedError during initialization."""
        config_dict = get_base_yaml_dict()
        config_dict["Track_power"] = True
        yaml_filepath = temp_yaml_filepath_factory(config_dict, "track_power_true")

        # Generate turbine layout based on the config
        farm_params = config_dict["farm"]
        x_pos, y_pos = generate_square_grid(
            turbine=V80(),
            nx=farm_params["nx"],
            ny=farm_params["ny"],
            xDist=farm_params["xDist"],
            yDist=farm_params["yDist"],
        )

        with pytest.raises(
            NotImplementedError, match="Power tracking reward is not yet implemented."
        ):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                reset_init=True,
                seed=42,
                turbtype="None",
                n_passthrough=0.1,
                burn_in_passthroughs=0.01,
            )
