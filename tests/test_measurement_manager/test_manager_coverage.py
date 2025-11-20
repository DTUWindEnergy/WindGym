# ./tests/test_manager_coverage.py

import pytest
import numpy as np
import yaml
import itertools
from unittest.mock import MagicMock
import gymnasium as gym

from WindGym.core.measurement_manager import (
    MeasurementManager,
    MeasurementType,
    MeasurementSpec,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    NoisyWindFarmEnv,
    NoiseModel,
    HybridNoiseModel,
)


# Helper function to generate a valid YAML config
def get_base_config():
    """Returns a dictionary for a minimal, valid YAML configuration."""
    return {
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


class TestCoverageForMeasurementManager:
    """A suite of tests designed to achieve 100% coverage."""

    def test_noise_model_optional_hook(self):
        """
        Covers: `NoiseModel.reset_noise`'s default `pass` statement.
        This is covered by calling the method on a subclass (WhiteNoiseModel)
        that does not override it, thus executing the parent's method.
        """
        noise_model = WhiteNoiseModel({})
        noise_model.reset_noise(specs=[], rng=np.random.default_rng())
        assert True  # Test passes if no error was raised

    def test_hybrid_model_get_info(self):
        """
        Covers: `HybridNoiseModel.get_info` and its list comprehension.
        """
        hybrid_model = HybridNoiseModel(
            [
                WhiteNoiseModel({MeasurementType.WIND_SPEED: 0.1}),
                EpisodicBiasNoiseModel({}),
            ]
        )

        info = hybrid_model.get_info()

        assert info["noise_type"] == "hybrid"
        assert len(info["component_models"]) == 2
        component_types = {comp["noise_type"] for comp in info["component_models"]}
        assert "white" in component_types and "episodic_bias" in component_types

    def test_manager_with_no_noise_model_set(self, env_factory):
        """Covers: `if self.noise_model is None: ...`"""
        env = env_factory(get_base_config())
        manager = MeasurementManager(env)
        clean_obs = env.observation_space.sample()
        noisy_obs, info = manager.apply_noise(clean_obs)
        assert np.allclose(clean_obs, noisy_obs)
        # Only check for the presence and type of 'noise_info'
        assert "noise_info" in info
        assert info["noise_info"]["type"] == "none"

    def test_noisy_wrapper_close_method(self):
        """Covers: `NoisyWindFarmEnv.close()`"""

        class MockCloseableEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                self.close = MagicMock()
                # Mock the attributes needed by MeasurementManager to build specs
                self.farm_measurements = MagicMock()
                self.farm_measurements.n_turbines = 1
                self.farm_measurements.turb_mes = [MagicMock()]
                self.farm_measurements.turb_mes[0].ws = MagicMock(
                    current=True, rolling_mean=False, history_N=0
                )
                self.farm_measurements.turb_mes[0].wd = MagicMock(
                    current=True, rolling_mean=False, history_N=0
                )
                self.farm_measurements.turb_mes[0].yaw = MagicMock(
                    current=True, rolling_mean=False, history_N=0
                )
                self.farm_measurements.turb_mes[0].power = MagicMock(
                    current=True, rolling_mean=False, history_N=0
                )
                self.farm_measurements.turb_mes[0].include_TI = True
                self.farm_measurements.farm_mes = MagicMock(
                    observed_variables=lambda: 0
                )  # No farm level for this test
                self.farm_measurements.turb_ws = True
                self.farm_measurements.turb_wd = True
                self.farm_measurements.turb_TI = True
                self.farm_measurements.turb_power = True
                self.farm_measurements.farm_ws = False
                self.farm_measurements.farm_wd = False
                self.farm_measurements.farm_TI = False
                self.farm_measurements.farm_power = False

                # Set min/max values for scaling mocks
                self.farm_measurements.turb_mes[0].ws_min = 0.0
                self.farm_measurements.turb_mes[0].ws_max = 20.0
                self.farm_measurements.turb_mes[0].wd_min = 0.0
                self.farm_measurements.turb_mes[0].wd_max = 360.0
                self.farm_measurements.turb_mes[0].yaw_min = -45.0
                self.farm_measurements.turb_mes[0].yaw_max = 45.0
                self.farm_measurements.turb_mes[0].TI_min = 0.0
                self.farm_measurements.turb_mes[0].TI_max = 1.0
                self.farm_measurements.turb_mes[0].power_max = 2000000.0

            def step(self, action):
                pass

            def reset(self, *, seed=None, options=None):
                pass

            @property
            def unwrapped(self):
                return self

        mock_base_env = MockCloseableEnv()
        # Mock a minimal manager that doesn't need a real env during instantiation
        dummy_manager = MeasurementManager(
            mock_base_env
        )  # Pass the mock env to the manager
        dummy_manager.reset_noise = (
            MagicMock()
        )  # Mock reset_noise as it might try to use env attributes
        dummy_manager.apply_noise = MagicMock(
            return_value=(np.zeros(1), {})
        )  # Mock apply_noise
        wrapped_env = NoisyWindFarmEnv(lambda **k: mock_base_env, dummy_manager)
        wrapped_env.close()
        mock_base_env.close.assert_called_once()

    @pytest.mark.parametrize(
        "turb_ws, turb_wd, turb_ti, turb_power, farm_ws, farm_wd, farm_ti, farm_power, use_rolling_mean",
        [
            (False, False, False, False, False, False, False, False, False),
            (True, True, True, True, True, True, True, True, False),
            (True, True, True, True, True, True, True, True, True),
        ],
    )
    def test_all_build_from_env_flags(
        self,
        env_factory,
        turb_ws,
        turb_wd,
        turb_ti,
        turb_power,
        farm_ws,
        farm_wd,
        farm_ti,
        farm_power,
        use_rolling_mean,
    ):
        """
        Covers all `if fm.feature:` and `if mes_obj.rolling_mean:` branches.
        """
        config = get_base_config()
        config["mes_level"] = {
            "turb_ws": turb_ws,
            "turb_wd": turb_wd,
            "turb_TI": turb_ti,
            "turb_power": turb_power,
            "farm_ws": farm_ws,
            "farm_wd": farm_wd,
            "farm_TI": farm_ti,
            "farm_power": farm_power,
        }

        # Set history_N based on use_rolling_mean for all types if rolling mean is used
        n_hist = 2 if use_rolling_mean else 0
        for key in ["ws_mes", "wd_mes", "yaw_mes", "power_mes"]:
            prefix = key.split("_")[0]
            config[key] = {
                f"{prefix}_current": True,  # Ensure current is always on for simplicity of counting
                f"{prefix}_rolling_mean": use_rolling_mean,
                f"{prefix}_history_N": n_hist,
                f"{prefix}_history_length": 10,
                f"{prefix}_window_length": 2,
            }

        env = env_factory(config)
        manager = MeasurementManager(env)
        specs = manager.specs

        expected_count = 0
        n_turbines = 2

        # Dynamically calculate expected count based on config settings
        # Each 'current' adds 1, each 'history' adds 'history_N' if rolling_mean is true
        if turb_ws:
            expected_count += n_turbines * (
                config["ws_mes"]["ws_current"]
                + config["ws_mes"]["ws_rolling_mean"] * config["ws_mes"]["ws_history_N"]
            )
        if turb_wd:
            expected_count += n_turbines * (
                config["wd_mes"]["wd_current"]
                + config["wd_mes"]["wd_rolling_mean"] * config["wd_mes"]["wd_history_N"]
            )
        if turb_ti:  # TI is a single value, not history based
            expected_count += n_turbines
        if turb_power:
            expected_count += n_turbines * (
                config["power_mes"]["power_current"]
                + config["power_mes"]["power_rolling_mean"]
                * config["power_mes"]["power_history_N"]
            )

        # Yaw is implicitly always available for each turbine in MesClass.py,
        # its values are controlled by yaw_mes flags.
        # It's not gated by mes_level["turb_yaw"] or similar.
        expected_count += n_turbines * (
            config["yaw_mes"]["yaw_current"]
            + config["yaw_mes"]["yaw_rolling_mean"] * config["yaw_mes"]["yaw_history_N"]
        )

        if farm_ws:
            expected_count += (
                config["ws_mes"]["ws_current"]
                + config["ws_mes"]["ws_rolling_mean"] * config["ws_mes"]["ws_history_N"]
            )
        if farm_wd:
            expected_count += (
                config["wd_mes"]["wd_current"]
                + config["wd_mes"]["wd_rolling_mean"] * config["wd_mes"]["wd_history_N"]
            )
        if farm_ti:  # TI is a single value, not history based
            expected_count += 1
        if farm_power:
            expected_count += (
                config["power_mes"]["power_current"]
                + config["power_mes"]["power_rolling_mean"]
                * config["power_mes"]["power_history_N"]
            )

        assert len(specs) == expected_count
