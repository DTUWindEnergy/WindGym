# In tests/test_measurement_manager/test_confounding_noise.py

import pytest
import numpy as np
from unittest.mock import MagicMock

# FIX 3: Add missing import for NoisyPyWakeAgent
from WindGym.Agents import NoisyPyWakeAgent
from WindGym.core.measurement_manager import (
    EpisodicBiasNoiseModel,
    WhiteNoiseModel,
    NoisyWindFarmEnv,
    MeasurementManager,
    MeasurementSpec,
    MeasurementType,
    NoiseModel,
)


def test_episodic_bias_confounds_yaw_with_wd():
    """
    Unit test to confirm that yaw measurement bias is correctly
    confounded by the wind direction measurement bias in the new _resample_bias method.
    """
    # 1. ARRANGE
    WD_BIAS_PHYSICAL = 10.0
    YAW_BIAS_PHYSICAL = -2.0

    mock_rng = MagicMock(spec=np.random.Generator)
    mock_rng.uniform.side_effect = [WD_BIAS_PHYSICAL, YAW_BIAS_PHYSICAL]

    # FIX: Set up static scaling methods, as they are not available in a unit test context.
    # This is required because apply_noise will call _handle_circular_noise for the 'wd' spec.
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    noise_model = EpisodicBiasNoiseModel(
        {
            # The ranges don't matter due to the mock, but the keys must be present.
            MeasurementType.WIND_DIRECTION: (0, 0),
            MeasurementType.YAW_ANGLE: (0, 0),
        }
    )

    specs = [
        MeasurementSpec(
            name="turb_0/wd_current",
            measurement_type=MeasurementType.WIND_DIRECTION,
            index_range=(0, 1),
            min_val=260,
            max_val=280,
            turbine_id=0,
            is_circular=True,
        ),
        MeasurementSpec(
            name="turb_0/yaw_current",
            measurement_type=MeasurementType.YAW_ANGLE,
            index_range=(1, 2),
            min_val=-30,
            max_val=30,
            turbine_id=0,
        ),
    ]
    clean_obs = np.zeros(2, dtype=np.float32)

    # 2. ACT
    # This calls the NEW _resample_bias method, which uses the mocked rng
    noise_model.reset_noise(specs, mock_rng)
    # The apply_noise method now uses the correctly pre-computed confounding vector
    noisy_obs = noise_model.apply_noise(clean_obs, specs, mock_rng)

    # 3. ASSERT
    expected_final_yaw_bias_physical = YAW_BIAS_PHYSICAL - WD_BIAS_PHYSICAL
    yaw_spec = specs[1]
    expected_scaled_yaw_delta = (expected_final_yaw_bias_physical * 2.0) / (
        yaw_spec.max_val - yaw_spec.min_val
    )
    expected_noisy_yaw_obs = 0.0 + expected_scaled_yaw_delta

    assert noisy_obs[1] == pytest.approx(expected_noisy_yaw_obs)

    # Teardown: It's good practice to clear static variables after the test
    NoiseModel._unscale_value_static = None
    NoiseModel._scale_value_static = None


def test_white_noise_confounds_yaw_with_wd():
    """
    Unit test to confirm confounding in WhiteNoiseModel using a mocked RNG.
    """
    # 1. ARRANGE
    WD_NOISE_PHYSICAL = 5.0
    YAW_NOISE_PHYSICAL = -1.0

    mock_rng = MagicMock(spec=np.random.Generator)
    mock_rng.normal.side_effect = [
        np.array([WD_NOISE_PHYSICAL]),  # For wd_current
        np.array([YAW_NOISE_PHYSICAL]),  # For yaw_current
    ]

    # FIX 2: Manually set the static scaling methods for the unit test
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    noise_model = WhiteNoiseModel(
        {
            MeasurementType.WIND_DIRECTION: 2.0,
            MeasurementType.YAW_ANGLE: 0.5,
        }
    )

    specs = [
        MeasurementSpec(
            name="turb_0/wd_current",
            measurement_type=MeasurementType.WIND_DIRECTION,
            index_range=(0, 1),
            min_val=260,
            max_val=280,
            turbine_id=0,
            is_circular=True,
        ),
        MeasurementSpec(
            name="turb_0/yaw_current",
            measurement_type=MeasurementType.YAW_ANGLE,
            index_range=(1, 2),
            min_val=-30,
            max_val=30,
            turbine_id=0,
        ),
    ]
    clean_obs = np.zeros(2, dtype=np.float32)

    # 2. ACT
    noisy_obs = noise_model.apply_noise(clean_obs, specs, mock_rng)

    # 3. ASSERT
    expected_final_yaw_noise_physical = (
        YAW_NOISE_PHYSICAL - WD_NOISE_PHYSICAL
    )  # -1.0 - 5.0 = -6.0
    yaw_spec = specs[1]
    expected_scaled_yaw_delta = (expected_final_yaw_noise_physical * 2.0) / (
        yaw_spec.max_val - yaw_spec.min_val
    )

    assert noisy_obs[1] == pytest.approx(expected_scaled_yaw_delta)


def test_agent_perception_is_confounded(smoke_test_env):
    """
    Integration test to ensure that an agent's perception of yaw is
    correctly confounded by wind direction noise in a full environment.
    """
    # 1. ARRANGE
    WD_BIAS_PHYSICAL = 15.0

    manager = MeasurementManager(smoke_test_env, seed=123)
    bias_model = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_DIRECTION: (WD_BIAS_PHYSICAL, WD_BIAS_PHYSICAL),
            MeasurementType.YAW_ANGLE: (0.0, 0.0),  # No intrinsic yaw bias
        }
    )
    manager.set_noise_model(bias_model)

    noisy_env = NoisyWindFarmEnv(
        base_env_class=lambda **kwargs: smoke_test_env, measurement_manager=manager
    )

    agent = NoisyPyWakeAgent(
        measurement_manager=manager,
        x_pos=smoke_test_env.x_pos,
        y_pos=smoke_test_env.y_pos,
        turbine=smoke_test_env.turbine,
    )

    # 2. ACT
    obs, info = noisy_env.reset()
    obs, _, _, _, info = noisy_env.step(agent.predict(obs)[0])

    # 3. ASSERT
    # FIX 3 (Robustness): Find yaw spec dynamically instead of hardcoding index
    yaw_spec = next((s for s in manager.specs if s.name == "turb_0/yaw_current"), None)
    assert yaw_spec is not None, "Could not find yaw spec for turb_0"
    yaw_idx = yaw_spec.index_range[0]

    true_yaw_offset_scaled = info["clean_obs"][yaw_idx]
    true_yaw_physical = manager._unscale_value(
        np.array([true_yaw_offset_scaled]), yaw_spec.min_val, yaw_spec.max_val
    ).item()

    sensed_yaw_physical = info["obs_sensed/turb_0/yaw_current"]

    expected_sensed_yaw = true_yaw_physical - WD_BIAS_PHYSICAL

    assert sensed_yaw_physical == pytest.approx(expected_sensed_yaw, abs=0.1)

    noisy_env.close()
