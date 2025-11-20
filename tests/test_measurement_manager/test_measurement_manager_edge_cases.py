# ./tests/test_measurement_manager_edge_cases.py

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock, patch

from WindGym.core.measurement_manager import (
    MeasurementManager,
    MeasurementType,
    MeasurementSpec,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    NoisyWindFarmEnv,
    NoiseModel,  # Ensure NoiseModel is imported
)


# --- Helper function to create a valid base configuration ---
def get_minimal_valid_config():
    """Helper to provide a complete, default YAML configuration dictionary."""
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


# --- Edge Case Tests ---


def test_runtime_error_for_unsetAttribute_scaling_functions():
    """
    Covers: `raise RuntimeError("Scaling functions not set on NoiseModel base class.")`
    This happens if _handle_circular_noise is called but _unscale_value_static
    and _scale_value_static haven't been set by MeasurementManager.
    """
    # Store original static methods
    original_unscale = NoiseModel._unscale_value_static
    original_scale = NoiseModel._scale_value_static

    # Temporarily unset the static methods to simulate the error condition
    NoiseModel._unscale_value_static = None
    NoiseModel._scale_value_static = None

    spec = MeasurementSpec(
        name="circular_val",  # Needs to be circular to call _handle_circular_noise
        measurement_type=MeasurementType.WIND_DIRECTION,  # Or any circular type
        index_range=(0, 1),
        min_val=0.0,
        max_val=360.0,
        is_circular=True,
    )
    # Instantiate a concrete NoiseModel subclass
    # FIX: Ensure the noise_std_devs dict contains the measurement_type (WIND_DIRECTION)
    # so that the `if spec.measurement_type in self.noise_std_devs:` check passes in apply_noise,
    # leading to the call of _handle_circular_noise.
    concrete_noise_model = WhiteNoiseModel(
        noise_std_devs={MeasurementType.WIND_DIRECTION: 1.0}
    )

    # Call apply_noise, which will attempt to call _handle_circular_noise
    # with `is_circular=True` spec. This will trigger the RuntimeError.
    clean_obs = np.array([0.0], dtype=np.float32)
    rng = np.random.default_rng(42)  # Needs an rng for apply_noise

    with pytest.raises(RuntimeError) as excinfo:
        concrete_noise_model.apply_noise(clean_obs, [spec], rng)

    assert "Scaling functions not set on NoiseModel base class" in str(excinfo.value)

    # Restore original static methods to avoid interfering with other tests
    NoiseModel._unscale_value_static = original_unscale
    NoiseModel._scale_value_static = original_scale


def test_whitenoisemodel_circular_path_coverage():
    """
    Covers the `if spec.is_circular:` branch within `WhiteNoiseModel.apply_noise`
    when it correctly calls `_handle_circular_noise`.
    """
    # Ensure static scaling methods are set as MeasurementManager would do
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    std_devs = {MeasurementType.WIND_DIRECTION: 1.0}
    noise_model = WhiteNoiseModel(std_devs)

    # Create a circular spec
    circular_spec = MeasurementSpec(
        name="wd",
        measurement_type=MeasurementType.WIND_DIRECTION,
        index_range=(0, 1),
        min_val=0.0,
        max_val=360.0,
        is_circular=True,
    )
    clean_obs = np.array(
        [0.0], dtype=np.float32
    )  # Scaled value (e.g., 0.0 means 180 degrees)

    # Create a *mocked* Generator object explicitly, and then its 'normal' method
    mock_rng_instance = MagicMock(spec=np.random.Generator)
    mock_rng_instance.normal.return_value = (
        10.0  # Make it return a predictable non-zero value
    )

    # We use this mocked instance as the rng passed to apply_noise
    noisy_obs = noise_model.apply_noise(clean_obs, [circular_spec], mock_rng_instance)

    # clean_obs (scaled 0.0) -> physical 180 degrees (from min 0, max 360)
    # noise_to_add_physical is 10.0 degrees
    # noisy_physical = 180 + 10 = 190 degrees
    # wrapped_physical = 190 % 360 = 190 degrees
    # 190 degrees scaled back to [-1, 1] range: 2 * (190 - 0) / (360 - 0) - 1 = 0.0555...
    expected_noisy_obs_val = MeasurementManager._scale_value(
        np.array([190.0]), 0.0, 360.0
    ).item()

    assert np.isclose(noisy_obs[0], expected_noisy_obs_val)
    mock_rng_instance.normal.assert_called_once_with(
        0, 1.0, size=1
    )  # Verify random.normal was called as expected


def test_noismodel_handle_circular_noise_else_branch():
    """
    Covers the `else:` branch in `NoiseModel._handle_circular_noise`
    (where `spec.is_circular` is False).
    This branch is theoretically not called by WhiteNoiseModel.apply_noise in normal flow
    for non-circular specs (as apply_noise handles it directly for those).
    So, we call it directly on a concrete instance.
    """
    # Ensure static scaling methods are set (as MeasurementManager would do)
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    scaled_values = np.array([0.5], dtype=np.float32)  # A scaled input value
    noise_unscaled_physical = np.array(
        [2.0], dtype=np.float32
    )  # Some physical noise to add

    # A non-circular spec
    non_circular_spec = MeasurementSpec(
        name="non_circular_val",
        measurement_type=MeasurementType.GENERIC,  # Not WIND_DIRECTION, so typically not circular
        index_range=(0, 1),
        min_val=0.0,
        max_val=10.0,
        is_circular=False,  # Crucially False, to hit the 'else' branch
    )

    # Instantiate a concrete NoiseModel subclass to call its inherited method
    concrete_noise_model = WhiteNoiseModel(
        noise_std_devs={}
    )  # Empty dict is fine as apply_noise is not called

    # Call the protected method directly with a non-circular spec
    result = concrete_noise_model._handle_circular_noise(
        scaled_values, noise_unscaled_physical, non_circular_spec
    )

    # The expected behavior of this 'else' branch is `values_scaled + noise_unscaled_physical`.
    assert np.allclose(result, scaled_values + noise_unscaled_physical)


def test_episodic_bias_scaled_bias_delta_array_zero_span():
    """
    Covers the `if span == 0:` branch within `EpisodicBiasNoiseModel._resample_bias`
    for `scaled_bias_delta_array = np.full(..., 0.0, ...)`
    """
    # Create a spec with zero span for a measurement type that EpisodicBiasNoiseModel applies bias to
    spec_with_zero_span = MeasurementSpec(
        "fixed_val",
        MeasurementType.WIND_SPEED,
        (0, 1),
        10.0,
        10.0,  # min_val == max_val
    )
    # Set bias_ranges to target this spec's measurement type
    bias_model = EpisodicBiasNoiseModel({MeasurementType.WIND_SPEED: (-5.0, 5.0)})

    # Ensure static scaling methods are set as MeasurementManager would do
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    rng = np.random.default_rng(42)

    # We need to *inspect* _resample_bias or check the result in current_bias_vector
    bias_model.reset_noise([spec_with_zero_span], rng)

    # After reset_noise, _resample_bias should have been called.
    # The current_bias_vector should have a 0.0 for this spec due to span==0.
    assert np.allclose(bias_model.current_bias_vector[0], 0.0)
    # The actual sampled unscaled bias value should be non-zero (from the range -5.0 to 5.0)
    # This confirms that the zero-span logic overrode a potentially non-zero bias.
    assert (
        bias_model.current_unscaled_biases_by_spec_name[spec_with_zero_span.name] != 0.0
    )


def test_episodic_bias_physical_bias_value_for_slice_zero_span():
    """
    Covers the `if span == 0:` branch within `EpisodicBiasNoiseModel.apply_noise`
    for `physical_bias_value_for_slice = np.full_like(..., 0.0)` for circular types.
    """
    # This case requires a circular spec with zero span
    spec_circular_zero_span = MeasurementSpec(
        "fixed_wd",
        MeasurementType.WIND_DIRECTION,
        (0, 1),
        270.0,
        270.0,
        is_circular=True,
    )
    bias_model = EpisodicBiasNoiseModel(
        {MeasurementType.WIND_DIRECTION: (-10.0, 10.0)}
    )  # Non-zero bias range

    # Ensure static scaling methods are set
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    rng = np.random.default_rng(42)
    # First, reset_noise to populate current_bias_vector.
    # The _resample_bias will set scaled_bias_delta_array to 0 for this zero-span spec.
    bias_model.reset_noise([spec_circular_zero_span], rng)

    clean_obs = np.array([0.0], dtype=np.float32)  # Arbitrary scaled input

    # In apply_noise, when it iterates over specs, for this spec:
    # scaled_bias_delta_for_slice will be 0.0 (from _resample_bias because span was 0)
    # The `if spec.is_circular:` block will be hit.
    # Inside, `span == 0` for the unscaling part will be true for this spec (as min_val == max_val).
    # This should trigger `physical_bias_value_for_slice = np.full_like(..., 0.0)`.
    noisy_obs = bias_model.apply_noise(clean_obs, [spec_circular_zero_span], rng)

    # The result should be clean_obs because the effective bias for a zero-span circular measurement is 0
    assert np.allclose(noisy_obs, clean_obs)
    # Verify that the unscaled bias value was non-zero but the scaled one was 0.
    assert (
        bias_model.current_unscaled_biases_by_spec_name[spec_circular_zero_span.name]
        != 0.0
    )


def test_episodic_bias_initial_apply_noise_shape_mismatch_coverage():
    """
    Covers the `if self.current_bias_vector is None or self.current_bias_vector.shape != observations.shape:`
    branch in `EpisodicBiasNoiseModel.apply_noise` when a shape mismatch occurs.
    """
    # Set up with one spec, so initial bias vector has size 1
    spec_initial = MeasurementSpec("val_1", MeasurementType.GENERIC, (0, 1), 0, 10)
    bias_model = EpisodicBiasNoiseModel({MeasurementType.GENERIC: (1.0, 1.0)})

    # Ensure static scaling methods are set
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    rng = np.random.default_rng(42)

    # First call: current_bias_vector is None, so it gets initialized to shape (1,)
    obs_initial = np.array([0.0], dtype=np.float32)
    bias_model.apply_noise(obs_initial, [spec_initial], rng)
    assert bias_model.current_bias_vector.shape == (1,)

    # Second call: provide observations of a different shape to trigger re-sampling
    obs_mismatch = np.array([0.0, 0.0], dtype=np.float32)  # Shape (2,)
    spec_mismatch_1 = MeasurementSpec("val_1", MeasurementType.GENERIC, (0, 1), 0, 10)
    spec_mismatch_2 = MeasurementSpec("val_2", MeasurementType.GENERIC, (1, 2), 0, 10)

    # _resample_bias will be called again due to shape mismatch
    # It will use the provided specs ([spec_mismatch_1, spec_mismatch_2])
    noisy_obs_mismatch = bias_model.apply_noise(
        obs_mismatch, [spec_mismatch_1, spec_mismatch_2], rng
    )

    # Assert that the bias vector was indeed re-sampled to the new shape
    assert bias_model.current_bias_vector.shape == (2,)
    # Verify noise was applied (meaning the loop in apply_noise ran)
    assert not np.allclose(noisy_obs_mismatch, obs_mismatch)


def test_measurement_manager_scale_value_zero_span():
    """
    Covers `if span == 0: return unscaled_value * 0.0` in `MeasurementManager._scale_value`.
    """
    unscaled_val = np.array([15.0])
    min_val = 10.0
    max_val = 10.0  # Zero span
    scaled_value = MeasurementManager._scale_value(unscaled_val, min_val, max_val)
    assert np.allclose(scaled_value, 0.0)  # Should return 0.0 as per the code


def test_get_physical_values_from_obs_for_logging_with_multi_element_spec():
    """
    Covers the `else` branch in `_get_physical_values_from_obs_for_logging`
    (`physical_values[spec.name] = self._unscale_value(...)`)
    when `val_scaled.size > 1`.

    For this test, we will manually create a `MeasurementSpec` that covers
    multiple indices to force `val_scaled.size > 1`.
    """

    # Create a dummy environment that MeasurementManager can initialize with
    class MockEnvMinimal:
        def __init__(self):
            # Minimal mocks required by MeasurementManager's __init__ for _build_from_env
            self.farm_measurements = MagicMock()
            self.farm_measurements.n_turbines = 1
            # Mock `get_mes_names` from MesClass.py behavior if it were a real Mes instance
            # For this test, we want to control specs manually, so we make it simple.
            self.farm_measurements.turb_ws = False
            self.farm_measurements.turb_wd = False
            self.farm_measurements.turb_TI = False
            self.farm_measurements.turb_power = False
            self.farm_measurements.farm_ws = False
            self.farm_measurements.farm_wd = False
            self.farm_measurements.farm_TI = False
            self.farm_measurements.farm_power = False

            # Minimal values to avoid errors during MeasurementManager init
            self.yaw_min = -45
            self.yaw_max = 45
            self.maxturbpower = 1000000
            self.n_turb = 1  # Required for MesClass.farm_mes init
            self.ws_scaling_min = 0.0
            self.ws_scaling_max = 30.0
            self.wd_scaling_min = 0.0
            self.wd_scaling_max = 360.0
            self.ti_scaling_min = 0.0
            self.ti_scaling_max = 1.0
            self.yaw_scaling_min = -45.0
            self.yaw_scaling_max = 45.0
            self.ti_sample_count = 30  # Required by farm_mes
            self.dt = 1  # minimal
            self.dt_env = 1  # minimal
            self.sim_steps_per_env_step = 1
            self.x_pos = np.array([0])
            self.y_pos = np.array([0])
            self.yaml_path = None  # To avoid trying to load config.yaml

        @property
        def observation_space(self):
            # Define an observation space that can accommodate our custom spec below
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    mock_env = MockEnvMinimal()
    manager = MeasurementManager(mock_env)  # Initialize MeasurementManager

    # Manually override the specs generated by _build_from_env with our custom multi-element spec
    # This spec represents, for example, 3 components of a wind vector at a turbine.
    manager.specs = [
        MeasurementSpec(
            name="turb_0/wind_vector",
            measurement_type=MeasurementType.GENERIC,  # Use GENERIC as it's not specific to scalar WS/WD
            index_range=(0, 3),  # This spans 3 elements in the observation vector
            min_val=-20.0,  # Example min physical value
            max_val=20.0,  # Example max physical value
            turbine_id=0,
        )
    ]

    # Create a scaled observation vector for this 3-element spec
    # e.g., scaled values of -1, 0, 1
    obs_vector = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    # Expected physical values for -1.0, 0.0, 1.0 scaled in [-20, 20] range:
    # -1.0 -> -20.0
    # 0.0  -> 0.0
    # 1.0  -> 20.0
    expected_physical_values_array = np.array([-20.0, 0.0, 20.0], dtype=np.float32)

    physical_values = manager._get_physical_values_from_obs_for_logging(obs_vector)

    assert "turb_0/wind_vector" in physical_values
    # Now, the value for "turb_0/wind_vector" should be a numpy array of size 3
    assert isinstance(physical_values["turb_0/wind_vector"], np.ndarray)
    assert physical_values["turb_0/wind_vector"].shape == (3,)
    assert np.allclose(
        physical_values["turb_0/wind_vector"], expected_physical_values_array
    )


def test_episodic_bias_resample_bias_no_specs():
    """
    Covers the `if not specs: self.current_bias_vector = np.array([], dtype=np.float32); return`
    branch in `EpisodicBiasNoiseModel._resample_bias`.
    """
    bias_model = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_SPEED: (-1.0, 1.0)  # Bias range for some type
        }
    )

    # Ensure static scaling methods are set as MeasurementManager would do.
    # While this specific branch doesn't use them, other parts of the class do,
    # and it's good practice for consistency in tests involving NoiseModels.
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    rng = np.random.default_rng(42)

    # Call _resample_bias with an empty list of specs
    bias_model._resample_bias([], rng)

    # Assert that current_bias_vector is an empty numpy array
    assert bias_model.current_bias_vector is not None
    assert bias_model.current_bias_vector.size == 0
    assert bias_model.current_bias_vector.shape == (0,)
    # Also verify that current_unscaled_biases_by_spec_name is empty
    assert bias_model.current_unscaled_biases_by_spec_name == {}


def test_episodic_bias_apply_noise_returns_copy_if_no_specs():
    """
    Covers the `if self.current_bias_vector is None or self.current_bias_vector.size == 0: return observations.copy()`
    branch in `EpisodicBiasNoiseModel.apply_noise` when _resample_bias results in an empty bias vector.
    """
    bias_model = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_SPEED: (
                -1.0,
                1.0,
            )  # Define some bias ranges to initialize
        }
    )

    # Ensure static scaling methods are set
    NoiseModel._unscale_value_static = MeasurementManager._unscale_value
    NoiseModel._scale_value_static = MeasurementManager._scale_value

    rng = np.random.default_rng(42)

    # Initially, current_bias_vector is None.
    # Call apply_noise with an empty list of specs.
    # This will cause _resample_bias to be called with empty specs,
    # leading to self.current_bias_vector.size == 0.
    # Subsequently, the inner 'if' condition in apply_noise will be met.

    initial_observations = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    returned_observations = bias_model.apply_noise(initial_observations, [], rng)

    # Assert that the returned observations are a copy of the original (no noise applied)
    assert np.array_equal(returned_observations, initial_observations)
    assert (
        returned_observations is not initial_observations
    )  # Ensure it's a copy, not the same object

    # Also assert that the bias vector is indeed empty, confirming the path
    assert bias_model.current_bias_vector.size == 0
    assert bias_model.current_unscaled_biases_by_spec_name == {}
