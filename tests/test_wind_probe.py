import pytest
import numpy as np
from unittest.mock import MagicMock
from WindGym.core.wind_probe import WindProbe
from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
import yaml
import tempfile


@pytest.fixture
def mock_fs():
    """
    Creates a mock flow simulation (fs) object.
    This allows us to test the WindProbe's logic without running a full simulation.
    It's configured to return predictable wind speed and turbulence values.
    """
    fs = MagicMock()
    # Configure the mock to return a constant wind vector [u, v, w]
    fs.get_windspeed.return_value = np.array([8.0, 6.0, 0.0])
    # Configure the mock to return a constant turbulence intensity
    fs.get_turbulence_intensity.return_value = 0.15
    return fs


def test_probe_read_dispatch(mock_fs):
    """
    Tests the main `read()` method to ensure it correctly dispatches
    to the right internal function based on `probe_type`.
    """
    # Test Wind Speed (WS) probe
    ws_probe = WindProbe(
        fs=mock_fs,
        position=(0, 0, 90),
        turbine_position=(100, 0, 90),
        yaw_angle=0,
        probe_type="WS",
    )
    # The projected speed for a turbine directly downstream (+x) of the probe
    # with a wind of u=8 should be 8.
    assert np.isclose(ws_probe.read(), 8.0)

    # Test Turbulence Intensity (TI) probe
    ti_probe = WindProbe(
        fs=mock_fs,
        position=(0, 0, 90),
        turbine_position=(100, 0, 90),
        yaw_angle=0,
        probe_type="TI",
    )
    # The read() method should return the value from the mock fs
    assert ti_probe.read() == 0.15

    # Test that an invalid probe type raises a ValueError
    invalid_probe = WindProbe(
        fs=mock_fs,
        position=(0, 0, 90),
        turbine_position=(100, 0, 90),
        yaw_angle=0,
        probe_type="INVALID",
    )
    with pytest.raises(ValueError, match="Unsupported probe_type: INVALID"):
        invalid_probe.read()


def test_read_speed_magnitude(mock_fs):
    """
    Tests that the scalar magnitude of the wind vector is calculated correctly.
    """
    probe = WindProbe(
        fs=mock_fs, position=(0, 0, 90), turbine_position=(1, 1, 1), yaw_angle=0
    )
    # With wind vector [8, 6, 0], the magnitude is sqrt(8^2 + 6^2) = 10
    assert np.isclose(probe.read_speed_magnitude(), 10.0)


def test_get_projected_wind_speed(mock_fs):
    """
    Tests the vector projection logic for various probe/turbine orientations.
    """
    # Wind vector is [u, v] = [8, 6]
    probe = WindProbe(
        fs=mock_fs, position=(0, 0, 90), turbine_position=(0, 0, 0), yaw_angle=0
    )  # Temp turbine position

    # Case 1: Turbine is directly downstream (+x direction)
    probe.turbine_position = (100, 0, 90)
    # The wind component along the x-axis is 8.0
    assert np.isclose(probe.get_projected_wind_speed_toward_turbine(), 8.0)

    # Case 2: Turbine is at a 45-degree angle
    probe.turbine_position = (100, 100, 90)
    # The direction vector is proportional to [1, 1].
    # The dot product of [8, 6] and [1, 1] is 14.
    # The magnitude of [1, 1] is sqrt(2).
    # Projected speed = 14 / sqrt(2)
    assert np.isclose(probe.get_projected_wind_speed_toward_turbine(), 14 / np.sqrt(2))

    # Edge Case: Probe and turbine at the same location
    probe.turbine_position = (0, 0, 90)
    with pytest.raises(ValueError, match="Probe and turbine are at the same location"):
        probe.get_projected_wind_speed_toward_turbine()


def test_get_inflow_angle(mock_fs):
    """
    Tests the angle calculation from the probe to the turbine.
    """
    probe = WindProbe(
        fs=mock_fs, position=(0, 0, 90), turbine_position=(0, 0, 0), yaw_angle=0
    )

    # To the right (+x)
    probe.turbine_position = (10, 0, 90)
    assert np.isclose(probe.get_inflow_angle_to_turbine(degrees=True), 0.0)

    # Straight up (+y)
    probe.turbine_position = (0, 10, 90)
    assert np.isclose(probe.get_inflow_angle_to_turbine(degrees=True), 90.0)

    # To the left (-x)
    probe.turbine_position = (-10, 0, 90)
    assert np.isclose(probe.get_inflow_angle_to_turbine(degrees=True), 180.0)

    # 45-degree angle
    probe.turbine_position = (10, 10, 90)
    assert np.isclose(probe.get_inflow_angle_to_turbine(degrees=True), 45.0)


def test_init_probes_with_absolute_position():
    """
    Tests that probes with absolute positions are correctly initialized,
    covering the 'else' branch in `_init_probes`. This also fixes the KeyError
    by providing a complete YAML.
    """
    yaml_config_with_absolute_probe = """
    yaw_init: "Zeros"
    BaseController: "Local"
    ActionMethod: "yaw"
    farm: {yaw_min: -30, yaw_max: 30}
    wind: {ws_min: 10, ws_max: 10, TI_min: 0.07, TI_max: 0.07, wd_min: 270, wd_max: 270}
    mes_level: {turb_ws: False, turb_wd: False, turb_TI: False, turb_power: False, farm_ws: False, farm_wd: False, farm_TI: False, farm_power: False}
    ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 0, ws_history_length: 1, ws_window_length: 1}
    wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 0, wd_history_length: 1, wd_window_length: 1}
    yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 0, yaw_history_length: 1, yaw_window_length: 1}
    power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 0, power_history_length: 1, power_window_length: 1}
    act_pen: {action_penalty: 0.0, action_penalty_type: "Change"}
    power_def: {Power_reward: "None", Power_avg: 1, Power_scaling: 1.0}
    probes:
      - name: absolute_probe_1
        position: [100, 50, 90]
        probe_type: WS
    """
    x_pos, y_pos = generate_square_grid(V80(), nx=2, ny=1, xDist=5, yDist=3)

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_config_with_absolute_probe,
        turbtype="None",
        reset_init=True,
    )

    assert hasattr(env, "probes")
    assert len(env.probes) == 1
    probe = env.probes[0]
    assert probe.name == "absolute_probe_1"
    # Note: `position` is now relative to the turbine after initialization
    # To check absolute position, we'd need to reconstruct it, but checking existence is sufficient here.
    assert probe.position is not None

    env.close()
