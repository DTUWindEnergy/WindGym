# tests/test_hawc2_derating.py
"""Tests for the HAWC2-backend derating path (mocked, no h2lib / HAWC2 needed).

Covers the htc validation (`check_htc_supports_derating`), the derate sensor
registration and its d <-> dr% mapping, the `_apply_derating` write path, the
rated-mode pass-through, and that baseline turbines stay greedy.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.core.derating import check_htc_supports_derating

HTC_SRC = Path(__file__).parent.parent / "examples/HawcFiles/htc/DTU10mw_derate.htc"
VANILLA_HTC = (
    Path(__file__).parent.parent / "examples/HawcFiles/htc/python_yaw_control.htc"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def derate_htc(tmp_path):
    """Copy of the shipped derate htc (constants 79=2, 80=-1, 104=1)."""
    dst = tmp_path / "DTU10mw_derate.htc"
    shutil.copy(HTC_SRC, dst)
    return dst


def htc_variant(htc_path, replacements=(), drop_containing=None):
    """Write a sibling htc with string replacements / dropped lines applied."""
    text = Path(htc_path).read_text()
    for old, new in replacements:
        assert old in text, f"variant anchor {old!r} not found in htc"
        text = text.replace(old, new)
    if drop_containing is not None:
        text = "\n".join(
            line for line in text.splitlines() if drop_containing not in line
        )
    dst = Path(htc_path).with_name("variant_" + Path(htc_path).name)
    dst.write_text(text)
    return dst


def rated_htc(htc_path):
    """Derate htc with constant 104 = 0 (dr is % of rated power)."""
    return htc_variant(htc_path, [("constant 104    1.0", "constant 104    0.0")])


def make_config(**overrides):
    """Minimal dict config for a mocked HAWC2 derating env."""
    config = {
        "yaw_action": True,
        "derate_action": True,
        "derate_reference": "available",
        "derate_min": 0.0,
        "derate_max": 1.0,
        "derate_method": "absolute",
        "ActionMethod": "yaw",
        "BaseController": "Local",
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
        "power_def": {
            "Power_reward": "Power_avg",
            "Power_avg": 1,
            "Power_scaling": 1.0,
        },
        "mes_level": {
            "turb_ws": True,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
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
            "power_current": True,
            "power_rolling_mean": False,
            "power_history_N": 0,
            "power_history_length": 1,
            "power_window_length": 1,
        },
    }
    config.update(overrides)
    return config


@pytest.fixture
def mock_hawc2_wind_turbines():
    """Mock HAWC2WindTurbinesW (agent + baseline) like test_hawc2_integration."""

    def mock_factory(*args, **kwargs):
        collection_mock = MagicMock()
        x_pos = args[0] if args else kwargs.get("x", [])
        y_pos = args[1] if len(args) > 1 else kwargs.get("y", [])
        num_turbines = len(x_pos)

        z_pos = np.full_like(np.asarray(x_pos, dtype=float), 90.0)
        collection_mock.positions_east_north = np.array([x_pos, y_pos, z_pos])
        collection_mock.N = num_turbines
        collection_mock.yaw = np.zeros(num_turbines)
        collection_mock.step_handlers = []
        collection_mock.diameter.return_value = np.full(num_turbines, V80().diameter())
        collection_mock.power.return_value = np.full(num_turbines, 1e6)
        collection_mock.add_sensor = MagicMock()
        collection_mock.yaw_tilt.return_value = (
            np.zeros(num_turbines),
            np.zeros(num_turbines),
        )
        type(collection_mock).positions_xyz = PropertyMock(
            return_value=np.array([x_pos, y_pos, z_pos])
        )
        type(collection_mock).rotor_positions_xyz = PropertyMock(
            return_value=np.array([x_pos, y_pos, z_pos])
        )
        type(collection_mock).rotor_avg_windspeed = PropertyMock(
            return_value=np.zeros((num_turbines, 3))
        )
        collection_mock.h2 = MagicMock(close=MagicMock(), write_output=MagicMock())
        htc_list = []
        for i in range(num_turbines):
            mock_htc_obj = MagicMock()
            type(mock_htc_obj.output.filename).values = PropertyMock(
                return_value=[f"mock_results_file_{i}"]
            )
            mock_htc_obj.modelpath = "/mock/path/"
            htc_list.append(mock_htc_obj)
        collection_mock.htc_lst = htc_list
        collection_mock.sensors = MagicMock()

        single_mock = MagicMock()
        single_mock.ct.return_value = 0.8
        single_mock.diameter.return_value = np.array([V80().diameter()])
        single_mock.rotor_avg_ti.return_value = np.array([0.07])
        type(single_mock).rotor_avg_windspeed = PropertyMock(
            return_value=np.array([10.0, 0.0, 0.0])
        )
        single_mock.axisymetric_induction.side_effect = lambda r_input: np.full(
            (len(r_input), 1), 0.1
        )
        collection_mock.__getitem__.return_value = single_mock

        return collection_mock

    with (
        patch(
            "WindGym.wind_farm_env.HAWC2WindTurbinesW", side_effect=mock_factory
        ) as mock_class_env,
        patch(
            "WindGym.core.baseline_manager.HAWC2WindTurbinesW", side_effect=mock_factory
        ),
        patch("dynamiks.flow_simulation.FlowSimulation.step"),
    ):
        yield mock_class_env


def make_env(htc_path, config=None, **kwargs):
    """Mocked 2-turbine HAWC2 env (requires mock_hawc2_wind_turbines active)."""
    d = V80().diameter()
    return WindFarmEnv(
        turbine=V80(),
        x_pos=[0.0, 5.0 * d],
        y_pos=[0.0, 0.0],
        config=config if config is not None else make_config(),
        HTC_path=str(htc_path),
        reset_init=True,
        n_passthrough=0.1,
        burn_in_passthroughs=0.1,
        turbtype="None",
        **kwargs,
    )


def get_derate_sensor_call(wts):
    """Return the add_sensor call that registered the exposed 'derate' sensor."""
    for call in wts.add_sensor.call_args_list:
        name = call.kwargs.get("name", call.args[0] if call.args else None)
        if name == "derate":
            return call
    return None


# ---------------------------------------------------------------------------
# 1-4: htc validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validation_accepts_avail_htc(derate_htc):
    check_htc_supports_derating(derate_htc, "available")


@pytest.mark.unit
def test_validation_rejects_htc_without_derate_controller():
    with pytest.raises(ValueError, match="derate controller"):
        check_htc_supports_derating(VANILLA_HTC, "available")


@pytest.mark.unit
def test_validation_reference_mode_matching(derate_htc):
    # avail htc (104=1) + "rated" -> mismatch
    with pytest.raises(ValueError, match="constant 104"):
        check_htc_supports_derating(derate_htc, "rated")
    # rated htc (104=0) + "available" -> mismatch, + "rated" -> OK
    rated = rated_htc(derate_htc)
    with pytest.raises(ValueError, match="constant 104"):
        check_htc_supports_derating(rated, "available")
    check_htc_supports_derating(rated, "rated")
    # absent 104 is the controller default 0 -> "rated" OK, "available" rejected
    absent = htc_variant(rated, drop_containing="constant 104")
    check_htc_supports_derating(absent, "rated")
    with pytest.raises(ValueError, match="constant 104"):
        check_htc_supports_derating(absent, "available")


@pytest.mark.unit
def test_validation_rejects_fixed_derate_and_disabled_strategy(derate_htc):
    fixed = htc_variant(derate_htc, [("constant 80    -1.0", "constant 80    5.0")])
    with pytest.raises(ValueError, match="constant 80"):
        check_htc_supports_derating(fixed, "available")
    disabled = htc_variant(derate_htc, [("constant 79     2.0", "constant 79     0.0")])
    with pytest.raises(ValueError, match="constant 79"):
        check_htc_supports_derating(disabled, "available")


# ---------------------------------------------------------------------------
# 5-7: sensor registration, mapping and _apply_derating write path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_derate_sensor_registration_and_mapping(derate_htc, mock_hawc2_wind_turbines):
    env = make_env(derate_htc)
    call = get_derate_sensor_call(env.wts)
    assert call is not None, "no exposed 'derate' sensor registered on the wts"
    assert call.kwargs.get("expose") is True

    # setter: d -> dr% = (1 - d) * 100, sent as a *plain list* (MultiH2Lib only
    # distributes lists element-wise over the per-turbine processes)
    wt = MagicMock()
    call.kwargs["setter"](wt, np.array([0.3, 0.0]))
    (channel, values), _ = wt.h2.set_variable_sensor_value.call_args
    assert channel == 2
    assert isinstance(values, list)
    np.testing.assert_allclose(values, [70.0, 100.0])

    # getter: dr% -> d, tolerant of (n, 1)-shaped sensor arrays
    wt.sensors.derate_getter = np.array([[70.0], [100.0]])
    np.testing.assert_allclose(call.kwargs["getter"](wt), [0.3, 0.0])

    env.close()


@pytest.mark.unit
def test_derate_sensor_initialized_to_zero(derate_htc, mock_hawc2_wind_turbines):
    env = make_env(derate_htc)
    np.testing.assert_array_equal(env.wts.sensors.derate, np.zeros(env.n_turb))
    env.close()


@pytest.mark.unit
def test_apply_derating_writes_fraction_to_sensor(derate_htc, mock_hawc2_wind_turbines):
    env = make_env(derate_htc)
    # action layout [yaw | derate]; raw 0.0 -> d = 0.5, raw -1.0 -> d = 0.0
    action = np.array([0.0, 0.0, 0.0, -1.0])
    env._apply_derating(action)
    np.testing.assert_allclose(env.wts.sensors.derate, [0.5, 0.0])
    np.testing.assert_allclose(env.current_derate, [0.5, 0.0])
    env.close()


# ---------------------------------------------------------------------------
# 8: rated-mode pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rated_mode_passes_command_through(derate_htc, mock_hawc2_wind_turbines):
    env = make_env(rated_htc(derate_htc), config=make_config(derate_reference="rated"))
    action = np.array([0.0, 0.0, 0.2, -1.0])  # raw 0.2 -> cmd 0.6
    env._apply_derating(action)
    # HAWC2 rated mode: no available-power invariant conversion, the DTUWEC
    # controller applies the rated cap natively
    np.testing.assert_allclose(env.derate_command, [0.6, 0.0])
    np.testing.assert_allclose(env.current_derate, env.derate_command)
    np.testing.assert_allclose(env.wts.sensors.derate, [0.6, 0.0])
    env.close()


# ---------------------------------------------------------------------------
# 9: baseline turbines stay greedy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_baseline_turbines_get_no_derate_sensor(derate_htc, mock_hawc2_wind_turbines):
    env = make_env(derate_htc, Baseline_comp=True)
    assert env.wts_baseline is not None
    assert get_derate_sensor_call(env.wts) is not None
    assert get_derate_sensor_call(env.wts_baseline) is None
    env.close()
