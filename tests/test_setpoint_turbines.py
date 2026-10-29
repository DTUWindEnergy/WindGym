"""SetpointPyWakeWindTurbines: absolute per-turbine power cap on top of an
inertia-filtered available power, with a lagged effective derate feeding the
Ct table. No DWM: a fake flow simulation (time, dt, step_handlers), hand-set
rotor-average sensors and the synthetic exact derating table of
``test_derating.py`` (power = (1 - d) * P_avail(ws), ct = 0.8 (1 - d)^1.5)."""

from types import SimpleNamespace

import numpy as np
import pytest
from dynamiks.wind_turbines import PyWakeWindTurbines
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular

from WindGym.backend.setpoint_turbines import SetpointPyWakeWindTurbines
from WindGym.core.turbine_params import normalize_turbine_params

DT = 5.0
X = np.array([0.0, 400.0, 800.0])
Y = np.zeros(3)
RATED = 2.0e6


def p_avail(ws):
    return RATED * np.clip((np.asarray(ws, float) - 3.0) / 9.0, 0.0, 1.0) ** 3


@pytest.fixture(scope="module")
def turbine():
    ws = np.linspace(0.0, 30.0, 61)
    derate = np.linspace(0.0, 0.8, 17)
    power = p_avail(ws)[:, None] * (1.0 - derate[None, :])
    ct = 0.8 * (1.0 - derate[None, :]) ** 1.5 * np.ones_like(ws)[:, None]
    pctf = PowerCtNDTabular(
        input_keys=["ws", "derate"], value_lst=[ws, derate], power_arr=power,
        power_unit="W", ct_arr=ct, default_value_dict={"derate": 0.0},
    )
    for gi in pctf.interp:
        gi.bounds = "limit"
    return WindTurbine(name="SyntheticDerating", diameter=80.0, hub_height=70.0,
                       powerCtFunction=pctf)


def _set_inflow(wts, ws_waked, ws_free=None):
    if "rotor_avg_windspeed" not in wts.sensors.dict:
        wts.add_sensor("rotor_avg_windspeed", expose=True, ext_lst=["u", "v", "w"])
        wts.add_sensor("rotor_avg_freestream", expose=True, ext_lst=["u", "v", "w"])
    ws_waked = np.broadcast_to(np.asarray(ws_waked, float), (3,))
    ws_free = ws_waked if ws_free is None else np.broadcast_to(np.asarray(ws_free, float), (3,))
    wts.rotor_avg_windspeed = np.c_[ws_waked, np.zeros(3), np.zeros(3)]
    wts.rotor_avg_freestream = np.c_[ws_free, np.zeros(3), np.zeros(3)]


def make_plain(turbine, ws=9.0, derate=0.0):
    """dynamiks class with the env's plain-storage derate sensor (fraction mode)."""
    wts = PyWakeWindTurbines(x=X, y=Y, windTurbine=turbine)
    wts.add_sensor("derate", expose=True)
    wts.sensors.derate = np.broadcast_to(np.asarray(derate, float), (3,))
    _set_inflow(wts, ws)
    return wts


def make_setpoint(turbine, params=None, ws=9.0, dt=DT, attach=True):
    tp = normalize_turbine_params(params or {}, n_turb=3, dt_sim=dt)
    wts = SetpointPyWakeWindTurbines(x=X, y=Y, windTurbine=turbine, params=tp)
    _set_inflow(wts, ws)
    fs = SimpleNamespace(time=0.0, dt=dt, step_handlers=list(wts.step_handlers))
    if attach:
        wts.attach_to(fs)
    return wts, fs


def step(fs):
    fs.time = fs.time + fs.dt
    for h in fs.step_handlers:
        h(fs)


# ------------------------------------------------------------- registration --
def test_sensors_and_handlers(turbine):
    wts, fs = make_setpoint(turbine)
    for name in ("power_cap", "available_power", "derate"):
        assert name in wts.sensors.dict
    assert np.all(np.isinf(wts.power_cap)) and np.all(wts.power_cap > 0)
    assert wts.d_table_max == pytest.approx(0.8)
    assert fs.step_handlers[0] == wts.step_actuators
    assert wts.step_handlers[0] == wts.step_power_lag
    assert wts.step_handlers[-1] is wts.sensors


def test_derate_sensor_is_read_only(turbine):
    wts, _ = make_setpoint(turbine)
    with pytest.raises(ValueError, match="derived from power_cap"):
        wts.sensors.derate = np.zeros(3)


# ------------------------------------------------------------------ uncapped --
def test_uncapped_tau_zero_equals_plain_at_derate_zero(turbine):
    plain = make_plain(turbine)
    wts, fs = make_setpoint(turbine)
    step(fs)
    np.testing.assert_array_equal(wts.power(), plain.power())
    np.testing.assert_array_equal(wts.ct(), plain.ct())
    np.testing.assert_array_equal(wts.available_power, plain.power())
    np.testing.assert_array_equal(wts.effective_derate, np.zeros(3))
    np.testing.assert_array_equal(wts.sensors.derate, np.zeros(3))
    np.testing.assert_array_equal(wts.sensors.available_power, plain.power())


# ------------------------------------------------------------- cap semantics --
def test_cap_holds_power_flat_under_a_wind_step(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    wts.power_cap = np.full(3, 5.0e5)
    step(fs)
    np.testing.assert_allclose(wts.power(), 5.0e5)
    np.testing.assert_allclose(wts.effective_derate, 1.0 - 5.0e5 / p_avail(9.0))
    _set_inflow(wts, 12.0)
    step(fs)
    np.testing.assert_allclose(wts.power(), 5.0e5)  # cap acts at once, no jitter
    np.testing.assert_allclose(wts.effective_derate, 1.0 - 5.0e5 / p_avail(12.0))
    np.testing.assert_allclose(wts.available_power, p_avail(12.0))


def test_cap_above_available_is_inactive(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    wts.power_cap = np.array([np.inf, 1.0e6, 3.0e5])  # P_qs(9) ~ 5.93e5 W
    step(fs)
    expected_d = np.array([0.0, 0.0, 1.0 - 3.0e5 / p_avail(9.0)])
    np.testing.assert_allclose(wts.effective_derate, expected_d)
    np.testing.assert_allclose(wts.power(), np.minimum(wts.power_cap, p_avail(9.0)))


def test_effective_derate_is_clipped_to_the_table(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    wts.power_cap = np.full(3, 1.0e4)  # would need d = 0.983 > 0.8
    step(fs)
    np.testing.assert_allclose(wts.effective_derate, 0.8)
    np.testing.assert_allclose(wts.power(), 1.0e4)  # the cap is still delivered


def test_below_cut_in_gives_zero_derate_no_nan(turbine):
    wts, fs = make_setpoint(turbine, ws=1.0)
    wts.power_cap = np.full(3, 5.0e5)
    step(fs)
    assert np.all(np.isfinite(wts.effective_derate)) and np.all(wts.effective_derate == 0.0)
    np.testing.assert_array_equal(wts.power(), np.zeros(3))
    np.testing.assert_array_equal(wts.available_power, np.zeros(3))
    assert np.all(np.isfinite(wts.ct()))


def test_derate_is_relative_to_yawed_available_power(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    wts.reset_yaw_state(np.array([0.0, 20.0, 20.0]))
    wts.power_cap = np.array([np.inf, np.inf, 4.0e5])
    step(fs)
    plain = make_plain(turbine)
    plain.yaw = np.array([0.0, 20.0, 20.0])
    p_yawed = plain.power()
    np.testing.assert_array_equal(wts.available_power, p_yawed)
    np.testing.assert_allclose(wts.effective_derate, [0.0, 0.0, 1.0 - 4.0e5 / p_yawed[2]])


# -------------------------------------------------------- equivalence tau=0 --
def test_tau_zero_matches_plain_class_with_equivalent_derate(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    cap = np.array([np.inf, 4.0e5, 2.0e5])
    wts.power_cap = cap
    step(fs)
    d = np.where(cap < p_avail(9.0), 1.0 - cap / p_avail(9.0), 0.0)
    plain = make_plain(turbine, derate=d)
    np.testing.assert_array_equal(wts.effective_derate, d)
    np.testing.assert_array_equal(wts.ct(), plain.ct())  # same table call at the same d
    np.testing.assert_allclose(wts.power(), plain.power(), rtol=1e-9)
    np.testing.assert_array_equal(wts.rotor_avg_induction(), plain.rotor_avg_induction())


# ---------------------------------------------------------------- filters --
def test_available_power_follows_first_order_lag_exactly(turbine):
    tau = 10.0
    wts, fs = make_setpoint(turbine, {"tau_power": tau}, ws=9.0)
    step(fs)  # first call initialises the state at the quasi-steady value
    np.testing.assert_array_equal(wts.available_power, np.full(3, p_avail(9.0)))
    _set_inflow(wts, 12.0)
    step(fs)
    a = 1.0 - np.exp(-DT / tau)
    expected = p_avail(9.0) + a * (p_avail(12.0) - p_avail(9.0))
    np.testing.assert_array_equal(wts.available_power, np.full(3, expected))
    np.testing.assert_array_equal(wts.power(), wts.available_power)  # uncapped
    np.testing.assert_array_equal(wts.effective_derate, np.zeros(3))


def test_cap_change_moves_power_at_once_and_derate_with_tau_ct(turbine):
    tau_ct = 10.0
    wts, fs = make_setpoint(turbine, {"tau_ct": tau_ct}, ws=9.0)
    step(fs)
    np.testing.assert_array_equal(wts.effective_derate, np.zeros(3))
    wts.power_cap = np.full(3, 4.0e5)
    step(fs)
    np.testing.assert_allclose(wts.power(), 4.0e5)  # same step
    a = 1.0 - np.exp(-DT / tau_ct)
    d_cmd = 1.0 - 4.0e5 / p_avail(9.0)
    d1 = 0.0 + a * (d_cmd - 0.0)
    np.testing.assert_array_equal(wts.effective_derate, np.full(3, d1))
    step(fs)
    d2 = d1 + a * (d_cmd - d1)
    np.testing.assert_array_equal(wts.effective_derate, np.full(3, d2))
    # Ct is the table at the LAGGED derate, not at the command
    np.testing.assert_array_equal(wts.ct(), make_plain(turbine, derate=d2).ct())
    assert np.all(wts.ct() > make_plain(turbine, derate=d_cmd).ct())


def test_ct_gain_applies_on_top_of_the_lagged_table(turbine):
    wts, fs = make_setpoint(turbine, {"ct_gain": [0.9, 1.0, 1.1]}, ws=9.0)
    wts.power_cap = np.full(3, 4.0e5)
    step(fs)
    d = wts.effective_derate
    np.testing.assert_array_equal(
        wts.ct(), make_plain(turbine, derate=d).ct() * np.array([0.9, 1.0, 1.1]))


def test_cp_gain_scales_available_power(turbine):
    wts, fs = make_setpoint(turbine, {"cp_gain": [1.2, 1.0, 0.8]}, ws=9.0)
    step(fs)
    np.testing.assert_allclose(wts.available_power, p_avail(9.0) * np.array([1.2, 1.0, 0.8]))


# ----------------------------------------------------- before the first step --
def test_queries_before_the_first_handler_call_are_quasi_steady(turbine):
    wts, fs = make_setpoint(turbine, {"tau_power": 10.0, "tau_ct": 10.0}, ws=9.0)
    wts.power_cap = np.array([np.inf, 4.0e5, 4.0e5])
    d_cmd = np.array([0.0, 1.0 - 4.0e5 / p_avail(9.0), 1.0 - 4.0e5 / p_avail(9.0)])
    np.testing.assert_allclose(wts.effective_derate, d_cmd)
    np.testing.assert_allclose(wts.power(), np.minimum(wts.power_cap, p_avail(9.0)))
    np.testing.assert_allclose(wts.available_power, p_avail(9.0))
    np.testing.assert_allclose(wts.ct(), make_plain(turbine, derate=d_cmd).ct())
    step(fs)  # the first step initialises the states without a transient
    np.testing.assert_allclose(wts.available_power, p_avail(9.0))
    np.testing.assert_allclose(wts.effective_derate, d_cmd)


def test_step_before_attach_raises(turbine):
    wts, fs = make_setpoint(turbine, attach=False)
    with pytest.raises(RuntimeError, match="attach_to"):
        wts.step_power_lag(fs)


# ------------------------------------------------------- misc interfaces --
def test_power_without_wakes_is_quasi_static_at_the_effective_derate(turbine):
    wts, fs = make_setpoint(turbine, {"tau_ct": 10.0}, ws=9.0)
    _set_inflow(wts, 9.0, ws_free=12.0)
    wts.power_cap = np.full(3, 4.0e5)
    step(fs)
    d = wts.effective_derate
    plain = make_plain(turbine, ws=9.0, derate=d)
    _set_inflow(plain, 9.0, ws_free=12.0)
    np.testing.assert_array_equal(wts.power(include_wakes=False), plain.power(include_wakes=False))


def test_idx_proxy(turbine):
    wts, fs = make_setpoint(turbine, ws=9.0)
    wts.power_cap = np.array([np.inf, 4.0e5, 2.0e5])
    step(fs)
    for i in range(3):
        # the dynamiks proxy appends an axis to an int-indexed call: (1, 1) like upstream
        assert wts[i].power().shape == wts[i].ct().shape == (1, 1)
        np.testing.assert_array_equal(wts[i].power().ravel(), wts.power()[[i]])
        np.testing.assert_array_equal(wts[i].ct().ravel(), wts.ct()[[i]])
    assert wts.power(idx=1).shape == wts.ct(idx=1).shape == (1,)
    np.testing.assert_array_equal(wts.power(idx=slice(1, 3)), wts.power()[1:])


def test_sensor_dump_reports_delivered_power_and_lagged_derate(turbine):
    wts, fs = make_setpoint(turbine, {"tau_ct": 10.0}, ws=9.0)
    wts.power_cap = np.full(3, 4.0e5)
    step(fs)
    np.testing.assert_allclose(wts.sensors.power, 4.0e5)
    np.testing.assert_array_equal(wts.sensors.derate, wts.effective_derate)
    np.testing.assert_array_equal(wts.sensors.power_cap, np.full(3, 4.0e5))
    np.testing.assert_array_equal(wts.sensors.available_power, wts.available_power)
