"""ParamPyWakeWindTurbines: per-turbine gains, yaw actuator lag/dead time and
power lag on top of dynamiks' PyWakeWindTurbines. No DWM: a fake flow
simulation (time, dt, step_handlers) and hand-set rotor-average sensors."""

from types import SimpleNamespace

import numpy as np
import pytest
from dynamiks.wind_turbines import PyWakeWindTurbines
from py_wake.deficit_models.utils import ct2a_madsen
from py_wake.examples.data.hornsrev1 import V80

from WindGym.backend.param_turbines import ParamPyWakeWindTurbines
from WindGym.core.turbine_params import normalize_turbine_params

DT = 5.0
X = np.array([0.0, 400.0, 800.0])
Y = np.zeros(3)


def _set_inflow(wts, ws_waked, ws_free=None):
    if "rotor_avg_windspeed" not in wts.sensors.dict:
        wts.add_sensor("rotor_avg_windspeed", expose=True, ext_lst=["u", "v", "w"])
        wts.add_sensor("rotor_avg_freestream", expose=True, ext_lst=["u", "v", "w"])
    ws_waked = np.broadcast_to(np.asarray(ws_waked, float), (3,))
    ws_free = (
        ws_waked
        if ws_free is None
        else np.broadcast_to(np.asarray(ws_free, float), (3,))
    )
    wts.rotor_avg_windspeed = np.c_[ws_waked, np.zeros(3), np.zeros(3)]
    wts.rotor_avg_freestream = np.c_[ws_free, np.zeros(3), np.zeros(3)]


def make_plain(ws=9.0):
    wts = PyWakeWindTurbines(x=X, y=Y, windTurbine=V80())
    _set_inflow(wts, ws)
    return wts


def make_param(params, ws=9.0, dt=DT):
    tp = normalize_turbine_params(params, n_turb=3, dt_sim=dt)
    wts = ParamPyWakeWindTurbines(x=X, y=Y, windTurbine=V80(), params=tp)
    _set_inflow(wts, ws)
    fs = SimpleNamespace(time=0.0, dt=dt, step_handlers=list(wts.step_handlers))
    wts.attach_to(fs)
    return wts, fs


def step(fs):
    """One sim step: the fake FlowSimulation runs every handler in order."""
    fs.time = fs.time + fs.dt
    for h in fs.step_handlers:
        h(fs)


# ---------------------------------------------------------------- nominal ---
@pytest.mark.parametrize("yaw", [0.0, 10.0, -20.0])
def test_nominal_params_are_bitwise_identical_to_plain(yaw):
    plain = make_plain()
    wts, fs = make_param({})
    plain.yaw = np.full(3, yaw)
    wts.reset_yaw_state(np.full(3, yaw))
    step(fs)
    np.testing.assert_array_equal(wts.power(), plain.power())
    np.testing.assert_array_equal(
        wts.power(include_wakes=False), plain.power(include_wakes=False)
    )
    np.testing.assert_array_equal(wts.ct(), plain.ct())
    np.testing.assert_array_equal(
        wts.rotor_avg_induction(), plain.rotor_avg_induction()
    )
    np.testing.assert_array_equal(wts.yaw_tilt(), plain.yaw_tilt())
    np.testing.assert_array_equal(wts.yaw, plain.yaw)


def test_handlers_registered():
    wts, fs = make_param({})
    assert fs.step_handlers[0] == wts.step_actuators  # before update_particles
    assert wts.step_handlers[0] == wts.step_power_lag  # before the sensors dumper
    assert wts.step_handlers[-1] is wts.sensors


# ------------------------------------------------------------------ gains ---
def test_cp_gain_scales_power_both_branches():
    plain = make_plain()
    wts, fs = make_param({"cp_gain": [1.2, 1.0, 0.8]})
    step(fs)
    g = np.array([1.2, 1.0, 0.8])
    np.testing.assert_allclose(wts.power(), plain.power() * g, rtol=1e-14)
    np.testing.assert_allclose(
        wts.power(include_wakes=False), plain.power(include_wakes=False) * g, rtol=1e-14
    )
    np.testing.assert_array_equal(wts.ct(), plain.ct())


def test_ct_gain_scales_ct_and_induction():
    plain = make_plain()
    wts, fs = make_param({"ct_gain": [0.9, 1.0, 1.1]})
    g = np.array([0.9, 1.0, 1.1])
    np.testing.assert_allclose(wts.ct(), plain.ct() * g, rtol=1e-14)
    np.testing.assert_allclose(
        wts.rotor_avg_induction(), ct2a_madsen(plain.ct() * g), rtol=1e-14
    )
    np.testing.assert_array_equal(wts.power(), plain.power())


def test_yaw_exp_cos_law():
    plain = make_plain()
    plain.yaw = np.array([20.0, 20.0, 0.0])
    wts, fs = make_param({"yaw_exp": [1.0, 0.0, 1.0]})
    wts.reset_yaw_state(np.array([20.0, 20.0, 0.0]))
    step(fs)
    expect = plain.power() * np.array([np.cos(np.deg2rad(20.0)), 1.0, 1.0])
    np.testing.assert_allclose(wts.power(), expect, rtol=1e-14)
    assert wts.power()[2] == plain.power()[2]  # exact no-op at yaw 0


def test_scalar_idx_through_getitem_proxy():
    wts, fs = make_param({"cp_gain": [1.2, 1.0, 0.8], "ct_gain": 0.9})
    step(fs)
    for i in range(3):
        assert wts[i].power()[0] == wts.power()[i]
        assert wts[i].ct()[0] == wts.ct()[i]
        assert wts[i].rotor_avg_induction()[0] == wts.rotor_avg_induction()[i]


# ------------------------------------------------------------ yaw actuator ---
def test_yaw_lag_exact_first_order_discretisation():
    wts, fs = make_param({"tau_yaw": 30.0})
    wts.reset_yaw_state(np.zeros(3))
    wts.yaw = np.full(3, 10.0)  # command
    a = 1.0 - np.exp(-DT / 30.0)
    act = 0.0
    for _ in range(6):
        step(fs)
        act = act + a * (10.0 - act)
        np.testing.assert_allclose(wts.yaw, np.full(3, act), rtol=1e-15)
        np.testing.assert_allclose(wts.sensors.yaw, np.full(3, act), rtol=1e-15)
        np.testing.assert_allclose(wts.yaw_tilt()[0], np.full(3, act), rtol=1e-15)
    np.testing.assert_array_equal(wts.yaw_command, np.full(3, 10.0))
    assert 0 < act < 10


def test_dead_time_shifts_by_whole_steps_per_turbine():
    wts, fs = make_param({"delay_yaw": [0.0, 5.0, 10.0]})
    wts.reset_yaw_state(np.zeros(3))
    wts.yaw = np.full(3, 10.0)
    step(fs)
    np.testing.assert_array_equal(wts.yaw, [10.0, 0.0, 0.0])
    step(fs)
    np.testing.assert_array_equal(wts.yaw, [10.0, 10.0, 0.0])
    step(fs)
    np.testing.assert_array_equal(wts.yaw, [10.0, 10.0, 10.0])


def test_setter_twice_per_step_is_not_pushed_twice():
    wts, fs = make_param({"delay_yaw": 5.0})
    wts.reset_yaw_state(np.zeros(3))
    wts.yaw = np.full(3, 5.0)
    wts.yaw = np.full(3, 10.0)  # env writes the command twice per step
    step(fs)
    np.testing.assert_array_equal(wts.yaw, 0.0)  # one-step dead time
    step(fs)
    np.testing.assert_array_equal(wts.yaw, 10.0)  # the LAST write of that step


def test_shift_yaw_frame_moves_command_actual_and_history():
    wts, fs = make_param({"tau_yaw": 30.0, "delay_yaw": 5.0})
    wts.reset_yaw_state(np.zeros(3))
    wts.yaw = np.full(3, 10.0)
    step(fs)
    step(fs)
    err_before = wts.yaw_command - wts.yaw
    wts.shift_yaw_frame(3.0)
    np.testing.assert_allclose(wts.yaw_command - wts.yaw, err_before, atol=1e-12)
    cmd, act = wts.yaw_command.copy(), wts.yaw.copy()
    # the delayed command still in the buffer was shifted too: no transient
    # beyond the lag's own response when the (shifted) command is held
    wts.yaw = cmd
    for _ in range(100):  # 500 s = 16.7 tau -> residual < 1e-6 deg
        step(fs)
    np.testing.assert_allclose(wts.yaw, cmd, atol=1e-5)
    assert np.all(act != cmd)


def test_reset_yaw_state_has_no_transient():
    wts, fs = make_param({"tau_yaw": 60.0, "delay_yaw": 10.0})
    wts.reset_yaw_state(np.array([5.0, -5.0, 0.0]))
    np.testing.assert_array_equal(wts.yaw, [5.0, -5.0, 0.0])
    for _ in range(4):
        step(fs)
        np.testing.assert_array_equal(wts.yaw, [5.0, -5.0, 0.0])


# -------------------------------------------------------------- power lag ---
def test_power_lag_waked_only_and_per_turbine():
    wts, fs = make_param({"tau_power": [20.0, 0.0, 0.0]}, ws=8.0)
    wts.reset_yaw_state(np.zeros(3))
    step(fs)  # initialises the lag state at ws 8
    p8 = V80().power(np.array([8.0]))[0]
    p10 = V80().power(np.array([10.0]))[0]
    np.testing.assert_allclose(wts.power(), p8)
    _set_inflow(wts, 10.0)
    # before the next step the live turbines already see ws 10; the lagged one holds
    np.testing.assert_allclose(wts.power()[1:], p10)
    np.testing.assert_allclose(wts.power()[0], p8)
    np.testing.assert_allclose(wts.power(include_wakes=False), p10)  # quasi-static
    np.testing.assert_allclose(wts.ct(), V80().ct(np.array([10.0]))[0])  # ct not lagged
    a = 1.0 - np.exp(-DT / 20.0)
    state = p8
    for _ in range(5):
        step(fs)
        state = state + a * (p10 - state)
        np.testing.assert_allclose(wts.power()[0], state, rtol=1e-13)
        np.testing.assert_allclose(wts.power()[1:], p10)
    assert p8 < wts.power()[0] < p10


def test_step_power_lag_without_attach_raises():
    tp = normalize_turbine_params({"tau_power": 20.0}, n_turb=3, dt_sim=DT)
    wts = ParamPyWakeWindTurbines(x=X, y=Y, windTurbine=V80(), params=tp)
    _set_inflow(wts, 9.0)
    with pytest.raises(RuntimeError, match="attach_to"):
        wts.step_power_lag(SimpleNamespace(time=DT, dt=DT))


def test_attach_to_checks_dt():
    tp = normalize_turbine_params({"tau_yaw": 20.0}, n_turb=3, dt_sim=DT)
    wts = ParamPyWakeWindTurbines(x=X, y=Y, windTurbine=V80(), params=tp)
    with pytest.raises(AssertionError):
        wts.attach_to(SimpleNamespace(time=0.0, dt=1.0, step_handlers=[]))


# --------------------------------------------------------------- yaw_bias ---
@pytest.mark.parametrize("yaw", [0.0, 10.0, -20.0])
def test_yaw_bias_zero_is_bitwise_identical(yaw):
    a, fa = make_param({})
    b, fb = make_param({"yaw_bias": 0.0})
    for w in (a, b):
        w.reset_yaw_state(yaw)
    assert np.array_equal(a.power(), b.power()) and np.array_equal(a.ct(), b.ct())
    assert np.array_equal(a.yaw_tilt(), b.yaw_tilt())


@pytest.mark.parametrize("bias", [1.0, -2.5])
def test_yaw_bias_turns_the_physics_not_the_command_frame(bias):
    """The flow sees yaw - bias (power, Ct, the particles' yaw); the yaw sensor,
    which the env observes and its greedy controllers steer, stays commanded."""
    w, fs = make_param({"yaw_bias": bias})
    ref, _ = make_param({})
    w.reset_yaw_state(15.0)
    ref.reset_yaw_state(15.0 - bias)
    np.testing.assert_allclose(w.yaw, 15.0)                       # commanded frame
    np.testing.assert_allclose(w.sensors.yaw, 15.0)
    np.testing.assert_allclose(w.yaw_tilt()[0], 15.0 - bias)      # what the wake is emitted with
    np.testing.assert_allclose(w.power(), ref.power())            # = a turbine at yaw - bias
    np.testing.assert_allclose(w.ct(), ref.ct())
    # stepping keeps both frames: the command moves, the physics follows minus the bias
    w.yaw = np.full(3, 5.0)
    step(fs)
    np.testing.assert_allclose(w.yaw, 5.0)
    np.testing.assert_allclose(w.yaw_tilt()[0], 5.0 - bias)


def test_yaw_bias_at_zero_command_is_a_misaligned_rotor():
    """Commanded 0 with a bias loses power like a rotor yawed by -bias."""
    w, _ = make_param({"yaw_bias": 8.0})
    ref, _ = make_param({})
    w.reset_yaw_state(0.0)
    ref.reset_yaw_state(0.0)
    assert np.all(w.power() < ref.power())
