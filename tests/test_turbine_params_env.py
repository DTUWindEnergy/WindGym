"""WindFarmEnv routing of the turbine-parameter DR keys (plan Phase 1.4).

3 V80 in a row, dynamiks backend, no turbulence, baseline farm on; the
fast golden-episode settings (dt_sim 1, dt_env 2, short burn-in)."""

import numpy as np
import pytest
from dynamiks.wind_turbines import PyWakeWindTurbines
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.backend.param_turbines import ParamPyWakeWindTurbines
from test_golden_episodes import _dynamiks_config, _row

pytestmark = pytest.mark.integration

PERTURBED = {"cp_gain": [1.2, 1.0, 0.8], "yaw_exp": 1.0, "ct_gain": 0.9,
             "tau_yaw": 30.0, "delay_yaw": 10.0, "tau_power": 20.0}


def make_env(action_method="wind", **env_kwargs):
    x, y = _row(V80(), 3)
    cfg = _dynamiks_config(
        power_def={"Power_reward": "Wake_recovery", "Power_avg": 5, "Power_scaling": 1.0},
        ActionMethod=action_method,
    )
    return WindFarmEnv(
        turbine=V80(), x_pos=x, y_pos=y, config=cfg, backend="dynamiks",
        turbtype="None", Baseline_comp=True, yaw_init="Zeros",
        dt_sim=1, dt_env=2, delay=4, n_passthrough=0.5, burn_in_passthroughs=0.05,
        fill_window=1, reset_init=False, **env_kwargs,
    )


def test_keys_route_to_agent_and_baseline_farms():
    env = make_env(dwm_params=PERTURBED)
    assert env.turbine_params is None  # nothing resolved before the first reset
    env.reset(seed=0)
    agent, base = env.fs.windTurbines, env.fs_baseline.windTurbines
    assert type(agent) is ParamPyWakeWindTurbines and type(base) is ParamPyWakeWindTurbines
    for k in PERTURBED:
        np.testing.assert_array_equal(agent.params.values[k], base.params.values[k])
    tp = env.turbine_params
    np.testing.assert_array_equal(tp["cp_gain"], [1.2, 1.0, 0.8])
    np.testing.assert_array_equal(tp["ct_gain"], [0.9, 0.9, 0.9])
    assert set(tp) == {"yaw_exp", "cp_gain", "ct_gain", "tau_yaw", "delay_yaw", "tau_power"}
    # both farms were attached (step_actuators first in the handler lists)
    assert env.fs.step_handlers[0] == agent.step_actuators
    assert env.fs_baseline.step_handlers[0] == base.step_actuators
    env.close()


def test_nominal_reset_uses_plain_class():
    env = make_env()
    env.reset(seed=0)
    assert type(env.fs.windTurbines) is PyWakeWindTurbines
    assert type(env.fs_baseline.windTurbines) is PyWakeWindTurbines
    assert env.turbine_params is not None and np.all(env.turbine_params["cp_gain"] == 1.0)
    env.close()


def test_cp_gain_scales_power_exactly_without_changing_the_flow():
    nominal = make_env()
    gained = make_env(dwm_params={"cp_gain": [1.2, 1.0, 1.0]})
    nominal.reset(seed=3)
    gained.reset(seed=3)
    p0 = nominal.fs.windTurbines.power()
    p1 = gained.fs.windTurbines.power()
    np.testing.assert_allclose(p1, p0 * np.array([1.2, 1.0, 1.0]), rtol=1e-12)
    np.testing.assert_array_equal(gained.fs.windTurbines.ct(), nominal.fs.windTurbines.ct())
    # the baseline farm received the same draw
    np.testing.assert_allclose(
        gained.fs_baseline.windTurbines.power(),
        nominal.fs_baseline.windTurbines.power() * np.array([1.2, 1.0, 1.0]), rtol=1e-12)
    nominal.close(); gained.close()


def test_reset_options_long_array_sliced_short_raises_scalar_broadcast():
    env = make_env()
    env.reset(seed=0, options={"dwm_params": {"cp_gain": [1.1, 1.1, 1.1, 1.1, 1.1]}})
    np.testing.assert_array_equal(env.turbine_params["cp_gain"], [1.1, 1.1, 1.1])
    env.reset(seed=0, options={"dwm_params": {"ct_gain": 0.9}})
    np.testing.assert_array_equal(env.turbine_params["ct_gain"], [0.9, 0.9, 0.9])
    with pytest.raises(ValueError, match="cp_gain.*len 2.*n_turb 3"):
        env.reset(seed=0, options={"dwm_params": {"cp_gain": [1.1, 1.1]}})
    env.close()


def test_yaw_lag_visible_in_info_vs_command():
    env = make_env(action_method="wind", dwm_params={"tau_yaw": 60.0})
    env.reset(seed=0)
    for _ in range(3):
        _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    cmd, act = env.yaw_command, np.asarray(info["yaw angles agent"])
    assert np.all(cmd > 0) and np.all(act > 0)
    assert np.all(act < cmd), (cmd, act)  # actual lags the command
    np.testing.assert_array_equal(act, env.fs.windTurbines.yaw)
    env.close()


def test_wd_ramp_keeps_tracking_error_continuous():
    # delta-yaw action held at 0 -> the command only moves with the wd frame;
    # with a (slow) actuator the command/actual error must stay zero, i.e. the
    # frame shift is applied to the actual yaw and the dead-time buffer too.
    env = make_env(action_method="yaw", dwm_params={"tau_yaw": 300.0, "delay_yaw": 3.0},
                   wd_function=lambda t: 270.0 + 0.2 * t)
    env.reset(seed=0)
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        err = env.yaw_command - np.asarray(info["yaw angles agent"])
        assert np.abs(err).max() < 1e-9, err
    assert env.wd > 270.5  # the ramp actually moved the frame
    env.close()


def test_hawc2_path_rejects_turbine_keys():
    x, y = _row(V80(), 3)
    with pytest.raises(ValueError, match="HAWC2 turbines are the target model"):
        WindFarmEnv(turbine=V80(), x_pos=x, y_pos=y, config=_dynamiks_config(),
                    backend="dynamiks", turbtype="None", HTC_path="/model/htc/x.htc",
                    reset_init=False, dwm_params={"cp_gain": 1.1})


def test_pywake_backend_rejects_turbine_keys():
    x, y = _row(V80(), 3)
    with pytest.raises(NotImplementedError, match="dynamiks"):
        WindFarmEnv(turbine=V80(), x_pos=x, y_pos=y, config=_dynamiks_config(),
                    backend="pywake", turbtype="None", dt_sim=1, dt_env=1,
                    reset_init=False, dwm_params={"cp_gain": 1.1})


def test_unknown_key_message_unchanged():
    x, y = _row(V80(), 3)
    with pytest.raises(ValueError, match="Unknown dwm_params keys"):
        WindFarmEnv(turbine=V80(), x_pos=x, y_pos=y, config=_dynamiks_config(),
                    backend="dynamiks", turbtype="None", reset_init=False,
                    dwm_params={"bogus": 1.0})
    env = make_env()
    with pytest.raises(ValueError, match="Unknown dwm_params keys in reset options"):
        env.reset(seed=0, options={"dwm_params": {"bogus": 1.0}})


# --- windtunnel: validation split -------------------------------------------
# windtunnel validates dwm_params VALUES (validate_dwm_params: finite, > 0,
# scalar). Turbine keys must bypass that check (yaw_exp may be negative, every
# key may be a per-turbine array) and be validated by normalize_turbine_params.


def test_negative_yaw_exp_and_arrays_pass_alongside_dwm_keys():
    env = make_env(dwm_params={"yaw_exp": -0.17, "ct_gain": [0.9, 1.0, 1.1], "k1": 0.1})
    env.reset(seed=0)
    np.testing.assert_array_equal(env.turbine_params["yaw_exp"], [-0.17] * 3)
    np.testing.assert_array_equal(env.turbine_params["ct_gain"], [0.9, 1.0, 1.1])
    assert env.active_dwm_params["k1"] == 0.1
    env.close()


def test_turbine_value_out_of_bounds_raises_at_construction():
    with pytest.raises(ValueError, match="cp_gain.*outside"):
        make_env(dwm_params={"cp_gain": 5.0})  # reset_init=False: no reset ran


def test_dwm_values_still_validated_next_to_turbine_keys():
    with pytest.raises(ValueError, match=r"dwm_params\['k1'\].*> 0"):
        make_env(dwm_params={"cp_gain": 1.1, "k1": -1.0})
    env = make_env()
    with pytest.raises(ValueError, match=r"dwm_params\['k1'\] in reset options"):
        env.reset(seed=0, options={"dwm_params": {"yaw_exp": -0.1, "k1": -1.0}})
    env.close()
