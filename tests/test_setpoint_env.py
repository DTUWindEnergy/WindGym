"""WindFarmEnv in setpoint mode (``derate_reference="setpoint"``): both farms run
``SetpointPyWakeWindTurbines``, the agent's derate command sets an absolute
power cap, the baseline farm stays uncapped, and the info dict carries the
available power, the cap and the (lagged) effective derate.

3 synthetic derating turbines in a row, dynamiks backend, no turbulence,
baseline farm on; the fast golden-episode settings (dt_sim 1, dt_env 2)."""

import numpy as np
import pytest
from dynamiks.wind_turbines import PyWakeWindTurbines

from WindGym import WindFarmEnv
from WindGym.backend.setpoint_turbines import SetpointPyWakeWindTurbines
from WindGym.core.actuation import compute_derate
from WindGym.core.derating import check_htc_supports_derating
from test_golden_episodes import _derating_turbine, _dynamiks_config, _row

pytestmark = pytest.mark.integration

RATED = 2.0e6


def make_env(derate_reference="setpoint", backend="dynamiks", cfg_extra=None, **env_kwargs):
    turbine = _derating_turbine()
    x, y = _row(turbine, 3)
    cfg = _dynamiks_config(
        derate_action=True,
        derate_min=0.0,
        derate_max=0.8,
        derate_method="absolute",
        derate_reference=derate_reference,
        ActionMethod="wind",
        power_def={"Power_reward": "Baseline", "Power_avg": 5, "Power_scaling": 1.0},
    )
    # 12 m/s: the front turbine sits at rated (2 MW), so caps from commands
    # >= 0.1 are active there; waked turbines see less and may be uncapped.
    cfg["wind"] = {k: (12.0 if "ws" in k else v) for k, v in cfg["wind"].items()}
    cfg.update(cfg_extra or {})
    kw = dict(
        turbine=turbine, x_pos=x, y_pos=y, config=cfg, backend=backend,
        turbtype="None", Baseline_comp=True, yaw_init="Zeros",
        dt_sim=1, dt_env=2, delay=4, n_passthrough=0.5, burn_in_passthroughs=0.05,
        fill_window=1, reset_init=False,
    )
    kw.update(env_kwargs)
    return WindFarmEnv(**kw)


def action(env, derate_cmd):
    """Zero yaw action plus a derate command (fraction of rated) in [0, 0.8]."""
    a = np.zeros(2 * env.n_turb, dtype=np.float32)
    frac = np.asarray(derate_cmd, float) / 0.8  # derate_min 0, derate_max 0.8
    a[env.n_turb:] = 2.0 * frac - 1.0
    return a


# ------------------------------------------------------------------ wiring --
def test_both_farms_get_the_setpoint_class_and_the_baseline_is_uncapped():
    env = make_env()
    env.reset(seed=0)
    agent, base = env.fs.windTurbines, env.fs_baseline.windTurbines
    assert type(agent) is SetpointPyWakeWindTurbines
    assert type(base) is SetpointPyWakeWindTurbines
    assert "derate" in agent.sensors.dict and "derate" in base.sensors.dict
    assert env.derate_rated_power == pytest.approx(RATED)
    _, _, _, _, info = env.step(action(env, 0.3))
    assert np.all(np.isinf(base.power_cap))
    np.testing.assert_array_equal(info["available_powers_baseline"], info["baseline_powers"])
    np.testing.assert_array_equal(base.effective_derate, np.zeros(3))
    for key in ("available_powers", "power_caps", "available_powers_baseline"):
        assert info[key].shape == info["powers"].shape
    assert np.all(np.isfinite(info["available_powers"]))
    assert "available power agent" in info and "power cap agent" in info
    env.close()


def test_command_sets_the_cap_and_tau_zero_derate_is_the_exact_ratio():
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(action(env, [0.0, 0.3, 0.6]))
    np.testing.assert_allclose(info["power cap agent"], (1.0 - np.array([0.0, 0.3, 0.6])) * RATED)
    np.testing.assert_allclose(info["derate command"], [0.0, 0.3, 0.6])
    np.testing.assert_allclose(info["power_caps"][-1], info["power cap agent"])
    avail = info["available_powers"]
    cap = info["power_caps"]
    expected = np.clip(np.where(avail > 0, 1.0 - np.minimum(cap, avail) / avail, 0.0), 0.0, 0.8)
    np.testing.assert_allclose(info["derates"], expected, atol=1e-6)
    np.testing.assert_allclose(info["powers"], np.minimum(cap, avail), rtol=1e-6)
    np.testing.assert_allclose(info["derate agent"], info["derates"][-1], atol=1e-6)
    # headroom: the shadow farm is never capped
    assert np.all(info["baseline_powers"] >= info["powers"] - 1e-6)
    env.close()


def test_tau_ct_lags_the_effective_derate_but_not_the_power():
    env = make_env(dwm_params={"tau_ct": 10.0})
    env.reset(seed=0)
    _, _, _, _, info = env.step(action(env, 0.5))
    avail, cap = info["available_powers"], info["power_caps"]
    d_cmd = np.clip(1.0 - np.minimum(cap, avail) / avail, 0.0, 0.8)
    capped = d_cmd[-1] > 0
    assert capped[0], "front turbine at rated must be capped by a 1 MW cap"
    # the power is capped at once ...
    np.testing.assert_allclose(info["powers"], np.minimum(cap, avail), rtol=1e-6)
    # ... while the effective derate approaches d_cmd with a = 1 - exp(-1/10) per substep
    d = info["derates"]
    assert np.all(d[0][capped] < d_cmd[0][capped]) and np.all(d[-1][capped] > d[0][capped])
    a = 1.0 - np.exp(-1.0 / 10.0)
    for j in range(1, d.shape[0]):
        np.testing.assert_allclose(d[j], d[j - 1] + a * (d_cmd[j] - d[j - 1]), atol=1e-5)
    env.close()


def test_derate_step_sim_slews_the_command_not_the_effective_derate():
    env = make_env(dwm_params={"tau_ct": 10.0})
    env.derate_step_sim = 0.05
    env.reset(seed=0)
    _, _, _, _, info = env.step(action(env, 0.6))
    caps = info["power_caps"]
    cmd_applied = 1.0 - caps / RATED  # the slewed command, fraction of rated
    steps = np.diff(cmd_applied, axis=0)
    assert np.all(steps <= 0.05 + 1e-6) and np.all(steps >= -1e-6)  # float32 info arrays
    np.testing.assert_allclose(cmd_applied[0], 0.05, atol=1e-6)
    np.testing.assert_allclose(info["derate command"], 0.6)  # the raw command is not slewed
    assert np.all(info["derates"][-1] < cmd_applied[-1])  # d_eff lags the slewed command
    env.close()


def test_tau_power_filters_the_available_power_of_both_farms():
    env = make_env(dwm_params={"tau_power": 20.0})
    env.reset(seed=0)
    agent, base = env.fs.windTurbines, env.fs_baseline.windTurbines
    assert np.all(agent.params.a_power < 1.0) and np.all(base.params.a_power < 1.0)
    _, _, _, _, info = env.step(action(env, 0.0))
    np.testing.assert_array_equal(info["available_powers_baseline"], info["baseline_powers"])
    env.close()


def test_reward_change_penalty_uses_the_command_in_setpoint_mode():
    env = make_env(dwm_params={"tau_ct": 100.0})
    env.reset(seed=0)
    env.step(action(env, 0.0))
    env.step(action(env, 0.6))
    # with a 100 s Ct lag the effective derate barely moved, but the command did
    assert np.all(env.current_derate < 0.2)
    np.testing.assert_allclose(env.derate_applied, 0.6)
    np.testing.assert_allclose(env.old_derate, 0.0)
    env.close()


# ------------------------------------------------- available-power obs --
AVAIL_HIST = {"avail_current": False, "avail_rolling_mean": True, "avail_history_N": 3,
              "avail_history_length": 10, "avail_window_length": 1}


def _turb_blocks(env, obs):
    """Per-turbine observation blocks (they come first, farm-level values after)."""
    w = env.farm_measurements.turb_mes[0].observed_variables()
    return obs[: env.n_turb * w].reshape(env.n_turb, w)


def test_avail_mes_is_off_by_default():
    env = make_env()
    env.reset(seed=0)
    assert all(not (t.avail.current or t.avail.rolling_mean)
               for t in env.farm_measurements.turb_mes)
    assert env.farm_measurements.get_avail_turb().size == 0
    env.close()


def test_avail_mes_widens_each_turbine_block_by_its_samples():
    plain = make_env()
    env = make_env(cfg_extra={"avail_mes": AVAIL_HIST})
    assert env.observation_space.shape[0] == plain.observation_space.shape[0] + 3 * env.n_turb
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, *_ = env.step(action(env, 0.3))
    assert obs.shape == env.observation_space.shape
    plain.close()
    env.close()


def test_observed_avail_is_the_scaled_available_power_after_the_derate_slots():
    plain = make_env()
    env = make_env(cfg_extra={"avail_mes": {"avail_current": True}})
    plain.reset(seed=0)
    env.reset(seed=0)
    obs_plain, *_, info_plain = plain.step(action(plain, [0.0, 0.3, 0.6]))
    obs, *_, info = env.step(action(env, [0.0, 0.3, 0.6]))
    np.testing.assert_array_equal(info["powers"], info_plain["powers"])
    # last substep's inertia-filtered available power, scaled like power
    avail = info["available_powers"][-1]
    expected = np.clip(2.0 * avail / env.maxturbpower - 1.0, -1.0, 1.0)
    t0 = env.farm_measurements.turb_mes[0]
    k = sum(len(t0.__getattribute__(f"get_{n}")()) for n in ("ws", "wd", "yaw", "derate"))
    blocks, blocks_plain = _turb_blocks(env, obs), _turb_blocks(plain, obs_plain)
    np.testing.assert_allclose(blocks[:, k], expected, atol=1e-6)
    np.testing.assert_array_equal(np.delete(blocks, k, axis=1), blocks_plain)
    np.testing.assert_array_equal(obs[blocks.size:], obs_plain[blocks_plain.size:])
    plain.close()
    env.close()


def test_avail_mes_outside_setpoint_mode_raises():
    with pytest.raises(ValueError, match="avail_mes"):
        make_env(derate_reference="available", cfg_extra={"avail_mes": {"avail_current": True}})


def test_noise_manager_refuses_the_avail_observation():
    from WindGym.core.measurement_manager import MeasurementManager

    env = make_env(cfg_extra={"avail_mes": {"avail_current": True},
                              "derate_mes": {"derate_current": False}})
    with pytest.raises(NotImplementedError, match="available power"):
        MeasurementManager(env)
    env.close()


# ---------------------------------------------------------- unchanged paths --
def test_available_reference_keeps_the_plain_class():
    env = make_env(derate_reference="available")
    env.reset(seed=0)
    assert type(env.fs.windTurbines) is PyWakeWindTurbines
    assert type(env.fs_baseline.windTurbines) is PyWakeWindTurbines
    _, _, _, _, info = env.step(action(env, 0.3))
    assert "available_powers" not in info and "power cap agent" not in info
    np.testing.assert_allclose(info["derates"], 0.3)
    env.close()


def test_compute_derate_setpoint_reference_passes_the_command_through():
    kw = dict(method="absolute", step_base=None, step_env=0.1, derate_min=0.0,
              derate_max=0.8, reference="setpoint", rated_power=RATED,
              current_powers=np.array([1.0e6, 1.0e6]), current_derate=np.zeros(2),
              step_sim=None, hawc2=False)
    cmd, applied = compute_derate(np.array([-1.0, 0.0]), **kw)
    np.testing.assert_allclose(cmd, [0.0, 0.4])
    np.testing.assert_array_equal(applied, cmd)
    cmd, applied = compute_derate(np.array([-1.0, 0.0]), **{**kw, "step_sim": 0.1})
    np.testing.assert_allclose(applied, [0.0, 0.1])


# ------------------------------------------------------------------ refusals --
def test_pywake_backend_rejects_setpoint_mode():
    with pytest.raises(NotImplementedError, match="setpoint"):
        make_env(backend="pywake", dt_env=1, delay=1)  # pywake needs dt_env == dt_sim


def test_tau_ct_outside_setpoint_mode_raises():
    with pytest.raises(ValueError, match="tau_ct"):
        make_env(derate_reference="available", dwm_params={"tau_ct": 10.0})
    env = make_env(derate_reference="available")
    with pytest.raises(ValueError, match="tau_ct"):
        env.reset(seed=0, options={"dwm_params": {"tau_ct": 10.0}})
    env.close()


def test_hawc2_setpoint_reference_maps_to_constant_104_zero(tmp_path):
    """Any non-'available' reference needs htc constant 104 = 0 (fraction of rated)."""
    pytest.importorskip("wetb")
    import shutil
    from test_hawc2_derating import HTC_SRC, rated_htc

    src = tmp_path / "DTU10mw_derate.htc"
    shutil.copy(HTC_SRC, src)
    htc = rated_htc(src)
    check_htc_supports_derating(htc, "setpoint")
    with pytest.raises(ValueError, match="constant 104"):
        check_htc_supports_derating(htc, "available")
