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


# ------------------------------------------------ share inner loop (round 3) --
P_REF = 3.0e6


def make_share_env(p_ref=P_REF, turbtype="Random", cfg_extra=None, **kw):
    """Setpoint mode, derate_method 'share', tracking a constant P_ref, dt_env 2 (no delay)."""
    extra = dict(derate_method="share", Track_power=True,
                 power_def={"Power_reward": "None", "Power_avg": 1, "Power_scaling": 1.0})
    extra.update(cfg_extra or {})
    return make_env(cfg_extra=extra, turbtype=turbtype, delay=2,
                    power_ref_function=lambda t, env: p_ref, **kw)


def share_action(env, derate_raw):
    a = np.zeros(2 * env.n_turb, dtype=np.float32)
    a[env.n_turb:] = derate_raw
    return a


@pytest.mark.parametrize("raw, shares", [([1.0, -1.0, -1.0], [1.0, 0.0, 0.0]),
                                         ([0.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3])])
def test_share_mode_delivers_the_setpoint_every_sim_step_under_turbulence(raw, shares):
    # the paper's turbine parameters: the inertia filters the available power
    env = make_share_env(dwm_params={"tau_power": 6.0, "tau_ct": 10.0})
    env.reset(seed=3)
    errs, caps_moved = [], []
    for _ in range(10):
        _, _, _, _, info = env.step(share_action(env, raw))
        p_farm = info["powers"].sum(axis=1)
        assert np.all(info["available_powers"].sum(axis=1) > P_REF)  # tracking regime
        errs.append(np.abs(p_farm - P_REF) / (env.n_turb * RATED))
        caps_moved.append(np.ptp(info["power_caps"], axis=0))
        np.testing.assert_allclose(info["derate shares"], shares, atol=1e-6)
    errs = np.concatenate(errs)
    # P_hat is one sim step old: the only error is the change of the uncapped
    # turbines' available power within one second
    assert np.max(errs) < 0.02, np.max(errs)
    assert np.mean(errs) < 0.005, np.mean(errs)
    # the caps follow the turbulence within an env step (an inner loop, not a held command)
    assert np.max(np.concatenate(caps_moved)) > 1e3
    env.close()


def _feasible_low_ref():
    """A reference T1 alone cannot reach (it sheds at most 0.8 of its available
    power) but the farm can: halfway between the farm floor and T1's limit."""
    probe = make_share_env(p_ref=50e6, turbtype="None")
    _, info = probe.reset(seed=0)
    avail = np.asarray(info["available power agent"], float)
    probe.close()
    t1_limit = avail.sum() - 0.8 * avail[0]
    return 0.5 * (0.2 * avail.sum() + t1_limit)


def test_share_mode_t1_only_spills_past_the_floor_to_the_others():
    p_ref = _feasible_low_ref()
    env = make_share_env(p_ref=p_ref, turbtype="None")
    env.reset(seed=0)
    for _ in range(3):
        _, _, _, _, info = env.step(share_action(env, [1.0, -1.0, -1.0]))
    p, avail = info["powers"][-1], info["available_powers"][-1]
    np.testing.assert_allclose(p[0], 0.2 * avail[0], rtol=1e-3)
    assert np.all(p[1:] < avail[1:] - 1e3)
    np.testing.assert_allclose(p.sum(), p_ref, rtol=1e-3)
    env.close()


def test_share_mode_no_deficit_leaves_the_turbines_uncapped():
    env = make_share_env(p_ref=50e6, turbtype="None")
    env.reset(seed=0)
    _, _, _, _, info = env.step(share_action(env, [1.0, -1.0, -1.0]))
    np.testing.assert_allclose(info["power_caps"], RATED)
    np.testing.assert_allclose(env.derate_applied, 0.0)
    env.close()


def test_share_mode_change_penalty_acts_on_the_shares():
    env = make_share_env()
    env.reset(seed=3)
    np.testing.assert_allclose(env._penalised_derate, 1 / 3)  # equal shares before any action
    env.step(share_action(env, [1.0, -1.0, -1.0]))
    env.step(share_action(env, [1.0, -1.0, -1.0]))
    # same shares, different caps: no change for the penalty
    np.testing.assert_allclose(env.old_derate, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(env._penalised_derate, [1.0, 0.0, 0.0])
    env.close()


def test_share_mode_honours_derate_step_sim():
    p_ref = _feasible_low_ref()
    env = make_share_env(p_ref=p_ref, turbtype="None", cfg_extra={"derate_step_sim": 0.05})
    env.reset(seed=0)
    _, _, _, _, info = env.step(share_action(env, [1.0, -1.0, -1.0]))
    # 2 sim steps of at most 0.05 each from 0
    assert np.all(env.derate_applied <= 0.1 + 1e-9)
    assert env.derate_applied[0] == pytest.approx(0.1)
    np.testing.assert_allclose(info["derate shares"], [1.0, 0.0, 0.0])
    for _ in range(15):  # slewed up to the inner loop's target: T1 at its floor
        _, _, _, _, info = env.step(share_action(env, [1.0, -1.0, -1.0]))
    np.testing.assert_allclose(info["powers"][-1].sum(), p_ref, rtol=1e-3)
    env.close()


def test_share_mode_needs_tracking():
    with pytest.raises(ValueError, match="share.*Track_power"):
        make_env(cfg_extra={"derate_method": "share"})


# ---------------------------------------------------------- yaw deadband --
def _yaw_action(env, yaw_deg):
    a = action(env, 0.0)
    a[: env.n_turb] = np.asarray(yaw_deg) / env.yaw_max  # symmetric yaw range
    return a


def test_yaw_deadband_holds_small_targets_and_releases_large_ones():
    env = make_env(cfg_extra={"yaw_deadband": 0.375}, yaw_step_sim=1.0)
    env.reset(seed=0)
    env.step(_yaw_action(env, [0.3, -0.3, 0.0]))
    np.testing.assert_allclose(env.yaw_command, 0.0, atol=1e-6)
    env.step(_yaw_action(env, [1.0, -0.3, 0.0]))
    np.testing.assert_allclose(env.yaw_command, [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(env.fs_baseline.windTurbines.yaw, 0.0, atol=1e-6)
    env.close()


def test_yaw_deadband_default_off():
    env = make_env(yaw_step_sim=1.0)
    env.reset(seed=0)
    env.step(_yaw_action(env, [0.3, -0.3, 0.0]))
    np.testing.assert_allclose(env.yaw_command, [0.3, -0.3, 0.0], atol=1e-5)
    env.close()


# ------------------------------------------------ error integral / history --
def test_error_integral_and_history_widen_the_tracking_tail():
    base = make_share_env(turbtype="None")
    feat = make_share_env(turbtype="None", cfg_extra={"track_def": {
        "track_obs_error_integral": {"tau_s": 60}, "track_obs_error_history": 6}})
    w0, w1 = base.observation_space.shape[0], feat.observation_space.shape[0]
    assert w1 == w0 + 7
    assert feat.farm_measurements.track_error_integral_alpha == pytest.approx(2 / 60)
    obs, _ = feat.reset(seed=0)
    assert obs.shape == (w1,)
    obs, _, _, _, info = feat.step(share_action(feat, [0.0, 0.0, 0.0]))
    tail = obs[-7:]
    assert np.all(np.abs(tail) <= 1.0)
    # history newest first: the latest error, scaled by 0.05 * farm max power
    fm = feat.farm_measurements
    e = fm._track_error_val
    assert tail[1] == pytest.approx(np.clip(e / (0.05 * fm.power_max * 3), -1, 1), abs=1e-6)
    base.close()
    feat.close()


def test_share_mode_zero_share_turbines_stay_uncapped():
    # T1-only: T2/T3 give nothing, so they are not capped at last second's
    # available power (that would clip gusts and add a spurious derate)
    # P_ref 5 MW: a deficit of ~0.7 MW (available ~5.7 MW) that T1 (headroom ~1.5 MW) covers alone
    env = make_share_env(p_ref=5.0e6, dwm_params={"tau_power": 6.0, "tau_ct": 10.0})
    env.reset(seed=3)
    for _ in range(5):
        _, _, _, _, info = env.step(share_action(env, [1.0, -1.0, -1.0]))
        deficit = info["available_powers"].sum(axis=1) - 5.0e6
        assert np.all((deficit > 0) & (deficit < 0.8 * info["available_powers"][:, 0]))
        np.testing.assert_allclose(info["power_caps"][:, 1:], RATED)
        np.testing.assert_allclose(info["powers"][:, 1:], info["available_powers"][:, 1:])
        assert np.all(info["power_caps"][:, 0] < RATED)
    np.testing.assert_allclose(env.derate_applied[1:], 0.0)
    env.close()


@pytest.mark.parametrize("share_reference, lags", [("observed", True), ("scored", False), (None, True)])
def test_share_reference_observed_lags_one_env_step(share_reference, lags):
    # "observed" (default): the loop delivers the reference the agent has seen,
    # i.e. one env step behind the one the step is scored against; "scored" has
    # that one-step preview
    hi, lo = 5.2e6, 4.8e6
    extra = dict(derate_method="share", Track_power=True,
                 power_def={"Power_reward": "None", "Power_avg": 1, "Power_scaling": 1.0})
    if share_reference is not None:
        extra["share_reference"] = share_reference
    env = make_env(cfg_extra=extra, turbtype="None", delay=2,
                   power_ref_function=lambda t, env: hi if t < 7 else lo)
    env.reset(seed=0)
    delivered, scored = [], []
    for _ in range(8):
        _, _, _, _, info = env.step(share_action(env, [0.0, 0.0, 0.0]))
        delivered.append(info["powers"][-1].sum())
        scored.append(info["Power reference"])
    k = scored.index(lo)  # first step scored against the lower reference
    assert k >= 1
    np.testing.assert_allclose(delivered[k], hi if lags else lo, rtol=2e-3)
    np.testing.assert_allclose(delivered[k + 1], lo, rtol=2e-3)
    env.close()


# -------------------------------------------- relative derate (round 4) --
def make_relative_env(cfg_extra=None, **kw):
    """Setpoint mode, derate_method 'relative' (cap_i = (1 - d_i) * P_hat_i), dt_env 2."""
    extra = dict(derate_method="relative")
    extra.update(cfg_extra or {})
    return make_env(cfg_extra=extra, delay=2, **kw)


def relative_action(env, d):
    """Zero yaw action plus a relative derate d in [0, 0.8] (derate_min 0, derate_max 0.8)."""
    a = np.zeros(2 * env.n_turb, dtype=np.float32)
    a[env.n_turb:] = 2.0 * np.asarray(d, float) / 0.8 - 1.0
    return a


def test_relative_mode_caps_each_turbine_at_one_minus_d_of_its_available_power():
    env = make_relative_env(turbtype="Random", dwm_params={"tau_power": 6.0, "tau_ct": 10.0})
    env.reset(seed=3)
    d = np.array([0.4, 0.3, 0.2])
    for _ in range(4):
        p_hat_prev = np.array(env.current_available_power, dtype=np.float64)
        _, _, _, _, info = env.step(relative_action(env, d))
        avail, caps = info["available_powers"], info["power_caps"]
        # P_hat is one sim step old: the cap at substep j uses substep j-1's available power
        p_hat = np.vstack([p_hat_prev, avail[:-1]])
        np.testing.assert_allclose(caps, (1.0 - d) * p_hat, rtol=1e-5)
        np.testing.assert_allclose(info["derate command"], d, atol=1e-6)
    env.close()


def test_relative_mode_zero_d_leaves_the_turbine_uncapped():
    env = make_relative_env(turbtype="Random")
    env.reset(seed=3)
    _, _, _, _, info = env.step(relative_action(env, [0.0, 0.3, 0.0]))
    np.testing.assert_allclose(info["power_caps"][:, [0, 2]], RATED)
    np.testing.assert_allclose(env.derate_applied[[0, 2]], 0.0)
    assert np.all(info["power_caps"][:, 1] < info["available_powers"][:, 1])
    env.close()


def test_relative_mode_equal_d_is_proportional_derating():
    """The same d on every turbine delivers P_ref when d = 1 - P_ref / sum(P_hat)."""
    env = make_relative_env(turbtype="None")
    _, info = env.reset(seed=0)
    p_hat = np.asarray(info["available power agent"], float)
    p_ref = 0.7 * p_hat.sum()
    d = 1.0 - p_ref / p_hat.sum()
    for _ in range(3):
        _, _, _, _, info = env.step(relative_action(env, np.full(3, d)))
    np.testing.assert_allclose(info["powers"][-1].sum(), p_ref, rtol=1e-3)
    np.testing.assert_allclose(info["powers"][-1], 0.7 * info["available_powers"][-1], rtol=1e-3)
    env.close()


def test_relative_mode_honours_derate_step_sim_and_penalises_d():
    env = make_relative_env(turbtype="None", cfg_extra={"derate_step_sim": 0.05})
    env.reset(seed=0)
    np.testing.assert_allclose(env._penalised_derate, 0.0)  # no derate before any action
    d = np.array([0.5, 0.5, 0.5])
    _, _, _, _, info = env.step(relative_action(env, d))
    # 2 sim steps of at most 0.05 (fraction of rated) each from 0
    assert np.all(env.derate_applied <= 0.1 + 1e-9)
    assert env.derate_applied[0] == pytest.approx(0.1)
    np.testing.assert_allclose(info["derate command"], d, atol=1e-6)  # d itself is not slewed
    # the change penalty acts on d, the agent's decision, not on the slewed cap
    np.testing.assert_allclose(env._penalised_derate, d, atol=1e-6)
    env.step(relative_action(env, d))
    np.testing.assert_allclose(env.old_derate, d, atol=1e-6)
    for _ in range(15):  # slewed up to (1 - d) P_hat
        _, _, _, _, info = env.step(relative_action(env, d))
    np.testing.assert_allclose(info["powers"][-1], 0.5 * info["available_powers"][-1], rtol=1e-3)
    env.close()


def test_relative_mode_needs_the_setpoint_reference():
    with pytest.raises(ValueError, match="relative.*setpoint"):
        make_env(derate_reference="rated", cfg_extra={"derate_method": "relative"})
