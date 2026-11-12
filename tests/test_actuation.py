"""Unit tests for WindGym.core.actuation (pure yaw / derate command helpers)."""

import numpy as np
import pytest

from WindGym.core.actuation import (
    allocate_deficit,
    compute_derate,
    latch_yaw_action,
    lowpass_step,
    shares_from_action,
    step_yaw_command,
    tilt_from_action,
)

pytestmark = pytest.mark.unit

YAW_KW = dict(yaw_step_sim=1.0, yaw_min=-30.0, yaw_max=30.0, rate_limit=True)


def test_yaw_method_spends_budget_one_step_at_a_time():
    cmd = np.zeros(1)
    remaining = np.array([2.5], dtype=np.float32)
    seen = []
    for _ in range(4):
        cmd, remaining = step_yaw_command("yaw", None, cmd, remaining, **YAW_KW)
        seen.append(float(cmd[0]))
    assert seen == [1.0, 2.0, 2.5, 2.5]
    assert remaining[0] == 0.0
    assert remaining.dtype == np.float32


def test_yaw_method_clips_to_bounds():
    cmd, remaining = step_yaw_command(
        "yaw", None, np.array([29.5]), np.array([5.0]), **YAW_KW
    )
    assert cmd[0] == 30.0
    assert remaining[0] == 4.0  # the budget still shrinks by yaw_step_sim


def test_wind_method_maps_action_and_rate_limits():
    action = np.array([1.0, -1.0, 0.0])
    cmd, _ = step_yaw_command("wind", action, np.zeros(3), None, **YAW_KW)
    np.testing.assert_allclose(cmd, [1.0, -1.0, 0.0])


def test_wind_method_without_rate_limit_jumps_to_setpoint():
    kw = dict(YAW_KW, rate_limit=False)
    action = np.array([1.0, -1.0, 0.0])
    cmd, remaining = step_yaw_command("wind", action, np.zeros(3), None, **kw)
    np.testing.assert_allclose(cmd, [30.0, -30.0, 0.0])
    assert remaining is None


def test_absolute_not_implemented_and_unknown_rejected():
    with pytest.raises(NotImplementedError, match="absolute method is not implemented"):
        step_yaw_command("absolute", np.zeros(1), np.zeros(1), None, **YAW_KW)
    with pytest.raises(ValueError, match="ActionMethod must be yaw, wind or absolute"):
        step_yaw_command("bogus", np.zeros(1), np.zeros(1), None, **YAW_KW)


DERATE_KW = dict(
    method="absolute",
    step_base=None,
    step_env=0.1,
    derate_min=0.0,
    derate_max=0.8,
    reference="available",
    rated_power=2.0e6,
    current_powers=np.array([1.0e6, 1.0e6]),
    current_derate=np.zeros(2),
    step_sim=None,
    hawc2=False,
)


def test_absolute_affine_map():
    cmd, applied = compute_derate(np.array([-1.0, 1.0]), **DERATE_KW)
    np.testing.assert_allclose(cmd, [0.0, 0.8])
    np.testing.assert_allclose(applied, cmd)
    assert cmd.dtype == np.float64 and applied.dtype == np.float64


def test_step_method_adds_bounded_delta():
    kw = dict(DERATE_KW, method="step", step_base=np.array([0.5, 0.75]))
    cmd, _ = compute_derate(np.array([1.0, 1.0]), **kw)
    np.testing.assert_allclose(cmd, [0.6, 0.8])  # second is clipped to derate_max


def test_rated_reference_dead_zone():
    # cmd=0.5 -> target 1 MW == available power -> derate 0 (dead zone edge)
    kw = dict(DERATE_KW, reference="rated")
    cmd, applied = compute_derate(np.array([0.25, 0.25]), **kw)
    np.testing.assert_allclose(cmd, [0.5, 0.5])
    np.testing.assert_allclose(applied, [0.0, 0.0], atol=1e-12)
    # cmd=0.75 -> target 0.5 MW -> derate 0.5 of the 1 MW available
    _, applied = compute_derate(np.array([0.875, 0.875]), **kw)
    np.testing.assert_allclose(applied, [0.5, 0.5])


def test_rated_reference_passthrough_for_hawc2():
    kw = dict(DERATE_KW, reference="rated", hawc2=True)
    cmd, applied = compute_derate(np.array([0.875, 0.875]), **kw)
    np.testing.assert_allclose(applied, cmd)


def test_step_sim_slews_toward_setpoint():
    kw = dict(DERATE_KW, step_sim=0.05, current_derate=np.array([0.1, 0.7]))
    _, applied = compute_derate(np.array([1.0, -1.0]), **kw)
    np.testing.assert_allclose(applied, [0.15, 0.65])


# ------------------------------------------------------------------ deadband --
def test_wind_method_deadband_holds_a_small_target_change():
    kw = dict(YAW_KW, deadband=0.375)
    a = np.array([0.3 / 30.0, -0.3 / 30.0])  # targets +0.3 / -0.3 deg
    cmd, _ = step_yaw_command("wind", a, np.zeros(2), None, **kw)
    np.testing.assert_array_equal(cmd, [0.0, 0.0])


def test_wind_method_deadband_releases_a_large_target_change():
    kw = dict(YAW_KW, deadband=0.375)
    a = np.array([0.5 / 30.0, 0.0])  # target +0.5 deg (outside), 0 (inside: already there)
    cmd, _ = step_yaw_command("wind", a, np.zeros(2), None, **kw)
    np.testing.assert_allclose(cmd, [0.5, 0.0], atol=1e-6)


def test_wind_method_deadband_stops_short_of_the_target():
    # moving to 2 deg at 1 deg/step: stops at 1.0, then 2.0 is 1 deg away -> moves;
    # a target 0.3 deg away holds
    kw = dict(YAW_KW, deadband=0.375)
    cmd = np.array([1.7])
    cmd, _ = step_yaw_command("wind", np.array([2.0 / 30.0]), cmd, None, **kw)
    np.testing.assert_allclose(cmd, [1.7])
    cmd, _ = step_yaw_command("wind", np.array([2.5 / 30.0]), cmd, None, **kw)
    np.testing.assert_allclose(cmd, [2.5], atol=1e-6)


def test_wind_method_zero_deadband_is_the_old_behaviour():
    a = np.array([0.3 / 30.0])
    cmd, _ = step_yaw_command("wind", a, np.zeros(1), None, **dict(YAW_KW, deadband=0.0))
    old, _ = step_yaw_command("wind", a, np.zeros(1), None, **YAW_KW)
    np.testing.assert_array_equal(cmd, old)
    np.testing.assert_allclose(cmd, [0.3], atol=1e-6)


# ------------------------------------------------------------- share inner loop --
def test_shares_from_action_maps_and_normalises():
    np.testing.assert_allclose(shares_from_action(np.array([1.0, -1.0, 0.0])), [2 / 3, 0, 1 / 3])
    np.testing.assert_allclose(shares_from_action(np.array([-1.0, -1.0])), [0.5, 0.5])
    np.testing.assert_allclose(shares_from_action(np.array([5.0, -3.0])), [1.0, 0.0])  # clipped


def test_allocate_deficit_no_deficit_means_no_caps():
    p = np.array([4.0, 6.0])
    assert allocate_deficit(p, np.array([0.5, 0.5]), 0.0, 0.2) is None
    assert allocate_deficit(p, np.array([0.5, 0.5]), -1.0, 0.2) is None


def test_allocate_deficit_equal_shares_by_hand():
    r = allocate_deficit(np.array([4.0, 6.0]), np.array([0.5, 0.5]), 2.0, 0.2)
    np.testing.assert_allclose(r, [1.0, 1.0])


def test_allocate_deficit_t1_only_by_hand():
    r = allocate_deficit(np.array([4.0, 6.0, 5.0]), np.array([1.0, 0.0, 0.0]), 2.0, 0.2)
    np.testing.assert_allclose(r, [2.0, 0.0, 0.0])


def test_allocate_deficit_saturation_redistributes_by_share():
    # shares [0.5, 0.25, 0.25] of 6: wants [3, 1.5, 1.5]; T1 can give 0.8*2 = 1.6,
    # the excess 1.4 goes to T2/T3 by their shares (0.7 each): [1.6, 2.2, 2.2]
    r = allocate_deficit(np.array([2.0, 6.0, 6.0]), np.array([0.5, 0.25, 0.25]), 6.0, 0.2)
    np.testing.assert_allclose(r, [1.6, 2.2, 2.2])
    assert r.sum() == pytest.approx(6.0)


def test_allocate_deficit_t1_only_saturation_spills_by_headroom():
    # T1 gives 1.6 of 3; the zero-share turbines take the rest (1.4) in proportion
    # to their headroom 0.8*[6, 2] = [4.8, 1.6]: [1.05, 0.35]
    r = allocate_deficit(np.array([2.0, 6.0, 2.0]), np.array([1.0, 0.0, 0.0]), 3.0, 0.2)
    np.testing.assert_allclose(r, [1.6, 1.05, 0.35])


def test_allocate_deficit_beyond_the_floor_takes_everything_it_can():
    p = np.array([2.0, 6.0])
    r = allocate_deficit(p, np.array([0.5, 0.5]), 100.0, 0.2)
    np.testing.assert_allclose(r, 0.8 * p)


def test_allocate_deficit_cascading_saturation():
    # equal shares of 3.3 over [1, 2, 10] with floor 0.2 (headroom [0.8, 1.6, 8]):
    # 1.1 each -> T1 saturates at 0.8 (+0.3 spare: 0.15 each) -> T2 at 1.25, T3 1.25;
    # all within headroom
    r = allocate_deficit(np.array([1.0, 2.0, 10.0]), np.full(3, 1 / 3), 3.3, 0.2)
    np.testing.assert_allclose(r, [0.8, 1.25, 1.25])


# ------------------------------------------------ tilt mapping (round 5) --
def test_tilt_from_action_maps_to_zero_one_and_all_zero_is_uniform():
    np.testing.assert_allclose(tilt_from_action(np.array([1.0, -1.0, 0.0])), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(tilt_from_action(np.array([5.0, -3.0])), [1.0, 0.0])  # clipped
    # every action at -1: no tilt information, proportional
    np.testing.assert_allclose(tilt_from_action(np.array([-1.0, -1.0, -1.0])), [1.0, 1.0, 1.0])


def test_tilt_weights_equal_actions_are_proportional_shares():
    p_hat = np.array([2.0, 1.0, 0.5])
    for a in (-1.0, -0.3, 0.0, 1.0):
        c = tilt_from_action(np.full(3, a))
        w = c * p_hat / (c * p_hat).sum()
        np.testing.assert_allclose(w, p_hat / p_hat.sum(), rtol=1e-12)


# ------------------------------------------------------ yaw hold (round 5) --
def test_latch_yaw_action_latches_every_hold_steps_and_holds_in_between():
    held, count = None, 0
    seen = []
    for k in range(7):
        held, count = latch_yaw_action(np.array([float(k)]), held, count, hold_steps=3)
        seen.append(float(held[0]))
    assert seen == [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0]


def test_latch_yaw_action_copies_the_action():
    a = np.array([0.5])
    held, _ = latch_yaw_action(a, None, 0, hold_steps=2)
    a[0] = -1.0
    assert held[0] == 0.5


# ------------------------------------------- penalty low-pass (round 5) --
def test_lowpass_step_by_hand():
    # a = 1 - exp(-dt / tau); dt = tau ln 2 gives a = 0.5
    out = lowpass_step(np.array([0.0, 1.0]), np.array([1.0, 0.0]), dt=5.0 * np.log(2.0), tau=5.0)
    np.testing.assert_allclose(out, [0.5, 0.5])
    out = lowpass_step(out, np.array([1.0, 0.0]), dt=5.0 * np.log(2.0), tau=5.0)
    np.testing.assert_allclose(out, [0.75, 0.25])
