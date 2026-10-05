"""Unit tests for WindGym.core.actuation (pure yaw / derate command helpers)."""

import numpy as np
import pytest

from WindGym.core.actuation import compute_derate, step_yaw_command

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
