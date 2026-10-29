"""core.yaw_frame: the est-frame <-> simulator-frame yaw conversion shared by the
training env (yaw_frame_source="est"), the LESRL DWM harness and the LES script.

Convention (wind_farm_env._shift_yaw_frame, pywake_lut.LutYawController.target_yaw):
yaw is wind-relative and the nacelle is fixed in the ground frame, so a yaw
quantity expressed in a frame whose wind direction is ``phi`` equals the same
nacelle's yaw in the simulator frame ``wd_sim`` plus wrap180(phi - wd_sim).
"""
import numpy as np
import pytest

from WindGym.core.yaw_frame import (
    ctrl_to_sim,
    frame_offset,
    sim_to_ctrl,
    slew_toward,
    wrap180,
)


def test_wrap180_wraps_into_half_open_interval():
    assert wrap180(0.0) == 0.0
    assert wrap180(180.0) == -180.0
    assert wrap180(-180.0) == -180.0
    assert wrap180(359.0) == -1.0
    assert wrap180(-359.0) == 1.0
    np.testing.assert_allclose(wrap180(np.array([270.0 - 235.0, 235.0 - 270.0])), [35.0, -35.0])


def test_frame_offset_matches_lut_sign_convention():
    # pywake_lut: target_grid = gamma(phi) - wrap180(phi - 270); frame_offset is that wrap term
    assert frame_offset(phi_ctrl=275.0, wd_sim=270.0) == 5.0
    assert frame_offset(phi_ctrl=235.0, wd_sim=270.0) == -35.0
    assert frame_offset(phi_ctrl=1.0, wd_sim=359.0) == 2.0      # the 0/360 wrap
    assert frame_offset(phi_ctrl=270.0, wd_sim=270.0) == 0.0


def test_ctrl_to_sim_is_the_lut_conversion():
    # LES grid anchored at 270: a controller that believes the wind is at 235 and wants to be
    # aligned (target 0 in its frame) must command +35 on the grid (= -wrap180(235 - 270)).
    assert ctrl_to_sim(0.0, phi_ctrl=235.0, wd_sim=270.0) == 35.0
    np.testing.assert_allclose(ctrl_to_sim(np.array([20.0, -20.0, 0.0]), 275.0, 270.0), [15.0, -25.0, -5.0])


def test_sim_to_ctrl_inverts_ctrl_to_sim():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.uniform(-40, 40, size=3)
        phi, wd = rng.uniform(0, 360), rng.uniform(0, 360)
        np.testing.assert_allclose(sim_to_ctrl(ctrl_to_sim(y, phi, wd), phi, wd), y, atol=1e-12)


def test_sim_to_ctrl_matches_training_frame_shift():
    # training: a true-wd change of +5 with a fixed nacelle shifts yaw_command by +5
    # (_shift_yaw_frame); the same nacelle seen from a frame 5 deg further clockwise reads +5.
    assert sim_to_ctrl(0.0, phi_ctrl=275.0, wd_sim=270.0) == 5.0


def test_slew_toward_limits_the_per_step_change_circularly():
    assert slew_toward(270.0, 275.0, 1.0) == 271.0
    assert slew_toward(270.0, 268.0, 1.0) == 269.0
    assert slew_toward(270.0, 270.4, 1.0) == 270.4
    assert slew_toward(359.0, 1.0, 5.0) == pytest.approx(361.0)   # unwrapped, shortest way round
    assert slew_toward(270.0, float("nan"), 1.0) == 270.0        # estimator not warm yet: hold
    with pytest.raises(ValueError):
        slew_toward(270.0, 275.0, -1.0)


def test_residual_target_adds_a_bounded_delta_to_the_prior():
    from WindGym.core.yaw_frame import residual_target, wind_action_from_target
    prior = np.array([20.0, -20.0, 0.0])
    np.testing.assert_allclose(residual_target(np.zeros(3), prior, 10.0, -30.0, 30.0), prior)
    np.testing.assert_allclose(residual_target(np.array([1.0, -1.0, 0.5]), prior, 10.0, -30.0, 30.0),
                               [30.0, -30.0, 5.0])
    np.testing.assert_allclose(residual_target(np.array([1.0, 1.0, 1.0]), prior, 15.0, -30.0, 30.0),
                               [30.0, -5.0, 15.0])      # clipped to the band
    with pytest.raises(ValueError):
        residual_target(np.zeros(3), prior, -1.0, -30.0, 30.0)
    # the "wind" ActionMethod encoding of an absolute target (step_yaw_command's inverse)
    np.testing.assert_allclose(wind_action_from_target(np.array([-30.0, 0.0, 30.0]), -30.0, 30.0),
                               [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(wind_action_from_target(np.array([45.0]), -30.0, 30.0), [1.0])
