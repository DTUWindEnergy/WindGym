# test_operating_point.py
"""Tests for the steady-state operating-point lookup (pitch / rotor RPM).

The interpolation and env-plumbing tests use a small synthetic table; the
shipped full DTU10MW surrogate is only exercised when its .nc is present
(examples/data is repo-only, not part of the installed package).
"""

from pathlib import Path

import numpy as np
import pytest

from WindGym.core import OperatingPointLookup

from test_derating import derating_turbine, make_env  # noqa: F401

FULL_SURROGATE_NC = (
    Path(__file__).resolve().parents[1]
    / "examples/data/dtu10mw_derating_yaw_surrogate_full.nc"
)

ROTOR_DIAMETER = 80.0


@pytest.fixture(scope="module")
def synthetic_lookup():
    """Analytic table: pitch = 10*derate + yaw/10, tsr = 8 - 5*derate."""
    ws = np.linspace(4.0, 20.0, 9)
    yaw = np.linspace(-30.0, 30.0, 7)
    derate = np.linspace(0.0, 0.8, 5)
    W, Y, D = np.meshgrid(ws, yaw, derate, indexing="ij")
    pitch = 10.0 * D + Y / 10.0
    tsr = 8.0 - 5.0 * D
    return OperatingPointLookup(
        ws=ws,
        yaw=yaw,
        derate=derate,
        pitch=pitch,
        tsr=tsr,
        rotor_diameter=ROTOR_DIAMETER,
    )


def test_pitch_rpm_values_and_units(synthetic_lookup):
    ws = np.array([10.0, 10.0])
    yaw = np.array([0.0, 10.0])
    derate = np.array([0.0, 0.4])

    pitch, rpm = synthetic_lookup.pitch_rpm(ws, yaw, derate)

    assert pitch.shape == rpm.shape == (2,)
    np.testing.assert_allclose(pitch, [0.0, 5.0], atol=1e-9)
    # rpm = tsr * ws / R * 60 / (2 pi), R = 40 m
    expected_tsr = np.array([8.0, 6.0])
    np.testing.assert_allclose(
        rpm, expected_tsr * 10.0 / 40.0 * 60.0 / (2.0 * np.pi), rtol=1e-9
    )


def test_out_of_grid_clamps_to_edge(synthetic_lookup):
    inside = synthetic_lookup.pitch_rpm(
        np.array([20.0]), np.array([30.0]), np.array([0.8])
    )
    outside = synthetic_lookup.pitch_rpm(
        np.array([35.0]), np.array([60.0]), np.array([1.5])
    )
    # Both pitch AND rpm must clamp: rpm pairs the edge tsr with the edge ws,
    # never extrapolating tsr * ws off the table.
    np.testing.assert_allclose(inside, outside)


@pytest.mark.skipif(
    not FULL_SURROGATE_NC.exists(), reason="full DTU10MW surrogate .nc not present"
)
def test_full_surrogate_physical_trends():
    lut = OperatingPointLookup.from_netcdf(FULL_SURROGATE_NC, rotor_diameter=178.3)
    derates = np.array([0.0, 0.2, 0.5, 0.8])
    pitch, rpm = lut.pitch_rpm(
        np.full(4, 10.0), np.zeros(4), derates
    )
    # Deeper derate at ws=10 -> blades pitch up, rotor slows (min-Omega floor).
    assert np.all(np.diff(pitch) > 0)
    assert np.all(np.diff(rpm) <= 1e-9)
    assert pitch[0] == pytest.approx(-1.98, abs=0.1)
    assert rpm[0] == pytest.approx(7.80, abs=0.1)


@pytest.fixture(scope="module")
def env_lookup(derating_turbine):  # noqa: F811
    """Synthetic lookup matching the synthetic derating turbine's envelope."""
    ws = np.linspace(0.0, 30.0, 16)
    yaw = np.array([-45.0, 45.0])
    derate = np.linspace(0.0, 0.8, 5)
    shape = (ws.size, yaw.size, derate.size)
    pitch = np.broadcast_to(10.0 * derate, shape)
    tsr = np.broadcast_to(8.0 - 5.0 * derate, shape)
    return OperatingPointLookup(
        ws=ws,
        yaw=yaw,
        derate=derate,
        pitch=pitch,
        tsr=tsr,
        rotor_diameter=derating_turbine.diameter(),
    )


def test_env_reports_pitch_rpm_with_lookup(derating_turbine, env_lookup):  # noqa: F811
    env = make_env(derating_turbine, op_lookup=env_lookup)
    try:
        env.reset(seed=0)
        # reset's warm-up already populated the operating point.
        assert env.current_pitch is not None and env.current_rpm is not None

        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["pitches"].shape == info["powers"].shape
        assert info["rpms"].shape == info["powers"].shape
        assert np.all(np.isfinite(info["pitches"]))
        assert np.all(np.isfinite(info["rpms"]))
        np.testing.assert_allclose(info["derates"][-1], env.current_derate)
        # derate > 0 after the step -> synthetic table says pitch = 10*derate.
        np.testing.assert_allclose(
            info["pitches"][-1], 10.0 * env.current_derate, rtol=1e-5
        )
    finally:
        env.close()


def test_env_without_lookup_has_no_op_keys(derating_turbine):  # noqa: F811
    env = make_env(derating_turbine)
    try:
        env.reset(seed=0)
        assert env.current_pitch is None and env.current_rpm is None

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "pitches" not in info
        assert "rpms" not in info
        # derates is always reported (zeros for a zero action).
        assert "derates" in info
    finally:
        env.close()
