"""dwm_params['secondary_steering'] / ss_gain / ss_decay_D.

The mechanism itself (dynamiks.dwm.secondary_steering) is tested in dynamiks;
here: the key is routed, validated and wired into make_dwm, and in a
3-turbine row it moves the third turbine's power while leaving the yawed
turbine and the one directly behind it untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from dynamiks.dwm.secondary_steering import SecondarySteeringGenerator
from py_wake.examples.data.dtu10mw import DTU10MW
from test_dwm_model_params import (
    CONFIG,
    _FakeWTs,
    integration_env_kwargs,
    mann_turbulence_field,
)

from WindGym import WindFarmEnv
from WindGym.core import dwm_defaults
from WindGym.core.dwm_defaults import DWM_PARAM_SPEC, make_dwm, validate_dwm_params
from WindGym.utils.generate_layouts import generate_square_grid


def test_spec_entries():
    assert DWM_PARAM_SPEC["secondary_steering"].kind == "bool"
    assert DWM_PARAM_SPEC["secondary_steering"].default is False
    assert DWM_PARAM_SPEC["ss_gain"].default == 1.0
    assert DWM_PARAM_SPEC["ss_decay_D"].default == dwm_defaults.SS_DECAY_D
    for k in ("secondary_steering", "ss_gain", "ss_decay_D"):
        assert DWM_PARAM_SPEC[k].group == "make_dwm"
        assert k in WindFarmEnv._CLOSURE_PARAM_KEYS


def test_bool_validation():
    validate_dwm_params({"secondary_steering": True})
    validate_dwm_params(
        {"secondary_steering": False, "ss_gain": 0.5, "ss_decay_D": 3.0}
    )
    validate_dwm_params({"secondary_steering": np.bool_(True)})
    for bad in (1, 0, "yes", "True", 1.0, None):
        with pytest.raises(ValueError):
            validate_dwm_params({"secondary_steering": bad})
    with pytest.raises(ValueError):
        validate_dwm_params({"ss_gain": 0})
    with pytest.raises(ValueError):
        validate_dwm_params({"ss_decay_D": -1})


def _spy(monkeypatch, **overrides):
    captured = {}

    class FakeDeficitGen:
        def __init__(self, **kw):
            captured["deficit"] = kw

    class FakeSS:
        def __init__(self, base, **kw):
            captured["ss"] = dict(base=base, **kw)

    class FakeMotion:
        def __init__(self, **kw):
            captured["motion"] = kw

    class FakeFS:
        def __init__(self, *, interpolation=None, lateral_cutoff=None, **kw):
            captured["fs"] = kw

    monkeypatch.setattr(dwm_defaults, "jDWMAinslieGenerator", FakeDeficitGen)
    monkeypatch.setattr(dwm_defaults, "SecondarySteeringGenerator", FakeSS)
    monkeypatch.setattr(dwm_defaults, "HillVortexParticleMotion", FakeMotion)
    monkeypatch.setattr(dwm_defaults, "DWMFlowSimulation", FakeFS)
    make_dwm(
        site=None,
        windTurbines=_FakeWTs(),
        wind_direction=270.0,
        dt=1.0,
        addedTurbulenceModel=None,
        **overrides,
    )
    return captured


def test_make_dwm_wiring(monkeypatch):
    off = _spy(monkeypatch)
    assert "ss" not in off
    assert isinstance(off["fs"]["particleDeficitGenerator"], object)
    # the motion model is unchanged: include_wakes already on, own wake excluded
    assert off["motion"]["include_wakes"] is True
    assert off["motion"]["include_own_wake"] is False

    on = _spy(monkeypatch, secondary_steering=True, ss_gain=0.7, ss_decay_D=4.0)
    assert on["ss"]["gain"] == 0.7 and on["ss"]["decay_D"] == 4.0
    assert type(on["fs"]["particleDeficitGenerator"]).__name__ == "FakeSS"
    assert type(on["ss"]["base"]).__name__ == "FakeDeficitGen"
    with pytest.raises(TypeError):
        _spy(monkeypatch, secondary_steering="yes")


def test_real_generator_is_wrapped(mann_turbulence_field):  # noqa: F811  # noqa: F811
    """With the real classes: the flag wraps whichever deficit generator is
    selected, and r_max reads through for the lateral cutoff."""
    from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator

    captured = {}
    real_fs = dwm_defaults.DWMFlowSimulation

    class Capture:
        def __init__(self, **kw):
            captured.update(kw)
            raise RuntimeError("stop")

    dwm_defaults.DWMFlowSimulation = Capture
    try:
        for model in ("ainslie", "bastankhah"):
            with pytest.raises(RuntimeError):
                make_dwm(
                    site=None,
                    windTurbines=_FakeWTs(),
                    wind_direction=270.0,
                    dt=1.0,
                    addedTurbulenceModel=None,
                    deficit_model=model,
                    secondary_steering=True,
                )
            gen = captured["particleDeficitGenerator"]
            assert isinstance(gen, SecondarySteeringGenerator)
            if model == "ainslie":
                assert isinstance(gen.base, jDWMAinslieGenerator)
                assert gen.r_max == gen.base.r_max
            assert gen.projectionModel is gen.base.projectionModel
    finally:
        dwm_defaults.DWMFlowSimulation = real_fs


def _three_in_a_row(mann_turbulence_field, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate",
        lambda *a, **kw: mann_turbulence_field,
    )
    turbine = DTU10MW()
    x_pos, y_pos = generate_square_grid(turbine=turbine, nx=3, ny=1, xDist=4, yDist=4)
    return {
        "turbine": turbine,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "config": CONFIG,
        "turbtype": "Random",
        "dt_sim": 1,
        "dt_env": 1,
        "n_passthrough": 1.5,
        "burn_in_passthroughs": 0.0001,
        "reset_init": False,
    }


def _powers(env_kwargs, n_steps=320, yaw=(25.0, 0.0, 0.0), **dwm_params):
    """WT1 at 25 deg. The vortex field reaches WT2 after one 4 D travel and
    WT2's deflected wake reaches WT3 after another, so this needs ~2.5
    travels at ~10 m/s on the DTU10MW (4 D = 712 m): 320 steps."""
    env = WindFarmEnv(**env_kwargs, dwm_params=dwm_params or None)
    try:
        env.reset(seed=0)
        assert isinstance(
            env.fs.particleDeficitGenerator, SecondarySteeringGenerator
        ) == bool(dwm_params.get("secondary_steering", False))
        env.fs.windTurbines.yaw = np.array(yaw, float)
        out = []
        for _ in range(n_steps):
            env.fs.step()
            out.append(np.asarray(env.fs.windTurbines.power(), float).copy())
        return np.array(out)
    finally:
        env.close()


@pytest.mark.integration
def test_secondary_steering_moves_the_third_turbine(mann_turbulence_field, monkeypatch):  # noqa: F811
    kw = _three_in_a_row(mann_turbulence_field, monkeypatch)
    off = _powers(kw)
    on = _powers(kw, secondary_steering=True)
    rel = np.abs(on[-60:] - off[-60:]).mean(0) / off[-60:].mean(0)
    # The yawed turbine sees the same flow (power is u-only, its own wake is
    # excluded from its own advection); the 'Random' turbtype's ambient field
    # is however query-order dependent and differs at a couple of isolated
    # steps late in the run, so WT1 is held to a tolerance, not equality.
    assert rel[0] < 0.005
    # WT2 is only touched indirectly: WT1's particles beyond WT2 advect through
    # WT2's now-deflected wake, and the centreline at WT2 is interpolated from
    # particles on both sides of it.
    assert rel[1] < 0.05
    # WT2's wake is now deflected, so WT3 is the turbine that moves (~9 %).
    assert rel[2] > 0.02
    assert rel[2] > 2 * rel[1]


@pytest.mark.integration
def test_unyawed_farm_is_bit_identical(mann_turbulence_field, monkeypatch):  # noqa: F811
    kw = _three_in_a_row(mann_turbulence_field, monkeypatch)
    off = _powers(kw, n_steps=200, yaw=(0.0, 0.0, 0.0))
    on = _powers(kw, n_steps=200, yaw=(0.0, 0.0, 0.0), secondary_steering=True)
    np.testing.assert_array_equal(on, off)
