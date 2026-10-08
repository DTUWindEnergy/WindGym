"""Tests for the extended ``dwm_params`` key set (model-structure knobs).

Beyond the calibrated closure trio (``k1``, ``k2``, ``d_particle``) and the
Mann-box statistics, ``dwm_params`` now accepts

- categorical model choices: ``viscosity_model``, ``superposition``, ``x_speed``
- numeric solver-grid knobs: ``r_max``, ``n_r``, ``dx``, ``lateral_cutoff``

The unit tests spy on the dynamiks constructors that ``make_dwm`` calls so no
flow simulation is built; the integration tests (marked) run a real env.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dynamiks.dwm.particle_motion_models import XSpeed
from dynamiks.dwm.superposition import MixedSum
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from jDWM.EddyViscosityModel import IEC, keck, larsen, madsen
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.hornsrev1 import V80
from py_wake.superposition_models import LinearSum

import WindGym.core.dwm_defaults as dwm_defaults
from WindGym import WindFarmEnv
from WindGym.core.turbine_params import TURBINE_PARAM_SPECS
from WindGym.core.dwm_defaults import (
    AINSLIE_DX,
    AINSLIE_N_R,
    AINSLIE_R_MAX,
    DWM_PARAM_SPEC,
    K1,
    K2,
    SUPERPOSITION_MODELS,
    VISCOSITY_MODELS,
    X_SPEEDS,
    make_dwm,
    validate_dwm_params,
)
from WindGym.utils.generate_layouts import generate_square_grid
from test_mann_param_routing import _MINIMAL_YAML


# ---------------------------------------------------------------------------
# make_dwm unit tests (constructors spied, nothing simulated)
# ---------------------------------------------------------------------------
class _FakeWTs:
    """Just enough of PyWakeWindTurbines for make_dwm's particle-count math."""

    rotor_positions_east_north = np.array([[0.0, 400.0], [0.0, 0.0], [70.0, 70.0]])

    def diameter(self):
        return np.array([80.0, 80.0])


def _spy_make_dwm(monkeypatch, **overrides) -> dict:
    """Call make_dwm with spied constructors; return the captured kwargs."""
    captured: dict = {}

    class FakeDeficitGen:
        def __init__(self, **kw):
            captured["deficit"] = kw

    class FakeMotion:
        def __init__(self, **kw):
            captured["motion"] = kw

    class FakeFS:
        # Accept the speedup kwargs by name so _dwm_speed_kwargs forwards
        # them instead of warning about an old dynamiks.
        def __init__(self, *, interpolation=None, lateral_cutoff=None, **kw):
            captured["fs"] = {**kw, "interpolation": interpolation, "lateral_cutoff": lateral_cutoff}

    monkeypatch.setattr(dwm_defaults, "jDWMAinslieGenerator", FakeDeficitGen)
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


def test_spec_matches_env_key_sets():
    """DWM_PARAM_SPEC (plus TURBINE_PARAM_SPECS for the turbine-model keys) is
    the single source of truth for the env's key sets."""
    make_dwm_keys = {k for k, s in DWM_PARAM_SPEC.items() if s.group == "make_dwm"}
    mann_keys = {k for k, s in DWM_PARAM_SPEC.items() if s.group == "mann"}
    assert WindFarmEnv._CLOSURE_PARAM_KEYS == make_dwm_keys
    assert WindFarmEnv._MANN_PARAM_KEYS == mann_keys
    assert WindFarmEnv._TURBINE_PARAM_KEYS == set(TURBINE_PARAM_SPECS)
    assert WindFarmEnv._WTS_PARAM_KEYS == {"rotor_power"}
    assert WindFarmEnv._DWM_PARAM_KEYS == set(DWM_PARAM_SPEC) | set(TURBINE_PARAM_SPECS)
    assert mann_keys == {"mann_L", "mann_GAMMA", "mann_AE", "mann_Nxyz", "mann_dxyz"}
    assert make_dwm_keys == {
        "k1", "k2", "d_particle", "viscosity_model", "superposition", "x_speed",
        "r_max", "n_r", "dx", "lateral_cutoff",
        "boundary_condition", "deflection_c", "meandering_d",
    }
    # Every make_dwm-group key is an actual make_dwm keyword.
    import inspect
    params = inspect.signature(make_dwm).parameters
    assert make_dwm_keys <= set(params)


def test_make_dwm_defaults_unchanged(monkeypatch):
    """No new keys passed -> byte-identical to the calibrated setup."""
    cap = _spy_make_dwm(monkeypatch)
    visc = cap["deficit"]["viscosity_model"]
    assert isinstance(visc, keck)
    assert visc.k1 == K1 and visc.k2 == K2
    assert cap["deficit"]["r_max"] == AINSLIE_R_MAX == 3
    assert cap["deficit"]["n_r"] == AINSLIE_N_R == 52
    assert cap["deficit"]["dx"] == AINSLIE_DX == 0.1
    assert cap["motion"]["x_speed"] is XSpeed.Particle
    assert isinstance(cap["fs"]["superpositionModel"], MixedSum)


@pytest.mark.parametrize(
    "name, cls, k1_attr",
    [("keck", keck, "k1"), ("madsen", madsen, "k1"), ("larsen", larsen, "kamb"),
     ("iec", IEC, "k1"), ("IEC", IEC, "k1"), ("Madsen", madsen, "k1")],
)
def test_viscosity_registry(monkeypatch, name, cls, k1_attr):
    """Each choice builds the right jDWM class; explicit k1 lands on its coefficient name."""
    cap = _spy_make_dwm(monkeypatch, viscosity_model=name, k1=0.123, k2=0.045)
    visc = cap["deficit"]["viscosity_model"]
    assert isinstance(visc, cls)
    assert getattr(visc, k1_attr) == pytest.approx(0.123)
    assert visc.k2 == pytest.approx(0.045)


def test_viscosity_model_default_coefficients(monkeypatch):
    """k1/k2 = None means 'use the selected model's own coefficients'."""
    cap = _spy_make_dwm(monkeypatch, viscosity_model="madsen")
    visc = cap["deficit"]["viscosity_model"]
    ref = madsen(TI=1.0)
    assert (visc.k1, visc.k2) == (ref.k1, ref.k2)
    assert visc.k1 != K1  # not silently running madsen with keck's coefficient


def test_superposition_and_x_speed_resolution(monkeypatch):
    cap = _spy_make_dwm(monkeypatch, superposition="linear", x_speed="Global")
    assert isinstance(cap["fs"]["superpositionModel"], LinearSum)
    assert cap["motion"]["x_speed"] is XSpeed.Global
    cap = _spy_make_dwm(monkeypatch, x_speed="rotor")
    assert cap["motion"]["x_speed"] is XSpeed.Rotor
    assert set(SUPERPOSITION_MODELS) == {"mixed", "linear"}
    assert set(X_SPEEDS) == {"particle", "global", "rotor"}
    assert set(VISCOSITY_MODELS) == {"keck", "madsen", "larsen", "iec"}


def test_grid_kwargs_forwarded(monkeypatch):
    cap = _spy_make_dwm(monkeypatch, r_max=4, n_r=101, dx=0.05, lateral_cutoff=1.5)
    assert cap["deficit"]["r_max"] == 4.0
    assert cap["deficit"]["n_r"] == 101 and isinstance(cap["deficit"]["n_r"], int)
    assert cap["deficit"]["dx"] == 0.05


@pytest.mark.parametrize(
    "kwargs, listed",
    [({"viscosity_model": "banana"}, "madsen"),
     ({"superposition": "squared"}, "linear"),
     ({"x_speed": "sideways"}, "rotor")],
)
def test_invalid_choice_raises(monkeypatch, kwargs, listed):
    """An unknown categorical value raises and the message lists the choices."""
    with pytest.raises(ValueError, match=listed):
        _spy_make_dwm(monkeypatch, **kwargs)


def test_validate_dwm_params_numeric():
    # good
    validate_dwm_params({"k1": 0.1, "n_r": 52, "dx": np.float64(0.1), "lateral_cutoff": None})
    validate_dwm_params({"n_r": 52.0, "viscosity_model": "Keck"})
    validate_dwm_params({})
    # bad
    with pytest.raises(ValueError, match="Unknown dwm_params keys"):
        validate_dwm_params({"k_seven": 1.0})
    with pytest.raises(ValueError, match="k1"):
        validate_dwm_params({"k1": -0.1})
    with pytest.raises(ValueError, match="k1"):
        validate_dwm_params({"k1": float("nan")})
    with pytest.raises(ValueError, match="n_r"):
        validate_dwm_params({"n_r": 52.5})
    with pytest.raises(ValueError, match="n_r"):
        validate_dwm_params({"n_r": 0})
    with pytest.raises(ValueError, match="dx"):
        validate_dwm_params({"dx": "fast"})
    with pytest.raises(ValueError, match="viscosity_model"):
        validate_dwm_params({"viscosity_model": 3})
    with pytest.raises(ValueError, match="d_particle"):
        validate_dwm_params({"d_particle": None})
    with pytest.raises(ValueError, match="in reset options"):
        validate_dwm_params({"nope": 1}, where=" in reset options")


# ---------------------------------------------------------------------------
# Env routing (turbtype="None", reset_init=False, make_dwm spied)
# ---------------------------------------------------------------------------
class _StopReset(Exception):
    pass


def _build_env(temp_yaml_file_factory, **kwargs):
    yaml_path = temp_yaml_file_factory(_MINIMAL_YAML, "dwm_model_params")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
    return WindFarmEnv(
        turbine=V80(), x_pos=x_pos, y_pos=y_pos, config=yaml_path, seed=123,
        turbtype="None", n_passthrough=0.01, burn_in_passthroughs=0.0001,
        reset_init=False, **kwargs,
    )


def test_env_routes_model_keys_to_make_dwm(monkeypatch, temp_yaml_file_factory):
    env = _build_env(
        temp_yaml_file_factory,
        dwm_params={"viscosity_model": "madsen", "n_r": 40, "lateral_cutoff": 2.0},
    )
    captured: dict = {}

    def spy(**kw):
        captured.update(kw)
        raise _StopReset()

    monkeypatch.setattr("WindGym.wind_farm_env.make_dwm", spy)
    try:
        with pytest.raises(_StopReset):
            env.reset(options={"dwm_params": {"x_speed": "rotor"}})
        assert captured["viscosity_model"] == "madsen"
        assert captured["n_r"] == 40
        assert captured["x_speed"] == "rotor"
        # per-episode lateral_cutoff overrides the ctor value, no TypeError
        assert captured["lateral_cutoff"] == 2.0
        assert captured["interpolation"] == env.interpolation
    finally:
        env.close()


def test_env_rejects_invalid_categorical_at_init_and_reset(monkeypatch, temp_yaml_file_factory):
    with pytest.raises(ValueError, match="viscosity_model"):
        _build_env(temp_yaml_file_factory, dwm_params={"viscosity_model": "banana"})

    env = _build_env(temp_yaml_file_factory)
    called = []
    monkeypatch.setattr("WindGym.wind_farm_env.make_dwm", lambda **kw: called.append(kw))
    try:
        with pytest.raises(ValueError, match="superposition"):
            env.reset(options={"dwm_params": {"superposition": "squared"}})
        with pytest.raises(ValueError, match="n_r"):
            env.reset(options={"dwm_params": {"n_r": -3}})
        assert called == []
    finally:
        env.close()


def test_active_dwm_params_property(monkeypatch, temp_yaml_file_factory):
    env = _build_env(temp_yaml_file_factory, dwm_params={"k1": 0.05, "viscosity_model": "iec"})
    monkeypatch.setattr("WindGym.wind_farm_env.make_dwm", lambda **kw: (_ for _ in ()).throw(_StopReset()))
    try:
        assert env.active_dwm_params == {"k1": 0.05, "viscosity_model": "iec"}
        with pytest.raises(_StopReset):
            env.reset(options={"dwm_params": {"k1": 0.07}})
        assert env.active_dwm_params == {"k1": 0.07, "viscosity_model": "iec"}
        env.active_dwm_params["k1"] = 999  # a copy, not the live dict
        assert env.active_dwm_params["k1"] == 0.07
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Integration: real DWM with each model choice
# ---------------------------------------------------------------------------
CONFIG = Path("examples/EnvConfigs/2turb.yaml")


@pytest.fixture(scope="module")
def mann_turbulence_field():
    return MannTurbulenceField.generate(
        alphaepsilon=0.1, L=33.6, Gamma=3.9, Nxyz=(1024, 128, 32),
        dxyz=(3.0, 3.0, 3.0), seed=1234,
    )


@pytest.fixture
def integration_env_kwargs(mann_turbulence_field, monkeypatch):
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate",
        lambda *a, **kw: mann_turbulence_field,
    )
    turbine = DTU10MW()
    x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=4, yDist=4)
    return dict(
        turbine=turbine, x_pos=x_pos, y_pos=y_pos, config=CONFIG, turbtype="Random",
        dt_sim=1, dt_env=1, n_passthrough=1.5, burn_in_passthroughs=0.0001,
        reset_init=False,
    )


def _rollout(env, n_steps, seed=0, dwm_params=None):
    env.reset(seed=seed, options={"dwm_params": dwm_params} if dwm_params else None)
    zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    out = []
    for _ in range(n_steps):
        obs, _, term, trunc, _ = env.step(zero)
        out.append(obs)
        if term or trunc:
            break
    return np.concatenate(out)


@pytest.mark.integration
@pytest.mark.parametrize(
    "dwm_params",
    [{"viscosity_model": m} for m in ("keck", "madsen", "larsen", "iec")]
    + [{"superposition": s} for s in ("mixed", "linear")]
    + [{"x_speed": x} for x in ("particle", "global", "rotor")],
    ids=lambda d: "=".join(next(iter(d.items()))),
)
def test_each_model_choice_runs(integration_env_kwargs, dwm_params):
    env = WindFarmEnv(**integration_env_kwargs, dwm_params=dwm_params)
    try:
        obs = _rollout(env, 5)
        assert np.all(np.isfinite(obs))
    finally:
        env.close()


@pytest.mark.integration
def test_model_switch_changes_simulation(integration_env_kwargs):
    env = WindFarmEnv(**integration_env_kwargs)
    try:
        a = _rollout(env, 40, dwm_params={"viscosity_model": "keck"})
        b = _rollout(env, 40, dwm_params={"viscosity_model": "madsen"})
        assert not np.allclose(a, b)
    finally:
        env.close()
