"""Tests for the wake-structure and inflow overrides.

``dwm_params`` keys ``boundary_condition``, ``deflection_c``, ``meandering_d``,
``mann_Nxyz``, ``mann_dxyz`` and ``mann_AE=None``, and the ``mean_wind``
constructor argument. Every one defaults to the calibrated setup, so the
default path is unchanged; the tests below pin that as well as the overrides.
"""

from __future__ import annotations

import numpy as np
import pytest
from dynamiks.dwm.particle_motion_models import CutOffFrq
from jDWM import BoundaryCondition

import WindGym.core.dwm_defaults as dwm_defaults
import WindGym.core.turbulence_manager as tm_module
from WindGym import WindFarmEnv
from WindGym.core.dwm_defaults import (
    BOUNDARY_CONDITIONS,
    MANN_DXYZ,
    MANN_NXYZ,
    validate_dwm_params,
)
from WindGym.core.particle_motion import DEFLECTION_C, ScaledHillVortexParticleMotion
from WindGym.core.turbulence_manager import TurbulenceManager
from test_dwm_model_params import (  # noqa: F401  (fixtures)
    _spy_make_dwm,
    integration_env_kwargs,
    mann_turbulence_field,
)


# ---------------------------------------------------------------------------
# make_dwm wiring (constructors spied, nothing simulated)
# ---------------------------------------------------------------------------
def _spy_with_scaled(monkeypatch, **overrides) -> dict:
    """_spy_make_dwm, also capturing the scaled-deflection motion model."""
    captured: dict = {}

    class FakeScaled:
        def __init__(self, c, **kw):
            captured["scaled"] = {"c": c, **kw}

    monkeypatch.setattr(dwm_defaults, "ScaledHillVortexParticleMotion", FakeScaled)
    captured.update(_spy_make_dwm(monkeypatch, **overrides))
    return captured


def test_defaults_are_the_calibrated_setup(monkeypatch):
    """No new keys -> jDWM's madsen boundary, no meandering filter, stock motion model."""
    cap = _spy_with_scaled(monkeypatch)
    assert cap["deficit"]["boundaryConditionModel"] is BoundaryCondition.madsen
    assert cap["motion"]["temporal_filter"] is None
    assert "scaled" not in cap  # the stock HillVortexParticleMotion, not the copy


@pytest.mark.parametrize("name", sorted(BOUNDARY_CONDITIONS) + ["IEC", "Madsen"])
def test_boundary_condition_registry(monkeypatch, name):
    cap = _spy_with_scaled(monkeypatch, boundary_condition=name)
    assert cap["deficit"]["boundaryConditionModel"] is BOUNDARY_CONDITIONS[name.lower()]


@pytest.mark.parametrize("name", sorted(BOUNDARY_CONDITIONS))
def test_boundary_condition_registry_entries_are_accepted_by_jdwm(name):
    """Every registry entry must be a BoundaryCondition *class*: the spy test
    above only checks the lookup, but jDWMAinslieGenerator asserts
    issubclass(..., BoundaryCondition) at construction, so a helper function
    in the registry would only fail at the first reset."""
    from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator

    gen = jDWMAinslieGenerator(boundaryConditionModel=BOUNDARY_CONDITIONS[name])
    assert isinstance(gen.boundaryConditionModel, BoundaryCondition.BoundaryCondition)


def test_meandering_d_builds_cutoff_filter(monkeypatch):
    cap = _spy_with_scaled(monkeypatch, meandering_d=4.0)
    f = cap["motion"]["temporal_filter"]
    assert isinstance(f, CutOffFrq) and f.d == 4.0


def test_deflection_c_selects_scaled_motion(monkeypatch):
    cap = _spy_with_scaled(monkeypatch, deflection_c=0.3, meandering_d=4.0)
    assert "motion" not in cap
    assert cap["scaled"]["c"] == 0.3
    assert isinstance(cap["scaled"]["temporal_filter"], CutOffFrq)


def test_validation_of_new_keys():
    validate_dwm_params(
        {
            "boundary_condition": "IEC",
            "deflection_c": 0.3,
            "meandering_d": None,
            "mann_AE": None,
            "mann_Nxyz": (64, 32, 16),
            "mann_dxyz": [0.05, 0.05, 0.05],
        }
    )
    for bad in (
        {"boundary_condition": "banana"},
        {"deflection_c": 0.0},
        {"meandering_d": -1.0},
        {"mann_Nxyz": (64, 32)},
        {"mann_Nxyz": (64, 32, 16.5)},
        {"mann_Nxyz": "643216"},
        {"mann_dxyz": (0.05, 0.0, 0.05)},
        {"mann_L": None},
    ):
        with pytest.raises(ValueError):
            validate_dwm_params(bad)


# ---------------------------------------------------------------------------
# Mann box: grid overrides and mann_AE=None
# ---------------------------------------------------------------------------
class _FakeField:
    def __init__(self, **kw):
        self.kw = kw
        self.scaled_to = None

    def scale_TI(self, TI, U):
        self.scaled_to = (TI, U)


@pytest.fixture
def fake_mann(monkeypatch):
    monkeypatch.setattr(
        tm_module.MannTurbulenceField,
        "generate",
        staticmethod(lambda **kw: _FakeField(**kw)),
    )
    tm = TurbulenceManager(turbulence_type="MannGenerate")
    tm.np_random = np.random.default_rng(0)
    return tm


def test_mann_grid_and_scale_to_ti(fake_mann):
    tf, _ = fake_mann._generate_mann_generate(
        ws=5.5,
        ti=0.06,
        rotor_diameter=1.1,
        mann_overrides={
            "mann_L": 0.5,
            "mann_GAMMA": 1.0,
            "mann_AE": None,
            "mann_Nxyz": (512, 96, 48),
            "mann_dxyz": (0.05, 0.05, 0.05),
        },
    )
    assert tf.kw["L"] == 0.5 and tf.kw["Gamma"] == 1.0
    assert tf.kw["Nxyz"] == (512, 96, 48) and tf.kw["dxyz"] == (0.05, 0.05, 0.05)
    assert tf.scaled_to == (0.06, 5.5)


def test_numeric_mann_ae_stays_authoritative(fake_mann):
    tf, _ = fake_mann._generate_mann_generate(
        ws=5.5,
        ti=0.06,
        rotor_diameter=1.1,
        mann_overrides={"mann_L": 0.5, "mann_AE": 0.01},
    )
    assert tf.kw["alphaepsilon"] == 0.01
    assert tf.kw["Nxyz"] == MANN_NXYZ and tf.kw["dxyz"] == MANN_DXYZ
    assert tf.scaled_to is None


def test_missing_mann_ae_still_raises(fake_mann):
    with pytest.raises(ValueError, match="mann_AE=None"):
        fake_mann._generate_mann_generate(
            ws=5.5, ti=0.06, rotor_diameter=1.1, mann_overrides={"mann_L": 0.5}
        )


# ---------------------------------------------------------------------------
# mean_wind
# ---------------------------------------------------------------------------
class _ShapeWind:
    """A MeanWind stand-in: amplitude ws times a lateral shape."""

    def __init__(self, ws):
        self.ws = ws

    def __call__(self, xyz, uvw, time):
        uvw[0] += self.ws * (1.0 + 0.01 * np.asarray(xyz[1]))
        return uvw


def _create_sites(mean_wind, **kw):
    tm = TurbulenceManager(turbulence_type="Random")
    tm.np_random = np.random.default_rng(0)
    return tm.create_sites(
        ws=5.5,
        wd=270.0,
        ti=0.06,
        wd_list=[270.0] * 50,
        dt_sim=0.1,
        turbine_positions=np.array([[0.0, 0.0], [5.5, 0.0]]),
        rotor_diameter=1.1,
        n_passthrough=1,
        burn_in_passthroughs=0.1,
        create_baseline=True,
        mean_wind=mean_wind,
        **kw,
    )


def test_mean_wind_on_agent_and_baseline_site():
    mw = _ShapeWind(ws=99.0)
    site, site_base, *_ = _create_sites(mw)
    assert site.add_mean_windspeed is mw and site_base.add_mean_windspeed is mw
    assert mw.ws == 5.5  # refreshed to the episode's ws


def test_no_mean_wind_keeps_metmast_mean_wind():
    site, *_ = _create_sites(None)
    assert site.add_mean_windspeed == site.mean_wind


def test_mean_wind_with_veer_raises():
    with pytest.raises(ValueError, match="veer"):
        _create_sites(_ShapeWind(5.5), veer_rate=0.01)


# ---------------------------------------------------------------------------
# Integration: the scaled copy is upstream's code at c = 0.4
# ---------------------------------------------------------------------------
def _yawed_downstream_power(env_kwargs, n_steps=60, **dwm_params):
    env = WindFarmEnv(**env_kwargs, dwm_params=dwm_params or None)
    try:
        env.reset(seed=0)
        env.fs.windTurbines.yaw = np.array([25.0, 0.0])
        out = []
        for _ in range(n_steps):
            env.fs.step()
            out.append(np.asarray(env.fs.windTurbines.power(), float).copy())
        return np.array(out)
    finally:
        env.close()


@pytest.mark.integration
def test_scaled_motion_matches_upstream_at_default_c(
    integration_env_kwargs,  # noqa: F811  (pytest fixture, imported above)
    monkeypatch,
):
    stock = _yawed_downstream_power(integration_env_kwargs)
    monkeypatch.setattr(
        dwm_defaults,
        "HillVortexParticleMotion",
        lambda **kw: ScaledHillVortexParticleMotion(c=DEFLECTION_C, **kw),
    )
    copy = _yawed_downstream_power(integration_env_kwargs)
    np.testing.assert_array_equal(copy, stock)


@pytest.mark.integration
def test_deflection_c_changes_downstream_power(integration_env_kwargs):  # noqa: F811
    stock = _yawed_downstream_power(integration_env_kwargs)
    weak = _yawed_downstream_power(integration_env_kwargs, deflection_c=0.2)
    assert not np.allclose(stock[:, 1], weak[:, 1])
