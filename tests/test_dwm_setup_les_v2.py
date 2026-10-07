"""LES_V2: the DWM re-calibration v2 eval point (RECAL_V2 3.8 / 3.11) as a
registered DWMSetup. It differs from LES_CALIBRATED only in k2, d_particle,
hill_vortex_factor and the Mann inflow spec (fitted L / Gamma / alpha-eps)."""
import dataclasses

import pytest
from py_wake.examples.data.hornsrev1 import V80

from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import RandomTurbulence

from WindGym.core.calibrated_values import (
    DWM_SETUPS,
    LES_CALIBRATED,
    LES_CALIBRATED_MANN,
    LES_V2,
    LES_V2_MANN,
    resolve_dwm_setup,
)
from WindGym.core.dwm_defaults import make_dwm, make_wts
from WindGym.core.particle_motion import ScaledHillVortexParticleMotion

pytestmark = pytest.mark.unit

CHANGED = {"k2": 0.020, "d_particle": 0.2, "hill_vortex_factor": 0.30}


def test_les_v2_is_registered():
    assert DWM_SETUPS["les_v2"] is LES_V2
    assert resolve_dwm_setup("les_v2") is LES_V2


def test_les_v2_changed_fields():
    for name, value in CHANGED.items():
        assert getattr(LES_V2, name) == pytest.approx(value), name
    assert LES_V2.mann is LES_V2_MANN
    assert LES_V2.k1 == pytest.approx(0.0914)
    assert LES_V2.viscosity_model == "keck"
    assert LES_V2.boundary_condition == LES_CALIBRATED.boundary_condition == "madsen"


def test_les_v2_inflow_is_the_fitted_mann_box():
    assert LES_V2_MANN.L == pytest.approx(72.7)
    assert LES_V2_MANN.gamma == pytest.approx(2.0)
    assert LES_V2_MANN.alphaepsilon == pytest.approx(0.0129)
    assert LES_V2_MANN.nxyz == LES_CALIBRATED_MANN.nxyz == (1024, 256, 128)
    assert LES_V2_MANN.dxyz == LES_CALIBRATED_MANN.dxyz == (3.2, 3.2, 3.2)


def test_every_other_field_equals_les_calibrated():
    for f in dataclasses.fields(LES_V2):
        if f.name in CHANGED or f.name == "mann":
            continue
        assert getattr(LES_V2, f.name) == getattr(LES_CALIBRATED, f.name), f.name


def test_make_dwm_les_v2_uses_scaled_hill_vortex_c0p3():
    wts = make_wts(x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80(), setup=LES_V2)
    site = TurbulenceFieldSite(ws=9.0, turbulenceField=RandomTurbulence(ti=0.07, ws=9.0))
    fs = make_dwm(site=site, windTurbines=wts, wind_direction=270.0, dt=1.0,
                  addedTurbulenceModel=None, setup=LES_V2)
    pm = fs.particleMotionModel
    assert isinstance(pm, ScaledHillVortexParticleMotion) and pm.c == pytest.approx(0.30)
