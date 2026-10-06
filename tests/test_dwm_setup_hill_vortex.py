"""DWMSetup.hill_vortex_factor: the self-induction factor of HillVortexParticleMotion
(dynamiks hard-codes 0.4). At the default the plain dynamiks class is used
(bit-identical); otherwise WindGym's subclass carrying the factor."""
import dataclasses

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80

from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import RandomTurbulence

from WindGym.core.dwm_defaults import DWMSetup, make_dwm, make_wts
from WindGym.core.particle_motion import ScaledHillVortexParticleMotion

pytestmark = pytest.mark.unit


def _fs(setup):
    wts = make_wts(x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80(), setup=setup)
    site = TurbulenceFieldSite(ws=9.0, turbulenceField=RandomTurbulence(ti=0.07, ws=9.0))
    return make_dwm(site=site, windTurbines=wts, wind_direction=270.0, dt=1.0,
                    addedTurbulenceModel=None, setup=setup)


def test_default_factor_uses_plain_dynamiks_class():
    assert DWMSetup().hill_vortex_factor == 0.4
    fs = _fs(DWMSetup())
    assert type(fs.particleMotionModel) is HillVortexParticleMotion


def test_other_factor_uses_scaled_subclass():
    fs = _fs(dataclasses.replace(DWMSetup(), hill_vortex_factor=0.6))
    pm = fs.particleMotionModel
    assert isinstance(pm, ScaledHillVortexParticleMotion) and pm.c == 0.6


def test_scaled_subclass_at_0p4_matches_plain_step():
    a = _fs(DWMSetup())
    b = _fs(dataclasses.replace(DWMSetup(), hill_vortex_factor=0.4 + 1e-12))
    assert isinstance(b.particleMotionModel, ScaledHillVortexParticleMotion)
    for fs in (a, b):
        fs.windTurbines.yaw = np.array([20.0, 0.0])
        for _ in range(5):
            fs.step()
    np.testing.assert_allclose(a.windTurbines.rotor_avg_windspeed, b.windTurbines.rotor_avg_windspeed, rtol=1e-9)


def test_hill_vortex_factor_is_a_per_episode_closure_override():
    """Finding 8: the DR posterior carries hill_vortex_factor, so make_dwm must take
    it as a per-episode override like k1/k2/d_particle, and the env must accept it
    in dwm_params."""
    from WindGym.wind_farm_env import WindFarmEnv

    assert "hill_vortex_factor" in WindFarmEnv._CLOSURE_PARAM_KEYS
    wts = make_wts(x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80(), setup=DWMSetup())
    site = TurbulenceFieldSite(ws=9.0, turbulenceField=RandomTurbulence(ti=0.07, ws=9.0))
    fs = make_dwm(site=site, windTurbines=wts, wind_direction=270.0, dt=1.0, addedTurbulenceModel=None,
                  setup=DWMSetup(), hill_vortex_factor=0.55)
    assert isinstance(fs.particleMotionModel, ScaledHillVortexParticleMotion) and fs.particleMotionModel.c == 0.55
