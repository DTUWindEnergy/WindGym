"""DWMSetup.boundary_condition: Ainslie near-wake boundary condition passed as
boundaryConditionModel= to jDWMAinslieGenerator ("madsen" default = dynamiks default)."""
import dataclasses

import pytest
from jDWM import BoundaryCondition
from py_wake.examples.data.hornsrev1 import V80

from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import RandomTurbulence

from WindGym.core.dwm_defaults import DWMSetup, make_dwm, make_wts

pytestmark = pytest.mark.unit


def _fs(setup):
    wts = make_wts(x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80(), setup=setup)
    site = TurbulenceFieldSite(ws=9.0, turbulenceField=RandomTurbulence(ti=0.07, ws=9.0))
    return make_dwm(site=site, windTurbines=wts, wind_direction=270.0, dt=1.0,
                    addedTurbulenceModel=None, setup=setup)


def test_default_is_madsen():
    assert DWMSetup().boundary_condition == "madsen"
    fs = _fs(DWMSetup())
    assert isinstance(fs.particleDeficitGenerator.boundaryConditionModel, BoundaryCondition.madsen)


def test_iec_variant():
    fs = _fs(dataclasses.replace(DWMSetup(), boundary_condition="iec"))
    assert isinstance(fs.particleDeficitGenerator.boundaryConditionModel, BoundaryCondition.IEC)


def test_unknown_boundary_raises():
    with pytest.raises(ValueError, match="boundary_condition"):
        _fs(dataclasses.replace(DWMSetup(), boundary_condition="bogus"))
