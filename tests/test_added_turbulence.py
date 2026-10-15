"""TurbulenceManager(added_turbulence=...): the wake-added turbulence box's grid.

"synchronized" (the default, the original behaviour) puts dynamiks' box on the
main Mann box's grid; with the G1's 0.05 m grid that is 1.4 cells per the box's
D/16 length scale, i.e. white noise at the grid scale. "resolved" gives the box
its own HAWC2-style grid, 128 cells per diameter.
"""

from __future__ import annotations

import numpy as np
import pytest
from dynamiks.dwm.added_turbulence_models import (
    AutoScalingIsotropicMannTurbulence,
    BranlardScaling,
    SynchronizedAutoScalingIsotropicMannTurbulence,
)

from WindGym.core.turbulence_manager import (
    RESOLVED_ADDED_TURB_LXYZ,
    RESOLVED_ADDED_TURB_NXYZ,
    TurbulenceManager,
)


def test_default_is_the_original_model():
    m = TurbulenceManager(turbulence_type="MannGenerate")._added_turb_model()
    assert type(m) is SynchronizedAutoScalingIsotropicMannTurbulence
    assert isinstance(m.scaling, BranlardScaling)


def test_invalid_choice():
    with pytest.raises(ValueError):
        TurbulenceManager(turbulence_type="MannGenerate", added_turbulence="fine")


class _FakeTurbines:
    def diameter(self):
        return np.array([1.1, 1.1, 1.1])


class _FakeSite:
    turbulence_transport_speed = 5.5


class _FakeFS:
    site = _FakeSite()
    windTurbines = _FakeTurbines()


def test_resolved_box_resolves_its_length_scale():
    m = TurbulenceManager(
        turbulence_type="MannGenerate", added_turbulence="resolved"
    )._added_turb_model()
    assert type(m) is AutoScalingIsotropicMannTurbulence
    assert isinstance(m.scaling, BranlardScaling)
    m.initialize(_FakeFS())
    D = 1.1
    np.testing.assert_allclose(
        m.dxyz,
        np.array(RESOLVED_ADDED_TURB_LXYZ) / np.array(RESOLVED_ADDED_TURB_NXYZ) * D,
    )
    assert m.Nxyz == RESOLVED_ADDED_TURB_NXYZ
    # the Mann length scale D/16 spans ~8 cells: coherent over neighbouring cells
    assert (D / 16) / m.dxyz[1] > 7
    u = m.mannTurbulenceField.uvw[0]
    a = np.moveaxis(u, 1, 0)
    r1 = np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1]
    assert r1 > 0.7
    # the box is reused periodically along x (HAWC2's practice)
    Lx = m.Nxyz[0] * m.dxyz[0]
    v0 = m.mannTurbulenceField(np.array([0.3]), np.array([0.2]), np.array([0.2]))
    v1 = m.mannTurbulenceField(np.array([0.3 + Lx]), np.array([0.2]), np.array([0.2]))
    np.testing.assert_allclose(v0, v1, atol=1e-6)
