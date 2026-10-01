"""Dev-dynamiks-only features must fail with a clear error on PyPI dynamiks.

The flags are monkeypatched to False, so these guards are exercised on every
dynamiks build, not only on the ones that lack the feature.
"""

import numpy as np
import pytest

from WindGym.core import dwm_defaults, turbulence_manager
from WindGym.core.dynamiks_features import require
from WindGym.core.turbulence_manager import TurbulenceManager

pytestmark = pytest.mark.unit

MATCH = "dev_dynamiks branch"


def test_require_passes_when_available():
    require(True, "anything")


def test_require_names_feature_and_branch():
    with pytest.raises(ImportError, match=r"wind veer needs the dev dynamiks build"):
        require(False, "wind veer")


def test_precursor_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(turbulence_manager, "HAS_PRECURSOR", False)
    with pytest.raises(ImportError, match=MATCH):
        TurbulenceManager("Precursor", turbulence_box_path="precursor.nc")


def test_memmap_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(turbulence_manager, "HAS_MEMMAP", False)
    with pytest.raises(ImportError, match=MATCH):
        TurbulenceManager("None", memmap_boxes=True)


def _create_sites(veer_rate):
    tm = TurbulenceManager("None")
    tm.np_random = np.random.default_rng(0)
    return tm.create_sites(
        ws=8.0,
        wd=270.0,
        ti=0.06,
        wd_list=[270.0, 270.0],
        dt_sim=1.0,
        turbine_positions=np.array([[0.0, 0.0], [400.0, 0.0]]),
        rotor_diameter=80.0,
        n_passthrough=1,
        burn_in_passthroughs=0,
        veer_rate=veer_rate,
        veer_ref_height=70.0,
    )


def test_veer_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(turbulence_manager, "HAS_VEER", False)
    with pytest.raises(ImportError, match=MATCH):
        _create_sites(veer_rate=0.01)


def test_zero_veer_needs_no_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(turbulence_manager, "HAS_VEER", False)
    _create_sites(veer_rate=0.0)


@pytest.mark.parametrize(
    "kw", [{"interpolation": "linear"}, {"lateral_cutoff": 1.5}], ids=str
)
def test_dwm_speedups_raise_without_dev_dynamiks(monkeypatch, kw):
    from py_wake.examples.data.hornsrev1 import V80

    monkeypatch.setattr(dwm_defaults, "HAS_INTERP", False)
    wts = dwm_defaults.make_wts(x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80())
    with pytest.raises(ImportError, match=MATCH):
        dwm_defaults.make_dwm(
            site=None,
            windTurbines=wts,
            wind_direction=270.0,
            dt=1.0,
            addedTurbulenceModel=None,
            **kw,
        )
