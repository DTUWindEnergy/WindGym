"""Dev-dynamiks-only features must fail with a clear error on PyPI dynamiks.

The flags are monkeypatched to False, so these guards are exercised on every
dynamiks build, not only on the ones that lack the feature.
"""

import numpy as np
import pytest

from WindGym.core import dwm_defaults, turbulence_manager
from WindGym.core.calibrated_values import LES_CALIBRATED
from WindGym.core.dwm_defaults import DWMSetup
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


def _wts(setup=DWMSetup()):
    from py_wake.examples.data.hornsrev1 import V80

    return dwm_defaults.make_wts(
        x=[0.0, 400.0], y=[0.0, 0.0], windTurbine=V80(), setup=setup
    )


def _make_dwm(setup=DWMSetup(), **kw):
    return dwm_defaults.make_dwm(
        site=None,
        windTurbines=_wts(),
        wind_direction=270.0,
        dt=1.0,
        addedTurbulenceModel=None,
        setup=setup,
        **kw,
    )


@pytest.mark.parametrize(
    "kw", [{"interpolation": "linear"}, {"lateral_cutoff": 1.5}], ids=str
)
def test_dwm_speedups_raise_without_dev_dynamiks(monkeypatch, kw):
    monkeypatch.setattr(dwm_defaults, "HAS_INTERP", False)
    with pytest.raises(ImportError, match=MATCH):
        _make_dwm(**kw)


def test_ti_sensor_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(dwm_defaults, "HAS_TI_SENSOR", False)
    with pytest.raises(ImportError, match=MATCH):
        _wts(DWMSetup(ti_sensor_window_s=600))


def test_branlard_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(dwm_defaults, "HAS_BRANLARD", False)
    with pytest.raises(ImportError, match=MATCH):
        dwm_defaults.make_added_turbulence_model(
            DWMSetup(added_turbulence_scaling="branlard")
        )


def test_mixed_sum_raises_without_dev_dynamiks(monkeypatch):
    monkeypatch.setattr(dwm_defaults, "HAS_MIXED_SUM", False)
    with pytest.raises(ImportError, match=MATCH):
        _make_dwm(DWMSetup(superposition="mixed"))


def test_calibrated_setup_raises_without_dev_dynamiks(monkeypatch):
    """Opting in to LES_CALIBRATED on PyPI dynamiks fails loudly, not silently."""
    for flag in ("HAS_TI_SENSOR", "HAS_BRANLARD", "HAS_MIXED_SUM"):
        monkeypatch.setattr(dwm_defaults, flag, False)
    with pytest.raises(ImportError, match=MATCH):
        _wts(LES_CALIBRATED)


def test_default_setup_needs_no_dev_dynamiks(monkeypatch):
    from WindGym.presets import three_turbine_row

    for flag in ("HAS_INTERP", "HAS_TI_SENSOR", "HAS_BRANLARD", "HAS_MIXED_SUM"):
        monkeypatch.setattr(dwm_defaults, flag, False)
    dwm_defaults.make_added_turbulence_model(DWMSetup())
    env = three_turbine_row(turbtype="None", seed=0, reset_init=True)
    env.step(env.action_space.sample())
    env.close()
