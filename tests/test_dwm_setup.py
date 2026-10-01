"""DWMSetup selection: plain-dynamiks default, opt-in presets by name or object."""

import dataclasses

import pytest

from WindGym.core.calibrated_values import (
    DWM_SETUPS,
    LES_CALIBRATED,
    resolve_dwm_setup,
)
from WindGym.core.dwm_defaults import DWMSetup, MannBoxSpec
from WindGym.presets import three_turbine_row

pytestmark = pytest.mark.unit


def test_default_is_plain_dynamiks():
    assert resolve_dwm_setup(None) == DWMSetup()
    assert DWMSetup() != LES_CALIBRATED


def test_resolve_by_name_and_object():
    assert resolve_dwm_setup("les_calibrated") is LES_CALIBRATED
    custom = dataclasses.replace(DWMSetup(), d_particle=0.3)
    assert resolve_dwm_setup(custom) is custom


def test_unknown_name_lists_presets():
    with pytest.raises(ValueError, match="les_calibrated"):
        resolve_dwm_setup("calibrated")


def test_mann_spacing_d_relative_or_absolute():
    assert MannBoxSpec().spacing(80.0) == (4.0, 8.0, 8.0)
    assert MannBoxSpec(dxyz=(3.2, 3.2, 3.2)).spacing(80.0) == (3.2, 3.2, 3.2)


def _env(**kw):
    return three_turbine_row(turbtype="None", seed=0, reset_init=False, **kw)


def test_env_default_and_config_key():
    env = _env()
    assert env.dwm_setup == DWMSetup()
    assert env.turbulence_manager.dwm_setup is env.dwm_setup
    env.close()

    custom_name = "default"
    env = _env(config_overrides={"dwm_setup": custom_name})
    assert env.dwm_setup is DWM_SETUPS[custom_name]
    env.close()


def test_env_kwarg_overrides_config_key():
    custom = dataclasses.replace(DWMSetup(), d_particle=0.3)
    env = _env(config_overrides={"dwm_setup": "default"}, dwm_setup=custom)
    assert env.dwm_setup is custom
    env.close()
