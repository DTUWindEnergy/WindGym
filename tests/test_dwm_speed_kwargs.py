# tests/test_dwm_speed_kwargs.py
"""Tests for `_dwm_speed_kwargs`, the shim that only forwards the proj/fast
DWM speedup kwargs (`interpolation`, `lateral_cutoff`) to `DWMFlowSimulation`
when the installed `dynamiks` actually supports them.

No flow simulation or env construction is exercised here — just the kwarg
filtering and the once-per-process warning.
"""

import warnings

import pytest

import WindGym.wind_farm_env as wind_farm_env
from WindGym.wind_farm_env import _dwm_speed_kwargs


@pytest.fixture(autouse=True)
def _reset_warned_flag():
    """Each test gets a fresh "have we warned yet" flag."""
    original = wind_farm_env._dwm_speed_kwargs_warned
    wind_farm_env._dwm_speed_kwargs_warned = False
    yield
    wind_farm_env._dwm_speed_kwargs_warned = original


def test_drops_unsupported_kwargs_and_warns(monkeypatch):
    class FakeDWMFlowSimulationOld:
        def __init__(self, site, windTurbines, wind_direction, dt):
            pass

    monkeypatch.setattr(
        wind_farm_env, "DWMFlowSimulation", FakeDWMFlowSimulationOld
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kwargs = _dwm_speed_kwargs(interpolation="linear", lateral_cutoff=1.5)

    assert kwargs == {}
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "interpolation" in message
    assert "lateral_cutoff" in message


def test_passes_through_supported_kwargs(monkeypatch):
    class FakeDWMFlowSimulationNew:
        def __init__(
            self, site, windTurbines, wind_direction, dt, interpolation, lateral_cutoff
        ):
            pass

    monkeypatch.setattr(
        wind_farm_env, "DWMFlowSimulation", FakeDWMFlowSimulationNew
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kwargs = _dwm_speed_kwargs(interpolation="linear", lateral_cutoff=1.5)

    assert kwargs == {"interpolation": "linear", "lateral_cutoff": 1.5}
    assert len(caught) == 0


def test_warns_only_once_per_process(monkeypatch):
    class FakeDWMFlowSimulationOld:
        def __init__(self, site, windTurbines, wind_direction, dt):
            pass

    monkeypatch.setattr(
        wind_farm_env, "DWMFlowSimulation", FakeDWMFlowSimulationOld
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _dwm_speed_kwargs(interpolation="linear", lateral_cutoff=1.5)
        _dwm_speed_kwargs(interpolation="pchip", lateral_cutoff=None)

    assert len(caught) == 1
