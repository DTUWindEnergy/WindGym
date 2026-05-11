"""Tests for Mann-box parameter routing in WindFarmEnv.reset().

Verifies that DR-time ``dwm_params`` mixing closure keys (``k1``, ``k2``,
``d_particle``) with Mann-box keys (``mann_L``, ``mann_GAMMA``, ``mann_AE``)
is partitioned correctly:

- Closure keys flow into ``make_dwm`` only.
- Mann keys flow into ``TurbulenceManager.create_sites`` as ``mann_overrides``.
- Using Mann keys with a non-``MannGenerate`` turbtype raises immediately so a
  misconfigured DR run fails on the first reset rather than silently
  discarding the calibration's joint posterior over Mann statistics.
"""
from __future__ import annotations

import pytest
from py_wake.examples.data.hornsrev1 import V80

from dynamiks.sites.turbulence_fields import MannTurbulenceField

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid


@pytest.fixture
def tiny_mann_generate(monkeypatch):
    """Replace ``MannTurbulenceField.generate`` with a tiny-grid version.

    Conftest's ``mock_mann_methods`` fixture is currently broken against the
    installed dynamiks (its ``MannTurbulenceField(field, coords)`` call uses a
    stale 2-positional signature). We side-step it with a real ``.generate``
    call on a (32, 16, 8) grid — fast enough for unit tests, and the resulting
    object satisfies anything MetmastSite does with it during __init__.
    """
    real_generate = MannTurbulenceField.generate.__func__ if hasattr(
        MannTurbulenceField.generate, "__func__"
    ) else MannTurbulenceField.generate

    def fast_generate(**kwargs):
        kwargs["Nxyz"] = (32, 16, 8)
        kwargs["dxyz"] = (8.0, 8.0, 8.0)
        return real_generate(**kwargs)

    monkeypatch.setattr(MannTurbulenceField, "generate", staticmethod(fast_generate))


# Minimal self-contained YAML — same shape as test_env_reward_functions.py's
# helper, kept inline so this file has no cross-test dependencies.
_MINIMAL_YAML = """
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "yaw"
Track_power: False
farm:
  yaw_min: -30
  yaw_max: 30
  xDist: 5
  yDist: 3
  nx: 2
  ny: 1
wind:
  ws_min: 9
  ws_max: 9
  TI_min: 0.07
  TI_max: 0.07
  wd_min: 270
  wd_max: 270
act_pen:
  action_penalty: 0.0
  action_penalty_type: "Change"
power_def:
  Power_reward: "None"
  Power_avg: 1
  Power_scaling: 1.0
mes_level:
  turb_ws: True
  turb_wd: False
  turb_TI: False
  turb_power: True
  farm_ws: False
  farm_wd: False
  farm_TI: False
  farm_power: True
ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
"""


def _build_env(turbtype, temp_yaml_file_factory):
    """Construct a minimal WindFarmEnv with ``reset_init=False`` (no auto-reset)."""
    yaml_path = temp_yaml_file_factory(_MINIMAL_YAML, "mann_routing")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
    return WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        seed=123,
        turbtype=turbtype,
        n_passthrough=0.01,
        burn_in_passthroughs=0.0001,
        reset_init=False,  # don't auto-reset; we'll trigger manually
    )


def test_unknown_dwm_param_key_still_raises(temp_yaml_file_factory):
    """The pre-existing typo guard must still fire after Mann keys are added."""
    env = _build_env("None", temp_yaml_file_factory)
    try:
        with pytest.raises(ValueError, match="Unknown dwm_params keys"):
            env.reset(options={"dwm_params": {"banana": 1.0}})
    finally:
        env.close()


def test_mann_keys_under_non_mann_generate_turbtype_raises(temp_yaml_file_factory):
    """Mann keys with turbtype=='None' (or other non-MannGenerate) must fail fast.

    Per-episode Mann statistics only take effect under ``MannGenerate``; other
    branches don't rebuild the box from (L, Γ, αε). Silently ignoring them
    would discard the calibrated joint, so the env raises immediately.
    """
    env = _build_env("None", temp_yaml_file_factory)
    try:
        with pytest.raises(ValueError, match="MannGenerate"):
            env.reset(options={"dwm_params": {"mann_AE": 0.005}})
    finally:
        env.close()


def test_partition_routes_keys_to_correct_subsystems(
    tiny_mann_generate, monkeypatch, temp_yaml_file_factory,
):
    """Mixed dwm_params: closure keys → make_dwm, Mann keys → create_sites.

    Uses ``mock_mann_methods`` (from conftest) to short-circuit the expensive
    Mann-box generation, and a spy on ``make_dwm`` that captures kwargs then
    raises to abort the rest of reset (which would need a real DWMFlowSimulation
    object that we don't want to construct in a unit test).
    """
    env = _build_env("MannGenerate", temp_yaml_file_factory)

    captured_create_sites: dict = {}
    captured_make_dwm: dict = {}

    real_create_sites = env.turbulence_manager.create_sites

    def create_sites_spy(**kwargs):
        captured_create_sites.update(kwargs)
        return real_create_sites(**kwargs)

    monkeypatch.setattr(env.turbulence_manager, "create_sites", create_sites_spy)

    class _StopReset(Exception):
        """Raised by the make_dwm spy to abort reset after capturing kwargs."""

    def make_dwm_spy(**kwargs):
        captured_make_dwm.update(kwargs)
        raise _StopReset()

    # The env binds ``make_dwm`` at import time, so patch the binding in the env
    # module (not the source module).
    monkeypatch.setattr("WindGym.wind_farm_env.make_dwm", make_dwm_spy)

    try:
        with pytest.raises(_StopReset):
            env.reset(
                options={"dwm_params": {"k1": 0.05, "mann_L": 50.0, "mann_AE": 0.01}}
            )

        # Mann keys routed to the turbulence manager.
        assert captured_create_sites.get("mann_overrides") == {
            "mann_L": 50.0,
            "mann_AE": 0.01,
        }
        # Closure key routed to make_dwm; Mann keys must NOT appear here (they
        # would crash dwm_defaults.make_dwm whose signature only accepts the
        # closure trio).
        assert captured_make_dwm.get("k1") == 0.05
        for forbidden in ("mann_L", "mann_GAMMA", "mann_AE"):
            assert forbidden not in captured_make_dwm, (
                f"{forbidden} leaked into make_dwm kwargs; partition is broken."
            )
    finally:
        env.close()


def test_pure_closure_overrides_leave_mann_overrides_empty(
    tiny_mann_generate, monkeypatch, temp_yaml_file_factory,
):
    """Legacy DR (closure-only) must still pass mann_overrides=None to create_sites.

    Guards against regression where adding the partition silently activates the
    skip-``scale_TI`` branch for pre-Mann calibrations.
    """
    env = _build_env("MannGenerate", temp_yaml_file_factory)

    captured_create_sites: dict = {}
    real_create_sites = env.turbulence_manager.create_sites

    def create_sites_spy(**kwargs):
        captured_create_sites.update(kwargs)
        return real_create_sites(**kwargs)

    monkeypatch.setattr(env.turbulence_manager, "create_sites", create_sites_spy)

    class _StopReset(Exception):
        pass

    def make_dwm_spy(**kwargs):
        raise _StopReset()

    monkeypatch.setattr("WindGym.wind_farm_env.make_dwm", make_dwm_spy)

    try:
        with pytest.raises(_StopReset):
            env.reset(options={"dwm_params": {"k1": 0.05, "k2": 0.02}})
        # No Mann keys → mann_overrides should be falsy so create_sites takes the
        # legacy scale_TI path.
        assert not captured_create_sites.get("mann_overrides")
    finally:
        env.close()
