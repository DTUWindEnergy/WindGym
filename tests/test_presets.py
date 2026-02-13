"""Tests for WindGym.presets factory functions."""

import numpy as np
import pytest

from WindGym.presets import nine_turbine_grid, three_turbine_row, two_by_two_grid

PRESETS = [
    (three_turbine_row, 3),
    (two_by_two_grid, 4),
    (nine_turbine_grid, 9),
]


@pytest.fixture(params=PRESETS, ids=["3-row", "2x2-grid", "3x3-grid"])
def preset_env(request):
    """Create a preset environment and ensure cleanup."""
    factory, expected_n_turb = request.param
    env = factory(n_passthrough=0.01, burn_in_passthroughs=0.0001, fill_window=1)
    yield env, expected_n_turb
    env.close()


def test_turbine_count(preset_env):
    """Preset creates the correct number of turbines."""
    env, expected = preset_env
    assert env.n_turb == expected


def test_valid_observation_after_reset(preset_env):
    """Reset returns a valid observation within the observation space."""
    env, _ = preset_env
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert not np.any(np.isnan(obs))
    assert env.observation_space.contains(obs)


def test_step_without_error(preset_env):
    """A zero-action step succeeds and returns valid outputs."""
    env, _ = preset_env
    env.reset()
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert not np.any(np.isnan(obs))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_backend_override():
    """kwargs are forwarded — backend can be overridden to pywake (already default)."""
    env = three_turbine_row(
        backend="pywake",
        n_passthrough=0.01,
        burn_in_passthroughs=0.0001,
        fill_window=1,
    )
    try:
        assert env.backend == "pywake"
        obs, _ = env.reset()
        assert obs is not None
    finally:
        env.close()
