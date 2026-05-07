"""Unit tests for DWMRandomizationWrapper.

These tests use a tiny stub env (no DWM physics, no Mann box) — the wrapper's
contract with the env is just ``reset(options=)`` forwarding, so we don't
need a real WindFarmEnv to verify it.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from WindGym.wrappers import DWMRandomizationWrapper


class _StubEnv(gym.Env):
    """Records the last ``options=`` passed to ``reset()``."""

    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.last_options = None
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        self.last_options = options
        self.reset_count += 1
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _const_sampler(value):
    """Sampler that always returns the same dict (a copy, so mutation is safe)."""
    return lambda rng: dict(value)


def test_wrapper_passes_sampled_params_to_env():
    env = _StubEnv()
    wrapped = DWMRandomizationWrapper(env, sampler=_const_sampler({"k1": 0.05}))
    wrapped.reset()
    assert env.last_options == {"dwm_params": {"k1": 0.05}}


def test_wrapper_resamples_each_reset():
    env = _StubEnv()
    seq = iter([{"k1": 0.01}, {"k1": 0.02}, {"k1": 0.03}])
    wrapped = DWMRandomizationWrapper(env, sampler=lambda rng: next(seq))
    wrapped.reset()
    k_first = env.last_options["dwm_params"]["k1"]
    wrapped.reset()
    k_second = env.last_options["dwm_params"]["k1"]
    wrapped.reset()
    k_third = env.last_options["dwm_params"]["k1"]
    assert (k_first, k_second, k_third) == (0.01, 0.02, 0.03)


def test_wrapper_uses_rng_for_reproducibility():
    """Same seed → same param stream across two independent wrapper instances."""

    def random_sampler(rng):
        return {"k1": float(rng.uniform(0.0, 1.0))}

    a = DWMRandomizationWrapper(_StubEnv(), sampler=random_sampler, seed=42)
    b = DWMRandomizationWrapper(_StubEnv(), sampler=random_sampler, seed=42)
    a.reset()
    b.reset()
    assert (
        a.env.last_options["dwm_params"]["k1"]
        == b.env.last_options["dwm_params"]["k1"]
    )


def test_wrapper_reseeds_when_caller_passes_seed():
    """A caller-supplied seed deterministically pins the param stream."""

    def random_sampler(rng):
        return {"k1": float(rng.uniform())}

    w = DWMRandomizationWrapper(_StubEnv(), sampler=random_sampler, seed=0)
    w.reset(seed=99)
    k1_first = w.env.last_options["dwm_params"]["k1"]
    w.reset(seed=99)
    k1_second = w.env.last_options["dwm_params"]["k1"]
    assert k1_first == k1_second  # both started fresh from seed=99


def test_caller_options_dwm_params_override_sampler_keys():
    env = _StubEnv()
    wrapped = DWMRandomizationWrapper(
        env,
        sampler=_const_sampler({"k1": 0.10, "k2": 0.02}),
    )
    # caller overrides only k1; k2 still comes from the sampler
    wrapped.reset(options={"dwm_params": {"k1": 0.99}})
    assert env.last_options["dwm_params"] == {"k1": 0.99, "k2": 0.02}


def test_wrapper_preserves_other_options_keys():
    """Non-dwm_params options should pass through untouched."""
    env = _StubEnv()
    wrapped = DWMRandomizationWrapper(env, sampler=_const_sampler({"k1": 0.05}))
    wrapped.reset(options={"layout_idx": 3})
    assert env.last_options == {"layout_idx": 3, "dwm_params": {"k1": 0.05}}


def test_sampler_returning_non_dict_raises():
    wrapped = DWMRandomizationWrapper(_StubEnv(), sampler=lambda rng: [1, 2, 3])
    with pytest.raises(TypeError, match="sampler must return dict"):
        wrapped.reset()


def test_non_callable_sampler_raises():
    with pytest.raises(TypeError, match="sampler must be callable"):
        DWMRandomizationWrapper(_StubEnv(), sampler=42)


def test_last_theta_is_none_before_first_reset():
    w = DWMRandomizationWrapper(_StubEnv(), sampler=_const_sampler({"k1": 0.05}))
    assert w.last_theta is None


def test_last_theta_reflects_applied_params_after_reset():
    w = DWMRandomizationWrapper(
        _StubEnv(),
        sampler=_const_sampler({"k1": 0.10, "k2": 0.02}),
    )
    w.reset()
    assert w.last_theta == {"k1": 0.10, "k2": 0.02}


def test_last_theta_includes_caller_overrides():
    """last_theta should reflect what was actually applied, including overrides."""
    w = DWMRandomizationWrapper(
        _StubEnv(),
        sampler=_const_sampler({"k1": 0.10, "k2": 0.02}),
    )
    w.reset(options={"dwm_params": {"k1": 0.99}})  # override k1
    assert w.last_theta == {"k1": 0.99, "k2": 0.02}


def test_last_theta_returns_a_copy():
    """Mutating the returned dict must not affect future reads."""
    w = DWMRandomizationWrapper(_StubEnv(), sampler=_const_sampler({"k1": 0.05}))
    w.reset()
    snapshot = w.last_theta
    snapshot["k1"] = 999.0  # try to corrupt internal state
    assert w.last_theta == {"k1": 0.05}
