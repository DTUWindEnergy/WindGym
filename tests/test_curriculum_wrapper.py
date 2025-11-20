"""
Unit tests for CurriculumWrapper

pytest, numpy, gymnasium are the only run-time deps.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
import pytest


# --------------------------------------------------------------------------- #
#  Tiny dummy env + stub PyWakeAgent                                          #
# --------------------------------------------------------------------------- #
class DummyEnv(gym.Env):
    """Deterministic, 2-turbine env exposing only what CurriculumWrapper uses."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )

        self.x_pos = [0.0, 1.0]
        self.y_pos = [0.0, 1.0]
        self.turbine = object()
        self.yaw_min, self.yaw_max = -45.0, 45.0

        # constant global wind
        self.ws, self.wd, self.ti = 10.0, 270.0, 0.05

    def reset(self, **_):
        return np.zeros(1, dtype=np.float32), {
            "yaw angles agent": np.array([10.0, -5.0])
        }

    def step(self, action):  # noqa: D401
        return (
            np.zeros(1, dtype=np.float32),
            10.0,
            False,
            False,
            {"yaw angles agent": np.array([10.0, -5.0])},
        )


class DummyPyWakeAgent:
    """Just delivers a fixed ‘optimal’ yaw vector; no optimisation."""

    def __init__(self, **_):
        self.optimized_yaws = np.array([5.0, -5.0])

    def update_wind(self, *_):
        pass

    def optimize(self):
        pass


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cw_mod():
    """Import once, share for all tests."""
    import importlib

    return importlib.import_module("WindGym.wrappers.curriculum_wrapper")


@pytest.fixture
def patched_cw(monkeypatch, cw_mod):
    """Patch heavy PyWakeAgent → zero-cost stub."""
    monkeypatch.setattr(cw_mod, "PyWakeAgent", DummyPyWakeAgent, raising=True)
    return cw_mod


@pytest.fixture
def wrapper(patched_cw):
    """CurriculumWrapper with constant env-weight 0.3."""

    def weight_fn(_step):
        return 0.3

    return patched_cw.CurriculumWrapper(DummyEnv(), n_envs=2, weight_function=weight_fn)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
AGENT = np.array([10.0, -5.0])
PYWAKE = np.array([5.0, -5.0])
DIFF = AGENT - PYWAKE
L2 = np.linalg.norm(DIFF)
MAX_L2 = np.sqrt(2) * 90.0  # 2 turbines, yaw-range 90°


def expected_similarity(kind: str, huber_kappa=1.0, exp_alpha=1.0) -> float:
    if kind == "l2":
        return -L2
    if kind == "l1":
        return -np.mean(np.abs(DIFF))
    if kind == "mse":
        return -np.mean(DIFF**2)
    if kind == "normalized_l2":
        return 1 - L2 / MAX_L2
    if kind == "exponential":
        return float(np.exp(-exp_alpha * L2))
    if kind == "cosine":
        denom = np.linalg.norm(AGENT) * np.linalg.norm(PYWAKE)
        return float(np.dot(AGENT, PYWAKE) / denom)
    if kind == "huber":
        abs_d = np.abs(DIFF)
        quad = abs_d <= huber_kappa
        loss = np.where(quad, 0.5 * DIFF**2, huber_kappa * (abs_d - 0.5 * huber_kappa))
        return -float(np.mean(loss))
    raise ValueError


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "similarity_type",
    ["l2", "l1", "mse", "normalized_l2", "exponential", "cosine", "huber"],
)
def test_similarity_and_reward(wrapper, similarity_type):
    """All similarity modes + blended reward equation."""
    wrapper.similarity_type = similarity_type
    wrapper.reset()
    _, rew, *_ = wrapper.step(action=np.zeros(2, dtype=np.float32))

    sim = expected_similarity(similarity_type)
    exp_rew = 0.3 * 10.0 + 0.7 * sim  # env_w = 0.3 at first step
    assert np.isclose(rew, exp_rew, atol=1e-6)


def test_yaw_check_goal(wrapper):
    """Check action-to-yaw scaling (yaw_check='goal')."""
    wrapper.yaw_check = "goal"  # keep default similarity 'normalized_l2'
    wrapper.reset()

    act = np.array([0.0, 0.5], dtype=np.float32)  # --> [0, 22.5] deg
    _, rew, *_ = wrapper.step(act)

    scaled = (act + 1) / 2 * 90 - 45
    assert np.allclose(scaled, [0.0, 22.5], atol=1e-6)

    diff = scaled - PYWAKE
    sim = 1 - np.linalg.norm(diff) / MAX_L2
    exp_rew = 0.3 * 10 + 0.7 * sim
    assert np.isclose(rew, exp_rew, atol=1e-6)


def test_weight_clipping(wrapper):
    """weight_function returning >1 must be clipped to 1."""
    wrapper.weight_function = lambda *_: 5.0
    wrapper.reset()
    _, rew, *_ = wrapper.step(np.zeros(2))
    assert rew == 10.0  # pure env-reward


def test_cosine_zero_div(patched_cw):
    """cosine similarity → 0 if either norm is 0."""
    env = DummyEnv()
    env.reset = lambda **_: (np.zeros(1), {"yaw angles agent": np.zeros(2)})
    w = patched_cw.CurriculumWrapper(env, n_envs=2, similarity_type="cosine")
    w.pywake_agent.optimized_yaws = np.zeros(2)
    w.reset()
    _, rew, *_ = w.step(np.zeros(2))
    assert rew == 10.0  # env_w defaults to 1.0


def test_bad_similarity_type(wrapper):
    """Unknown similarity_type raises ValueError."""
    wrapper.similarity_type = "does_not_exist"
    wrapper.reset()
    with pytest.raises(ValueError):
        wrapper.step(np.zeros(2))
