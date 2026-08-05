import numpy as np
import pytest
from gymnasium.spaces import Box

from WindGym.wrappers.parallel_PettingZoo_wrapper import (
    ParallelPettingZooMultiprocessingWrapper,
)

AGENTS = ["agent_0", "agent_1"]


class DummyParallelEnv:
    """Minimal PettingZoo ParallelEnv stand-in for testing the multiprocessing wrapper."""

    def __init__(self, value=0, render_mode=None):
        self.value = value
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents = list(AGENTS)
        self._step_count = 0
        self._seed = None

    def observation_space(self, agent):
        return Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs = {agent: np.full(3, self.value, dtype=np.float32) for agent in self.agents}
        info = {"env_value": self.value}
        return obs, info

    def step(self, actions):
        self._step_count += 1
        obs = {
            agent: np.full(3, self.value + self._step_count, dtype=np.float32)
            for agent in self.agents
        }
        rewards = {agent: float(self.value) for agent in self.agents}
        dones = {agent: self._step_count >= 3 for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        infos = {agent: {"step": self._step_count} for agent in self.agents}
        return obs, rewards, dones, truncs, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return np.full((4, 4, 3), self.value, dtype=np.uint8)
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed


def make_env_fns(n_envs, render_mode=None):
    return [
        (lambda v=i: DummyParallelEnv(value=v, render_mode=render_mode))
        for i in range(n_envs)
    ]


@pytest.fixture
def wrapper():
    w = ParallelPettingZooMultiprocessingWrapper(make_env_fns(3))
    yield w
    w.close()


@pytest.mark.integration
def test_initialization(wrapper):
    assert wrapper.num_envs == 3
    assert wrapper.possible_agents == AGENTS
    assert wrapper.agents == AGENTS
    assert isinstance(wrapper.observation_space("agent_0"), Box)
    assert isinstance(wrapper.action_space("agent_0"), Box)


@pytest.mark.integration
def test_reset(wrapper):
    obs, info = wrapper.reset()

    assert set(obs.keys()) == set(AGENTS)
    for agent in AGENTS:
        assert len(obs[agent]) == 3
        np.testing.assert_array_equal(
            np.array(obs[agent]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        )

    assert info["env_value"] == [0, 1, 2]


@pytest.mark.integration
def test_step(wrapper):
    wrapper.reset()
    actions = {agent: [np.array([0.1])] * 3 for agent in AGENTS}

    obs, rewards, dones, truncs, infos = wrapper.step(actions)

    for agent in AGENTS:
        np.testing.assert_array_equal(
            np.array(obs[agent]), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        )
        assert rewards[agent] == [0.0, 1.0, 2.0]
        assert dones[agent] == [False, False, False]
        assert truncs[agent] == [False, False, False]
        assert [i["step"] for i in infos[agent]] == [1, 1, 1]


@pytest.mark.integration
def test_step_done_envs(wrapper):
    wrapper.reset()
    actions = {agent: [np.array([0.1])] * 3 for agent in AGENTS}

    for _ in range(3):
        _, _, dones, _, _ = wrapper.step(actions)

    for agent in AGENTS:
        assert dones[agent] == [True, True, True]


@pytest.mark.integration
def test_seed_smoke(wrapper):
    # Smoke test: seeding every sub-env should not raise or hang.
    wrapper.seed(42)


@pytest.mark.integration
def test_render_grid_rgb_array():
    w = ParallelPettingZooMultiprocessingWrapper(
        make_env_fns(3, render_mode="rgb_array")
    )
    try:
        grid = w.render_grid(mode="rgb_array", grid_shape=(1, 3))
        assert grid.shape == (4, 12, 3)
    finally:
        w.close()


@pytest.mark.integration
def test_render_grid_human(monkeypatch):
    from PIL import Image

    shown = []
    monkeypatch.setattr(Image.Image, "show", lambda self: shown.append(self))

    w = ParallelPettingZooMultiprocessingWrapper(
        make_env_fns(3, render_mode="rgb_array")
    )
    try:
        result = w.render_grid(mode="human", grid_shape=(1, 3))
        assert result is None
        assert len(shown) == 1
    finally:
        w.close()


@pytest.mark.integration
def test_close_is_idempotent(wrapper):
    wrapper.close()
    wrapper.close()
    for p in wrapper.processes:
        assert not p.is_alive()
