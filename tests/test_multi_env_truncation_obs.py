"""WindFarmEnvMulti returns the REAL final observation/info on the truncating
step (rebuilt from the parent's flat obs after the parent tore the sims down),
and forwards extra kwargs (max_time_steps, ...) to WindFarmEnv."""
import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80

from WindGym.wind_env_multi import WindFarmEnvMulti


def _cfg():
    return {
        "yaw_init": "Zeros", "BaseController": "Local", "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {"ws_min": 10, "ws_max": 10, "TI_min": 0.07, "TI_max": 0.07,
                 "wd_min": 270, "wd_max": 270},
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 2, "Power_scaling": 1.0},
        "mes_level": {"turb_ws": True, "turb_wd": False, "turb_TI": False,
                      "turb_power": True, "farm_ws": True, "farm_wd": True,
                      "farm_TI": False, "farm_power": True, "ti_sample_count": 2},
        "ws_mes": {"ws_current": True, "ws_rolling_mean": True, "ws_history_N": 1,
                   "ws_history_length": 2, "ws_window_length": 1},
        "wd_mes": {"wd_current": True, "wd_rolling_mean": False, "wd_history_N": 1,
                   "wd_history_length": 2, "wd_window_length": 1},
        "yaw_mes": {"yaw_current": True, "yaw_rolling_mean": True, "yaw_history_N": 1,
                    "yaw_history_length": 2, "yaw_window_length": 1},
        "power_mes": {"power_current": True, "power_rolling_mean": False,
                      "power_history_N": 1, "power_history_length": 2,
                      "power_window_length": 1},
    }


@pytest.fixture
def env():
    e = WindFarmEnvMulti(turbine=V80(), x_pos=[0, 400], y_pos=[0, 0], config=_cfg(),
                         backend="pywake", turbtype="None", dt_sim=1, dt_env=1,
                         max_time_steps=3)
    yield e
    e.close()


def test_kwargs_forwarded_to_parent(env):
    assert env.max_time_steps == 3          # reached WindFarmEnv.__init__
    assert env.n_passthrough == 999_999_999  # the passthrough method is disabled


def test_parent_obs_slicing_matches_multi_obs(env):
    obs, _ = env.reset(seed=1)
    rebuilt = env._obs_multi_from_parent(env._get_obs())
    for a in env.possible_agents:
        np.testing.assert_allclose(rebuilt[a], obs[a])
        assert obs[a].shape == (env.obs_var,)


def test_truncating_step_returns_real_obs_and_infos(env):
    env.reset(seed=1)
    acts = {a: np.array([0.3], np.float32) for a in env.possible_agents}
    for _ in range(2):
        obs, _, _, truncs, _ = env.step(acts)
        assert not any(truncs.values())
    obs, rewards, terms, truncs, infos = env.step(acts)
    assert all(truncs.values()) and not any(terms.values())
    assert env.fs is None                      # parent cleaned up ...
    for i, a in enumerate(env.possible_agents):
        assert obs[a].shape == (env.obs_var,)
        assert np.any(obs[a] != 0)             # ... but the obs is the real last one
        assert np.isfinite(infos[a]["Power turbine agent"])
        assert "yaw angles agent" in infos[a]
    assert env.agents == []
