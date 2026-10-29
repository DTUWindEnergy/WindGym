"""WindFarmEnv(yaw_frame_source="est"): the agent reads and commands yaw in the
frame of its OWN wind-direction estimate (what a deployed controller does),
while the simulator keeps its true-wind frame. See core/yaw_frame.py.
"""
import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.core.episode_timing import max_wd_step
from test_utils import get_fast_pywake_config


class _BiasedEstimateEnv(WindFarmEnv):
    """The pywake adapter measures wd == true wd (v = w = 0), so the estimator
    would be exact; bias it by +5 deg to make the frame conversion visible."""

    BIAS = 5.0

    @property
    def wd_est(self):
        return float(self.wd) + self.BIAS


def _env(cls=_BiasedEstimateEnv, **kw):
    d = V80().diameter()
    kw.setdefault("wd_est_tau", 30.0)
    kw.setdefault("max_turb_move", 1000.0)   # frame slew never binds unless a test lowers it
    return cls(turbine=V80(), x_pos=np.arange(2) * 5.0 * d, y_pos=np.zeros(2),
               config=get_fast_pywake_config(), backend="pywake", reset_init=False,
               n_passthrough=1, fill_window=False, yaw_step_sim=5.0, **kw)


def test_default_frame_is_the_true_wind():
    env = _env(cls=WindFarmEnv)
    env.reset(seed=0)
    assert env.yaw_frame_source == "true"
    assert env.frame_offset == 0.0
    env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(env.current_yaw_ctrl, env.current_yaw)
    env.close()


def test_est_frame_refuses_an_env_without_an_estimator():
    env = _env(cls=WindFarmEnv, wd_est_tau=None, yaw_frame_source="est")
    with pytest.raises(ValueError, match="estimator"):
        env.reset(seed=0)
    env.close()


def test_est_frame_rejects_unknown_source():
    with pytest.raises(ValueError):
        _env(yaw_frame_source="grid")


def test_est_frame_converts_observation_and_target():
    env = _env(yaw_frame_source="est")
    env.reset(seed=0)
    # "wind" action 0 = target 0 in the CONTROLLER frame = aligned with the believed wind,
    # which sits 5 deg clockwise of the true wind: the simulator-frame command goes to -5.
    for _ in range(3):
        env.step(np.zeros(2, dtype=np.float32))
    assert env.frame_offset == pytest.approx(5.0)
    np.testing.assert_allclose(env.yaw_command, -5.0, atol=1e-9)
    np.testing.assert_allclose(env.current_yaw_ctrl, 0.0, atol=1e-9)          # what the agent sees
    np.testing.assert_allclose(env.current_yaw, -5.0, atol=1e-9)              # physical (info["yaws"], yaw_a)
    # the measurement deque (the obs) holds the controller-frame yaw
    yaw_obs = env.farm_measurements.turb_mes[0].yaw.measurements[-1]
    assert yaw_obs == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_est_frame_yaw_band_applies_in_the_controller_frame():
    env = _env(yaw_frame_source="est")
    env.reset(seed=0)
    for _ in range(12):   # yaw_max 30 at 5 deg / sim step
        env.step(np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(env.current_yaw_ctrl, 30.0, atol=1e-9)   # the band of the agent's frame
    np.testing.assert_allclose(env.yaw_command, 25.0, atol=1e-9)         # simulator frame = 30 - 5
    env.close()


def test_est_frame_slews_at_the_true_wd_rate_limit():
    env = _env(yaw_frame_source="est", max_turb_move=2.0)
    env.reset(seed=0)
    lim = max_wd_step(np.column_stack([env.x_pos, env.y_pos]), env.D, 2.0)   # deg per sim step
    assert 0 < lim < 1.0
    env.step(np.zeros(2, dtype=np.float32))
    n_sim = env.sim_steps_per_env_step
    # reset's one warm-up env step + this step: the frame moved min(5, n_sim_total * lim)
    assert env.phi_frame - env.wd == pytest.approx(min(5.0, 2 * n_sim * lim), abs=1e-9)
    env.close()


def test_current_yaw_ctrl_is_derived_so_a_subclassed_measurement_still_steps():
    """Review 2026-10-29: tests (and users) override _take_measurements without calling super
    (tests/test_ti_calculation.py); the agent-frame yaw must not depend on that method
    having set an attribute."""

    class _Bare(WindFarmEnv):
        def _take_measurements(self):
            self.current_ws = np.full(self.n_turb, 10.0)
            self.current_wd = np.full(self.n_turb, 270.0)
            self.current_yaw = np.array(self.fs.windTurbines.yaw, dtype=float)
            self.current_powers = np.ones(self.n_turb)

    env = _env(cls=_Bare, wd_est_tau=None)
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(env.current_yaw_ctrl, env.current_yaw)
    env.close()
