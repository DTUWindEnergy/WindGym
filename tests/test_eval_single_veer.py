"""``eval_single_fast`` pins veer and assembles the GIF (proj/lesrl features).

The dev_dynamiks merge resolved ``agent_eval.py`` to the wdest side and dropped
the ``veer=`` / ``gif_fps=`` kwargs: an eval then drew veer from the config's
``veer_min``/``veer_max`` range instead of the requested value, and saved
frames were never assembled into a GIF.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80
from test_golden_episodes import _row
from test_utils import get_fast_pywake_config

from WindGym import AgentEval, AgentEvalFast, FarmEval

pytestmark = pytest.mark.integration

HAS_GIF_ENCODER = (
    shutil.which("ffmpeg") is not None
    or importlib.util.find_spec("imageio") is not None
)


class VeerProbe:
    """Random actions; records the veer rate of the live DWM site per step."""

    def __init__(self, env, seed=123):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.site_veer = []

    def predict(self, obs, deterministic=False):
        self.site_veer.append(getattr(self.env.site, "veer_rate", 0.0))
        shape = self.env.action_space.shape
        return self.rng.uniform(-1, 1, shape).astype(np.float32), None


def _veer_range_env():
    """dynamiks env whose config would draw veer from U(0, 2) deg/100 m."""
    config = get_fast_pywake_config()
    config["wind"] = dict(config["wind"], veer_min=0.0, veer_max=2.0)
    x, y = _row(V80(), 2)
    return FarmEval(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=config,
        turbtype="None",
        Baseline_comp=False,
        dt_sim=1,
        dt_env=10,
        n_passthrough=0.5,
        burn_in_passthroughs=0.05,
        fill_window=1,
        reset_init=False,
    )


def test_veer_kwarg_pins_episode_veer():
    env = _veer_range_env()
    model = VeerProbe(env)
    AgentEvalFast(env, model, ws=8.0, ti=0.07, wd=270.0, veer=1.6, t_sim=20, seed=1)
    assert env.veer == pytest.approx(1.6)
    # create_sites takes deg per metre
    assert model.site_veer and np.allclose(model.site_veer, 0.016)


def test_veer_defaults_to_zero_not_config_range():
    env = _veer_range_env()
    model = VeerProbe(env)
    AgentEvalFast(env, model, ws=8.0, ti=0.07, wd=270.0, t_sim=20, seed=1)
    assert env.veer == 0.0
    assert np.allclose(model.site_veer, 0.0)


@pytest.mark.skipif(not HAS_GIF_ENCODER, reason="needs ffmpeg or imageio")
def test_saved_frames_become_gif(tmp_path):
    env = _veer_range_env()
    model = VeerProbe(env)
    ds = AgentEvalFast(
        env,
        model,
        ws=8.0,
        ti=0.07,
        wd=270.0,
        t_sim=20,
        seed=1,
        save_figs=True,
        fig_dir=str(tmp_path),
        gif_fps=4,
    )
    gif = ds.attrs["gif_path"]
    assert os.path.isfile(gif)
    assert os.path.dirname(gif) == str(tmp_path)


def test_agent_eval_set_condition_veer_reaches_episode():
    env = _veer_range_env()
    model = VeerProbe(env)
    tester = AgentEval(env=env, model=model, t_sim=20, seed=1)
    tester.set_condition(ws=8.0, ti=0.07, wd=270.0, veer=1.6)
    tester.eval_single()
    assert env.veer == pytest.approx(1.6)
    assert np.allclose(model.site_veer, 0.016)


def test_missing_gif_encoder_keeps_eval_result(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no ffmpeg
    monkeypatch.setitem(sys.modules, "imageio", None)  # no imageio
    monkeypatch.setitem(sys.modules, "imageio.v2", None)
    env = _veer_range_env()
    with pytest.warns(UserWarning, match="GIF"):
        ds = AgentEvalFast(
            env,
            VeerProbe(env),
            ws=8.0,
            ti=0.07,
            wd=270.0,
            t_sim=20,
            seed=1,
            save_figs=True,
            fig_dir=str(tmp_path),
        )
    assert "gif_path" not in ds.attrs
    assert (tmp_path / "img_00000.png").is_file()


def test_veer_none_keeps_config_draw():
    env = _veer_range_env()
    AgentEvalFast(
        env, VeerProbe(env), ws=8.0, ti=0.07, wd=270.0, veer=None, t_sim=20, seed=1
    )
    assert 0.0 < env.veer <= 2.0


@pytest.mark.skipif(not HAS_GIF_ENCODER, reason="needs ffmpeg or imageio")
def test_loads_dataset_carries_gif_path(tmp_path, monkeypatch):
    """return_loads on HAWC2 turbines: the GIF path rides on the loads dataset."""
    import xarray as xr

    import WindGym.agent_eval as agent_eval

    loads = xr.Dataset({"dummy": ("time", np.zeros(3))})
    monkeypatch.setattr(agent_eval, "load_hawc2_loads_dataset", lambda env, **kw: loads)
    monkeypatch.setattr(agent_eval, "close_hawc2_and_cleanup", lambda env, **kw: None)

    class HawcFlagProbe(VeerProbe):
        # HTC_path is only read when the turbines are built at reset; flag the
        # env as HAWC2 afterwards so the eval takes the return_loads branch.
        def predict(self, obs, deterministic=False):
            self.env.HTC_path = "fake.htc"
            return super().predict(obs, deterministic)

    env = _veer_range_env()
    ds_load = AgentEvalFast(
        env,
        HawcFlagProbe(env),
        ws=8.0,
        ti=0.07,
        wd=270.0,
        t_sim=20,
        seed=1,
        save_figs=True,
        fig_dir=str(tmp_path),
        return_loads=True,
    )
    assert ds_load is loads
    assert os.path.isfile(ds_load.attrs["gif_path"])
