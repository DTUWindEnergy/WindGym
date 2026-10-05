"""Smoke tests for the live figures written by ``eval_single_fast(save_figs=True)``.

Each scenario writes 2-3 frames into ``tmp_path`` and checks the file names,
that every PNG decodes, and that no frame is blank. The frames themselves are
not committed; a one-off pixel diff against frames rendered from the
pre-refactor code is run separately (see the refactor notes).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from py_wake.examples.data.hornsrev1 import V80  # noqa: E402

from WindGym import AgentEvalFast, FarmEval  # noqa: E402
from test_eval_single_golden import (  # noqa: E402
    RandomModel,
    _derate_config,
    _op_lookup,
    _tracking_config,
)
from test_golden_episodes import _derating_turbine, _row  # noqa: E402
from test_utils import get_fast_pywake_config  # noqa: E402

pytestmark = pytest.mark.integration


def _pywake_env(config, turbine=None, n_turb=2, **kwargs):
    turbine = V80() if turbine is None else turbine
    x, y = _row(turbine, n_turb)
    return FarmEval(
        turbine=turbine,
        x_pos=x,
        y_pos=y,
        config=config,
        backend="pywake",
        n_passthrough=1,
        fill_window=False,
        reset_init=False,
        **kwargs,
    )


def make_yaw():
    return _pywake_env(get_fast_pywake_config()), {}


def make_debug():
    return _pywake_env(get_fast_pywake_config()), {"debug": True}


def make_derate_op_yaw():
    t = _derating_turbine()
    return _pywake_env(_derate_config(), turbine=t, op_lookup=_op_lookup(t)), {}


def make_derate_op_noyaw():
    t = _derating_turbine()
    cfg = _derate_config()
    cfg["yaw_action"] = False
    return _pywake_env(cfg, turbine=t, op_lookup=_op_lookup(t)), {}


def make_tracking():
    return _pywake_env(_tracking_config(), n_turb=3), {}


SCENARIOS = {
    "yaw": make_yaw,
    "debug": make_debug,
    "derate_op_yaw": make_derate_op_yaw,
    "derate_op_noyaw": make_derate_op_noyaw,
    "tracking": make_tracking,
}
T_SIM = 2  # -> 3 frames


def render_frames(name, out_dir):
    """Run one scenario, writing frames into ``out_dir``. Returns the frame paths."""
    env, extra = SCENARIOS[name]()
    model = RandomModel(env.action_space.shape)
    AgentEvalFast(
        env,
        model,
        1,
        ws=9.0,
        ti=0.06,
        wd=270.0,
        t_sim=T_SIM,
        seed=4,
        save_figs=True,
        fig_dir=str(out_dir),
        **extra,
    )
    return sorted(p for p in out_dir.iterdir() if p.suffix == ".png")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_live_figures(name, tmp_path):
    frames = render_frames(name, tmp_path)
    n_frames = T_SIM + 1
    assert [p.name for p in frames] == [f"img_{i:05d}.png" for i in range(n_frames)]
    for p in frames:
        img = mpimg.imread(p)
        assert img.ndim == 3 and img.shape[0] > 100 and img.shape[1] > 100
        assert img[..., :3].std() > 0.01, f"{p.name} is blank"
