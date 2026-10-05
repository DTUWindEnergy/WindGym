"""FarmEval / TF_files / WindManager.fix_conditions behaviour after the refactor."""

import numpy as np
import pytest
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from py_wake.examples.data.hornsrev1 import V80

from WindGym import FarmEval
from WindGym.core.wind_manager import WindManager
from test_turbulent_loading import create_mock_mann_field_instance
from test_utils import get_fast_pywake_config

pytestmark = pytest.mark.integration


class _CountingRng:
    """Wraps a Generator and counts uniform() draws."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.uniform_calls = 0

    def uniform(self, *a, **k):
        self.uniform_calls += 1
        return self._rng.uniform(*a, **k)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_fix_conditions_sets_degenerate_ranges_and_keeps_draw_count():
    wm = WindManager(ws_min=4, ws_max=25, wd_min=0, wd_max=360, ti_min=0.01, ti_max=0.2)
    wm.np_random = _CountingRng(0)
    wm.sample_conditions()
    assert wm.np_random.uniform_calls == 3  # ws, wd, ti (veer degenerate)

    wm.fix_conditions(ws=9, ti=0.06, wd=268)
    assert (wm.ws_min, wm.ws_max) == (9, 9)
    assert (wm.ti_min, wm.ti_max) == (0.06, 0.06)
    assert (wm.wd_min, wm.wd_max) == (268, 268)
    assert (wm.veer_min, wm.veer_max) == (0.0, 0.0)  # untouched

    cond = wm.sample_conditions()
    assert wm.np_random.uniform_calls == 6  # degenerate ranges still draw
    assert (cond.wind_speed, cond.wind_direction, cond.turbulence_intensity) == (
        9.0,
        268.0,
        0.06,
    )

    wm.fix_conditions(veer=0.5)
    assert (wm.veer_min, wm.veer_max) == (0.5, 0.5)
    assert wm.sample_conditions().veer == 0.5


def _farm_eval(**kwargs):
    d = V80().diameter()
    defaults = dict(
        turbine=V80(),
        x_pos=np.array([0.0, 5 * d]),
        y_pos=np.zeros(2),
        config=get_fast_pywake_config(),
        backend="pywake",
        reset_init=False,
        n_passthrough=1,
        fill_window=False,
    )
    defaults.update(kwargs)
    return FarmEval(**defaults)


def test_farm_eval_forwards_base_kwargs_and_defaults():
    env = _farm_eval(dt_sim=1, dt_env=1, delay=3, max_turb_move=7)
    assert env.delay == 3
    assert env.max_turb_move == 7
    assert env.turbtype == "MannGenerate"
    assert env.yaw_init == "Zeros"
    assert env.finite_episode is False
    assert env.kwargs["finite_episode"] is False
    assert env.kwargs["delay"] == 3


def test_farm_eval_yaml_path_is_raw_config(tmp_path):
    cfg = get_fast_pywake_config()
    env = _farm_eval(config=cfg)
    assert env.yaml_path is cfg
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    env = _farm_eval(config=str(p))
    assert env.yaml_path == str(p)


def test_farm_eval_clone_from_kwargs():
    env = _farm_eval(finite_episode=True)
    clone = type(env)(**env.kwargs)
    assert clone.finite_episode is True
    assert clone.turbtype == env.turbtype


def test_set_wind_vals_fixes_manager_and_inflow_attributes():
    env = _farm_eval()
    env.set_wind_vals(ws=9, ti=0.06, wd=268)
    assert (env.ws, env.ti, env.wd) == (9, 0.06, 268)
    assert (env.ws_inflow_min, env.ws_inflow_max) == (9, 9)
    assert (env.TI_inflow_min, env.TI_inflow_max) == (0.06, 0.06)
    assert (env.wd_inflow_min, env.wd_inflow_max) == (268, 268)
    wm = env.wind_manager
    assert (wm.ws_min, wm.ws_max, wm.ti_min, wm.ti_max, wm.wd_min, wm.wd_max) == (
        9,
        9,
        0.06,
        0.06,
        268,
        268,
    )
    env.reset(seed=0)
    assert (env.ws, env.ti, env.wd) == (9.0, 0.06, 268.0)


def test_tf_files_property_routes_to_turbulence_manager():
    env = _farm_eval(turbtype="None")
    assert env.TF_files == env.turbulence_manager.turbulence_files
    with pytest.warns(UserWarning, match="MannLoad"):
        env.TF_files = ["/x/TF_a.nc"]
    assert env.turbulence_manager.turbulence_files == ["/x/TF_a.nc"]
    assert env.TF_files == ["/x/TF_a.nc"]


def test_update_tf_selects_the_given_box(tmp_path, monkeypatch):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    file_a = box_dir / "TF_a.nc"
    file_a.write_text("dummy")
    file_b = tmp_path / "TF_b.nc"
    file_b.write_text("dummy")

    loaded = []

    def mock_from_netcdf(filename, **kwargs):
        loaded.append(filename)
        return create_mock_mann_field_instance(monkeypatch)

    monkeypatch.setattr(MannTurbulenceField, "from_netcdf", mock_from_netcdf)

    d = V80().diameter()
    env = FarmEval(
        turbine=V80(),
        x_pos=np.array([0.0, 5 * d]),
        y_pos=np.zeros(2),
        config=get_fast_pywake_config(),
        turbtype="MannLoad",
        TurbBox=str(box_dir),
        reset_init=False,
        n_passthrough=0.5,
        burn_in_passthroughs=0.05,
        fill_window=1,
    )
    assert env.TF_files == [str(file_a)]
    env.update_tf(str(file_b))
    env.reset(seed=1)
    assert env.turbulence_manager.tf_file == str(file_b)
    assert loaded == [str(file_b)]
