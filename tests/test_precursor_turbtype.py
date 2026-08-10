"""Tests for turbtype="Precursor": memmap-backed LES-precursor inflow."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80

import dynamiks
from dynamiks.sites._site import TurbulenceFieldSite
from dynamiks.sites.precursor import convert_precursor

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid

FIXTURE_NC = Path(dynamiks.__file__).parent.parent / 'tests' / 'test_files' / 'precursor.nc'
FARM = dict(hub_height=70.0, Lfarm=[500.0, 840.0, 460.0], yref=430.0)  # V80 hub, fixture box

def _config(**wind_overrides):
    import yaml
    cfg_path = Path(__file__).parent.parent / 'examples' / 'EnvConfigs' / '2turb.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["wind"].update({"ws_min": 9, "ws_max": 9, "TI_min": 0.07, "TI_max": 0.07,
                        "wd_min": 270, "wd_max": 270, **wind_overrides})
    return cfg


CONFIG = _config()


@pytest.fixture(scope="module")
def nc_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("precursor") / 'precursor.nc'
    shutil.copy(FIXTURE_NC, p)
    convert_precursor(str(p), **FARM, chunk_t=32)
    return str(p)


@pytest.fixture()
def env(nc_path):
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=4, yDist=4)
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=CONFIG,
        turbtype="Precursor",
        TurbBox=nc_path,
        Baseline_comp=True,
        dt_sim=1,
        dt_env=5,
        max_time_steps=5,
        n_passthrough=0.1,
        burn_in_passthroughs=0.0001,
        fill_window=False,
        reset_init=False,
    )
    yield env
    env.close()


def test_precursor_env_pins_wind_and_shares_memmap(env):
    env.reset(seed=1)
    meta = env.turbulence_manager.precursor_meta
    assert env.ws == pytest.approx(float(meta['advection_speed']))
    assert env.wd == 270.0
    assert env.ti == pytest.approx(float(meta['ti_hub']))

    assert isinstance(env.site, TurbulenceFieldSite)
    assert env.site.turbulence_transport_speed == pytest.approx(env.ws)
    # agent and baseline share the SAME read-only memmap (no per-farm copy)
    assert isinstance(env.site.turbulenceField.uvw, np.memmap)
    assert env.site.turbulenceField.uvw is env.site_base.turbulenceField.uvw
    # same episode window for agent and baseline (comparable inflow)
    assert env.site.turbulence_offset[0] == env.site_base.turbulence_offset[0]


def test_precursor_random_windows_differ_between_resets(env):
    env.reset(seed=1)
    w1 = env.turbulence_manager.window_offset_s
    env.reset(seed=2)
    w2 = env.turbulence_manager.window_offset_s
    assert w1 != w2
    # reproducible per seed
    env.reset(seed=1)
    assert env.turbulence_manager.window_offset_s == w1


def test_precursor_episode_runs_to_truncation(env):
    env.reset(seed=3)
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
        assert not terminated
    assert truncated


def test_precursor_rejects_mann_dr_keys(env):
    with pytest.raises(ValueError, match="Mann keys"):
        env.reset(seed=1, options={"dwm_params": {"mann_L": 29.4}})


def test_precursor_rejects_veer(nc_path):
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=4, yDist=4)
    cfg = _config(veer_min=2.0, veer_max=2.0)
    env = WindFarmEnv(turbine=V80(), x_pos=x_pos, y_pos=y_pos, config=cfg,
                      turbtype="Precursor", TurbBox=nc_path, dt_sim=1, dt_env=5,
                      max_time_steps=5, n_passthrough=0.1, burn_in_passthroughs=0.0001,
                      fill_window=False, reset_init=False)
    try:
        with pytest.raises(ValueError, match="[Vv]eer"):
            env.reset(seed=1)
    finally:
        env.close()
