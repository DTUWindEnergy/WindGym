"""MannLoad with memmap boxes (turb_memmap) and the shared Mann box spec.

Real (small) boxes, real envs: the memmap path must give the same turbulence as
the eager path, use a lazy field for BOTH the agent and the baseline site (no
deepcopy), and the MannGenerate spec must be configurable through the YAML.
"""

from pathlib import Path

import numpy as np
import pytest
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from dynamiks.views import Points

from WindGym.core import turbulence_manager as tm
from WindGym.presets import three_turbine_row

SMALL = dict(Nxyz=(256, 32, 32), dxyz=(4.0, 4.0, 4.0))


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("boxes")
    for seed in (0, 1):
        tf = MannTurbulenceField.generate(alphaepsilon=tm.MANN_ALPHAEPSILON, L=tm.MANN_L,
                                          Gamma=tm.MANN_GAMMA, seed=seed, **SMALL)
        tf.to_netcdf(filename=str(d / f"TF_seed_{seed}.nc"))
    return d


def _env(box_dir, memmap: bool, **kw):
    return three_turbine_row(
        turbtype="MannLoad", TurbBox=str(box_dir), seed=7, reset_init=False,
        config_overrides={"turb_memmap": memmap,
                          "wind": {"ws_min": 8, "ws_max": 8, "TI_min": 0.08, "TI_max": 0.08,
                                   "wd_min": 270, "wd_max": 270},
                          "power_def": {"Power_reward": "Baseline"}},
        **kw,
    )


def test_memmap_flag_reaches_turbulence_manager(box_dir):
    env = _env(box_dir, memmap=True)
    assert env.turb_memmap is True
    assert env.turbulence_manager.memmap_boxes is True
    assert env.turbulence_manager.mann_nxyz == tm.MANN_NXYZ
    env.close()
    env = _env(box_dir, memmap=False)
    assert env.turbulence_manager.memmap_boxes is False
    env.close()


def test_memmap_mannload_matches_eager_and_is_lazy_for_both_sites(box_dir):
    envs = {m: _env(box_dir, memmap=m) for m in (False, True)}
    fields = {}
    for m, env in envs.items():
        env.reset(seed=3)   # same seed -> same file choice, same ws/ti
        assert env.turbulence_manager.tf_file.endswith(".nc")
        fields[m] = (env.site.turbulenceField, env.site_base.turbulenceField)
    assert envs[False].turbulence_manager.tf_file == envs[True].turbulence_manager.tf_file
    for f in fields[False]:
        assert not f._memmap and isinstance(f.uvw, np.ndarray) and not isinstance(f.uvw, np.memmap)
    for f in fields[True]:
        assert f._memmap
        assert isinstance(getattr(f.uvw, "_uvw", f.uvw), np.memmap)
    assert MannTurbulenceField.sidecar_path(envs[True].turbulence_manager.tf_file).exists()
    # baseline of the memmap env is its own lazy handle, not a copy of the agent's
    assert fields[True][0] is not fields[True][1]
    # same scaled turbulence at the same points / time
    rng = np.random.default_rng(0)
    x, y, z = rng.uniform(50, 900, 200), rng.uniform(10, 110, 200), rng.uniform(20, 110, 200)
    for i in (0, 1):
        np.testing.assert_allclose(fields[False][i](x, y, z, time=0.0),
                                   fields[True][i](x, y, z, time=0.0), rtol=1e-5, atol=1e-6)
    # and the flow sims agree on the free-stream wind speed
    ws_eager = envs[False].fs.get_windspeed(Points(x, y, z), include_wakes=False)
    ws_mm = envs[True].fs.get_windspeed(Points(x, y, z), include_wakes=False)
    np.testing.assert_allclose(ws_eager, ws_mm, rtol=1e-5, atol=1e-4)
    for env in envs.values():
        env.close()


def test_memmap_env_steps(box_dir):
    env = _env(box_dir, memmap=True)
    obs, _ = env.reset(seed=1)
    for _ in range(3):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
    env.close()


def test_mann_generate_spec_is_configurable(monkeypatch):
    calls = {}
    real = MannTurbulenceField.generate

    def spy(**kw):
        calls.update(kw)
        return real(**kw)

    monkeypatch.setattr(MannTurbulenceField, "generate", staticmethod(spy))
    env = three_turbine_row(turbtype="MannGenerate", seed=5, reset_init=False,
                            config_overrides={"mann_nxyz": [128, 16, 16],
                                              "mann_dxyz_over_D": [0.1, 0.2, 0.2]})
    env.reset(seed=1)
    D = env.turbine.diameter()
    assert tuple(calls["Nxyz"]) == (128, 16, 16)
    np.testing.assert_allclose(calls["dxyz"], (0.1 * D, 0.2 * D, 0.2 * D))
    assert calls["alphaepsilon"] == tm.MANN_ALPHAEPSILON and calls["L"] == tm.MANN_L
    env.close()
