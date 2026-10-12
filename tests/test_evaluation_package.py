"""Unit tests for the WindGym.evaluation package (recorder, figures, HAWC2 loads)."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

import WindGym.agent_eval as agent_eval
from WindGym.agent_eval import AgentEval, eval_single_fast
from WindGym.evaluation import hawc2_loads
from WindGym.evaluation.hawc2_loads import (
    HAWC2_LOAD_COLUMNS,
    close_hawc2_and_cleanup,
    load_hawc2_loads_dataset,
)
from WindGym.evaluation.live_figures import resolve_fig_folder
from WindGym.evaluation.recorder import (
    FARM_DIMS,
    SCALAR_DIMS,
    TURB_DIMS,
    EpisodeRecorder,
)
from WindGym.evaluation.selection import select_action

pytestmark = pytest.mark.unit


# --- HAWC2 loads -------------------------------------------------------------


def test_gtsdf_module_identity_keeps_patch_target():
    assert hawc2_loads.gtsdf is agent_eval.gtsdf


def test_hawc2_load_columns_order():
    assert list(HAWC2_LOAD_COLUMNS) == [
        "Blade_Mx",
        "Blade_My",
        "Tower_Mx",
        "Tower_My",
        "Ae_rot_torque",
        "Ae_rot_power",
        "Ae_rot_thrust",
        "WSP_gl_coo_Vx",
        "WSP_gl_coo_Vy",
        "WSP_gl_coo_Vz",
        "yaw_a",
    ]
    assert HAWC2_LOAD_COLUMNS["Blade_Mx"] == 19 and HAWC2_LOAD_COLUMNS["yaw_a"] == 112


def _stub_hawc2_env(n_turb=2):
    htc_lst = [
        SimpleNamespace(
            modelpath="/model/",
            output=SimpleNamespace(filename=SimpleNamespace(values=[f"res/case/t{i}"])),
        )
        for i in range(n_turb)
    ]
    wts = SimpleNamespace(h2=MagicMock(), htc_lst=htc_lst)
    return SimpleNamespace(
        n_turb=n_turb,
        HTC_path="/model/htc/x.htc",
        wts=wts,
        wts_baseline=SimpleNamespace(h2=MagicMock()),
        _deleteHAWCfolder=MagicMock(),
        fs=object(),
        site=object(),
        farm_measurements=object(),
        fs_baseline=object(),
        site_base=object(),
    )


def test_load_hawc2_loads_dataset_from_stubbed_gtsdf(monkeypatch):
    env = _stub_hawc2_env()
    n_t = 5
    loaded = []

    def fake_load(path):
        loaded.append(path)
        data = np.tile(np.arange(120, dtype=float), (n_t, 1))  # data[:, c] == c
        return np.arange(n_t, dtype=float) * 0.1 + len(loaded), data, {}

    monkeypatch.setattr(hawc2_loads.gtsdf, "load", fake_load)
    ds = load_hawc2_loads_dataset(
        env, ws=8.0, wd=270, ti=0.07, turbbox="Default", model_step=3
    )

    env.wts.h2.write_output.assert_called_once()
    assert loaded == ["/model/res/case/t0.hdf5", "/model/res/case/t1.hdf5"]
    assert list(ds.data_vars) == list(HAWC2_LOAD_COLUMNS)
    for name, col in HAWC2_LOAD_COLUMNS.items():
        da = ds[name]
        assert da.dims == ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step")
        assert da.shape == (n_t, 2, 1, 1, 1, 1, 1)
        assert da.dtype == np.float64
        assert np.all(da.values == col)
    # time vector from turbine 0 (offset +1 from the first load)
    np.testing.assert_allclose(ds.time.values, np.arange(n_t) * 0.1 + 1)
    assert list(ds.coords) == [
        "ws",
        "wd",
        "turb",
        "time",
        "TI",
        "turbbox",
        "model_step",
    ]
    assert ds.model_step.values.tolist() == [3]


def test_close_hawc2_and_cleanup_closes_and_deletes():
    env = _stub_hawc2_env()
    close_hawc2_and_cleanup(env, cleanup=True, baseline_comp=True)
    env.wts.h2.close.assert_called_once()
    env.wts_baseline.h2.close.assert_called_once()
    env._deleteHAWCfolder.assert_called_once()
    for attr in ("fs", "site", "farm_measurements", "fs_baseline", "site_base"):
        assert not hasattr(env, attr)


def test_close_hawc2_without_cleanup_keeps_state():
    env = _stub_hawc2_env()
    close_hawc2_and_cleanup(env, cleanup=False, baseline_comp=False)
    env.wts.h2.close.assert_called_once()
    env.wts_baseline.h2.close.assert_not_called()
    env._deleteHAWCfolder.assert_not_called()
    assert hasattr(env, "fs")


# --- action selection --------------------------------------------------------


def test_select_action_predict_path():
    model = SimpleNamespace(predict=lambda obs, deterministic: (np.array([0.5]), None))
    a = select_action(model, np.zeros(3), deterministic=True, device="cpu")
    assert a.tolist() == [0.5]


def test_select_action_cleanrl_path():
    import torch

    class M:
        model_type = "CleanRL"

        def get_action(self, obs, deterministic):
            assert obs.shape == (1, 3)
            return torch.tensor([[0.25, -0.25]]), None, None

    a = select_action(M(), np.zeros(3), deterministic=False, device=torch.device("cpu"))
    assert a.shape == (2,) and a.tolist() == [0.25, -0.25]


def test_select_action_unknown_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        select_action(
            SimpleNamespace(model_type="Other"),
            np.zeros(1),
            deterministic=False,
            device="cpu",
        )


# --- recorder ----------------------------------------------------------------


def test_recorder_dataset_variable_order_and_coords():
    rec = EpisodeRecorder(
        4, 2, baseline=True, op_mode=True, log_derate=True, tracking=True
    )
    rec.time_plot[:] = np.arange(4)
    ds = rec.to_dataset(
        ws=8, wd=270.0, ti=0.07, turbbox="Default", model_step=1, deterministic=True
    )
    assert list(ds.data_vars) == [
        "powerF_a",
        "powerT_a",
        "yaw_a",
        "ws_a",
        "reward",
        "pitch_a",
        "rpm_a",
        "derate_a",
        "power_ref",
        "track_err",
        "track_mae",
        "powerF_b",
        "powerT_b",
        "yaw_b",
        "ws_b",
        "pct_inc",
    ]
    assert ds.powerF_a.dims == FARM_DIMS and ds.powerT_a.dims == TURB_DIMS
    assert ds.track_mae.dims == SCALAR_DIMS
    assert ds.powerF_a.shape == (4, 1, 1, 1, 1, 1, 1)
    assert ds.powerT_a.shape == (4, 2, 1, 1, 1, 1, 1, 1)
    assert all(ds[v].dtype == np.float32 for v in ds.data_vars)
    assert ds.ws.dtype == np.int64  # coords passed uncast
    assert ds.turbbox.dtype.kind == "U"
    assert ds.time.dtype.kind == "f"  # sim time in seconds; dt_sim may be sub-second
    assert list(ds.coords) == [
        "ws",
        "wd",
        "turb",
        "time",
        "TI",
        "turbbox",
        "model_step",
        "deterministic",
    ]


def test_recorder_keeps_sub_second_sim_time():
    """dt_sim < 1 s (the G1 runs at 0.125 s): the time coord must carry the
    fractional sim time, or every second repeats step_val times and the
    per-case datasets cannot be concatenated (duplicate index)."""
    step_val = 4
    rec = EpisodeRecorder(
        1 + 2 * step_val,
        1,
        baseline=False,
        op_mode=False,
        log_derate=False,
        tracking=False,
    )
    for i in range(2):
        t_arr = np.arange(1, step_val + 1) * 0.125 + i * 0.5
        info = {
            "powers": np.zeros((step_val, 1)),
            "yaws": np.zeros((step_val, 1)),
            "windspeeds": np.zeros((step_val, 1)),
            "time_array": t_arr,
        }
        rec.record_step(i, step_val, info, 0.0)
    ds = rec.to_dataset(
        ws=5.5,
        wd=270.0,
        ti=0.06,
        turbbox="MannGenerate",
        model_step=0,
        deterministic=True,
    )
    np.testing.assert_array_equal(ds.time.values, np.arange(0, 9) * 0.125)
    assert np.unique(ds.time.values).size == 9


def test_recorder_minimal_has_only_core_variables():
    rec = EpisodeRecorder(
        3, 1, baseline=False, op_mode=False, log_derate=False, tracking=False
    )
    ds = rec.to_dataset(
        ws=8.0, wd=270.0, ti=0.07, turbbox="Default", model_step=1, deterministic=False
    )
    assert list(ds.data_vars) == ["powerF_a", "powerT_a", "yaw_a", "ws_a", "reward"]
    assert not hasattr(rec, "powerF_b")


# --- figure folder -----------------------------------------------------------


def test_resolve_fig_folder():
    assert resolve_fig_folder("/tmp/x", "n", 8.0, 270) == os.path.join("/tmp/x", "")
    assert resolve_fig_folder(None, "run", 8.0, 270) == "./Temp_Figs_run_ws8.0_wd270/"


# --- AgentEval / eval_single_fast surface -------------------------------------


def test_run_simulation_signature():
    sig = inspect.signature(AgentEval.run_simulation)
    params = sig.parameters
    assert list(params)[:8] == [
        "self",
        "winddir",
        "windspeed",
        "TI",
        "box",
        "save_figs",
        "scale_obs",
        "debug",
    ]
    assert params["return_loads"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["seed"].kind is inspect.Parameter.KEYWORD_ONLY


def test_eval_single_fast_signature_unchanged():
    assert list(inspect.signature(eval_single_fast).parameters) == [
        "env",
        "model",
        "model_step",
        "ws",
        "ti",
        "wd",
        "turbbox",
        "save_figs",
        "scale_obs",
        "t_sim",
        "name",
        "debug",
        "deterministic",
        "return_loads",
        "cleanup",
        "seed",
        "fig_dir",
    ]


@pytest.mark.integration
def test_return_loads_without_htc_returns_none():
    from test_eval_single_golden import RandomModel, make_b

    env = make_b()
    out = eval_single_fast(
        env,
        RandomModel(env.action_space.shape),
        ws=9.0,
        ti=0.06,
        wd=270.0,
        t_sim=1,
        return_loads=True,
    )
    assert out is None
    assert isinstance(
        eval_single_fast(
            make_b(),
            RandomModel(env.action_space.shape),
            ws=9.0,
            ti=0.06,
            wd=270.0,
            t_sim=1,
        ),
        xr.Dataset,
    )


def test_eval_multiple_sets_conditions_once_per_episode(monkeypatch):
    ev = AgentEval(env=SimpleNamespace(), model=object(), name="n", t_sim=1, seed=0)
    ev.set_conditions(winddirs=[260, 280], windspeeds=[8], turbintensities=[0.05])
    calls = []
    monkeypatch.setattr(
        ev, "set_env_vals", lambda: calls.append((ev.ws, ev.wd, ev.ti, ev.turbbox))
    )
    monkeypatch.setattr(
        ev,
        "eval_single",
        lambda **kw: xr.Dataset({"x": (("wd",), [1.0])}, coords={"wd": [ev.wd]}),
    )
    ds = ev.eval_multiple()
    assert calls == [(8, 260, 0.05, "Default"), (8, 280, 0.05, "Default")]
    assert ds.wd.values.tolist() == [260, 280]
