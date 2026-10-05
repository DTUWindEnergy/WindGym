"""Golden characterization tests for ``eval_single_fast`` datasets.

Each scenario runs ``eval_single_fast`` on a ``FarmEval`` env with a stub
model whose actions come from an independent numpy generator, and compares the
returned ``xarray.Dataset`` against ``tests/golden/eval_single_<name>.nc``:
exact variable order, dims, dtype and shape per variable, values to
``rtol=1e-6``, and coordinate names / values.

Regenerate only on purpose (runs twice, refuses to write unless identical)::

    WINDGYM_UPDATE_GOLDEN=1 pytest tests/test_eval_single_golden.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from py_wake.examples.data.hornsrev1 import V80

from WindGym import AgentEvalFast, FarmEval
from WindGym.core.operating_point import OperatingPointLookup
from test_golden_episodes import _derating_turbine, _row
from test_utils import get_fast_pywake_config

pytestmark = pytest.mark.integration

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("WINDGYM_UPDATE_GOLDEN") == "1"


class RandomModel:
    """Stub agent: uniform random actions from a private generator."""

    def __init__(self, shape, seed=123):
        self.shape = shape
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):
        return self.rng.uniform(-1, 1, self.shape).astype(np.float32), None


def _tracking_config():
    return get_fast_pywake_config(
        Track_power=True,
        power_def={"Power_reward": "None", "Power_avg": 5, "Power_scaling": 1.0},
    )


def _derate_config():
    return get_fast_pywake_config(
        derate_action=True, derate_min=0.0, derate_max=0.8, derate_method="absolute"
    )


def _op_lookup(turbine):
    ws = np.linspace(0.0, 30.0, 16)
    yaw = np.array([-45.0, 45.0])
    derate = np.linspace(0.0, 0.8, 5)
    shape = (ws.size, yaw.size, derate.size)
    return OperatingPointLookup(
        ws=ws,
        yaw=yaw,
        derate=derate,
        pitch=np.broadcast_to(10.0 * derate, shape),
        tsr=np.broadcast_to(8.0 - 5.0 * derate, shape),
        rotor_diameter=turbine.diameter(),
    )


def make_a():
    """dynamiks, no turbulence, dt_env=10 (sub-step slicing), baseline."""
    x, y = _row(V80(), 2)
    return FarmEval(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=get_fast_pywake_config(),
        turbtype="None",
        Baseline_comp=True,
        dt_sim=1,
        dt_env=10,
        n_passthrough=0.5,
        burn_in_passthroughs=0.05,
        fill_window=1,
        reset_init=False,
    )


def make_b():
    """pywake power-tracking env."""
    x, y = _row(V80(), 3)
    return FarmEval(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=_tracking_config(),
        backend="pywake",
        n_passthrough=1,
        fill_window=False,
        reset_init=False,
    )


def make_c():
    """pywake derating env with a synthetic operating-point lookup."""
    turbine = _derating_turbine()
    x, y = _row(turbine, 3)
    return FarmEval(
        turbine=turbine,
        x_pos=x,
        y_pos=y,
        config=_derate_config(),
        backend="pywake",
        op_lookup=_op_lookup(turbine),
        n_passthrough=1,
        fill_window=False,
        reset_init=False,
    )


# name -> (factory, eval kwargs)
SCENARIOS = {
    "A": (make_a, dict(ws=8.0, ti=0.07, wd=270.0, t_sim=30, seed=1)),
    "B": (make_b, dict(ws=9.0, ti=0.06, wd=270.0, t_sim=5, seed=2)),
    "C": (make_c, dict(ws=9.0, ti=0.06, wd=270.0, t_sim=5, seed=3)),
    "B_int_ws": (make_b, dict(ws=8, ti=0.06, wd=270.0, t_sim=5, seed=2)),
}


def run_eval(factory, kwargs):
    env = factory()
    model = RandomModel(env.action_space.shape)
    ds = AgentEvalFast(env, model, 1, deterministic=True, **kwargs)
    return ds.load() if hasattr(ds, "load") else ds


def _coord_values(da):
    v = np.asarray(da.values)
    return v.astype(str) if v.dtype.kind in "UOS" else v


def assert_dataset_matches(actual: xr.Dataset, golden: xr.Dataset):
    assert list(actual.data_vars) == list(golden.data_vars)
    for name in golden.data_vars:
        a, g = actual[name], golden[name]
        assert a.dims == g.dims, name
        assert a.shape == g.shape, name
        assert a.dtype == g.dtype, (name, a.dtype, g.dtype)
        np.testing.assert_allclose(a.values, g.values, rtol=1e-6, err_msg=name)
    assert list(actual.coords) == list(golden.coords)
    for name in golden.coords:
        a, g = actual.coords[name], golden.coords[name]
        assert a.dims == g.dims, name
        np.testing.assert_array_equal(_coord_values(a), _coord_values(g), err_msg=name)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_eval_single_golden(name):
    factory, kwargs = SCENARIOS[name]
    path = GOLDEN_DIR / f"eval_single_{name}.nc"

    ds = run_eval(factory, kwargs)
    assert isinstance(ds, xr.Dataset)

    if UPDATE:
        ds2 = run_eval(factory, kwargs)
        assert_dataset_matches(ds2, ds)
        GOLDEN_DIR.mkdir(exist_ok=True)
        ds.to_netcdf(path)
        return

    assert path.exists(), f"missing golden {path}; run with WINDGYM_UPDATE_GOLDEN=1"
    with xr.open_dataset(path) as golden:
        golden = golden.load()
    assert_dataset_matches(ds, golden)


def test_coords_passed_uncast():
    """``ws=8`` (int) gives an int64 coordinate; ``turbbox`` stays a unicode coord."""
    ds = run_eval(make_b, dict(ws=8, ti=0.06, wd=270.0, t_sim=2, seed=2))
    assert ds.ws.dtype == np.int64
    assert ds.turbbox.dtype.kind == "U"
    assert ds.time.dtype.kind == "i"
    assert ds.powerF_a.dtype == np.float32
