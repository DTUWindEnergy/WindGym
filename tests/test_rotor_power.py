"""dwm_params["rotor_power"]: how a turbine turns the wind over its rotor into power.

Unit tests on hand-set rotor nodes (no DWM), then the env routing: constructor
and per-episode keys, both farms, and the composition with turbine-parameter DR."""

import numpy as np
import pytest
from dynamiks.wind_turbines import PyWakeWindTurbines
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.backend.param_turbines import ParamPyWakeWindTurbines
from WindGym.backend.rotor_power import (CubeMeanParamPyWakeWindTurbines,
                                         CubeMeanPowerWindTurbines)
from WindGym.core.dwm_defaults import (DWM_PARAM_SPEC, ROTOR_POWER_CHOICES, make_wts,
                                       validate_dwm_params)
from WindGym.core.turbine_params import normalize_turbine_params

X, Y = [0.0, 400.0, 800.0], [0.0, 0.0, 0.0]
# a known non-uniform set of rotor nodes, the same on every turbine
U = np.array([4.0, 6.0, 8.0, 10.0])
W = np.array([0.1, 0.2, 0.3, 0.4])
CUBE_MEAN = np.average(U ** 3, weights=W) ** (1 / 3)


def _stub_nodes(wts, u=U, w=W):
    """Feed fixed rotor nodes to power(); yaw-free kwargs."""
    uvw = np.zeros((3, wts.N, u.size))
    uvw[0] = u
    wts._get_rotor_windspeed = lambda include_wakes, idx=slice(None): (uvw[:, idx], w)
    wts.get_kwargs = lambda idx=slice(None): {}
    return wts


def _tp(params):
    return normalize_turbine_params(params, n_turb=3, dt_sim=1.0)


# ---------------------------------------------------------------- make_wts ---

def test_default_is_dynamiks_own_class():
    assert ROTOR_POWER_CHOICES == ("mean", "cubic")
    assert DWM_PARAM_SPEC["rotor_power"].default == "mean"
    assert type(make_wts(X, Y, V80())) is PyWakeWindTurbines
    assert type(make_wts(X, Y, V80(), rotor_power="mean")) is PyWakeWindTurbines


@pytest.mark.parametrize("rotor_power, params, cls", [
    ("cubic", None, CubeMeanPowerWindTurbines),
    ("cubic", {}, CubeMeanPowerWindTurbines),               # all-default vector = nominal
    ("mean", {"cp_gain": 1.1}, ParamPyWakeWindTurbines),
    ("cubic", {"cp_gain": 1.1}, CubeMeanParamPyWakeWindTurbines),
    ("CUBIC", None, CubeMeanPowerWindTurbines),             # choices are case-insensitive
])
def test_class_selection(rotor_power, params, cls):
    tp = None if params is None else _tp(params)
    assert type(make_wts(X, Y, V80(), turbine_params=tp, rotor_power=rotor_power)) is cls


def test_composed_mro_puts_cube_mean_under_the_dr_class():
    mro = CubeMeanParamPyWakeWindTurbines.__mro__
    assert mro.index(ParamPyWakeWindTurbines) < mro.index(CubeMeanPowerWindTurbines) \
        < mro.index(PyWakeWindTurbines)


def test_unknown_choice_raises():
    with pytest.raises(ValueError, match="rotor_power"):
        make_wts(X, Y, V80(), rotor_power="median")
    with pytest.raises(ValueError, match="rotor_power"):
        validate_dwm_params({"rotor_power": "median"})


# ------------------------------------------------------------------ power ---

def test_cubic_power_is_evaluated_at_the_cube_mean():
    wts = _stub_nodes(make_wts(X, Y, V80(), rotor_power="cubic"))
    np.testing.assert_allclose(wts.power(), V80().power(np.full(3, CUBE_MEAN)))
    assert np.all(wts.power() > V80().power(np.average(U, weights=W)))


def test_uniform_rotor_gives_the_same_power_either_way():
    u = np.full(4, 7.3)
    wts = _stub_nodes(make_wts(X, Y, V80(), rotor_power="cubic"), u=u)
    np.testing.assert_allclose(wts.power(), V80().power(np.full(3, 7.3)), rtol=1e-12)


def test_dr_gains_apply_on_top_of_the_cube_mean():
    """cp_gain scales the cube-mean power, not the arithmetic-mean one."""
    tp = _tp({"cp_gain": [1.2, 1.0, 0.8]})
    wts = _stub_nodes(make_wts(X, Y, V80(), turbine_params=tp, rotor_power="cubic"))
    np.testing.assert_allclose(wts.power(),
                               V80().power(np.full(3, CUBE_MEAN)) * [1.2, 1.0, 0.8])


# -------------------------------------------------------------------- env ---

def make_env(**env_kwargs):
    from test_golden_episodes import _dynamiks_config, _row
    x, y = _row(V80(), 3)
    cfg = _dynamiks_config(
        power_def={"Power_reward": "Wake_recovery", "Power_avg": 5, "Power_scaling": 1.0},
        ActionMethod="wind",
    )
    return WindFarmEnv(
        turbine=V80(), x_pos=x, y_pos=y, config=cfg, backend="dynamiks",
        turbtype="None", Baseline_comp=True, yaw_init="Zeros",
        dt_sim=1, dt_env=2, delay=4, n_passthrough=0.5, burn_in_passthroughs=0.05,
        fill_window=1, reset_init=False, **env_kwargs,
    )


def _farm_types(env):
    return type(env.fs.windTurbines), type(env.fs_baseline.windTurbines)


@pytest.mark.integration
def test_env_default_and_constructor_key_reach_both_farms():
    env = make_env()
    env.reset(seed=0)
    assert _farm_types(env) == (PyWakeWindTurbines, PyWakeWindTurbines)
    env.close()
    env = make_env(dwm_params={"rotor_power": "cubic"})
    env.reset(seed=0)
    assert _farm_types(env) == (CubeMeanPowerWindTurbines, CubeMeanPowerWindTurbines)
    env.close()


@pytest.mark.integration
def test_env_per_episode_override_and_fallback():
    env = make_env()
    env.reset(seed=0, options={"dwm_params": {"rotor_power": "cubic"}})
    assert _farm_types(env) == (CubeMeanPowerWindTurbines, CubeMeanPowerWindTurbines)
    env.reset(seed=1)
    assert _farm_types(env) == (PyWakeWindTurbines, PyWakeWindTurbines)
    env.close()


@pytest.mark.integration
def test_env_composes_with_turbine_parameter_dr():
    env = make_env(dwm_params={"rotor_power": "cubic", "cp_gain": 1.1})
    env.reset(seed=0)
    assert _farm_types(env) == (CubeMeanParamPyWakeWindTurbines,
                                CubeMeanParamPyWakeWindTurbines)
    env.close()


@pytest.mark.integration
def test_env_rejects_unknown_choice():
    with pytest.raises(ValueError, match="rotor_power"):
        make_env(dwm_params={"rotor_power": "median"})
