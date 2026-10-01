"""Golden characterization tests for seeded WindFarmEnv / FarmEval episodes.

Each scenario builds an env, runs a seeded episode with actions drawn from an
*independent* numpy generator (never the env RNG), and compares every recorded
quantity bit-for-bit against ``tests/golden/G*.npz``.

The goldens were recorded from the code on ``main`` before the env/eval
refactor; they pin the RNG draw order in ``reset()`` and the numerics of
``step()``. Regenerate only on purpose::

    WINDGYM_UPDATE_GOLDEN=1 pytest tests/test_golden_episodes.py

Regeneration runs each scenario twice and refuses to write unless both runs
are bit-identical.

The ``*_les_calibrated`` scenarios rerun G2/G4 under the opt-in LES-calibrated
DWM setup (with the linear / 1.5 speed-ups); they need the dev dynamiks build
and skip elsewhere.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular

from WindGym import FarmEval, WindFarmEnv
from WindGym.core import dynamiks_features as features
from WindGym.core.calibrated_values import LES_CALIBRATED
from test_utils import get_fast_pywake_config

pytestmark = pytest.mark.integration

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("WINDGYM_UPDATE_GOLDEN") == "1"

# Info keys recorded at every step (all present on every env in this module).
INFO_KEYS = (
    "yaw angles agent",
    "yaw angles measured",
    "Wind speed at turbines",
    "Wind direction at turbines",
    "Power agent",
    "Power pr turbine agent",
    "Turbulence intensity at turbines",
    "powers",
    "yaws",
    "windspeeds",
    "time_array",
)
# Extra keys recorded only when the env exposes them.
OPTIONAL_INFO_KEYS = (
    "Power baseline",
    "Power pr turbine baseline",
    "yaw angles base",
    "derate agent",
    "derate command",
    "Power reference",
    "Tracking error",
    "Power reference preview",
)


def _derating_turbine():
    """Synthetic turbine with a 'derate' input (same as tests/test_derating.py)."""
    rated = 2.0e6
    ws = np.linspace(0.0, 30.0, 61)
    derate = np.linspace(0.0, 0.8, 17)
    p_avail = rated * np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 3
    power = p_avail[:, None] * (1.0 - derate[None, :])
    ct = 0.8 * (1.0 - derate[None, :]) ** 1.5 * np.ones_like(ws)[:, None]
    pctf = PowerCtNDTabular(
        input_keys=["ws", "derate"],
        value_lst=[ws, derate],
        power_arr=power,
        power_unit="W",
        ct_arr=ct,
        default_value_dict={"derate": 0.0},
    )
    for gi in pctf.interp:
        gi.bounds = "limit"
    return WindTurbine(
        name="SyntheticDerating", diameter=80.0, hub_height=70.0, powerCtFunction=pctf
    )


def _row(turbine, n_turb, spacing_d=5.0):
    d = turbine.diameter()
    return np.arange(n_turb) * spacing_d * d, np.zeros(n_turb)


def _dynamiks_config(**overrides):
    cfg = get_fast_pywake_config()
    cfg["wind"] = {
        "ws_min": 8.0,
        "ws_max": 12.0,
        "TI_min": 0.05,
        "TI_max": 0.10,
        "wd_min": 260.0,
        "wd_max": 280.0,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Scenarios: name -> (env factory, seed, n_steps)
# ---------------------------------------------------------------------------


def make_g1():
    """pywake, ActionMethod yaw, fixed wind."""
    x, y = _row(V80(), 2)
    cfg = get_fast_pywake_config(ActionMethod="yaw")
    return WindFarmEnv(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=cfg,
        backend="pywake",
        reset_init=False,
        n_passthrough=1,
        fill_window=False,
    )


def make_g2(**env_kwargs):
    """dynamiks, no turbulence, baseline + Baseline reward, random yaw init, delay."""
    x, y = _row(V80(), 2)
    cfg = _dynamiks_config(
        power_def={"Power_reward": "Baseline", "Power_avg": 5, "Power_scaling": 1.0},
    )
    return WindFarmEnv(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=cfg,
        backend="dynamiks",
        turbtype="None",
        Baseline_comp=True,
        yaw_init="Random",
        dt_sim=1,
        dt_env=2,
        delay=4,
        n_passthrough=0.5,
        burn_in_passthroughs=0.05,
        fill_window=1,
        reset_init=False,
        **env_kwargs,
    )


def make_g3():
    """pywake, stepwise derating against rated power, power tracking with preview."""
    turbine = _derating_turbine()
    x, y = _row(turbine, 3)
    cfg = get_fast_pywake_config(
        derate_action=True,
        derate_min=0.0,
        derate_max=0.8,
        derate_method="step",
        derate_reference="rated",
        derate_step_sim=0.05,
        Track_power=True,
        track_def={"track_obs_preview": 3},
        power_def={"Power_reward": "None", "Power_avg": 5, "Power_scaling": 1.0},
    )
    cfg["wind"] = {k: (9.0 if "ws" in k else v) for k, v in cfg["wind"].items()}
    return WindFarmEnv(
        turbine=turbine,
        x_pos=x,
        y_pos=y,
        config=cfg,
        backend="pywake",
        reset_init=False,
        max_time_steps=5,
        delay=2,
        fill_window=False,
    )


def make_g4(**env_kwargs):
    """FarmEval, dynamiks, Random turbulence, baseline, pinned wind + defined yaws."""
    x, y = _row(V80(), 2)
    cfg = _dynamiks_config()
    env = FarmEval(
        turbine=V80(),
        x_pos=x,
        y_pos=y,
        config=cfg,
        turbtype="Random",
        Baseline_comp=True,
        yaw_init="Defined",
        n_passthrough=0.5,
        burn_in_passthroughs=0.05,
        fill_window=1,
        reset_init=False,
        **env_kwargs,
    )
    env.set_wind_vals(ws=9, ti=0.06, wd=268)
    env.set_yaw_vals([5.0, -5.0])
    return env


# The LES-calibrated setup plus the speed-ups it was run with in the sweeps.
LES_CALIBRATED_KW = dict(
    dwm_setup=LES_CALIBRATED, interpolation="linear", lateral_cutoff=1.5
)
HAS_DEV_DYNAMIKS = all(
    (
        features.HAS_INTERP,
        features.HAS_MIXED_SUM,
        features.HAS_TI_SENSOR,
        features.HAS_BRANLARD,
    )
)

SCENARIOS = {
    "G1": (make_g1, 11, 6, False),
    "G2": (make_g2, 3, 4, False),
    "G3": (make_g3, 7, 50, True),
    "G4": (make_g4, 5, 3, False),
    "G2_les_calibrated": (partial(make_g2, **LES_CALIBRATED_KW), 3, 4, False),
    "G4_les_calibrated": (partial(make_g4, **LES_CALIBRATED_KW), 5, 3, False),
}


# ---------------------------------------------------------------------------
# Episode recording
# ---------------------------------------------------------------------------


def run_episode(env, seed, n_steps, expect_truncation):
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=seed)
    rec = {
        "obs_reset": np.asarray(obs),
        "ws": np.asarray(env.ws),
        "wd": np.asarray(env.wd),
        "ti": np.asarray(env.ti),
        "veer": np.asarray(env.veer),
        "time_max": np.asarray(env.time_max),
        "t_developed": np.asarray(env.t_developed),
        "rated_power": np.asarray(env.rated_power),
    }
    keys = list(INFO_KEYS) + [k for k in OPTIONAL_INFO_KEYS if k in info]
    for k in keys:
        if k in info:
            rec[f"reset_info__{k}"] = np.asarray(info[k])

    obs_l, rew_l, trunc_l = [], [], []
    info_l = {k: [] for k in keys}
    truncated = False
    for _ in range(n_steps):
        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        obs_l.append(np.asarray(obs))
        rew_l.append(reward)
        trunc_l.append(truncated)
        for k in keys:
            info_l[k].append(np.asarray(info[k]))
        if truncated:
            break
    assert truncated == expect_truncation
    rec["obs"] = np.stack(obs_l)
    rec["reward"] = np.asarray(rew_l)
    rec["truncated"] = np.asarray(trunc_l)
    for k in keys:
        rec[f"info__{k}"] = np.stack(info_l[k])
    return rec


def _assert_same(a: dict, b: dict):
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        assert a[k].dtype == b[k].dtype, (k, a[k].dtype, b[k].dtype)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_episode(name):
    factory, seed, n_steps, expect_trunc = SCENARIOS[name]
    path = GOLDEN_DIR / f"{name}.npz"
    if name.endswith("_les_calibrated") and not HAS_DEV_DYNAMIKS:
        pytest.skip("LES-calibrated DWM setup needs the dev dynamiks build")

    env = factory()
    try:
        rec = run_episode(env, seed, n_steps, expect_trunc)
        if name.startswith("G4"):
            assert env.time_max == 100_000
            rec["time_max"] = np.asarray(env.time_max)
    finally:
        env.close()

    if UPDATE:
        env2 = factory()
        try:
            rec2 = run_episode(env2, seed, n_steps, expect_trunc)
            if name.startswith("G4"):
                rec2["time_max"] = np.asarray(env2.time_max)
        finally:
            env2.close()
        _assert_same(rec, rec2)
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(path, **rec)
        return

    assert path.exists(), f"missing golden {path}; run with WINDGYM_UPDATE_GOLDEN=1"
    with np.load(path) as g:
        golden = {k: g[k] for k in g.files}
    _assert_same(golden, rec)
