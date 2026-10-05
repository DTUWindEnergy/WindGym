"""Tests for the fixed rotor tilt setting (constructor kwarg / `farm: tilt`)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from py_wake.examples.data.hornsrev1 import V80

from WindGym.farm_eval import FarmEval

CONFIG_PATH = Path(__file__).parent.parent / "examples" / "EnvConfigs" / "Env1.yaml"


def _make_env(**kwargs):
    return FarmEval(
        turbine=V80(),
        x_pos=[0.0, 400.0],
        y_pos=[0.0, 0.0],
        config=str(CONFIG_PATH),
        yaw_init="Zeros",
        seed=0,
        turbtype="None",
        n_passthrough=0.1,
        burn_in_passthroughs=0.0001,
        **kwargs,
    )


def test_default_tilt_is_zero_and_untouched():
    env = _make_env()
    env.reset(seed=0)
    assert env.tilt == 0.0
    # Default path never writes the tilt sensor
    assert np.all(np.asarray(env.fs.windTurbines.tilt, dtype=float) == 0.0)


def test_tilt_kwarg_applied_to_fs_and_baseline():
    env = _make_env(tilt=5.0, Baseline_comp=True)
    _, info = env.reset(seed=0)
    assert env.tilt == 5.0
    assert info["Turbine tilt"] == 5.0
    assert np.all(np.asarray(env.fs.windTurbines.tilt, dtype=float) == 5.0)
    assert np.all(np.asarray(env.fs_baseline.windTurbines.tilt, dtype=float) == 5.0)


def test_tilt_from_config_farm_section():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["farm"]["tilt"] = 3.0
    env = FarmEval(
        turbine=V80(),
        x_pos=[0.0, 400.0],
        y_pos=[0.0, 0.0],
        config=config,
        yaw_init="Zeros",
        seed=0,
        turbtype="None",
        n_passthrough=0.1,
        burn_in_passthroughs=0.0001,
    )
    env.reset(seed=0)
    assert env.tilt == 3.0
    assert np.all(np.asarray(env.fs.windTurbines.tilt, dtype=float) == 3.0)


def test_tilt_kwarg_overrides_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["farm"]["tilt"] = 3.0
    env = FarmEval(
        turbine=V80(),
        x_pos=[0.0, 400.0],
        y_pos=[0.0, 0.0],
        config=config,
        yaw_init="Zeros",
        seed=0,
        turbtype="None",
        n_passthrough=0.1,
        burn_in_passthroughs=0.0001,
        tilt=5.0,
    )
    env.reset(seed=0)
    assert env.tilt == 5.0
