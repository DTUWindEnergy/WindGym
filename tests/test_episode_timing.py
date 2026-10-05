"""Unit tests for WindGym.core.episode_timing (pure geometry/time helpers)."""

import math

import numpy as np
import pytest

from WindGym.core.episode_timing import episode_time_parameters, max_wd_step

pytestmark = pytest.mark.unit


def test_two_turbines_scale_with_passthroughs():
    pos = np.array([[0.0, 0.0], [400.0, 0.0]])
    t_dev, t_max = episode_time_parameters(
        pos, rotor_diameter=80.0, ws=10.0, n_passthrough=5, burn_in_passthroughs=2
    )
    assert (t_dev, t_max) == (80, 200)
    assert isinstance(t_dev, int) and isinstance(t_max, int)


def test_fractional_passthroughs_are_ceiled():
    pos = np.array([[0.0, 0.0], [400.0, 0.0]])
    t_dev, t_max = episode_time_parameters(
        pos, rotor_diameter=80.0, ws=10.0, n_passthrough=0.5, burn_in_passthroughs=0.05
    )
    assert (t_dev, t_max) == (math.ceil(2.0), 20)


def test_single_turbine_uses_rotor_diameter():
    pos = np.array([[0.0, 0.0]])
    t_dev, t_max = episode_time_parameters(
        pos, rotor_diameter=80.0, ws=10.0, n_passthrough=5, burn_in_passthroughs=2
    )
    assert t_dev == 0
    assert t_max == math.ceil(80.0 * 5 / 10.0)


def test_time_max_floor_is_one_second():
    pos = np.array([[0.0, 0.0], [0.0, 0.0]])  # coincident -> zero inflow time
    _, t_max = episode_time_parameters(
        pos, rotor_diameter=80.0, ws=10.0, n_passthrough=5, burn_in_passthroughs=2
    )
    assert t_max == 1


def test_max_wd_step_two_turbines():
    pos = np.array([[0.0, 0.0], [400.0, 0.0]])
    expected = 2.0 * 360 / (2 * np.pi * 200.0)
    assert max_wd_step(pos, rotor_diameter=80.0, max_turb_move=2.0) == pytest.approx(
        expected
    )


def test_max_wd_step_single_turbine_uses_half_diameter():
    pos = np.array([[0.0, 0.0]])
    expected = 2.0 * 360 / (2 * np.pi * 40.0)
    assert max_wd_step(pos, rotor_diameter=80.0, max_turb_move=2.0) == pytest.approx(
        expected
    )
