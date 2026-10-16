"""dwm_params meandering_radius_scale: the particle's averaging disk."""

import numpy as np
import pytest

from WindGym.core.dwm_defaults import (PARTICLE_SPATIAL_AVG_N, _particle_spatial_filter,
                                       validate_dwm_params)
from py_wake.rotor_avg_models import CGIRotorAvg


def test_one_is_the_stock_disk_and_zero_is_the_point():
    stock = CGIRotorAvg(PARTICLE_SPATIAL_AVG_N)
    one = _particle_spatial_filter(1.0)
    np.testing.assert_allclose(one.nodes_x, stock.nodes_x)
    np.testing.assert_allclose(one.nodes_y, stock.nodes_y)
    half = _particle_spatial_filter(0.5)
    np.testing.assert_allclose(half.nodes_x, 0.5 * stock.nodes_x)
    zero = _particle_spatial_filter(0.0)
    assert np.all(zero.nodes_x == 0) and np.all(zero.nodes_y == 0)
    assert zero.nodes_weight is stock.nodes_weight or np.allclose(zero.nodes_weight, stock.nodes_weight)


def test_validation():
    validate_dwm_params({"meandering_radius_scale": 0.5})
    with pytest.raises(ValueError):
        _particle_spatial_filter(-0.1)
