"""Unit tests for WindGym.core.turbine_params (per-turbine DR parameter vector)."""

import numpy as np
import pytest

from WindGym.core.turbine_params import (
    TURBINE_PARAM_KEYS,
    TURBINE_PARAM_SPECS,
    TurbineParams,
    normalize_turbine_params,
)

DT = 5.0


def test_spec_keys_and_defaults():
    assert list(TURBINE_PARAM_SPECS) == [
        "yaw_exp",
        "cp_gain",
        "ct_gain",
        "tau_yaw",
        "delay_yaw",
        "tau_power",
    ]
    assert TURBINE_PARAM_KEYS == frozenset(TURBINE_PARAM_SPECS)
    defaults = {k: s.default for k, s in TURBINE_PARAM_SPECS.items()}
    assert defaults == {
        "yaw_exp": 0.0,
        "cp_gain": 1.0,
        "ct_gain": 1.0,
        "tau_yaw": 0.0,
        "delay_yaw": 0.0,
        "tau_power": 0.0,
    }
    for k, s in TURBINE_PARAM_SPECS.items():
        assert s.lo <= s.default <= s.hi, k


def test_defaults_are_nominal():
    tp = normalize_turbine_params({}, n_turb=3, dt_sim=DT)
    assert isinstance(tp, TurbineParams)
    assert tp.is_nominal
    assert tp.dt_sim == DT
    for k in TURBINE_PARAM_KEYS:
        v = tp.values[k]
        assert v.shape == (3,) and v.dtype == np.float64
        assert np.all(v == TURBINE_PARAM_SPECS[k].default)
    assert np.all(tp.a_yaw == 1.0) and np.all(tp.a_power == 1.0)
    assert np.all(tp.k_delay == 0) and tp.k_delay.dtype.kind == "i"


def test_none_is_treated_as_empty():
    assert normalize_turbine_params(None, n_turb=2, dt_sim=DT).is_nominal


def test_scalar_broadcasts_and_marks_non_nominal():
    tp = normalize_turbine_params({"cp_gain": 1.1}, n_turb=4, dt_sim=DT)
    assert not tp.is_nominal
    np.testing.assert_array_equal(tp.values["cp_gain"], np.full(4, 1.1))
    np.testing.assert_array_equal(tp.values["ct_gain"], np.ones(4))


def test_long_array_is_sliced_to_n_turb():
    tp = normalize_turbine_params(
        {"yaw_exp": [0.1, 0.2, 0.3, 0.4, 0.5]}, n_turb=3, dt_sim=DT
    )
    np.testing.assert_array_equal(tp.values["yaw_exp"], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"yaw_exp": [0.1, 0.2]}, "yaw_exp.*len 2.*n_turb 3"),
        ({"yaw_exp": np.zeros((3, 1))}, "yaw_exp.*ndim"),
        ({"bogus": 1.0}, "Unknown turbine param.*bogus"),
        ({"cp_gain": np.nan}, "cp_gain.*finite"),
        ({"cp_gain": [1.0, np.inf, 1.0]}, "cp_gain.*finite"),
        ({"ct_gain": 5.0}, r"ct_gain.*outside \[0.2, 1.6\]"),
        ({"tau_yaw": -1.0}, r"tau_yaw.*outside"),
    ],
)
def test_invalid_inputs_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        normalize_turbine_params(bad, n_turb=3, dt_sim=DT)


def test_filter_coefficients():
    tp = normalize_turbine_params(
        {"tau_yaw": [0.0, 30.0, 600.0], "tau_power": [20.0, 0.0, 5.0]},
        n_turb=3,
        dt_sim=DT,
    )
    assert tp.a_yaw[0] == 1.0 and tp.a_power[1] == 1.0  # exactly 1.0 where tau == 0
    np.testing.assert_allclose(
        tp.a_yaw[1:], 1.0 - np.exp(-DT / np.array([30.0, 600.0])), rtol=1e-15
    )
    np.testing.assert_allclose(
        tp.a_power[[0, 2]], 1.0 - np.exp(-DT / np.array([20.0, 5.0])), rtol=1e-15
    )


def test_k_delay_rounds_to_sim_steps():
    tp = normalize_turbine_params(
        {"delay_yaw": [7.5, 10.0, 12.5, 0.0]}, n_turb=4, dt_sim=DT
    )
    # np.rint: round half to even -> 7.5/5 = 1.5 -> 2, 12.5/5 = 2.5 -> 2
    np.testing.assert_array_equal(tp.k_delay, [2, 2, 2, 0])
    assert tp.k_delay.dtype.kind == "i"


def test_as_dict_round_trip_and_frozen():
    tp = normalize_turbine_params({"cp_gain": [0.9, 1.0]}, n_turb=2, dt_sim=DT)
    d = tp.as_dict()
    assert set(d) == TURBINE_PARAM_KEYS
    np.testing.assert_array_equal(d["cp_gain"], [0.9, 1.0])
    d["cp_gain"][0] = 5.0  # copy, not a view
    assert tp.values["cp_gain"][0] == 0.9
    with pytest.raises(Exception):
        tp.dt_sim = 1.0
