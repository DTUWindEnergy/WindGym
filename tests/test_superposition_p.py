"""dwm_params superposition_p: the exponent the u-deficits combine with."""

import numpy as np
import pytest
from dynamiks.dwm.superposition import MixedSum
from py_wake.superposition_models import LinearSum

from WindGym.core.dwm_defaults import _PowerSum, _superposition_model


def _deficits():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 1.5, size=(3, 3, 7))     # (wake, component, point)
    v[:, 1:] -= 0.75                             # v and w of either sign
    return v


def test_two_is_mixed_and_one_is_linear():
    d = _deficits()
    np.testing.assert_allclose(_PowerSum(2.0)(d.copy()), MixedSum()(d.copy()))
    np.testing.assert_allclose(_PowerSum(1.0)(d.copy()), d.sum(0))
    assert type(_superposition_model("mixed", 2.0)) is MixedSum
    assert isinstance(_superposition_model("mixed", 1.5), _PowerSum)
    assert type(_superposition_model("linear", 1.5)) is LinearSum


def test_between_the_two():
    d = _deficits()
    lo, mid, hi = (_PowerSum(p)(d.copy())[0] for p in (2.0, 1.5, 1.0))
    assert np.all(lo <= mid + 1e-12) and np.all(mid <= hi + 1e-12)


def test_validation():
    with pytest.raises(ValueError):
        _PowerSum(0.5)
