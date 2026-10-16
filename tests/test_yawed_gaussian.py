"""The yawed Gaussian particle deficit (dwm_params yaw_ct_exp / yaw_width_exp)."""

import numpy as np
import pytest
from dynamiks.dwm.projection_models import NoProjection
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit

from WindGym.core.dwm_defaults import _ScalarTIPyWakeDeficitGenerator, validate_dwm_params

D, HUB = 1.1, 0.825


def _profile(yaw, n=0.0, m=0.0):
    gen = _ScalarTIPyWakeDeficitGenerator(
        BastankhahGaussianDeficit(k=0.019, ceps=0.2, use_effective_ws=True),
        scale_with_freestream=False, projectionModel=NoProjection(),
        yaw_ct_exp=n, yaw_width_exp=m)
    return gen.new_particle_deficit(particle_position=np.zeros(3), ip=0, u_freestream=6.0, u_scale=6.0,
                                    D_src=D, TI=0.06, h=HUB, CT=0.75, yaw=yaw, tilt=0.0, type=0)


def _centre_and_widths(p, x=5 * D):
    xs = np.full(41, x)
    off = np.linspace(-1.5 * D, 1.5 * D, 41)
    dy = p._get_deficit_norm(xs, off, np.zeros(41))[0]
    dz = p._get_deficit_norm(xs, np.zeros(41), off)[0]
    centre = float(dy.max())
    sig = lambda d: float(np.sqrt((off**2 * d).sum() / d.sum()))
    return centre, sig(dy), sig(dz)


def test_off_is_the_stock_profile():
    from dynamiks.dwm.particle_deficit_profiles.pywake_deficit_wrapper import PyWakeDeficitProfile
    for yaw in (0.0, 30.0):
        p = _profile(yaw)
        xs, ys, zs = np.full(7, 4.0), np.linspace(-1, 1, 7), np.linspace(-0.5, 0.5, 7)
        np.testing.assert_allclose(p._get_deficit_norm(xs, ys, zs),
                                   PyWakeDeficitProfile._get_deficit_norm(p, xs, ys, zs))


def test_yawed_wake_is_weaker_and_laterally_narrower():
    c0, sy0, sz0 = _centre_and_widths(_profile(0.0, n=2.0, m=1.0))
    c30, sy30, sz30 = _centre_and_widths(_profile(30.0, n=2.0, m=1.0))
    c30_off, sy30_off, _ = _centre_and_widths(_profile(30.0))
    assert c30 < 0.95 * c0 and c30 < c30_off         # weaker than unyawed, and than the stock yawed
    assert sy30 < sy0 and sy30 < sy30_off              # laterally narrower
    assert sz30 == pytest.approx(sz0, rel=0.15)        # the vertical width hardly moves (only through ct)
    assert c0 == pytest.approx(_centre_and_widths(_profile(0.0))[0])   # unyawed: the stock wake


def test_gradient_is_consistent_with_the_profile():
    p = _profile(25.0, n=2.0, m=1.0)
    x, y, z = np.full(1, 5.5), np.array([0.3]), np.array([0.1])
    d, dudr = p._get_UVW_dUdr(x, y, z)
    h = 1e-5
    gy = (p._get_deficit_norm(x, y + h, z)[0] - p._get_deficit_norm(x, y - h, z)[0]) / (2 * h)
    gz = (p._get_deficit_norm(x, y, z + h)[0] - p._get_deficit_norm(x, y, z - h)[0]) / (2 * h)
    np.testing.assert_allclose(dudr, np.sqrt(gy**2 + gz**2) * D / 2, rtol=1e-4)


def test_keys_validate():
    validate_dwm_params({"deficit_model": "bastankhah", "yaw_ct_exp": 2.0, "yaw_width_exp": 1.0})
    with pytest.raises((TypeError, ValueError)):
        validate_dwm_params({"yaw_ct_exp": "two"})
