"""HubWindFreeWindCoupling: the controller's hub-wind general variables get written.

Mocked flow simulation + h2 handle, no h2lib / HAWC2 needed.
"""

from unittest.mock import MagicMock

import numpy as np

from dynamiks.wind_turbines.hawc2_windturbine import FreeWindCoupling
from WindGym.backend.hawc2_adapter import HubWindFreeWindCoupling


def _fake_flow(uvw_by_turbine):
    """get_windspeed(xyz, ...) -> (3, 1) for the turbine whose x matches."""
    fs = MagicMock()

    def get_windspeed(xyz, include_wakes=True, exclude_wake_from=(), **_):
        i = int(np.round(xyz[0, 0] / 1000.0))
        return np.asarray(uvw_by_turbine[i]).reshape(3, 1)

    fs.get_windspeed.side_effect = get_windspeed
    fs.wind_direction = 270
    fs.site.turbulence_transport_speed = 9.0
    fs.time = 5.0
    return fs


def _fake_wts(n):
    wts = MagicMock()
    wts.N = n
    wts.rotor_positions_xyz = np.array(
        [[1000.0 * i for i in range(n)], [0.0] * n, [170.0] * n]
    )
    wts.h2 = MagicMock()
    return wts


def test_step_writes_general_variables_1_to_3():
    uvw = {0: (9.0, 0.1, -0.2), 1: (6.5, 0.0, 0.0)}
    fs = _fake_flow(uvw)
    wts = _fake_wts(2)
    cpl = HubWindFreeWindCoupling()
    cpl.windTurbine = wts
    # skip the parent's windfield machinery (needs a real turbulence field)
    cpl.current_wind_direction = fs.wind_direction
    cpl.current_transport_speed = fs.site.turbulence_transport_speed
    cpl.last_windfield_update = fs.time
    cpl.windfield_update_interval = 5

    cpl.step(fs)

    calls = {
        c.args[0]: c.args[1] for c in wts.h2.set_variable_sensor_value.call_args_list
    }
    assert set(calls) == {1, 2, 3}
    np.testing.assert_allclose(calls[1], [9.0, 6.5])
    np.testing.assert_allclose(calls[2], [0.1, 0.0])
    np.testing.assert_allclose(calls[3], [-0.2, 0.0])
    # own wake excluded per turbine
    excl = [c.kwargs["exclude_wake_from"] for c in fs.get_windspeed.call_args_list]
    assert excl == [[0], [1]]


def test_is_a_free_wind_coupling():
    assert issubclass(HubWindFreeWindCoupling, FreeWindCoupling)
    assert HubWindFreeWindCoupling(
        general_variable_uvw=(5, 6, 7)
    ).general_variable_uvw == (5, 6, 7)
