"""How a turbine turns the wind over its rotor into power (``dwm_params["rotor_power"]``).

dynamiks samples the wind at the rotor-average nodes (``CGIRotorAvg``),
averages them ARITHMETICALLY into one speed and evaluates the power curve
there. It reads only a py_wake rotor-average model's nodes and weights, so
py_wake's own ``WSPowerRotorAvg(alpha=3)`` is silently an arithmetic mean too.
Power goes as u^3, so in a partial wake that under-predicts what the rotor
extracts: half the disk at 6 m/s and half at 4 m/s gives P(5 m/s) ~ 125, where
the rotor sees ~ mean(u^3) = 140.

``"cubic"`` evaluates the power curve at the cube-mean ``(sum w u^3)^(1/3)``
over the same nodes and weights -- FLORIS v2.3's ``Turbine.average_velocity``.
Thrust, and so the wake, keep dynamiks' arithmetic mean (thrust ~ u^2).

The cube-mean is a ``power()`` override on dynamiks' class, and composes with
``ParamPyWakeWindTurbines`` (turbine-parameter DR), whose ``_static_power``
calls ``super().power()``: ``CubeMeanParamPyWakeWindTurbines`` puts
``CubeMeanPowerWindTurbines`` between the two in the MRO, so a perturbed
turbine applies its Cp gain, yaw exponent and power lag to the cube-mean power.
"""

from __future__ import annotations

import numpy as np
from dynamiks.wind_turbines import PyWakeWindTurbines

from WindGym.backend.param_turbines import ParamPyWakeWindTurbines


class CubeMeanPowerWindTurbines(PyWakeWindTurbines):
    """``PyWakeWindTurbines`` whose power comes from the rotor's cube-mean wind."""

    def power(self, include_wakes=True, idx=slice(None)):
        uvw, w = self._get_rotor_windspeed(include_wakes, idx)
        u = np.maximum(np.asarray(uvw)[0], 0.0)
        ws = np.average(u ** 3, axis=-1, weights=w) ** (1.0 / 3.0)
        return self.windTurbine.power(ws, **self.get_kwargs(idx))


class CubeMeanParamPyWakeWindTurbines(ParamPyWakeWindTurbines, CubeMeanPowerWindTurbines):
    """Turbine-parameter DR on top of cube-mean power.

    MRO: this class, ``ParamPyWakeWindTurbines``, ``CubeMeanPowerWindTurbines``,
    ``PyWakeWindTurbines``. ``ParamPyWakeWindTurbines._static_power`` calls
    ``super().power()``, which resolves to the cube-mean.
    """


# rotor_power choice -> (nominal class, turbine-parameter DR class)
ROTOR_POWER_CLASSES: dict[str, tuple[type, type]] = {
    "mean": (PyWakeWindTurbines, ParamPyWakeWindTurbines),
    "cubic": (CubeMeanPowerWindTurbines, CubeMeanParamPyWakeWindTurbines),
}
