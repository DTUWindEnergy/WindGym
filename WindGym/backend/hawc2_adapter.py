"""HAWC2 turbine adapter: normalize power to WATTS.

Native HAWC2 reports rotor power in kilowatts (`HAWC2WindTurbines.power()` returns
`self.sensors.aero_power`, in kW), whereas the PyWake backends report watts. WindGym's power
accounting -- the info dict, the measurement/observation feed, the baseline farm, and the
`maxturbpower` normalization of the scaled power channel -- is all in watts. Mixing the two
makes the scaled power observation collapse to ~ -1 at fidelity level 3.

`HAWC2WindTurbinesW` fixes that at the single boundary: it subclasses dynamiks'
`HAWC2WindTurbines` and overrides `power()` to convert kW -> W, so every reader gets watts with
no per-call-site scaling. A plain subclass (not a closure monkeypatched onto the MultiH2Lib `h2`
handle) is pickling-safe under the spawn-based level-3 vector env.
"""

from dynamiks.wind_turbines.hawc2_windturbine import HAWC2WindTurbines


class HAWC2WindTurbinesW(HAWC2WindTurbines):
    """HAWC2 turbines whose `power()` is in WATTS (native HAWC2 sensor is kW).

    Keeps WindGym's power accounting + scaled power observation consistent with the PyWake
    backends. Only `power()` is touched; all other sensors/behaviour are inherited unchanged.
    """

    POWER_KW_TO_W = 1000.0

    def power(self, include_wakes=True, idx=slice(None)):
        return super().power(include_wakes=include_wakes, idx=idx) * self.POWER_KW_TO_W
