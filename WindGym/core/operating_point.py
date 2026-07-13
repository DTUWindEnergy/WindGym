"""Steady-state operating-point lookup: (ws, yaw, derate) -> (pitch, rpm).

The derating power/ct surrogate (see ``core.derating``) is the reduction of a
full HAWCStab2 table that, per grid point, also records the blade pitch [deg]
and tip-speed ratio the controller would settle at (the min-Ct solution that
meets the power target). This module interpolates that full table so the env
can report the steady-state pitch and rotor speed alongside power — same
fidelity as the surrogate power/ct values, not a dynamic controller sim.

Rotor speed is derived from the tip-speed ratio:
``rpm = tsr * ws / rotor_radius * 60 / (2 pi)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from py_wake.utils.grid_interpolator import GridInterpolator


class OperatingPointLookup:
    """Interpolates blade pitch [deg] and rotor speed [RPM] on a
    (ws, yaw, derate) grid.

    Uses the same ``GridInterpolator(bounds="limit")`` that
    ``PowerCtNDTabular`` uses internally, so out-of-grid queries clamp to the
    table edges exactly like the power/ct lookup does.
    """

    def __init__(self, ws, yaw, derate, pitch, tsr, rotor_diameter):
        ws = np.asarray(ws, dtype=float)
        # One interpolator with a trailing value dimension handles
        # (pitch, tsr) in a single call.
        self._interp = GridInterpolator(
            [ws, np.asarray(yaw, dtype=float), np.asarray(derate, dtype=float)],
            np.stack([np.asarray(pitch), np.asarray(tsr)], axis=-1),
            bounds="limit",
        )
        self.rotor_radius = rotor_diameter / 2.0
        self.ws_min = float(ws.min())
        self.ws_max = float(ws.max())

    @classmethod
    def from_netcdf(cls, nc_path, rotor_diameter):
        """Load from a full HAWCStab2 surrogate .nc with dims
        (ws, yaw, derating) and variables ``pitch`` [deg] and ``tsr``."""
        ds = xr.load_dataset(Path(nc_path)).transpose("ws", "yaw", "derating")
        return cls(
            ws=ds.ws.values,
            yaw=ds.yaw.values,
            derate=ds.derating.values,
            pitch=ds.pitch.values,
            tsr=ds.tsr.values,
            rotor_diameter=rotor_diameter,
        )

    def pitch_rpm(self, ws, yaw, derate):
        """Per-turbine steady-state (pitch [deg], rotor speed [RPM]).

        Args are arrays of shape (n_turb,): rotor-averaged wind speed, yaw
        angle [deg] and derate fraction. Returns two (n_turb,) arrays.
        """
        ws = np.asarray(ws, dtype=float)
        pts = np.stack(
            [ws, np.asarray(yaw, dtype=float), np.asarray(derate, dtype=float)],
            axis=-1,
        )
        pitch, tsr = self._interp(pts).T
        # tsr is clamped to the grid edge out of bounds, so pair it with the
        # clamped ws — otherwise rpm would extrapolate off the table.
        ws_clamped = np.clip(ws, self.ws_min, self.ws_max)
        rpm = tsr * ws_clamped / self.rotor_radius * 60.0 / (2.0 * np.pi)
        return pitch, rpm
