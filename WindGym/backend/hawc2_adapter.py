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

import os
import shutil

import numpy as np
from dynamiks.wind_turbines.hawc2_windturbine import HAWC2WindTurbines


class HAWC2WindTurbinesW(HAWC2WindTurbines):
    """HAWC2 turbines whose `power()` is in WATTS (native HAWC2 sensor is kW).

    Keeps WindGym's power accounting + scaled power observation consistent with the PyWake
    backends. Only `power()` is touched; all other sensors/behaviour are inherited unchanged.
    """

    POWER_KW_TO_W = 1000.0

    def power(self, include_wakes=True, idx=slice(None)):
        return super().power(include_wakes=include_wakes, idx=idx) * self.POWER_KW_TO_W


from dynamiks.wind_turbines.hawc2_windturbine import FreeWindCoupling  # noqa: E402


class HubWindFreeWindCoupling(FreeWindCoupling):
    """`FreeWindCoupling` that also feeds the controller its hub wind speed.

    dynamiks' default `FreeWindCoupling` passes the DWM wind field to HAWC2 as a
    micro-turbulence box but never writes the HAWC2 *general variables* 1-3.
    The IEA-22 htc (`LEShawc2files/htc/input_hawc_yaw_actuator_tipcorr.htc`,
    controller output block) wires exactly those into the DTU WE controller as
    the hub wind-speed vector, which the controller uses for its wind-speed
    dependent minimum-pitch schedule (`control/wpdata.100`). With the
    variables left at zero the schedule is stuck at its lowest-ws row, so the
    turbine runs at a wrong pitch in region II. The EllipSys/LES harness uses
    `ADCoupling`, whose `set_velocities` does write them
    (dynamiks `hawc2_windturbine.py`), so this subclass makes the DWM+HAWC2
    rung consistent with the LES+HAWC2 rung.

    Per flow step (after the parent's wind-field update) the rotor-centre wind
    speed of every turbine -- dynamiks uvw, wakes of the other turbines
    included, its own excluded -- is written into general variables
    `general_variable_uvw` (default 1, 2, 3 = u, v, w), mirroring
    `AeroSectionCoupling.set_velocities`. The controller low-pass filters the
    wind speed itself, so once per `dt_sim` is sufficient.
    """

    def __init__(self, windfield_update_interval=5, general_variable_uvw=(1, 2, 3)):
        FreeWindCoupling.__init__(
            self, windfield_update_interval=windfield_update_interval
        )
        self.general_variable_uvw = tuple(general_variable_uvw)

    def initialize(self, flowSimulation, windTurbine):
        FreeWindCoupling.initialize(self, flowSimulation, windTurbine)
        # Prime the controller before the first HAWC2 step (free stream, no wakes yet).
        self._set_hub_wind(flowSimulation, include_wakes=False)

    def hub_wind_uvw(self, flowSimulation, include_wakes=True):
        """(3, N) rotor-centre wind speed per turbine, own wake excluded."""
        wts = self.windTurbine
        xyz = np.asarray(wts.rotor_positions_xyz)
        uvw = np.zeros((3, wts.N))
        for i in range(wts.N):
            uvw[:, i] = np.asarray(
                flowSimulation.get_windspeed(
                    xyz[:, i : i + 1],
                    include_wakes=include_wakes,
                    exclude_wake_from=[i],
                )
            ).reshape(3)
        return uvw

    def _set_hub_wind(self, flowSimulation, include_wakes=True):
        uvw = self.hub_wind_uvw(flowSimulation, include_wakes=include_wakes)
        for slot, comp in zip(self.general_variable_uvw, uvw):
            try:
                self.windTurbine.h2.set_variable_sensor_value(slot, comp.tolist())
            except ChildProcessError:  # pragma: no cover - MPI ranks without a HAWC2
                pass

    def step(self, flowSimulation):
        FreeWindCoupling.step(self, flowSimulation)
        self._set_hub_wind(flowSimulation, include_wakes=True)


# ---------------------------------------------------------------------------
# Teardown helpers (used by WindFarmEnv's cleanup paths)
# ---------------------------------------------------------------------------


def safe_close_h2(wt, owner_pid) -> None:
    """Close a HAWC2 turbine's h2 connection defensively.

    The MultiH2Lib children can only be polled/closed from the process that spawned
    them, so closing from another process raises ``AssertionError: can only test a
    child process``. Gate on the owning pid and never let teardown raise (HAWC2's own
    ``atexit`` handler still closes the connection in the owning process).
    """
    if wt is None or not hasattr(wt, "h2"):
        return
    if os.getpid() != owner_pid:
        return
    try:
        wt.h2.close()
    except (AssertionError, OSError, EOFError):
        # Proxy child already gone / pipe closed / non-owning poll — teardown must not raise.
        pass


def delete_case_folders(wts) -> None:
    """Remove the res/, htc/ and log/ case subfolders for one set of HAWC2 turbines.

    ``output.filename`` points at ``res/<case>/...``; the htc and log folders mirror
    it with the leading ``res`` swapped. ``ignore_errors=True`` so an already-deleted
    folder does not raise during teardown.
    """
    modelpath = wts.htc_lst[0].modelpath
    res_rel = os.path.split(wts.htc_lst[0].output.filename.values[0])[0]
    for sub in ("res", "htc", "log"):
        # replace only the leading "res" (count=1) to avoid touching the case name
        folder = modelpath + res_rel.replace("res", sub, 1)
        shutil.rmtree(folder, ignore_errors=True)
