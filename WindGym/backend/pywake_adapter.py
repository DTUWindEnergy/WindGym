# --- PyWake steady-state "flow simulation" adapter -------------------------
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez, STF2017TurbulenceModel
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.site import UniformSite
from dynamiks.views import XYView

# from py_wake.utils.grid_types import XYGrid
from py_wake import HorizontalGrid
from dynamiks.utils.geometry import get_east_north_height, get_xyz
from py_wake.flow_map import Points


class _XArrayLikeUVW:
    """Minimal xarray-like container so your plotting code works:
    - uvw[0] -> 2D array of u-component (WS)
    - has .x.values and .y.values
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, u: np.ndarray):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        # Build [3, Ny, Nx] with u in [0], v=w=0
        Ny, Nx = u.shape
        uvw = np.zeros((3, Ny, Nx), dtype=float)
        uvw[0] = u  # u = WS on grid
        self._uvw = uvw

        class _Coord:
            def __init__(self, v):
                self.values = v

        self.x = _Coord(self._x)
        self.y = _Coord(self._y)

    def __getitem__(self, idx):
        return self._uvw[idx]


def _extract_turbine_names(turbine) -> list[str]:
    """
    Best-effort: try turbine.names (list) or turbine.name (str).
    Always returns a non-empty list of strings.
    """
    # Try attribute 'names'
    names = getattr(turbine, "names", None)
    if callable(names):
        names = names()
    if isinstance(names, (list, tuple, np.ndarray)) and len(names):
        return [str(n) for n in names]

    # Fall back to single 'name'
    name = getattr(turbine, "name", None)
    if callable(name):
        name = name()
    if name is None:
        name = "WT"
    return [str(name)]


@dataclass
class _AdapterWindTurbines:
    # World-frame ENZ positions
    positions_east_north: np.ndarray  # shape (3, N) = [east, north, z]
    turbine: any  # py_wake.wind_turbines.WindTurbines
    types: np.ndarray
    flowSimulation: any  # backref to the adapter
    # get_xyz_fn: callable | None = None    # injected get_xyz(east_north, wd, center_offset)

    def __post_init__(self):
        N = self.positions_east_north.shape[1]
        self.yaw = np.zeros(N, dtype=float)  # deg
        self._ws_eff = np.full(N, np.nan, dtype=float)
        self._power_eff = np.full(N, np.nan, dtype=float)
        self._ws_nowake = np.full(N, np.nan, dtype=float)

        # ---- NEW: DWM-like ._names ----
        base_names = _extract_turbine_names(
            self.turbine
        )  # e.g., ["V80"] or ["DTU10MW"]
        # Map by type index if present; default to the first name
        maxlen = max(len(s) for s in base_names)
        dtype = f"<U{maxlen if maxlen > 0 else 1}"
        names_arr = np.empty(N, dtype=dtype)
        for i in range(N):
            t = int(self.types[i]) if i < len(self.types) else 0
            if t < 0 or t >= len(base_names):
                t = 0
            names_arr[i] = base_names[t]
        self._names = names_arr

    # ---- Geometry-like helpers used by your code/plots ----
    @property
    def positions_xyz(self):
        # return np.vstack([self.x, self.y, np.full_like(self.x, self.hub_height)])
        return get_xyz(
            self.positions_east_north,
            self.flowSimulation.wind_direction,
            self.flowSimulation.center_offset,
        )

    @property
    def rotor_positions_xyz(self):
        # Same as hub centers for our purposes
        return self.positions_xyz

    def yaw_tilt(self):
        # PyWake plotting helper expects arrays (deg). Tilt=0
        return self.yaw.copy(), np.zeros_like(self.yaw)

    # ----- "Sensors" your env expects -----
    @property
    def rotor_avg_windspeed(self):
        u = np.asarray(self._ws_eff, float)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1)

    def get_rotor_avg_windspeed(self, include_wakes=True):
        u = np.asarray(self._ws_eff if include_wakes else self._ws_nowake, float)
        return np.vstack([u, np.zeros_like(u), np.zeros_like(u)])

    def power(self, include_wakes=True):
        if include_wakes:
            return np.asarray(self._power_eff, float)
        # No-wake reference must still honour any active derating, otherwise
        # the Wake_recovery headroom is biased for derated farms.
        derate = getattr(self.flowSimulation, "_derate", None)
        if derate is not None and np.any(np.asarray(derate) != 0):
            return np.asarray(self.turbine.power(self._ws_nowake, derate=derate), float)
        return np.asarray(self.turbine.power(self._ws_nowake), float)

    # Convenience passthroughs
    def diameter(self, type=0):
        return np.ones(self.types.size, float) * self.turbine.diameter()

    # self.turbine.diameter()

    def hub_height(self):
        return self.turbine.hub_height()


class PyWakeFlowSimulationAdapter:
    """Steady-state PyWake backend that mirrors the subset of DWMFlowSimulation
    your environment relies on.
    """

    def __init__(
        self,
        *,
        x,
        y,
        windTurbine,
        ws: float,
        wd: float,
        ti: float,
        dt: float = 1.0,
        model=None,
        wd_lst=None,
    ):
        """
        Args:
            x, y: turbine coordinates [m]
            windTurbine: a py_wake.wind_turbines.WindTurbines (or subclass)
            ws, wd, ti: global inflow (m/s, deg, -)
            dt: time increment for .step()
            model: optional PyWake WindFarmModel. Default: BastankhahGaussian
            wd_lst: optional wind direction per simulation step (deg). Entry i
                applies at time i*dt; interpolated like MetmastSite.
        """
        self.dt = float(dt)
        self.time = 0.0
        self.ws = float(ws)
        self.wd = float(wd)
        self.ti = float(ti)
        self.wind_direction = float(wd)  # used by your plotting call

        if wd_lst is not None:
            # unwrap so interpolation across 0/360 behaves like MetmastSite
            self._wd_lst = np.unwrap(np.asarray(wd_lst, float), period=360)
            self._time_lst = np.arange(len(self._wd_lst)) * self.dt
        else:
            self._wd_lst = None

        # World ENZ positions
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.full_like(x, windTurbine.hub_height(), dtype=float)
        positions_east_north = np.vstack([x, y, z])  # (3, N)

        # DWM-style center offset
        self.center_offset = (
            positions_east_north.max(1) + positions_east_north.min(1)
        ) / 2.0

        self._wt = windTurbine
        self._site = UniformSite(p_wd=[1.0], ti=self.ti)

        self._model = model or Blondel_Cathelain_2020(
            site=self._site,
            windTurbines=self._wt,
            turbulenceModel=STF2017TurbulenceModel(),
            deflectionModel=JimenezWakeDeflection(),
        )

        types = np.zeros(x.size, dtype=int)
        self.windTurbines = _AdapterWindTurbines(
            positions_east_north=positions_east_north,
            turbine=self._wt,
            types=types,
            flowSimulation=self,  # backref
        )

        # Per-turbine derating fractions (0 = no derating).  Updated each step
        # by _apply_derating() in wind_farm_env when derate_action=True.
        self._derate = np.zeros(x.size)

        # Prime steady-state
        self._compute_steady_state()

    # -------- public API to mirror DWMFlowSimulation --------
    @property
    def _wind_direction(self):
        # read-only alias mirroring DWMFlowSimulation's private attribute
        return self.wind_direction

    def _update_wd(self):
        """Set wd from the wd schedule at the current time (if provided)."""
        if self._wd_lst is not None:
            wd = float(np.interp(self.time, self._time_lst, self._wd_lst)) % 360
            self.wd = wd
            self.wind_direction = wd

    def run(self, T: int | float):
        """Advance 'time' by T (seconds) and recompute steady flow."""
        # steady -> nothing evolves but we update time for consistency
        self.time += float(T)
        self._update_wd()
        self._compute_steady_state()

    def step(self):
        """One time increment."""
        self.time += self.dt
        self._update_wd()
        self._compute_steady_state()

    def get_wind_direction(self, xyz=None, include_wakes=True):
        """Return the wind direction at given positions.

        In steady-state PyWake the inflow direction is spatially uniform,
        so this returns the global wind direction for every queried point.
        """
        if xyz is None:
            return np.array([self.wind_direction])
        xyz = np.asarray(xyz)
        if xyz.ndim == 1:
            return np.array([self.wind_direction])
        n_points = xyz.shape[1]
        return np.full(n_points, self.wind_direction, dtype=float)

    def get_windspeed(self, view, include_wakes=True, xarray=True):
        """Return an xarray-like UVW grid with u=WS, v=w=0 for plotting."""
        # view.x, view.y are 1D arrays
        x = np.asarray(view.x)
        y = np.asarray(view.y)
        grid = HorizontalGrid(x=x, y=y, h=self._wt.hub_height())

        extra = {}
        if np.any(self._derate != 0):
            extra["derate"] = self._derate

        # The view grid and positions_xyz are in the wind-aligned (XYView)
        # frame, where the wind blows along +x. In PyWake that is wd=270
        # (wind from west); passing wd=self.wd here would rotate the
        # already-rotated coordinates a second time.
        fm = self._model(
            x=self.windTurbines.positions_xyz[0],
            y=self.windTurbines.positions_xyz[1],
            wd=[270.0],
            ws=[self.ws],
            TI=self.ti,
            tilt=0,
            yaw=self.windTurbines.yaw,
            **extra,
        ).flow_map(grid=grid)

        ws_xy = fm.WS_eff.squeeze().T
        if xarray:
            return _XArrayLikeUVW(x, y, ws_xy)
        # fallback: raw ndarray with shape (3, Ny, Nx)
        Ny, Nx = ws_xy.shape
        uvw = np.zeros((3, Ny, Nx), float)
        uvw[0] = ws_xy
        return uvw

    # -------- internal helpers --------
    def _compute_steady_state(self):
        """Recompute steady-state wakes & turbine quantities for current yaw."""
        # With PyWake, pass arrays (wd, ws) as length-1 + per-turbine yaw
        # sim = self._model(self._x, self._y, wd=[self.wd], ws=[self.ws],
        #                   TI=self.ti, tilt=0,
        #                   yaw=self.windTurbines.yaw)
        extra = {}
        if np.any(self._derate != 0):
            extra["derate"] = self._derate

        sim = self._model(
            self.windTurbines.positions_east_north[0],
            self.windTurbines.positions_east_north[1],
            wd=[self.wd],
            ws=[self.ws],
            TI=self.ti,
            tilt=0,
            yaw=self.windTurbines.yaw,
            **extra,
        )

        # Effective WS per turbine (i, l, k) -> (N,)
        ws_eff = np.asarray(sim.WS_eff_ilk[:, 0, 0], float)
        # Power per turbine
        p_eff = np.asarray(sim.power_ilk[:, 0, 0], float)
        # No-wake WS (free stream at rotor)
        ws_nowake = np.full(ws_eff.shape, self.ws, float)

        self.windTurbines._ws_eff = ws_eff
        self.windTurbines._power_eff = p_eff
        self.windTurbines._ws_nowake = ws_nowake
