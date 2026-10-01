"""LES-precursor inflow source for ``turbtype="Precursor"``.

Wraps the memmap sidecar produced by ``dynamiks.sites.precursor`` (opened once
and shared by every reset) and owns the three precursor-specific pieces the
``TurbulenceManager`` needs: the per-episode ``PrecursorField``, the episode
feasibility checks with the random start-time window draw, and the
``TurbulenceFieldSite`` that advects the box across the farm.

The dynamiks/hipersim precursor modules are imported lazily inside the
methods so the rest of WindGym imports on installs without them.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np

from dynamiks.sites._site import TurbulenceFieldSite
from dynamiks.sites.mean_wind import ConstantWindSpeedProfile
from dynamiks.dwm.added_turbulence_models import (
    BranlardScaling,
    SynchronizedAutoScalingIsotropicMannTurbulence,
)


class PrecursorSource:
    """Shared read-only precursor box plus its per-episode state.

    Attributes:
        uvw: The memmapped (n_x, n_y, n_z, 3)-ish velocity box (read-only).
        meta: The sidecar metadata dict (``advection_speed``, ``Nxyz``,
            ``dxyz``, ``n_ramp``, ``t_data_s``, ``ti_hub``, ``ti_yz``, ...).
        window_offset_s: Start-time window (seconds into the box) drawn by
            ``check_episode`` for the current episode; exposed for
            logging/tests.
    """

    def __init__(self, path: Optional[Union[str, Path]]):
        if not path:
            raise FileNotFoundError(
                "Provide 'TurbBox' (precursor .nc or sidecar .npy/.meta.npz "
                "path) for turbtype='Precursor'."
            )
        from dynamiks.sites.precursor import load_sidecar

        self.uvw, self.meta = load_sidecar(path)
        self.window_offset_s = 0.0

    @classmethod
    def from_arrays(cls, uvw, meta) -> "PrecursorSource":
        """Build a source around already-loaded data (tests / custom loaders)."""
        self = cls.__new__(cls)
        self.uvw = uvw
        self.meta = meta
        self.window_offset_s = 0.0
        return self

    def field(self, ws: float) -> tuple:
        """Build a PrecursorField around the shared read-only memmap.

        The field object itself is cheap (per-reset advection state around the
        one memmap opened once). Same added-turbulence model as the Mann paths
        / MakeDWM_precursor.ipynb.

        Returns:
            tuple: (turbulence_field, added_turbulence_model)
        """
        from hipersim import Bounds
        from dynamiks.sites.precursor import PrecursorField

        meta = self.meta
        U = float(meta["advection_speed"])
        if abs(ws - U) > 0.05:
            raise ValueError(
                f"ws={ws:.3f} but the precursor advection speed is {U:.3f}. "
                "The env must pin ws from precursor_meta before create_sites "
                "(WindFarmEnv.reset does this for turbtype='Precursor')."
            )
        n_x, n_y, n_z = (int(v) for v in meta["Nxyz"])
        dx, dy, dz = (float(v) for v in meta["dxyz"])
        tf = PrecursorField(
            self.uvw,
            Nxyz=(n_x, n_y, n_z),
            dxyz=(dx, dy, dz),
            bounds=Bounds.Warning,
            ti_yz=meta["ti_yz"],
        )
        added_turb_model = SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(),
            cache_field=False,
        )
        return tf, added_turb_model

    def check_episode(
        self,
        np_random,
        *,
        turbine_positions: np.ndarray,
        rotor_diameter: float,
        veer_rate: float,
        wd_list: list,
        episode_time_budget_s: float,
    ) -> None:
        """Precursor-episode guards + random start-time window sampling.

        The box holds t_data_s seconds of LES data. At window offset tau and
        sim time T, the most-upstream probed point x_min runs off the back of
        the data when U*(tau+T) > t_data*U + x_min, so the episode budget must
        satisfy tau + budget <= t_data + x_min/U. (The prepended ramp covers
        the farm at T=0 and buys no extra time; downstream x > Lx clamps to
        the ramp/box edge under Bounds.Warning, same as the validated
        notebook.) tau is drawn uniformly from the remaining slack with the
        env-seeded rng, so agent and baseline share the episode's window and
        seeds reproduce it.
        """
        meta = self.meta
        if veer_rate:
            raise ValueError(
                "turbtype='Precursor' carries the LES shear/veer in the box "
                "itself; set veer to 0."
            )
        if np.ptp(wd_list) > 0:
            raise ValueError(
                "turbtype='Precursor' uses a TurbulenceFieldSite, which cannot "
                "express a time-varying wind direction series; use a constant "
                "wd (the env pins wd=270 for Precursor)."
            )
        n_y = int(meta["Nxyz"][1])
        dy = float(meta["dxyz"][1])
        farm_width = float(np.ptp(turbine_positions[:, 1])) + rotor_diameter
        box_width = (n_y - 1) * dy
        if farm_width > box_width:
            raise ValueError(
                f"Farm y-width ~{farm_width:.0f} m exceeds the precursor box "
                f"width {box_width:.0f} m."
            )
        U = float(meta["advection_speed"])
        n_ramp = int(meta["n_ramp"])
        dx = float(meta["dxyz"][0])
        x_margin = 2.0 * rotor_diameter  # probes/rotor points around turbines
        x_max = float(turbine_positions[:, 0].max()) + x_margin
        if x_max > n_ramp * dx:
            raise ValueError(
                f"Farm extends to x~{x_max:.0f} m but the precursor ramp only "
                f"covers [0, {n_ramp * dx:.0f}] m at episode start. Reconvert "
                f"with a larger --Lx or shift the layout."
            )
        x_min = float(turbine_positions[:, 0].min()) - x_margin
        t_usable = float(meta["t_data_s"]) + x_min / U
        slack = t_usable - float(episode_time_budget_s)
        if slack < 0:
            raise ValueError(
                f"Episode needs ~{episode_time_budget_s:.0f} s of inflow but the "
                f"precursor provides only ~{t_usable:.0f} s (t_data="
                f"{float(meta['t_data_s']):.0f} s, upstream margin "
                f"{-x_min:.0f} m). Reduce max_time_steps / burn-in."
            )
        self.window_offset_s = float(np_random.uniform(0.0, slack))

    def site(self, tf, turbine_positions: np.ndarray) -> TurbulenceFieldSite:
        """TurbulenceFieldSite advecting ``tf`` at the LES speed, centred on the farm."""
        from dynamiks.sites.precursor import n_ramp_offset

        meta = self.meta
        U = float(meta["advection_speed"])
        profile = ConstantWindSpeedProfile(wsTab=meta["wsTab"], zTab=meta["z"], Uadv=U)
        n_y = int(meta["Nxyz"][1])
        dy = float(meta["dxyz"][1])
        y_center = float(
            (turbine_positions[:, 1].max() + turbine_positions[:, 1].min()) / 2
        )
        # x: notebook offset convention + the sampled start-time window
        # (advancing the offset by U*tau == having advected tau seconds).
        # y: center the box on the farm.
        offset = [
            n_ramp_offset(meta) + U * self.window_offset_s,
            y_center - (n_y - 1) * dy / 2.0,
            0.0,
        ]
        return TurbulenceFieldSite(
            ws=profile,
            turbulenceField=tf,
            turbulence_transport_speed=U,
            turbulence_offset=offset,
        )
