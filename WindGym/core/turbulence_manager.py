"""
Turbulence field and site management module for WindGym environments.

This module handles turbulence field generation, site creation, and time calculations
for wind farm simulations. Supports multiple turbulence generation strategies.
"""

from typing import Union, Optional
from pathlib import Path
import numpy as np
import math
import copy
import gc

from dynamiks.sites.turbulence_fields import MannTurbulenceField, RandomTurbulence
from dynamiks.sites._site import MetmastSite, TurbulenceFieldSite
from dynamiks.sites.mean_wind import ConstantWindSpeedProfile
from dynamiks.dwm.added_turbulence_models import (
    AutoScalingIsotropicMannTurbulence,
    BranlardScaling,
    SynchronizedAutoScalingIsotropicMannTurbulence,
)

from .dwm_defaults import (
    MANN_AE,
    MANN_DXYZ,
    MANN_GAMMA,
    MANN_L,
    MANN_NXYZ,
)


class TurbulenceManager:
    """
    Manages turbulence field generation and site creation for wind farm simulations.

    Supports multiple turbulence generation strategies:
    - MannLoad: Load pre-generated Mann turbulence boxes from files
    - MannGenerate: Generate new Mann turbulence boxes on-the-fly
    - MannFixed: Generate a fixed Mann turbulence box (reproducible)
    - Random: Use random turbulence (faster, less realistic)
    - None: Zero turbulence (fastest, for testing)
    - Precursor: LES-precursor inflow from a memmap sidecar (see
      dynamiks.sites.precursor). turbulence_box_path is the precursor .nc
      (with a unique converted sidecar next to it) or the sidecar
      .npy/.meta.npz path itself. The read-only memmap is opened once here
      and shared by every site/reset (and, through the OS page cache, by
      every worker process on the node).
    """

    def __init__(
        self,
        turbulence_type: str,
        turbulence_box_path: Optional[Union[str, Path]] = None,
        max_turb_move: float = 2.0,
    ):
        """
        Initialize the turbulence manager.

        Args:
            turbulence_type: Type of turbulence ("MannLoad", "MannGenerate",
                           "MannFixed", "Random", "None")
            turbulence_box_path: Path to turbulence box files (required for MannLoad)
            max_turb_move: Maximum distance turbines can move in one timestep (m)
                          Used to calculate wind direction change rate limits
        """
        self.turbulence_type = turbulence_type
        self.turbulence_box_path = turbulence_box_path
        self.max_turb_move = max_turb_move

        # Random number generator (set by environment)
        self.np_random = None

        # Optional explicit Mann-box seed. When not None, MannGenerate uses THIS seed for the
        # turbulence box instead of drawing one from np_random. Lets an evaluator pin a
        # specific, reproducible, held-out box per episode (e.g. seeds >= 100000, which the
        # training draw `np_random.integers(0, 100000)` can never produce). None = original
        # behaviour (random box each reset).
        self.box_seed = None

        # Discovered turbulence files (for MannLoad)
        self.turbulence_files = []
        if turbulence_type == "MannLoad":
            if not turbulence_box_path:
                raise FileNotFoundError("Provide 'TurbBox' for turbtype='MannLoad'.")
            self.turbulence_files = self._discover_turbulence_files(turbulence_box_path)

        # Precursor sidecar: memmap + meta opened once, reused across resets.
        # The per-episode start-time window (seconds into the box) is sampled
        # in create_sites; exposed here for logging/tests.
        self.precursor_meta = None
        self._precursor_uvw = None
        self.window_offset_s = 0.0
        if turbulence_type == "Precursor":
            if not turbulence_box_path:
                raise FileNotFoundError(
                    "Provide 'TurbBox' (precursor .nc or sidecar .npy/.meta.npz "
                    "path) for turbtype='Precursor'."
                )
            from dynamiks.sites.precursor import load_sidecar

            self._precursor_uvw, self.precursor_meta = load_sidecar(turbulence_box_path)

    def create_sites(
        self,
        ws: float,
        wd: float,
        ti: float,
        wd_list: list,
        dt_sim: float,
        turbine_positions: np.ndarray,
        rotor_diameter: float,
        n_passthrough: int,
        burn_in_passthroughs: int,
        create_baseline: bool = False,
        mann_overrides: Optional[dict] = None,
        veer_rate: float = 0.0,
        veer_ref_height: float = 0.0,
        episode_time_budget_s: Optional[float] = None,
    ) -> tuple:
        """
        Create turbulence fields and sites for agent and optionally baseline.

        This method:
        1. Generates turbulence field based on turbulence_type
        2. Calculates t_developed and time_max based on farm geometry
        3. Creates MetmastSite with wind direction time series (or a
           TurbulenceFieldSite with a veered mean-wind profile if veer_rate != 0)
        4. Optionally creates baseline site (deep copy of turbulence field)

        Args:
            ws: Wind speed (m/s)
            wd: Wind direction (degrees)
            ti: Turbulence intensity (fraction)
            wd_list: Wind direction time series
            dt_sim: Simulation timestep (seconds)
            turbine_positions: Turbine positions [x, y] array (n_turb, 2)
            rotor_diameter: Rotor diameter (m)
            n_passthrough: Number of flow passthroughs for episode
            burn_in_passthroughs: Number of passthroughs for flow development
            create_baseline: Whether to create baseline site
            mann_overrides: Optional dict with per-episode Mann-box parameter
                overrides drawn from a calibrated posterior. Recognized keys:
                ``mann_L``, ``mann_GAMMA``, ``mann_AE``. Only honored by the
                ``MannGenerate`` branch (other branches raise if non-empty —
                this is defense-in-depth; ``WindFarmEnv.reset`` is the primary
                guard).
            veer_rate: Linear veer rate in deg per METER (callers pass
                deg-per-100m values divided by 100). Positive = wd increases
                with height. 0 disables veer (default, MetmastSite path).
            veer_ref_height: Height (m) where the veer offset is zero,
                normally hub height, so hub-height wd equals the nominal wd.
            episode_time_budget_s: Total seconds of simulation the episode will
                consume (burn-in + sensor fill + episode). Only used by the
                Precursor branch to bound the random start-time window; the env
                passes it because it overrides time_max after
                _calculate_time_parameters. Default None falls back to
                t_developed + time_max as computed here.

        Returns:
            tuple: (site, site_baseline, t_developed, time_max, added_turbulence_model)
                  site_baseline is None if create_baseline=False
        """
        if self.np_random is None:
            raise RuntimeError(
                "np_random must be set before creating sites. "
                "Call env.reset() to initialize the random generator."
            )

        # Generate turbulence field
        tf_agent, added_turbulence_model = self._generate_turbulence_field(
            ws=ws, ti=ti, rotor_diameter=rotor_diameter,
            mann_overrides=mann_overrides,
        )

        # Calculate time parameters based on farm geometry
        t_developed, time_max = self._calculate_time_parameters(
            turbine_positions=turbine_positions,
            rotor_diameter=rotor_diameter,
            ws=ws,
            n_passthrough=n_passthrough,
            burn_in_passthroughs=burn_in_passthroughs,
        )

        # Calculate wind direction change rate limit
        turb_pos = turbine_positions
        center = (turb_pos.max(0) + turb_pos.min(0)) / 2
        distances = np.sqrt(np.sum((turb_pos - center) ** 2, axis=1))
        max_dist = np.max(distances)
        # If only 1 turbine, max_dist is half rotor diameter
        max_dist = max(max_dist, rotor_diameter / 2)

        d_theta_lim = self.max_turb_move * 360 / (2 * np.pi * max_dist)

        if veer_rate and np.ptp(wd_list) > 0:
            raise ValueError(
                "Veer uses ConstantWindSpeedProfile, which does not support a "
                "time-varying wind direction series. Disable veer or use a "
                "constant wind direction."
            )

        if self.turbulence_type == "Precursor":
            self._check_precursor_episode(
                turbine_positions=turbine_positions,
                rotor_diameter=rotor_diameter,
                veer_rate=veer_rate,
                wd_list=wd_list,
                episode_time_budget_s=(
                    episode_time_budget_s
                    if episode_time_budget_s is not None
                    else t_developed + time_max
                ),
            )

        def _make_site(tf):
            if self.turbulence_type == "Precursor":
                from dynamiks.sites.precursor import n_ramp_offset

                meta = self.precursor_meta
                U = float(meta["advection_speed"])
                profile = ConstantWindSpeedProfile(
                    wsTab=meta["wsTab"], zTab=meta["z"], Uadv=U
                )
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
            if veer_rate:
                # Veered inflow: use ConstantWindSpeedProfile instead of
                # MetmastSite. MetmastSite hard-codes a height-independent
                # mean wind (dynamiks _site.py, mean_wind), so it cannot
                # express veer; the profile site cannot express a
                # time-varying wind direction. We trade the latter for the
                # former: veered episodes require a constant wd series
                # (guarded above).
                # wsTab is in the flow frame (+x downstream at nominal wd):
                # a pure rotation theta(z) = veer_rate*(z - hub) keeps
                # |U| = ws at every height and pins the hub-height wd to the
                # nominal sampled wd. v = -sin gives wd increasing with
                # height for positive veer_rate (NH meteorological veer).
                z_tab = np.linspace(0.0, veer_ref_height + 2.0 * rotor_diameter, 64)
                # Include the reference height as an exact node so the veer
                # offset (and hence v) is exactly zero at hub height.
                z_tab = np.union1d(z_tab, [veer_ref_height])
                d_theta = np.deg2rad(veer_rate) * (z_tab - veer_ref_height)
                ws_tab = np.array(
                    [
                        ws * np.cos(d_theta),
                        -ws * np.sin(d_theta),
                        np.zeros_like(d_theta),
                    ]
                )
                profile = ConstantWindSpeedProfile(wsTab=ws_tab, zTab=z_tab, Uadv=ws)
                return TurbulenceFieldSite(ws=profile, turbulenceField=tf)
            return MetmastSite(
                ws=ws,
                turbulenceField=tf,
                wd_lst=wd_list,
                dt=dt_sim,
                max_wd_step=d_theta_lim,
                update_interval=dt_sim,
            )

        # Create agent site
        site = _make_site(tf_agent)

        # Create baseline site if requested
        site_baseline = None
        if create_baseline:
            # Share the large `uvw` ndarray (~3.2 GB) by reference between the
            # agent and baseline fields, while deep-copying everything else so
            # each field keeps its own mutable `time_offset`/`ti` state.
            # Pre-seeding the memo with `uvw` mapped to itself makes deepcopy
            # skip (share) it. Safe only because `uvw` is read-only after
            # `_generate_turbulence_field` returns — `scale_TI` (its only
            # in-place mutator) has already run above. Revisit if any caller
            # scales TI per-field after `create_sites`.
            # RandomTurbulence has no `uvw` box and is cheap to copy outright.
            memo = (
                {id(tf_agent.uvw): tf_agent.uvw}
                if hasattr(tf_agent, "uvw")
                else None
            )
            tf_base = copy.deepcopy(tf_agent, memo)
            site_baseline = _make_site(tf_base)
            tf_base = None

        # Clean up
        tf_agent = None
        gc.collect()

        return site, site_baseline, t_developed, time_max, added_turbulence_model

    def _generate_turbulence_field(
        self, ws: float, ti: float, rotor_diameter: float,
        mann_overrides: Optional[dict] = None,
    ) -> tuple:
        """
        Generate turbulence field based on turbulence_type.

        Args:
            ws: Wind speed (m/s)
            ti: Turbulence intensity (fraction)
            rotor_diameter: Rotor diameter (m)
            mann_overrides: Per-episode Mann-box overrides (see ``create_sites``).
                Only the ``MannGenerate`` branch can honor these; other branches
                must reject non-empty overrides (defense-in-depth — the env
                already guards on construction-time ``turbtype``).

        Returns:
            tuple: (turbulence_field, added_turbulence_model)
        """
        if mann_overrides and self.turbulence_type != "MannGenerate":
            raise ValueError(
                f"mann_overrides={sorted(mann_overrides)} provided but "
                f"turbulence_type={self.turbulence_type!r} cannot honor them. "
                "Per-episode Mann statistics only take effect under "
                "turbtype='MannGenerate'."
            )

        if self.turbulence_type == "MannLoad":
            return self._generate_mann_load(ws, ti)

        elif self.turbulence_type == "MannGenerate":
            return self._generate_mann_generate(
                ws, ti, rotor_diameter, mann_overrides=mann_overrides,
            )

        elif self.turbulence_type == "MannFixed":
            return self._generate_mann_fixed(ws, ti)

        elif self.turbulence_type == "Random":
            return self._generate_random(ws, ti)

        elif self.turbulence_type == "None":
            return self._generate_none(ws)

        elif self.turbulence_type == "Precursor":
            return self._generate_precursor(ws)

        else:
            raise ValueError("Invalid turbulence type specified")

    def _generate_mann_load(self, ws: float, ti: float) -> tuple:
        """Load Mann turbulence box from pre-generated files."""
        if not self.turbulence_files:
            raise RuntimeError("No turbulence files discovered for MannLoad")

        # Select random turbulence file
        self.tf_file = self.np_random.choice(self.turbulence_files)
        tf = MannTurbulenceField.from_netcdf(filename=self.tf_file)
        tf.scale_TI(TI=ti, U=ws)

        added_turb_model = SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(),
            cache_field=False,
        )
        return tf, added_turb_model

    def _generate_mann_generate(
        self, ws: float, ti: float, rotor_diameter: float,
        mann_overrides: Optional[dict] = None,
    ) -> tuple:
        """Generate new Mann turbulence box (explicit box_seed if set, else random).

        ``rotor_diameter`` is unused (the calibrated grid in ``dwm_defaults``
        is uniform and not D-relative); kept on the signature for now so
        callers don't have to change.

        ``mann_overrides`` is a dict drawn from the calibrated posterior:
        ``{"mann_L": ..., "mann_GAMMA": ..., "mann_AE": ...}``. Missing keys
        fall back to the module-level defaults from ``dwm_defaults``.

        When any override is present, ``tf.scale_TI(...)`` is skipped so that
        αε directly sets the ambient TI of the box — matching the calibration's
        reference implementation in ``calibration/simulator.py:_build_site``
        (αε ≈ (TI_target/0.40)² at WS=9 for the canonical L=29.4, Γ=3.9 box).
        Without overrides, the legacy ``scale_TI`` path remains so non-DR runs
        are byte-identical to pre-change behaviour.
        """
        overrides = mann_overrides or {}
        mann_L_eff     = float(overrides.get("mann_L",     MANN_L))
        mann_gamma_eff = float(overrides.get("mann_GAMMA", MANN_GAMMA))
        mann_ae_eff    = float(overrides.get("mann_AE",    MANN_AE))

        if self.box_seed is not None:
            tf_seed = int(self.box_seed)
        else:
            tf_seed = int(self.np_random.integers(0, 100000))

        tf = MannTurbulenceField.generate(
            alphaepsilon=mann_ae_eff,
            L=mann_L_eff,
            Gamma=mann_gamma_eff,
            Nxyz=MANN_NXYZ,
            dxyz=MANN_DXYZ,
            seed=tf_seed,
        )
        if not overrides:
            # Legacy non-DR path: scale_TI renormalises the raw box to the env's
            # nominal ambient TI. With overrides, αε is authoritative — see
            # docstring above.
            tf.scale_TI(TI=ti, U=ws)

        added_turb_model = SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(), cache_field=False,
        )
        return tf, added_turb_model

    def _generate_mann_fixed(self, ws: float, ti: float) -> tuple:
        """Generate fixed Mann turbulence box (reproducible)."""
        tf_seed = 1234  # Fixed seed for reproducibility

        tf = MannTurbulenceField.generate(
            alphaepsilon=MANN_AE,
            L=MANN_L,
            Gamma=MANN_GAMMA,
            Nxyz=MANN_NXYZ,
            dxyz=MANN_DXYZ,
            seed=tf_seed,
        )
        tf.scale_TI(TI=ti, U=ws)

        added_turb_model = SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(), cache_field=False,
        )
        return tf, added_turb_model

    def _generate_random(self, ws: float, ti: float) -> tuple:
        """Generate random turbulence (fast, less realistic)."""
        tf_seed = self.np_random.integers(0, 100000)
        tf = RandomTurbulence(ti=ti, ws=ws, seed=tf_seed)

        added_turb_model = AutoScalingIsotropicMannTurbulence(cache_field=False)
        return tf, added_turb_model

    def _generate_none(self, ws: float) -> tuple:
        """Generate zero turbulence site (for testing)."""
        tf_seed = self.np_random.integers(2**31)
        tf = RandomTurbulence(ti=0, ws=ws, seed=tf_seed)

        added_turb_model = None
        return tf, added_turb_model

    def _generate_precursor(self, ws: float) -> tuple:
        """Build a PrecursorField around the shared read-only memmap.

        The field object itself is cheap (per-reset advection state around the
        one memmap opened in __init__). Same added-turbulence model as the
        Mann paths / MakeDWM_precursor.ipynb.
        """
        from hipersim import Bounds
        from dynamiks.sites.precursor import PrecursorField

        meta = self.precursor_meta
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
            self._precursor_uvw,
            Nxyz=(n_x, n_y, n_z),
            dxyz=(dx, dy, dz),
            bounds=Bounds.Warning,
            ti_yz=meta["ti_yz"],
        )
        added_turb_model = SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(), cache_field=False,
        )
        return tf, added_turb_model

    def _check_precursor_episode(
        self,
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
        meta = self.precursor_meta
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
        self.window_offset_s = float(self.np_random.uniform(0.0, slack))

    def _calculate_time_parameters(
        self,
        turbine_positions: np.ndarray,
        rotor_diameter: float,
        ws: float,
        n_passthrough: int,
        burn_in_passthroughs: int,
    ) -> tuple:
        """
        Calculate t_developed and time_max based on farm geometry.

        Args:
            turbine_positions: Turbine positions array (n_turb, 2)
            rotor_diameter: Rotor diameter (m)
            ws: Wind speed (m/s)
            n_passthrough: Number of passthroughs for episode
            burn_in_passthroughs: Number of passthroughs for flow development

        Returns:
            tuple: (t_developed, time_max) in seconds
        """
        n_turb = turbine_positions.shape[0]

        # Calculate maximum distance between any turbines

        diff = (
            turbine_positions[:, np.newaxis, :] - turbine_positions[np.newaxis, :, :]
        )  # (NT, NT, 2)
        distances = np.linalg.norm(diff, axis=-1)  # (NT, NT)
        max_distance = distances.max()

        t_inflow = max_distance / ws

        # Time for flow to develop
        t_developed = math.ceil(t_inflow * burn_in_passthroughs)

        # Maximum episode time
        time_max = math.ceil(t_inflow * n_passthrough)

        # Special case: single turbine uses rotor diameter
        if n_turb == 1:
            time_max = math.ceil((rotor_diameter * n_passthrough) / ws)

        # Ensure at least 1 second
        time_max = max(1, time_max)

        return t_developed, time_max

    def _discover_turbulence_files(self, root: Union[str, Path]) -> list[str]:
        """
        Discover turbulence box files (TF_*.nc) in the given directory.

        Args:
            root: Path to file or directory containing turbulence boxes

        Returns:
            list: Sorted list of turbulence file paths

        Raises:
            FileNotFoundError: If no turbulence files found
        """
        p = Path(root)

        # If root is a single file
        if p.is_file() and p.name.startswith("TF_") and p.suffix == ".nc":
            return [str(p)]

        # If root is a directory, search for TF_*.nc files
        if p.is_dir():
            files = sorted(str(f) for f in p.glob("TF_*.nc"))
            if files:
                return files

        raise FileNotFoundError(f"No TF_*.nc files found at: {root}")
