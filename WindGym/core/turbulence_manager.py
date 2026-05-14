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
from dynamiks.sites._site import MetmastSite
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

        # Discovered turbulence files (for MannLoad)
        self.turbulence_files = []
        if turbulence_type == "MannLoad":
            if not turbulence_box_path:
                raise FileNotFoundError("Provide 'TurbBox' for turbtype='MannLoad'.")
            self.turbulence_files = self._discover_turbulence_files(turbulence_box_path)

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
    ) -> tuple:
        """
        Create turbulence fields and sites for agent and optionally baseline.

        This method:
        1. Generates turbulence field based on turbulence_type
        2. Calculates t_developed and time_max based on farm geometry
        3. Creates MetmastSite with wind direction time series
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

        # Create agent site
        site = MetmastSite(
            ws=ws,
            turbulenceField=tf_agent,
            wd_lst=wd_list,
            dt=dt_sim,
            max_wd_step=d_theta_lim,
            update_interval=dt_sim,
        )

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
            memo = {id(tf_agent.uvw): tf_agent.uvw}
            tf_base = copy.deepcopy(tf_agent, memo)
            site_baseline = MetmastSite(
                ws=ws,
                turbulenceField=tf_base,
                wd_lst=wd_list,
                dt=dt_sim,
                max_wd_step=d_theta_lim,
                update_interval=dt_sim,
            )
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
            scaling=BranlardScaling()
        )
        return tf, added_turb_model

    def _generate_mann_generate(
        self, ws: float, ti: float, rotor_diameter: float,
        mann_overrides: Optional[dict] = None,
    ) -> tuple:
        """Generate new Mann turbulence box with random seed.

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

        tf_seed = self.np_random.integers(0, 100000)

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
