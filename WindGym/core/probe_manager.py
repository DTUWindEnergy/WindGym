"""
Probe management module for WindGym environments.

This module handles wind probe initialization, positioning, and rotation
to track wind conditions at specific locations in the wind farm.
"""

from typing import Optional, List, Dict
from collections import defaultdict
import numpy as np
import math

from .wind_probe import WindProbe


class ProbeManager:
    """
    Manages wind probes in the environment.

    Supports two placement modes:
    1. Free placement: Probes at fixed absolute positions
    2. Turbine-relative: Probes positioned relative to turbines, rotating with yaw

    Attributes:
        probes: List of all WindProbe objects
        turbine_probes: Dict mapping turbine_index to list of probes
    """

    def __init__(self, probes_config: Optional[List[Dict]] = None):
        """
        Initialize the probe manager.

        Args:
            probes_config: List of probe configuration dictionaries.
                          Each dict can contain:
                          - position: Absolute [x, y, z] position (free placement)
                          - turbine_index: Index of turbine to attach to
                          - relative_position: [x, y, z] relative to turbine
                          - include_wakes: Whether to include wake effects
                          - exclude_wake_from: List of turbine indices to exclude
                          - time: Specific time for probe reading
                          - probe_type: "WS" or "TI"
                          - name: Optional probe name
        """
        self.probes_config = probes_config or []
        self.probes: List[WindProbe] = []
        self.turbine_probes: Dict[int, List[WindProbe]] = defaultdict(list)

    def count_probes_per_turbine(self) -> Dict[int, int]:
        """
        Count how many probes are assigned to each turbine index.

        Returns:
            Dict mapping turbine_index to probe count
        """
        counts = defaultdict(int)
        for p in self.probes_config:
            tid = p.get("turbine_index")
            if tid is not None:
                counts[tid] += 1
        return dict(counts)

    def initialize_probes_free_placement(self, env) -> List[WindProbe]:
        """
        Initialize probes with free (absolute) placement.

        This mode is for probes at fixed positions that don't rotate with turbines.

        Args:
            env: Environment object (for WindProbe compatibility)

        Returns:
            List of initialized WindProbe objects
        """
        self.probes = []
        for i, p in enumerate(self.probes_config):
            probe = WindProbe(
                env=env,
                position=tuple(p["position"]),
                include_wakes=p.get("include_wakes", True),
                exclude_wake_from=p.get("exclude_wake_from", []),
                time=p.get("time", None),
            )
            probe.name = p.get("name", f"probe_{i}")
            self.probes.append(probe)

        return self.probes

    def initialize_probes(self, fs, yaw_angles) -> List[WindProbe]:
        """
        Initialize probes with turbine-relative positioning.

        Probes can be placed relative to turbines and will rotate with turbine yaw.

        Args:
            fs: Flow simulation object
            yaw_angles: Initial yaw angles (degrees), scalar or array

        Returns:
            List of initialized WindProbe objects
        """
        self.probes = []

        # Convert yaw_angles to array in radians
        yaw_angles = (
            np.full(len(fs.windTurbines.positions_xyz[0]), yaw_angles)
            if np.isscalar(yaw_angles)
            else np.array(yaw_angles)
        )
        yaw_angles = np.radians(yaw_angles)

        for i, p in enumerate(self.probes_config):
            if "turbine_index" in p and "relative_position" in p:
                # Turbine-relative placement (rotates with yaw)
                tid = p["turbine_index"]
                rel = p["relative_position"]
                yaw = yaw_angles[tid]

                rel_x, rel_y = rel[0], rel[1]
                rel_z = rel[2] if len(rel) > 2 else 0.0

                # Rotate relative position by turbine yaw
                rel_x_rot = rel_x * math.cos(yaw) - rel_y * math.sin(yaw)
                rel_y_rot = rel_x * math.sin(yaw) + rel_y * math.cos(yaw)

                # Turbine absolute position
                tp_x = fs.windTurbines.rotor_positions_xyz[0][tid]
                tp_y = fs.windTurbines.rotor_positions_xyz[1][tid]
                tp_z = fs.windTurbines.rotor_positions_xyz[2][tid]

                position = (tp_x + rel_x_rot, tp_y + rel_y_rot, tp_z + rel_z)
            else:
                # Free placement (fixed position)
                yaw = 0
                position = tuple(p["position"])
                tp_x, tp_y, tp_z = p.get("turbine_position", (0, 0, 0))

            probe = WindProbe(
                fs=fs,
                position=position,
                include_wakes=p.get("include_wakes", True),
                exclude_wake_from=p.get("exclude_wake_from", []),
                time=p.get("time", None),
                probe_type=p.get("probe_type"),
                yaw_angle=yaw,
                turbine_position=(tp_x, tp_y, tp_z),
            )

            # Enforce name and turbine index for lookup
            probe.name = p.get("name", f"probe_{i}")
            probe.turbine_index = p.get("turbine_index", None)

            self.probes.append(probe)

        # Group probes per turbine for easy access later
        self.turbine_probes = defaultdict(list)
        for probe in self.probes:
            if probe.turbine_index is not None:
                self.turbine_probes[probe.turbine_index].append(probe)

        return self.probes

    def update_probe_positions(self, fs, yaw_angles):
        """
        Update probe positions when turbines yaw.

        Only updates probes that are attached to turbines (turbine-relative).

        Args:
            fs: Flow simulation object
            yaw_angles: New yaw angles (degrees), array
        """
        # Convert yaw_angles to radians
        yaw_angles = np.radians(yaw_angles)

        for probe in self.probes:
            # Only update if probe has turbine_index and relative_position info
            if hasattr(probe, "turbine_index") and probe.turbine_index is not None:
                tid = probe.turbine_index

                # Find relative position from config
                rel = None
                for pconf in self.probes_config:
                    if pconf.get("name") == probe.name:
                        rel = pconf.get("relative_position")
                        break

                if rel is None:
                    continue  # Can't rotate without relative position

                yaw = yaw_angles[tid]

                rel_x, rel_y = rel[0], rel[1]
                rel_z = rel[2] if len(rel) > 2 else 0.0

                # Rotate relative position by new yaw angle
                rel_x_rot = rel_x * np.cos(yaw) - rel_y * np.sin(yaw)
                rel_y_rot = rel_x * np.sin(yaw) + rel_y * np.cos(yaw)

                # Get current turbine position
                tp_x = fs.windTurbines.rotor_positions_xyz[0][tid]
                tp_y = fs.windTurbines.rotor_positions_xyz[1][tid]
                tp_z = fs.windTurbines.rotor_positions_xyz[2][tid]

                # Update probe position and yaw
                new_position = (tp_x + rel_x_rot, tp_y + rel_y_rot, tp_z + rel_z)
                probe.yaw_angle = yaw
                probe.position = new_position

    def get_probe_readings(self) -> List[float]:
        """
        Get readings from all probes.

        Returns:
            List of probe readings (wind speed or turbulence intensity)
        """
        return [float(probe.read()) for probe in self.probes]

    def get_turbine_probe_readings(self, turbine_index: int) -> List[float]:
        """
        Get readings from probes attached to a specific turbine.

        Args:
            turbine_index: Index of turbine

        Returns:
            List of probe readings for that turbine
        """
        if turbine_index not in self.turbine_probes:
            return []

        return [float(probe.read()) for probe in self.turbine_probes[turbine_index]]

    def has_probes(self) -> bool:
        """Check if any probes are configured."""
        return len(self.probes) > 0
