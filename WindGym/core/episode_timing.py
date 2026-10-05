"""Episode timing helpers derived from farm geometry.

Pure functions shared by ``WindFarmEnv.reset`` and
``TurbulenceManager.create_sites``: how long the flow needs to develop before
an episode starts, how long the episode may run, and how fast the wind
direction may rotate without moving any turbine more than ``max_turb_move``
metres per step.
"""

from __future__ import annotations

import math

import numpy as np


def episode_time_parameters(
    turbine_positions: np.ndarray,
    rotor_diameter: float,
    ws: float,
    n_passthrough: float,
    burn_in_passthroughs: float,
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


def max_wd_step(
    turbine_positions: np.ndarray, rotor_diameter: float, max_turb_move: float
) -> float:
    """Wind-direction change rate limit (deg per step) so that no turbine moves
    more than ``max_turb_move`` metres around the farm centre in one step."""
    turb_pos = turbine_positions
    center = (turb_pos.max(0) + turb_pos.min(0)) / 2
    distances = np.sqrt(np.sum((turb_pos - center) ** 2, axis=1))
    max_dist = np.max(distances)
    # If only 1 turbine, max_dist is half rotor diameter
    max_dist = max(max_dist, rotor_diameter / 2)

    return max_turb_move * 360 / (2 * np.pi * max_dist)
