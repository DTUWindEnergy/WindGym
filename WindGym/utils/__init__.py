"""Utility functions and tools for WindGym environments."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from .test_env import test_env  # noqa: F401


def scale_val(
    val: NDArray[np.floating], min_val: float, max_val: float
) -> NDArray[np.floating]:
    """Scale the value from -1 to 1."""
    return 2 * (val - min_val) / (max_val - min_val) - 1


def circ_mean_deg(
    a: NDArray[np.floating], axis: int | None = None
) -> float | NDArray[np.floating]:
    """Circular mean of angles in degrees, in [0, 360).

    Unlike the arithmetic mean, this handles the 0/360 wrap correctly:
    circ_mean_deg([359, 1]) == 0, not 180.
    """
    rad = np.deg2rad(np.asarray(a, dtype=float))
    s = np.mean(np.sin(rad), axis=axis)
    c = np.mean(np.cos(rad), axis=axis)
    res = np.rad2deg(np.arctan2(s, c)) % 360.0
    # A tiny negative angle mod 360 rounds to exactly 360.0 in floating
    # point; map it back so the result truly lives in [0, 360).
    return np.where(res >= 360.0, 0.0, res)[()]


def defined_yaw(yaws: NDArray[np.floating], n_turb: int) -> NDArray[np.floating]:
    """Set the yaw values to a specified value.

    If the length of the yaw values is equal to the number of turbines, return the yaw values.
    If the length of the yaw values is 1, we assume that all turbines should have that yaw value.
    """
    if len(yaws) == n_turb:
        return yaws
    elif len(yaws) == 1:
        return np.ones(n_turb) * yaws[0]
    else:
        raise ValueError(
            f"The specified yaw values are not the right length. "
            f"Expected either 1 or {n_turb} values."
        )


__all__ = [
    "scale_val",
    "circ_mean_deg",
    "defined_yaw",
]
