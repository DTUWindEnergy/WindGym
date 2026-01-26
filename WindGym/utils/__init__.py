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
    "defined_yaw",
]
