"""Utility functions and tools for WindGym environments."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def scale_val(
    val: NDArray[np.floating], min_val: float, max_val: float
) -> NDArray[np.floating]:
    """Scale the value from -1 to 1."""
    return 2 * (val - min_val) / (max_val - min_val) - 1


def random_choice(
    np_random: np.random.Generator, min_val: float, max_val: float, n: int
) -> float:
    """Return a random choice between min_val and max_val."""
    return np_random.choice(np.linspace(min_val, max_val, n))


def random_uniform(
    np_random: np.random.Generator, min_val: float, max_val: float, n: int | None = None
) -> float:
    """Return a random value between min_val and max_val.

    Note: The n parameter is ignored for compatibility.
    """
    return np_random.uniform(low=min_val, high=max_val)


def return_zeros(n: int, **kwargs) -> NDArray[np.floating]:
    """Return a numpy array of zeros with the specified shape."""
    return np.zeros(n)


def randoms_uniform(
    np_random: np.random.Generator, min_val: float, max_val: float, n: int
) -> NDArray[np.floating]:
    """Return a numpy array of n random values between min_val and max_val."""
    return np_random.uniform(low=min_val, high=max_val, size=n)


def random_normal(
    np_random: np.random.Generator, mean: float, std: float, n: int
) -> NDArray[np.floating]:
    """Return a numpy array of n random values with a normal distribution."""
    return np_random.normal(loc=mean, scale=std, size=n)


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
    "random_choice",
    "random_uniform",
    "return_zeros",
    "randoms_uniform",
    "random_normal",
    "defined_yaw",
]
