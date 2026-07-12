"""
Power reference management for power-tracking environments.

This module generates and serves the farm-level power reference signal that a
tracking agent must follow (like a grid/AGC dispatch command). It follows the
WindManager/BaselineManager composition pattern: the environment owns a
PowerTrackingManager, injects its RNG at reset, and queries the reference per
env step.
"""

from typing import Callable, Optional
from collections import deque
import math
import numpy as np


class PowerTrackingManager:
    """
    Manages the farm power reference trajectory for one episode.

    The reference is resolved at env-step resolution: index i corresponds to
    simulation time t = i * delay seconds. The full episode trajectory (plus a
    preview tail) is precomputed at reset, either from a user-supplied
    callable or from the default constant-setpoint sampler.

    Args:
        ref_function: Optional callable(t_seconds, env) -> reference power in
            watts. Evaluated at env-step times; may use env.np_random,
            env.ws, env.n_turb, env.rated_power. None uses the default
            sampler: a constant setpoint per episode, uniform in
            ref_range * (rated_power * n_turb), where rated_power is the
            freestream power of one turbine at the episode wind speed.
        ref_range: (low, high) fraction range for the default sampler.
        preview_steps: Number of future setpoints exposed as observations.
    """

    def __init__(
        self,
        ref_function: Optional[Callable] = None,
        ref_range=(0.2, 0.8),
        preview_steps: int = 0,
    ):
        if len(ref_range) != 2:
            raise ValueError("ref_range must be a (low, high) pair")
        lo, hi = float(ref_range[0]), float(ref_range[1])
        if not (0.0 <= lo <= hi):
            raise ValueError(
                f"ref_range must satisfy 0 <= low <= high, got ({lo}, {hi})"
            )
        if preview_steps < 0:
            raise ValueError("preview_steps must be non-negative")

        self.ref_function = ref_function
        self.ref_range = (lo, hi)
        self.preview_steps = int(preview_steps)

        # Random number generator (set by environment at reset)
        self.np_random = None

        self.trajectory = None
        self.ref_deque = None
        self._env = None
        self._delay = None

    def reset_episode(self, env, time_max: float, delay: float, power_avg: int):
        """
        Precompute the reference trajectory for a new episode.

        Args:
            env: The environment (passed through to ref_function; the default
                sampler reads env.rated_power and env.n_turb from it).
            time_max: Episode length in seconds.
            delay: Seconds per env step (trajectory resolution).
            power_avg: Window size for the reference deque; must match the
                farm power deque so the reward compares equal-length means.
        """
        self._env = env
        self._delay = float(delay)

        # One entry per env step (indices 0..ceil(time_max/delay)), plus the
        # preview tail so previews near truncation stay in-bounds.
        n = math.ceil(time_max / delay) + 1 + self.preview_steps

        if self.ref_function is None:
            if self.np_random is None:
                raise RuntimeError(
                    "np_random must be set before sampling. "
                    "Call env.reset() to initialize the random generator."
                )
            lo, hi = self.ref_range
            setpoint = float(self.np_random.uniform(lo, hi)) * (
                env.rated_power * env.n_turb
            )
            self.trajectory = np.full(n, setpoint, dtype=np.float64)
        else:
            times = np.arange(n) * self._delay
            self.trajectory = np.array(
                [float(self.ref_function(t, env)) for t in times], dtype=np.float64
            )

        self.ref_deque = deque(maxlen=power_avg)

    def reference(self, step_idx: int) -> float:
        """
        Reference power (W) at env step *step_idx*.

        Indices past the precomputed horizon can occur (FarmEval runs with an
        effectively infinite time_max): without a callable the last value is
        held; with a callable the trajectory is lazily extended.
        """
        if self.trajectory is None:
            raise RuntimeError(
                "reset_episode must be called before querying the reference."
            )
        if step_idx < len(self.trajectory):
            return float(self.trajectory[step_idx])

        if self.ref_function is None:
            return float(self.trajectory[-1])

        # Lazily extend the trajectory up to and including step_idx.
        times = np.arange(len(self.trajectory), step_idx + 1) * self._delay
        extension = np.array(
            [float(self.ref_function(t, self._env)) for t in times], dtype=np.float64
        )
        self.trajectory = np.concatenate([self.trajectory, extension])
        return float(self.trajectory[step_idx])

    def preview(self, step_idx: int) -> np.ndarray:
        """References for env steps step_idx+1 .. step_idx+preview_steps."""
        return np.array(
            [self.reference(step_idx + 1 + k) for k in range(self.preview_steps)],
            dtype=np.float64,
        )

    def push(self, step_idx: int) -> float:
        """Append the reference at *step_idx* to the window deque, return it."""
        ref = self.reference(step_idx)
        self.ref_deque.append(ref)
        return ref
