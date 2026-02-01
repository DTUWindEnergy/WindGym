"""
PerTurbineObservationWrapper Template

This wrapper restructures the flat observation/action spaces of WindFarmEnv
into per-turbine.

-------------------------------------------------------
Properties:
    - n_turbines: int
    - turbine_positions: np.ndarray of shape (n_turbines, 2)
    - rotor_diameter: float
    - mean_wind_direction: float

Methods:
    - reset() -> obs of shape (n_turbines, obs_dim_per_turbine)
    - step(action) -> obs of shape (n_turbines, obs_dim_per_turbine)

The action input to step() will be shape (n_turbines, action_dim_per_turbine)
and needs to be flattened before passing to the base environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class PerTurbineObservationWrapper(gym.Wrapper):
    """
    Wrapper that restructures WindFarmEnv observations and actions
    from flat vectors to per-turbine format.

    Base env observation (flat): [t0_ws, t0_wd, t0_yaw, t1_ws, t1_wd, t1_yaw, ...]
    Wrapped observation: [[t0_ws, t0_wd, t0_yaw], [t1_ws, t1_wd, t1_yaw], ...]

    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The base WindFarmEnv (or vectorized version)
        """
        super().__init__(env)

        self._n_turbines = env.n_turb
        self._turbine_positions = np.column_stack([env.x_pos, env.y_pos])

        # Rotor diameter in meters (for position normalization)
        self._rotor_diameter = env.D

        # Observation dimension per turbine
        # Use config-based calculation if available (avoids requiring reset)
        if hasattr(env, "get_obs_dim_per_turbine"):
            self._obs_dim_per_turbine = env.get_obs_dim_per_turbine()
        else:
            self._obs_dim_per_turbine = len(
                env.farm_measurements.turb_mes[0].get_measurements()
            )

        # Action dimension per turbine
        self._action_dim_per_turbine = 1  # Usually just yaw

        # New observation space: (n_turbines, obs_dim_per_turbine)
        obs_low = -np.inf  # or get from base env
        obs_high = np.inf
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self._n_turbines, self._obs_dim_per_turbine),
            dtype=np.float32,
        )

    @property
    def n_turbines(self) -> int:
        """Number of turbines in the farm."""
        return self._n_turbines

    @property
    def turbine_positions(self) -> np.ndarray:
        """
        Turbine positions in meters.

        Returns:
            np.ndarray of shape (n_turbines, 2) with (x, y) coordinates
        """
        return self._turbine_positions

    @property
    def rotor_diameter(self) -> float:
        """Rotor diameter in meters (used for position normalization)."""
        return self._rotor_diameter

    @property
    def mean_wind_direction(self) -> float:
        """
        Current mean wind direction in degrees (meteorological convention).

        This is used for transforming positions to wind-relative coordinates.
        270° means wind comes from the West.

        Returns:
            float: Wind direction in degrees
        """
        return self.env.wd

    # =========================================================================
    # Core Methods
    # =========================================================================

    def _reshape_obs_to_per_turbine(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Reshape flat observation to per-turbine format.

        Args:
            flat_obs: Shape (n_turbines * obs_dim_per_turbine,) or
                      (batch, n_turbines * obs_dim_per_turbine) for vectorized

        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine) or
                 (batch, n_turbines, obs_dim_per_turbine)
        """
        """Reshape assuming obs is ordered by turbine."""
        return flat_obs.reshape(self._n_turbines, self._obs_dim_per_turbine)

    def _flatten_action(self, per_turbine_action: np.ndarray) -> np.ndarray:
        """
        Flatten per-turbine action back to format expected by base env.

        Args:
            per_turbine_action: Shape (n_turbines, action_dim_per_turbine)
                               or already flat (n_turbines * action_dim,)

        Returns:
            flat_action: Shape expected by base env
        """
        # Handle case where action is already flat
        if per_turbine_action.ndim == 1:
            return per_turbine_action

        # Flatten: (n_turbines, action_dim) -> (n_turbines * action_dim,)
        return per_turbine_action.reshape(-1)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and return per-turbine observation.

        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine)
            info: Dict with additional information
        """
        flat_obs, info = self.env.reset(seed=seed, options=options)

        # Reshape to per-turbine
        obs = self._reshape_obs_to_per_turbine(flat_obs)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step with per-turbine action.

        Args:
            action: Shape (n_turbines, action_dim_per_turbine) or flat

        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        # Flatten action for base env
        flat_action = self._flatten_action(action)

        # Step base env
        flat_obs, reward, terminated, truncated, info = self.env.step(flat_action)

        # Reshape observation
        obs = self._reshape_obs_to_per_turbine(flat_obs)

        return obs, reward, terminated, truncated, info
