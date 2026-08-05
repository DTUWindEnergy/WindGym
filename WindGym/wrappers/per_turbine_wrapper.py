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
    - action_dim_per_turbine: int (1 or 2)

Methods:
    - reset() -> obs of shape (n_turbines, obs_dim_per_turbine)
    - step(action) -> obs of shape (n_turbines, obs_dim_per_turbine)

The action input to step() is shape (n_turbines, action_dim_per_turbine),
with one row per turbine: [yaw_i], [derate_i], or [yaw_i, derate_i]
depending on the env's yaw_action/derate_action flags. The base env expects
actions grouped by variable ([yaw_0..yaw_n | derate_0..derate_n]), so the
wrapper transposes before flattening.
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

    Actions are per-turbine rows whose width follows the env's act_var:
        yaw only (default):            [yaw_i]
        derate only (yaw_action=False):[derate_i]
        yaw + derate:                  [yaw_i, derate_i]
    This is the same row convention as WindFarmEnvMulti agents. The base env
    expects actions grouped by variable ([yaw_0..yaw_n | derate_0..derate_n]),
    so 2-D actions are transposed before flattening.
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

        # The obs reshape assumes the flat obs is exactly the per-turbine
        # blocks back to back; farm-level measurements (e.g. mes_level
        # farm_ws=True) would break that silently.
        obs_var = env.unwrapped.obs_var
        if obs_var != self._n_turbines * self._obs_dim_per_turbine:
            raise ValueError(
                f"PerTurbineObservationWrapper requires a purely per-turbine "
                f"observation vector, but the env has obs_var={obs_var} != "
                f"n_turbines * obs_dim_per_turbine = "
                f"{self._n_turbines} * {self._obs_dim_per_turbine}. "
                f"Disable farm-level measurements (mes_level farm_* keys) "
                f"to use this wrapper."
            )

        # Action dimension per turbine: the env encodes yaw/derate/both here
        # (1 for yaw-only or derate-only, 2 for yaw + derate)
        self._action_dim_per_turbine = int(env.unwrapped.act_var)

        # New observation space: (n_turbines, obs_dim_per_turbine)
        obs_low = -np.inf  # or get from base env
        obs_high = np.inf
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self._n_turbines, self._obs_dim_per_turbine),
            dtype=np.float32,
        )

        # Per-turbine action space, mirroring the 2-D observation space.
        # Row layout: [yaw_i], [derate_i], or [yaw_i, derate_i].
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_turbines, self._action_dim_per_turbine),
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
    def action_dim_per_turbine(self) -> int:
        """Number of actions per turbine (1 for yaw-only or derate-only,
        2 for yaw + derate)."""
        return self._action_dim_per_turbine

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

        The base env groups actions by variable:
        [yaw_0..yaw_n | derate_0..derate_n], so a (n_turbines, act_var) input
        is transposed before flattening. A plain reshape would produce
        turbine-grouped ordering and scramble yaw/derate across turbines.

        Args:
            per_turbine_action: Shape (n_turbines, action_dim_per_turbine),
                               or already flat in base-env layout
                               (n_turbines * action_dim,)

        Returns:
            flat_action: Shape (n_turbines * action_dim,), grouped by variable
        """
        per_turbine_action = np.asarray(per_turbine_action)
        flat_dim = self._n_turbines * self._action_dim_per_turbine

        # Already flat: assumed to be in base-env (variable-grouped) layout
        if per_turbine_action.ndim == 1:
            if per_turbine_action.shape[0] != flat_dim:
                raise ValueError(
                    f"Flat action has length {per_turbine_action.shape[0]}, "
                    f"expected n_turbines * action_dim_per_turbine = "
                    f"{self._n_turbines} * {self._action_dim_per_turbine} "
                    f"= {flat_dim}."
                )
            return per_turbine_action

        expected_shape = (self._n_turbines, self._action_dim_per_turbine)
        if per_turbine_action.ndim != 2 or per_turbine_action.shape != expected_shape:
            raise ValueError(
                f"Per-turbine action has shape {per_turbine_action.shape}, "
                f"expected {expected_shape} "
                f"(one [yaw_i], [derate_i], or [yaw_i, derate_i] row per "
                f"turbine) or a flat vector of length {flat_dim}."
            )

        # (n_turbines, act_var) -> variable-grouped flat vector
        return per_turbine_action.T.reshape(-1)

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
