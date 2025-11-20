from gymnasium.vector.vector_env import ArrayType, VectorEnv
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.core import ActType, ObsType


class RecordEpisodeVals(gym.wrappers.vector.RecordEpisodeStatistics):
    """
    This wraps the RecordEpisodeStatistics Wrapper.
    It also adds a queue to store the mean power of the episodes. This is used for the logging during training.
    Could also be expanded upon to include more statistics if wanted.
    """

    def __init__(self, env: VectorEnv, buffer_length=100):
        super().__init__(env=env, buffer_length=buffer_length)

        # Moving windows for mean power
        self.mean_power_queue = deque(maxlen=buffer_length)
        self.mean_power_queue_nowake = deque(maxlen=buffer_length)
        self.mean_power_queue_baseline = deque(maxlen=buffer_length)

        # Per-episode accumulators
        self.episode_powers: np.ndarray = np.zeros(())
        self.episode_powers_nowake: np.ndarray = np.zeros(())
        self.episode_powers_baseline: np.ndarray = np.zeros(())
        self.last_dones: np.ndarray = np.zeros((), dtype=bool)

        # ---- Yaw tracking ----
        # Total yaw travel (per env) accumulated over the episode
        self.episode_yaw_travel: np.ndarray = np.zeros(())
        # Previous yaw angles (per env, per turbine)
        self.prev_yaws: np.ndarray | None = None
        # Moving window of per-episode total yaw travel (per env)
        self.total_yaw_travel_queue = deque(maxlen=buffer_length)
        # Name of the info key containing current yaw angles (deg), shape (num_envs, n_turbs)
        self.yaw_key = "yaw angles agent"

    def reset(self, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        self.episode_powers = np.zeros(self.num_envs)
        self.episode_powers_nowake = np.zeros(self.num_envs)
        self.episode_powers_baseline = np.zeros(self.num_envs)
        self.episode_yaw_travel = np.zeros(self.num_envs)

        self.last_dones = self.prev_dones.copy()

        # Initialize prev_yaws at the very start if available
        if isinstance(info, dict) and self.yaw_key in info:
            self.prev_yaws = np.asarray(info[self.yaw_key]).copy()
        else:
            self.prev_yaws = None

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = super().step(actions)

        # ----------------- Power accumulation -----------------
        self.episode_powers[self.last_dones] = 0
        self.episode_powers[~self.last_dones] += infos["Power agent"][~self.last_dones]

        if "Power agent nowake" in infos:
            self.episode_powers_nowake[self.last_dones] = 0
            self.episode_powers_nowake[~self.last_dones] += infos["Power agent nowake"][
                ~self.last_dones
            ]

        if "Power baseline" in infos:
            self.episode_powers_baseline[self.last_dones] = 0
            self.episode_powers_baseline[~self.last_dones] += infos["Power baseline"][
                ~self.last_dones
            ]

        # ----------------- Yaw travel accumulation -----------------
        # Expect infos[self.yaw_key] to be (num_envs, n_turbines) in degrees
        if self.yaw_key in infos:
            curr_yaws = np.asarray(infos[self.yaw_key])  # shape: (num_envs, n_turbines)

            # Reset per-episode yaw travel for envs that just finished last step
            self.episode_yaw_travel[self.last_dones] = 0

            if self.prev_yaws is None:
                # First time we see yaws
                self.prev_yaws = curr_yaws.copy()
            else:
                # Compute per-env total abs delta across turbines
                # Note: Only accumulate for ongoing episodes (~self.last_dones)
                current_dones = np.asarray(terminations) | np.asarray(truncations)
                # Current_dones are needed, as at reset, we go back to a random yaw angle, so this is to make sure we dont count that.
                delta = np.abs(curr_yaws - self.prev_yaws).sum(
                    axis=1
                )  # shape: (num_envs,)
                self.episode_yaw_travel[~current_dones] += delta[~current_dones]

                # Update previous yaws for next step
                self.prev_yaws = curr_yaws.copy()

        self.last_dones = self.prev_dones.copy()

        # ----------------- On episode end: push values to queues -----------------
        if np.any(self.prev_dones):
            done_idxs = np.where(self.prev_dones)[0]

            for i in done_idxs:
                # Mean powers
                self.mean_power_queue.append(
                    self.episode_powers[i] / max(1, self.episode_lengths[i])
                )
                if "Power agent nowake" in infos:
                    self.mean_power_queue_nowake.append(
                        self.episode_powers_nowake[i] / max(1, self.episode_lengths[i])
                    )
                if "Power baseline" in infos:
                    self.mean_power_queue_baseline.append(
                        self.episode_powers_baseline[i]
                        / max(1, self.episode_lengths[i])
                    )

                # Total yaw travel of the episode (sum over turbines & steps)
                # Units: degrees (assuming infos[yaw_key] is in degrees)
                if self.yaw_key in infos:
                    self.total_yaw_travel_queue.append(self.episode_yaw_travel[i])

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
