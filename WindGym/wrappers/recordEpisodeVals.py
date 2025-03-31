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

        # The Queue to store the mean power of the episodes.
        self.mean_power_queue = deque(maxlen=buffer_length)
        self.episode_powers: np.ndarray = np.zeros(())  #
        self.last_dones: np.ndarray = np.zeros((), dtype=bool)

    def reset(self, seed: int | list[int] | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        self.episode_powers = np.zeros(self.num_envs)  #
        self.last_dones = self.prev_dones.copy()

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

        self.episode_powers[self.last_dones] = 0  # my code
        self.episode_powers[~self.last_dones] += infos["Power agent"][
            ~self.last_dones
        ]  # Add the power to the episode power

        self.last_dones = self.prev_dones.copy()

        #
        if np.sum(self.prev_dones):
            for i in np.where(self.prev_dones):
                # The mean power is the total power divided by the number of steps in the episode.
                self.mean_power_queue.extend(
                    self.episode_powers[i] / self.episode_lengths[i]
                )  #

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
