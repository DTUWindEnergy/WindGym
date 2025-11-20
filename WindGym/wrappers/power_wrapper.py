import numpy as np
import gymnasium as gym
from WindGym.Agents import PyWakeAgent


class PowerWrapper(gym.Wrapper):
    """
    PowerWrapper wrapper for the WindGym environment.
    This wrapper adds a reward based on the power from PyWake.

    weight_function:
      function(step: int) -> float in [0,1], weighting env reward vs. similarity
      1 = env reward, 0 = similarity
    """

    def __init__(
        self,
        env: gym.Env,
        n_envs: int,
        weight_function=lambda step: 1.0,
    ):
        super().__init__(env)

        self.weight_function = weight_function
        self.n_envs = n_envs
        # initialize PyWake agent
        self.pywake_agent = PyWakeAgent(
            x_pos=self.env.x_pos,
            y_pos=self.env.y_pos,
            turbine=self.env.turbine,
            env=env,
        )

        # state
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment and the pywake agent.
        """
        obs, info = self.env.reset(**kwargs)
        # optimize reference yaw angles
        self.pywake_agent.update_wind(self.env.ws, self.env.wd, self.env.ti)

        # The baseline pywake power is with no yaw angles applied
        self.pywake_baseline_power = self.pywake_agent.calc_power(
            np.zeros_like(self.env.x_pos)
        )

        # reset curriculum state
        info.update(
            {
                "curriculum_weight": self.weight_function(self.current_step),
            }
        )
        return obs, info

    def step(self, action):
        """
        Take a step in the environment and calculate the reward based on the power from PyWake, and the normal WindGym reward.
        """
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        pywake_agent_power = self.pywake_agent.calc_power(self.env.current_yaw)

        if self.env.power_reward == "Baseline":
            wrapper_reward = pywake_agent_power / self.pywake_baseline_power - 1

        elif self.env.power_reward == "Power_avg":
            wrapper_reward = pywake_agent_power / self.env.n_turb / self.env.rated_power
        else:
            # TODO implement other power_reward types, or make this input to the deques somehow.
            raise ValueError(f"Unknown power_reward type: {self.env.power_reward}")

        # weight between 0 and 1
        env_w = float(np.clip(self.weight_function(self.current_step), 0.0, 1.0))
        new_reward = env_w * env_reward + (1 - env_w) * wrapper_reward

        # update info
        info.update(
            {
                "curriculum_weight": env_w,
                "wrapper_reward": wrapper_reward,
                "current_step": self.current_step,
            }
        )

        # Because we use multiple envs, each step is not a single step, but a batch of steps
        self.current_step += self.n_envs
        return obs, new_reward, terminated, truncated, info
