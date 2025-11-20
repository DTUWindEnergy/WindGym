import numpy as np
import gymnasium as gym
from WindGym.Agents import PyWakeAgent


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum wrapper for the WindGym environment.
    This wrapper adds a curriculum-based similarity reward between the agent's yaw
    vector and a reference ("good") yaw vector produced by a PyWakeAgent.
    yaw_check options:
        - 'current': use the current yaw angles of the agent
        - 'goal': use the yaw angles that would have been used, with no yaw step limits (only for wind actions)

    similarity_type options:
      - 'l2': negative L2 distance
      - 'l1': negative mean absolute error
      - 'mse': negative mean squared error
      - 'normalized_l2': 1 - (L2 distance / max_distance)
      - 'exponential': exp(-alpha * L2 distance)
      - 'cosine': cosine similarity
      - 'huber': negative Huber loss
    weight_function:
      function(step: int) -> float in [0,1], weighting env reward vs. similarity
      1 = env reward, 0 = similarity
    """

    def __init__(
        self,
        env: gym.Env,
        n_envs: int,
        similarity_type: str = "normalized_l2",
        yaw_check: str = "current",
        weight_function=lambda step: 1.0,
        huber_kappa: float = 1.0,
        exp_alpha: float = 1.0,
    ):
        super().__init__(env)

        self.similarity_type = similarity_type
        self.weight_function = weight_function
        self.huber_kappa = huber_kappa
        self.exp_alpha = exp_alpha
        self.n_envs = n_envs
        self.yaw_check = yaw_check
        # initialize PyWake agent
        self.pywake_agent = PyWakeAgent(
            x_pos=self.env.x_pos,
            y_pos=self.env.y_pos,
            turbine=self.env.turbine,
            env=env,
        )

        # maximum possible L2 distance: sqrt(N) * max_yaw_range
        self.n_turbines = len(self.env.x_pos)
        self.max_yaw_range = self.env.yaw_max - self.env.yaw_min
        self.max_l2 = np.sqrt(self.n_turbines) * self.max_yaw_range

        # state
        self.current_step = 0
        self.pywake_yaws = None

    def reset(self, **kwargs):
        """
        Reset the environment and the pywake agent.
        """
        obs, info = self.env.reset(**kwargs)
        # optimize reference yaw angles
        self.pywake_agent.update_wind(self.env.ws, self.env.wd, self.env.ti)
        self.pywake_agent.optimize()
        self.pywake_yaws = np.array(self.pywake_agent.optimized_yaws)

        # reset curriculum state
        info.update(
            {
                "pywake_yaws": self.pywake_yaws.copy(),
                "curriculum_weight": self.weight_function(self.current_step),
            }
        )
        return obs, info

    def step(self, action):
        """
        Take a step in the environment and calculate the reward based on the similarity between the yaw angles of the agent and the pywake agent.
        """
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        if self.yaw_check == "current":
            agent_yaws = info[
                "yaw angles agent"
            ]  # This is the current yaw of the agent
        elif self.yaw_check == "goal":
            agent_yaws = (action + 1.0) / 2.0 * (
                self.env.yaw_max - self.env.yaw_min
            ) + self.env.yaw_min  # Scales the action to the yaw range

        ref_yaws = self.pywake_yaws

        # compute yaw_diff metric depending on similarity_type
        diff = agent_yaws - ref_yaws

        # Calculate the similarity reward. This is the reward that is added to the base reward. It depends on the similarity between the yaw angles
        # if self.similarity_type== 'basic':
        #     yaw_diff = ((np.array(agent_yaws) - np.array(pywake_yaws)) ** 2).mean()
        #     similarity_reward = 1 / (1 + yaw_diff)
        # elif self.similarity_type == 'punish':
        #     THRESHOLD = 0.5  # tune this to control where punishment starts
        #     yaw_diff = ((np.array(agent_yaws) - np.array(pywake_yaws)) ** 2).mean()
        #     similarity_reward = np.minimum(1, 1 - ((max(0, yaw_diff - THRESHOLD)) ** 2))
        # else:
        #     raise(Exception('Bad similarity_type'))
        if self.similarity_type == "l2":
            similarity = -np.linalg.norm(diff)
        elif self.similarity_type == "l1":
            similarity = -np.mean(np.abs(diff))
        elif self.similarity_type == "mse":
            similarity = -np.mean(diff**2)
        elif self.similarity_type == "normalized_l2":
            similarity = 1 - (np.linalg.norm(diff) / self.max_l2)
        elif self.similarity_type == "exponential":
            similarity = float(np.exp(-self.exp_alpha * np.linalg.norm(diff)))
        elif self.similarity_type == "cosine":
            # avoid zero division
            denom = np.linalg.norm(agent_yaws) * np.linalg.norm(ref_yaws)
            similarity = (
                float(np.dot(agent_yaws, ref_yaws) / denom) if denom > 0 else 0.0
            )
        elif self.similarity_type == "huber":
            # Huber per turbine then mean
            abs_diff = np.abs(diff)
            quadratic = abs_diff <= self.huber_kappa
            loss = np.where(
                quadratic,
                0.5 * diff**2,
                self.huber_kappa * (abs_diff - 0.5 * self.huber_kappa),
            )
            similarity = -float(np.mean(loss))
        else:
            raise ValueError(f"Unknown similarity_type: {self.similarity_type}")

        # weight between 0 and 1
        env_w = float(np.clip(self.weight_function(self.current_step), 0.0, 1.0))
        new_reward = env_w * env_reward + (1 - env_w) * similarity

        # update info
        info.update(
            {
                "curriculum_weight": env_w,
                "similarity_reward": similarity,
                "current_step": self.current_step,
                "pywake_yaws": ref_yaws.copy(),
            }
        )

        # Because we use multiple envs, each step is not a single step, but a batch of steps
        self.current_step += self.n_envs
        return obs, new_reward, terminated, truncated, info
