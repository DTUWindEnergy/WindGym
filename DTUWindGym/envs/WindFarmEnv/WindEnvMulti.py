from pettingzoo import ParallelEnv

import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from .Wind_Farm_Env import WindFarmEnv


class CustomEnvironment(ParallelEnv, WindFarmEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        
        self.n_turbines = 2
        self.yaw_initial = 0

    def reset(self, seed=None, options=None):

        self.agents = range(self.n_turbines)
        self.timestep = 0

        self.turbine_1_yaw = self.yaw_initial
        self.turbine_2_yaw = self.yaw_initial


        observations = {
            a: (
                np.array([self.turbine_1_yaw,
                self.turbine_2_yaw,
                ], dtype=np.float32)
            )
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos


    def step(self, actions):
        # Execute actions
        T1_action = actions[0]
        T2_action = actions[1]

        self.turbine_1_yaw += T1_action
        self.turbine_2_yaw += T2_action



        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}


        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            print("The timestep is over 100?")
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                np.array([self.turbine_1_yaw,
                self.turbine_2_yaw,
                ], dtype=np.float32)
            )
            for a in self.agents
        }




        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=-45.0, high=45.0, shape=(self.n_turbines,), dtype=np.float32)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)