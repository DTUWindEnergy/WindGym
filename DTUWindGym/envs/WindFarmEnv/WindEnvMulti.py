from pettingzoo import ParallelEnv

import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Box
from .Wind_Farm_Env import WindFarmEnv
import gymnasium as gym


class WindFarmEnvMulti(ParallelEnv, WindFarmEnv):
    metadata = {
        "name": "MultiFarm_environment_v0",
    }

    def __init__(self, turbine, 
                 time_end = 5000,
                 TI_min_mes:float = 0.0, TI_max_mes:float = 0.50, 
                 TurbBox = "Default",
                 yaml_path = None,
                 Baseline_comp = False,  
                 yaw_init = None,
                 render_mode=None, seed = None,
                 ):
        #call the init function of the parent class.
        WindFarmEnv.__init__(self, turbine=turbine, 
                               time_end=time_end,
                               TI_min_mes=TI_min_mes, 
                               TI_max_mes=TI_max_mes, TurbBox=TurbBox, 
                               yaml_path=yaml_path, Baseline_comp=Baseline_comp, 
                               yaw_init=yaw_init, render_mode=render_mode, seed=seed)
        
        #I dont think the time limit wrapper works with the pettingzoo envs, so we just set the time limit here ourself. 
        # self.time_max = time_end #Maximum time of the simulation.

        #Then we change some minor things.
        self.act_var = 1
        #Define the observation and action space
        turbine_obs_var = self.farm_measurements.turb_mes[0].observed_variables() #number of observed variables for the turbine
        farm_obs_var = self.farm_measurements.farm_mes.observed_variables()
        self.obs_var = turbine_obs_var + farm_obs_var #number of observed variables for the agents


        self.timestep = 0
        self.possible_agents = ["turbine_" + str(r) for r in range(self.n_turb)]

        # a mapping between agent name and ID. Used to get the correct observations and infos.
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    def _get_obs_multi(self):
        """
        Concatenate the observations of the turbines and the farm, pr agent.
        Also clip it between -1 and 1.
        """
        observations = {
        a: (
            np.clip(np.concatenate([turbine_mes.get_measurements(scaled=True),
            self.farm_measurements.farm_mes.get_measurements(scaled=True),
            ]), -1.0, 1.0, dtype=np.float32)
        )
        for a, turbine_mes in zip(self.agents, self.farm_measurements.turb_mes)
        }

        return observations

    def _get_infos(self):
        """
        Return info dictionary. 
        """

        infos = {a: {"yaw angles agent": self.current_yaw[self.agent_name_mapping[a]],
                    "yaw angles measured": self.farm_measurements.turb_mes[self.agent_name_mapping[a]].get_yaw(),
                    "Wind speed Global": self.ws,
                    "Wind speed at turbine": self.current_ws[self.agent_name_mapping[a]],
                    "Wind speed at turbine measured": self.farm_measurements.turb_mes[self.agent_name_mapping[a]].get_ws(),
                    "Wind direction Global": self.wd,
                    "Wind direction at turbine": self.current_wd[self.agent_name_mapping[a]],
                    "Wind direction at turbine measured": self.farm_measurements.turb_mes[self.agent_name_mapping[a]].get_wd(),
                    "Wind direction at farm measured": self.farm_measurements.get_wd_farm(),
                    "Turbulence intensity": self.ti,
                    "Power agent": self.fs.windTurbines.power().sum(),
                    "Power turbine agent": self.fs.windTurbines.power()[self.agent_name_mapping[a]],
                    "Turbine x positions": self.fs.windTurbines.positions_xyz[0][self.agent_name_mapping[a]],
                    "Turbine y positions": self.fs.windTurbines.positions_xyz[1][self.agent_name_mapping[a]],
        } for a in self.agents}

        return infos

    def reset(self, seed=None, options=None):
        #call the reset function of the parent class.
        WindFarmEnv.reset(self, seed, options)

        #Then we unpack the observations and infos, and make them fit the parallel_env format.
        self.agents = copy(self.possible_agents) 
        self.timestep = 0

        # Get observations and infos
        observations = self._get_obs_multi()
        infos = self._get_infos()

        return observations, infos

    def _calc_reward(self):
        """
        Calculate the reward. 
        TODO think about this function.
        For now the reward is the power of the turbines.
        """

        #The reward is a combination of the turbine powers, plus the farm power.
        rewards = {a: np.mean(self.farm_measurements.turb_mes[self.agent_name_mapping[a]].power.measurements) / self.rated_power + np.mean(self.farm_measurements.farm_mes.power.measurements) / self.rated_power           
                   for a in self.agents}

        #This reward is simply the turbine power
        # rewards = {a: self.fs.windTurbines.power()[self.agent_name_mapping[a]] for a in self.agents}

        #This reward is the farm power
        # rewards = {a: np.mean(self.farm_measurements.farm_mes.power.measurements) / self.rated_power for a in self.agents}

        return rewards

    def step(self, actions):
        """
        The step function. 
        We unpack the actions, and call the step function of the parent class.
        """
        #Extract all actions
        all_action = np.array([yaw[0] for yaw in actions.values()])
        
        observation, reward, terminated, truncated, info = WindFarmEnv.step(self, all_action)

        
        # Get observations rewards and infos.
        observations = self._get_obs_multi() 
        rewards = self._calc_reward()
        infos = self._get_infos()

        #If we are at the end of the simulation, we truncate the agents.
        # Note that this is not the same as terminating the agents. 
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        if truncated:
            # terminated = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
        else:
            truncations = {a: False for a in self.agents}

        self.timestep += 1

        #We are never at some end point in wind farm simulations, so we always return False for terminated.
        terminations = {a: False for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _init_spaces(self):
        """
        This is done as to remove that functionality from the parent class.
        """
        pass

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=-1.0, high=1.0, shape=(self.obs_var,), dtype=np.float32)
        # return self._observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(self.act_var,), dtype=np.float32)
        # return self._action_spaces[agent]