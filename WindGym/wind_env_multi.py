from pettingzoo import ParallelEnv

import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Box
from .wind_farm_env import WindFarmEnv

"""
This is the multi agent version of the wind farm env. It just wraps the wind farm env and makes it behave in a way that pettingzoo can understand.
We use the parallel env from pettingzoo, as all agents act at the same time.
The main difference between this and the single agent version is that we have to unpack the actions and observations for each agent, and we have to make sure that the actions are in the right format for the wind farm env.
"""


class WindFarmEnvMulti(ParallelEnv, WindFarmEnv):
    metadata = {
        "name": "MultiFarm_environment_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        n_passthrough=20,
        ws_scaling_min: float = 0.0,
        ws_scaling_max: float = 30.0,
        wd_scaling_min: float = 0,
        wd_scaling_max: float = 360,
        ti_scaling_min: float = 0.0,
        ti_scaling_max: float = 1.0,
        yaw_scaling_min: float = -45,
        yaw_scaling_max: float = 45,
        TurbBox="Default",
        turbtype="MannGenerate",
        config=None,
        Baseline_comp=False,
        yaw_init=None,
        render_mode=None,
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step_sim=1,
        yaw_step_env=1,  # How many degrees the yaw angles can change pr. step
        fill_window=True,
        sample_site=None,
        HTC_path=None,
        reset_init=False,
        burn_in_passthroughs=2,
    ):
        self.n_turb = len(x_pos)  # n_turb needed before possible_agents
        self.possible_agents = ["turbine_" + str(r) for r in range(self.n_turb)]
        # a mapping between agent name and ID. Used to get the correct observations and infos.
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # call the init function of the parent class.
        WindFarmEnv.__init__(
            self,
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            n_passthrough=n_passthrough,
            ws_scaling_min=ws_scaling_min,
            ws_scaling_max=ws_scaling_max,
            wd_scaling_min=wd_scaling_min,
            wd_scaling_max=wd_scaling_max,
            ti_scaling_min=ti_scaling_min,
            ti_scaling_max=ti_scaling_max,
            yaw_scaling_min=yaw_scaling_min,
            yaw_scaling_max=yaw_scaling_max,
            TurbBox=TurbBox,
            turbtype=turbtype,
            config=config,
            Baseline_comp=Baseline_comp,
            yaw_init=yaw_init,
            render_mode=render_mode,
            seed=seed,
            dt_sim=dt_sim,
            dt_env=dt_env,
            yaw_step_sim=yaw_step_sim,
            yaw_step_env=yaw_step_env,
            fill_window=fill_window,
            sample_site=sample_site,
            HTC_path=HTC_path,
            reset_init=reset_init,
            burn_in_passthroughs=burn_in_passthroughs,
        )

        self.act_var = 1
        # Define the observation and action space
        # The obsevations pr turbine is:
        turbine_obs_var = self.farm_measurements.turb_mes[0].observed_variables()
        # The observations for the farm is:
        farm_obs_var = self.farm_measurements.farm_mes.observed_variables()
        # farm_obs_var = self.farm_measurements.farm_observed_variables
        # The observations for each agents is the number of observations for the turbine + the number of observations for the farm.
        self.obs_var = turbine_obs_var + farm_obs_var

        self.timestep = 0

    def render(self):
        return WindFarmEnv.render(self)

    def _get_obs_multi(self):
        """
        Concatenate the observations of the turbines and the farm, pr agent.
        Also clip it between -1 and 1.
        """
        if self.farm_measurements is None:
            # Environment has been truncated and cleaned up by parent, return zero-filled obs
            observations = {
                a: np.zeros(self.observation_space(a).shape, dtype=np.float32)
                for a in self.agents
            }
        else:
            observations = {
                a: (
                    np.clip(
                        np.concatenate(
                            [
                                turbine_mes.get_measurements(scaled=True),
                                self.farm_measurements.farm_mes.get_measurements(
                                    scaled=True
                                ),
                            ]
                        ),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                )
                for a, turbine_mes in zip(self.agents, self.farm_measurements.turb_mes)
            }
        return observations

    def _get_infos(self):
        """
        Return info dictionary.
        """

        infos = {}
        for a in self.agents:
            agent_idx = self.agent_name_mapping[a]
            if self.farm_measurements is None:
                # Environment has been truncated and cleaned up by parent, return empty info dict
                info_dict = {}
            else:
                info_dict = {
                    "yaw angles agent": self.current_yaw[agent_idx],
                    "yaw angles measured": self.farm_measurements.turb_mes[
                        agent_idx
                    ].get_yaw(),
                    "Wind speed Global": self.ws,
                    "Wind speed at turbine": self.current_ws[agent_idx],
                    "Wind speed at turbine measured": self.farm_measurements.turb_mes[
                        agent_idx
                    ].get_ws(),
                    "Wind speed at farm measured": self.farm_measurements.get_ws_farm(),
                    "Wind direction Global": self.wd,
                    "Wind direction at turbine": self.current_wd[agent_idx],
                    "Wind direction at turbine measured": self.farm_measurements.turb_mes[
                        agent_idx
                    ].get_wd(),
                    "Wind direction at farm measured": self.farm_measurements.get_wd_farm(),
                    "Turbulence intensity": self.ti,
                    "Power agent": self.fs.windTurbines.power().sum(),
                    "Power turbine agent": self.fs.windTurbines.power()[agent_idx],
                    "Turbine x positions": self.fs.windTurbines.positions_xyz[0][
                        agent_idx
                    ],
                    "Turbine y positions": self.fs.windTurbines.positions_xyz[1][
                        agent_idx
                    ],
                }
                if self.Baseline_comp:
                    info_dict["yaw angles base"] = self.fs_baseline.windTurbines.yaw[
                        agent_idx
                    ]  # Per turbine yaw
                    info_dict["Power baseline"] = (
                        self.fs_baseline.windTurbines.power().sum()
                    )  # Total farm power
                    info_dict["Power pr turbine baseline"] = (
                        self.fs_baseline.windTurbines.power()[agent_idx]
                    )  # Per turbine power
                    info_dict["Wind speed at turbines baseline"] = np.linalg.norm(
                        self.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
                    )[agent_idx]

            infos[a] = info_dict
        return infos

    def reset(self, seed=None, options=None):
        # Clear the agents list before calling parent reset,
        # as parent reset populates relevant internal state
        # and we then rebuild the agents list for the new episode.
        # This ensures that when _get_obs_multi/_get_infos are called below,
        # self.agents is correctly populated for the new episode.
        self.agents = copy(self.possible_agents)

        # call the reset function of the parent class.
        WindFarmEnv.reset(self, seed, options)

        # Then we unpack the observations and infos, and make them fit the parallel_env format.
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

        # The reward is a combination of the turbine powers, plus the farm power.
        # rewards = {
        #     a: np.mean(
        #         self.farm_measurements.turb_mes[
        #             self.agent_name_mapping[a]
        #         ].power.measurements
        #     )
        #     / self.rated_power
        #     + np.mean(self.farm_measurements.farm_mes.power.measurements)
        #     / self.rated_power
        #     for a in self.agents
        # }

        # This reward is simply the turbine power
        rewards = {
            a: self.fs.windTurbines.power().sum() / self.rated_power / self.n_turb
            for a in self.agents
        }

        return rewards

    def step(self, actions):
        """
        The step function.
        We unpack the actions, and call the step function of the parent class.
        """

        # Extract all actions
        all_action = np.array([yaw[0] for yaw in actions.values()])

        # Call the parent WindFarmEnv's step method
        parent_obs, parent_reward, parent_terminated, parent_truncated, parent_info = (
            WindFarmEnv.step(self, all_action)
        )

        # IMPORTANT: Check if the environment was truncated by the parent step call.
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

        # If it was, self.farm_measurements will be None.
        if parent_truncated:
            # Prepare outputs for a truncated state.
            # _get_obs_multi and _get_infos are designed to handle self.farm_measurements being None.
            observations = self._get_obs_multi()
            infos = self._get_infos()

            # For rewards, use the last reward from the parent step, as the episode is now ending.
            # You might want a specific reward for truncation here, but using parent_reward is a safe default.
            rewards = {
                a: parent_reward for a in self.possible_agents
            }  # Apply to all possible agents

            # Set truncation flags for all agents
            truncations = {a: True for a in self.possible_agents}
            terminations = {
                a: False for a in self.possible_agents
            }  # Truncation, not termination

            # Clear the list of active agents, as per PettingZoo convention for a finished episode.
            self.agents = []

            return observations, rewards, terminations, truncations, infos

        # If not truncated, proceed normally
        observations = self._get_obs_multi()
        infos = self._get_infos()

        # Rewards are the same for all agents in this environment.
        rewards = {a: parent_reward for a in self.agents}

        # Use the flags returned by the parent step
        truncations = {a: parent_truncated for a in self.agents}
        terminations = {a: parent_terminated for a in self.agents}

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
