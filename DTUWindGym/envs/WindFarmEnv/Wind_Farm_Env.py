from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os
import gc

# Dynamiks imports
from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField, RandomTurbulence
from dynamiks.wind_turbines import PyWakeWindTurbines
from dynamiks.views import XYView

from IPython import display

# DTUWindGym imports
from .WindEnv import WindEnv
from .MesClass import farm_mes
from .BasicControllers import local_yaw_controller, global_yaw_controller

from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from collections import deque
import itertools
import yaml

"""
This is the base for the wind farm environment. This is where the magic happens.
For now it only supports the PyWakeWindTurbines, but it should be easy to expand to other types of turbines.
"""
# TODO So some sources says that the rewards should be in the order of 10. Therefore we could try and do that.
# TODO Assert that the turbine is a subclass of the PyWakeWindTurbines
# TODO make it so that the turbines can be other then a square grid
# TODO user defined observed variables
# TODO thrust coefficient control
# TODO for now I have just hardcoded this scaling value (1 and 25 for the wind_speed min and max). This is beacuse the wind speed is chosen from the normal distribution, but becasue of the wakes and the turbulence, we canhave cases where we go above or below these values.
# TODO IMPORTANT: We need to make sure that the turbines are consistent with the returning of variables. So that independent of the self.wd
# TODO maybe we should take larger steps in the flow simulation. Or multiple. This coulde be done with fs.run() instead.


class WindFarmEnv(WindEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        turbine,
        n_passthrough=5,
        TI_min_mes: float = 0.0,
        TI_max_mes: float = 0.50,
        TurbBox="Default",
        turbtype="MannLoad",
        yaml_path=None,
        Baseline_comp=False,
        yaw_init=None,
        render_mode=None,
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step=1,  # How many degrees the yaw angles can change pr. step
        power_avg=1,
    ):
        """
        This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
        Args:
            turbine: PyWakeWindTurbines: The wind turbine that is used in the environment
            n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
            TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
            TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
            TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
            turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
            yaml_path: str: The path to the yaml file that contains the configuration of the environment. TODO make a default value for this?
            Baseline_comp: bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
            yaw_init: str: The method for initializing the yaw angles of the turbines. If 'Random', then the yaw angles will be random. Else they will be zeros.
            render_mode: str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
            seed: int: The seed for the environment. If None, then the seed will be random.
            dt_sim: float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
            dt_env: float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
            yaw_step: float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
        """

        # Predefined values
        # The power setpoint for the farm. This is used if the Track_power is True. (Not used yet)
        self.power_avg = power_avg
        self.power_setpoint = 0.0
        self.act_var = (
            1  # number of actions pr. turbine. For now it is just the yaw angles
        )
        self.dt = dt_sim  # DWM simulation timestep
        self.dt_sim = dt_sim
        self.dt_env = dt_env  # Environment timestep
        self.sim_steps_per_env_step = int(self.dt_env / self.dt_sim)
        if self.dt_env % self.dt_sim != 0:
            raise ValueError("dt_env must be a multiple of dt_sim")

        self.yaw_start = 15.0  # This is the limit for the initialization of the yaw angles. This is used to make sure that the yaw angles are not too large at the start, but still not zero
        # Max power pr turbine. Used in the measurement class
        self.maxturbpower = max(turbine.power(np.arange(10, 25, 1)))
        # The step size for the yaw angles. How manny degress the yaw angles can change pr. step
        self.yaw_step = yaw_step
        # The distance between the particles. This is used in the flow simulation.
        self.d_particle = 0.1

        self.turbtype = turbtype

        # Saves to self
        self.TI_min_mes = TI_min_mes
        self.TI_max_mes = TI_max_mes
        self.seed = seed
        self.TurbBox = TurbBox
        self.turbine = turbine
        # The maximum time of the simulation. This is used to make sure that the simulation doesnt run forever.
        self.time_max = 0
        # The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
        self.n_passthrough = n_passthrough
        self.timestep = 0

        self.TF_files = []
        # The initial yaw of the turbines. This is used if the yaw_init is "Defined"
        self.yaw_initial = [0]

        # Load the configuration
        self.load_config(yaml_path)

        self.n_turb = self.nx * self.ny  # The number of turbines

        # Deques that holds the power output of the farm and the baseline farm. This is used for the power reward
        self.farm_pow_deq = deque(maxlen=self.power_avg)
        self.base_pow_deq = deque(maxlen=self.power_avg)
        self.power_len = self.power_avg

        # Sets the yaw init method. If Random, then the yaw angles will be random. Else they will be zeros
        # If yaw_init is defined (it will be if we initialize from EnvEval) then set it like this. Else just use the value from the yaml
        if yaw_init is not None:
            # We only ever have this, IF we have set the value from
            if yaw_init == "Random":
                self._yaw_init = self._randoms_uniform
            elif yaw_init == "Defined":
                self._yaw_init = self._defined_yaw
            else:
                self._yaw_init = self._return_zeros
        else:
            if self.yaw_init == "Random":
                self._yaw_init = self._randoms_uniform
            elif self.yaw_init == "Defined":
                self._yaw_init = self._defined_yaw
            else:
                self._yaw_init = self._return_zeros

        # Define the power tracking reward function TODO Not implemented yet. Also make the power_setpoint an observable parameter
        if self.Track_power:
            self.power_setpoint = 42  # ???
            self._track_rew = self.track_rew_avg
            raise NotImplementedError("The Track_power is not implemented yet")
        else:
            self._track_rew = self.track_rew_none

        # Define the power production reward function
        if self.power_reward == "Baseline":
            self._power_rew = (
                self.power_rew_baseline
            )  # The baseline power reward function
        elif self.power_reward == "Power_avg":
            self._power_rew = self.power_rew_avg  # The power_avg reward function
        elif self.power_reward == "None":
            self._power_rew = self.power_rew_none  # The no power reward function
        elif self.power_reward == "Power_diff":
            # TODO rethink this way of doing it.
            self._power_rew = self.power_rew_diff  # The power_diff reward function
            # We set this to 10, to have some space in the middle.
            self._power_wSize = self.power_avg // 10
            if self.power_avg < 40:
                # Why 40? I just chose this as the minimum value. In reality 2 could have sufficed, but to save myself a headache, I set it to 10
                raise ValueError(
                    "The Power_avg must be larger then 40 for the Power_diff reward. Also it should probably be way larger my guy"
                )
        else:
            raise ValueError(
                "The Power_reward must be either Baseline, Power_avg, None or Power_diff"
            )

        # Read in the turb boxes
        if turbtype == "MannLoad":
            if os.path.exists(TurbBox) and os.path.isfile(TurbBox):
                # The TurbBox is a file, so we just add this to the list of files
                self.TF_files.append(TurbBox)
            else:
                # If the path exist, but is not a file, then we must be a directory
                # Therefore add all the files in the dir, to the list.
                try:
                    for f in os.listdir(TurbBox):
                        if f.split("_")[0] == "TF":
                            self.TF_files.append(os.path.join(TurbBox, f))
                except FileNotFoundError:
                    # If not then we change to generated turbulence
                    print(
                        "Coudnt find the turbulence box file(s), so we switch to generated turbulence"
                    )
                    self.turbtype = "MannGenerate"

        # If we need to have a "baseline" farm, then we need to set up the baseline controller
        # This could be moved to the Power_reward check, but I have a feeling this will be expanded in the future, when we include damage.
        if self.power_reward == "Baseline" or Baseline_comp:
            self.Baseline_comp = True
        else:
            self.Baseline_comp = False

        # #Initializing the measurements class with the specified values.
        self._init_farm_mes()

        # The maximum history length of the measurements
        self.hist_max = self.farm_measurements.max_hist()

        # Setting up the turbines:

        D = turbine.diameter()

        x = np.linspace(0, D * self.xDist * self.nx, self.nx)
        y = np.linspace(0, D * self.yDist * self.ny, self.ny)

        xv, yv = np.meshgrid(x, y, indexing="xy")

        self.x_pos = xv.flatten()
        self.y_pos = yv.flatten()
        self.y_pos += 200  # Note we move the farm 200 units up. This is done because I think I saw some weird behaviour with y = 0 :/

        self.wts = PyWakeWindTurbines(
            x=self.x_pos,
            y=self.y_pos,  # x and y position of two wind turbines
            windTurbine=self.turbine,
        )

        # Setting up the baseline controller if we need it
        if self.Baseline_comp:
            # If we compare to some baseline performance, then we also need a controller for that
            if self.BaseController == "Local":
                self._base_controller = local_yaw_controller
            elif self.BaseController == "Global":
                self._base_controller = global_yaw_controller
            else:
                raise ValueError(
                    "The BaseController must be either Local or Global... For now"
                )
            # Definde the turbines
            self.wts_baseline = PyWakeWindTurbines(
                x=self.x_pos,
                y=self.y_pos,  # x and y position of two wind turbines
                windTurbine=self.turbine,
            )

        # Define the observation and action space
        self.obs_var = self.farm_measurements.observed_variables()

        self._init_spaces()

        # We should have this here, to set the seeding correctly
        self.reset(seed=seed)

        # TODO the render mode is not implemented yet. I think?
        # Asserting that the render_mode is valid.
        # If render_mode is None, we will not render anything, if it is human, we will render the environment in a window, if it is rgb_array, we will render the environment as an array
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.reset()
            self.init_render()

    def load_config(self, config_path):
        """
        This loads in the yaml file, and sets a bunch of internal values.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)  # Load the YAML file

        # Set the attributes of the class based on the config file
        self.yaw_init = config.get("yaw_init")
        self.noise = config.get("noise")
        self.BaseController = config.get("BaseController")
        self.ActionMethod = config.get("ActionMethod")
        # self.Baseline_comp = config.get('Baseline_comp')
        self.Track_power = config.get("Track_power")

        # Unpack the farm params
        farm_params = config.get("farm")
        self.yaw_min = farm_params["yaw_min"]
        self.yaw_max = farm_params["yaw_max"]
        self.xDist = farm_params["xDist"]
        self.yDist = farm_params["yDist"]
        self.nx = farm_params["nx"]
        self.ny = farm_params["ny"]

        # Unpack the wind params
        wind_params = config.get("wind")
        self.ws_min = wind_params["ws_min"]
        self.ws_max = wind_params["ws_max"]
        self.TI_min = wind_params["TI_min"]
        self.TI_max = wind_params["TI_max"]
        self.wd_min = wind_params["wd_min"]
        self.wd_max = wind_params["wd_max"]

        self.act_pen = config.get("act_pen")
        self.power_def = config.get("power_def")
        self.mes_level = config.get("mes_level")
        self.ws_mes = config.get("ws_mes")
        self.wd_mes = config.get("wd_mes")
        self.yaw_mes = config.get("yaw_mes")
        self.power_mes = config.get("power_mes")

        # unpack some more, because we use these later.
        self.action_penalty = self.act_pen["action_penalty"]
        self.action_penalty_type = self.act_pen["action_penalty_type"]
        self.Power_scaling = self.power_def["Power_scaling"]
        # self.power_avg = self.power_def["Power_avg"]
        self.power_reward = self.power_def["Power_reward"]

    def _init_farm_mes(self):
        """
        This function initializes the farm measurements class.
        This id done partly due to modularity, but also because we can delete it from memory later, as I suspect this might be the source of the memory leak
        """
        # Initializing the measurements class with the specified values.
        # TODO if history_length is 1, then we dont need to save the history, and we can just use the current values.
        # TODO is history_N is 1 or larger, then it is kinda implied that the rolling_mean is true.. Therefore we can change the if self.rolling_mean: check in the Mes() class, to be a if self.history_N >= 1 check... or something like that
        self.farm_measurements = farm_mes(
            self.n_turb,
            self.noise,
            self.mes_level["turb_ws"],
            self.mes_level["turb_wd"],
            self.mes_level["turb_TI"],
            self.mes_level["turb_power"],
            self.mes_level["farm_ws"],
            self.mes_level["farm_wd"],
            self.mes_level["farm_TI"],
            self.mes_level["farm_power"],
            self.ws_mes["ws_current"],
            self.ws_mes["ws_rolling_mean"],
            self.ws_mes["ws_history_N"],
            self.ws_mes["ws_history_length"],
            self.ws_mes["ws_window_length"],
            self.wd_mes["wd_current"],
            self.wd_mes["wd_rolling_mean"],
            self.wd_mes["wd_history_N"],
            self.wd_mes["wd_history_length"],
            self.wd_mes["wd_window_length"],
            self.yaw_mes["yaw_current"],
            self.yaw_mes["yaw_rolling_mean"],
            self.yaw_mes["yaw_history_N"],
            self.yaw_mes["yaw_history_length"],
            self.yaw_mes["yaw_window_length"],
            self.power_mes["power_current"],
            self.power_mes["power_rolling_mean"],
            self.power_mes["power_history_N"],
            self.power_mes["power_history_length"],
            self.power_mes["power_window_length"],
            2.0,
            25.0,  # Max and min values for wind speed measuremenats
            # Max and min values for wind direction measurements   NOTE i have added 5 for some slack in the measurements. so the scaling is better.
            self.wd_min - 5,
            self.wd_max + 5,
            self.yaw_min,
            self.yaw_max,  # Max and min values for yaw measurements
            # Max and min values for the turbulence intensity measurements
            self.TI_min_mes,
            self.TI_max_mes,
            power_max=self.maxturbpower,
        )

    def _init_spaces(self):
        """
        This function initializes the observation and action spaces.
        This is done in a seperate function, so we can replace it in the multi agent version of the environment
        """
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=((self.obs_var),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=((self.n_turb * self.act_var),), dtype=np.float32
        )

    def init_render(self):
        plt.ion()

        x_turb, y_turb = self.fs.windTurbines.positions_xyz[:2]

        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.a = np.linspace(-200 + min(x_turb), 1000 + max(x_turb), 250)
        self.b = np.linspace(-200 + min(y_turb), 200 + max(y_turb), 250)

        self.view = XYView(
            z=self.turbine.hub_height(), x=self.a, y=self.b, ax=self.ax, adaptive=False
        )

        plt.close()

    def _update_measurements(self):
        """
        This function adds the current observations to the farm_measurements class
        """

        # Get the observation of the environment
        self.current_ws = np.linalg.norm(
            self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0
        )
        # TODO make sure the implementation is correct
        # self.current_ws = np.linalg.norm(self.fs.windTurbines.rotor_avg_windspeed, axis=1)
        # The current ws is the norm of the three components
        # The current wd is the invtan of the u/v components of the wind speed. Remember to add the "global" wind direction to this measurement, as we are rotating the farm
        u_speed = self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True)[1]
        # u_speed = self.fs.windTurbines.rotor_avg_windspeed[:,1]
        v_speed = self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True)[0]
        # v_speed = self.fs.windTurbines.rotor_avg_windspeed[:,0]
        self.current_wd = np.rad2deg(np.arctan(u_speed / v_speed)) + self.wd

        self.current_yaw = self.fs.windTurbines.yaw
        powers = self.fs.windTurbines.power()  # The Power pr turbine

        self.farm_measurements.add_measurements(
            self.current_ws, self.current_wd, self.current_yaw, powers
        )

    def _get_obs(self):
        """
        Gets the sensordata from the farm_measurements class, and scales it to be between -1 and 1
        If you want to implement your own handling of the observations, then you can do that here by overwriting this function
        """

        values = self.farm_measurements.get_measurements(scaled=True)
        return np.clip(values, -1.0, 1.0, dtype=np.float32)

    def _get_info(self):
        """
        Return info dictionary.
        If we have a baseline comparison, then we also return the baseline values.
        """
        return_dict = {
            "yaw angles agent": self.current_yaw,
            "yaw angles measured": self.farm_measurements.get_yaw_turb(),
            "Wind speed Global": self.ws,
            "Wind speed at turbines": self.current_ws,
            "Wind speed at turbines measured": self.farm_measurements.get_ws_turb(),
            "Wind speed at farm measured": self.farm_measurements.get_ws_farm(),
            "Wind direction Global": self.wd,
            "Wind direction at turbines": self.current_wd,
            "Wind direction at turbines measured": self.farm_measurements.get_wd_turb(),
            "Wind direction at farm measured": self.farm_measurements.get_wd_farm(),
            "Turbulence intensity": self.ti,
            "Power agent": self.fs.windTurbines.power().sum(),
            "Power pr turbine agent": self.fs.windTurbines.power(),
            "Turbine x positions": self.fs.windTurbines.positions_xyz[0],
            "Turbine y positions": self.fs.windTurbines.positions_xyz[1],
        }

        if self.Baseline_comp:
            return_dict["yaw angles base"] = self.fs_baseline.windTurbines.yaw
            return_dict["Power baseline"] = self.fs_baseline.windTurbines.power().sum()
            return_dict["Power pr turbine baseline"] = (
                self.fs_baseline.windTurbines.power()
            )
            # return_dict["Wind speed at turbines baseline"] = self.fs_baseline.windTurbines.rotor_avg_windspeed[:,0] #Just the largest component
            return_dict["Wind speed at turbines baseline"] = (
                self.fs_baseline.windTurbines.rotor_avg_windspeed(
                    include_wakes=True
                )[0, :]
            )  # Just the largest component
        return return_dict

    def _set_windconditions(self):
        """
        Sets the global windconditions for the environment
        """

        # The wind speed is a random number between ws_min and ws_max
        self.ws = self._random_uniform(self.ws_min, self.ws_max)
        # The turbulence intensity is a random number between TI_min and TI_max
        self.ti = self._random_uniform(self.TI_min, self.TI_max)
        # The wind direction is a random number between wd_min and wd_max
        self.wd = self._random_uniform(self.wd_min, self.wd_max)

    def _def_site(self):
        """
          We choose a random turbulence box and scale it to the correct TI and wind speed.
        This is repeated for the baseline if we have that.

        The turbulence box used for the simulation can be one of the following:
        - MannLoad: The turbulence box is loaded from predefined Mann turbulence box files.
        - MannGenerate: A random turbulence box is generated.
        - MannFixed: A fixed turbulence box is used with a constant seed.
        - Random: Specifies the 'box' as random turbulence.
        - None: Zero turbulence site.
        """

        if self.turbtype == "MannLoad":
            # Load the turbbox from predefined folder somewhere
            # selects one at random
            tf_file = self.np_random.choice(self.TF_files)
            # print("Loading Mann turbulence box: ", tf_file)

            tf_agent = MannTurbulenceField.from_netcdf(filename=tf_file)
            tf_agent.scale_TI(ti=self.ti, U=self.ws)

        elif self.turbtype == "MannGenerate":
            # Create the turbbox with a random seed.
            # TODO this can be improved in the future.
            TF_seed = self.np_random.integers(0, 100000)
            # print("Generating Mann turbulence box with seed: ", TF_seed)
            tf_agent = MannTurbulenceField.generate(
                alphaepsilon=0.1,  # use correct alphaepsilon or scale later
                L=33.6,  # length scale
                Gamma=3.9,  # anisotropy parameter
                # numbers should be even and should be large enough to cover whole farm in all dimensions and time, see above
                Nxyz=(8192, 512, 64),
                # should be small enough to capture variations needed for the wind the turbine model
                dxyz=(3.0, 3.0, 3.0),
                seed=TF_seed,  # seed for random generator
                # HighFreqComp=0, # the high frequency compensation is questionable and it is recommened to switch it off
                # double_xyz=(False, False, False), # turbulence periodicity is not expected to be an issue in a wind farm
            )
            tf_agent.scale_TI(ti=self.ti, U=self.ws)

        elif self.turbtype == "Random":
            # Specifies the 'box' as random turbulence
            raise NotImplementedError(
                "This turbulence type doenst work with the current dynamiks version"
            )
            TF_seed = self.np_random.integers(0, 100000)
            # print("Using Random turbulence with seed:", TF_seed)
            tf_agent = RandomTurbulence(ti=self.ti, ws=self.ws, seed=TF_seed)

        elif self.turbtype == "MannFixed":
            # print("Using fixed Mann turbulence box")
            # Generates a fixed mann box
            TF_seed = 1234  # Hardcoded for now
            tf_agent = MannTurbulenceField.generate(
                alphaepsilon=0.1,  # use correct alphaepsilon or scale later
                L=33.6,  # length scale
                Gamma=3.9,  # anisotropy parameter
                # numbers should be even and should be large enough to cover whole farm in all dimensions and time, see above
                Nxyz=(8192, 512, 64),
                # should be small enough to capture variations needed for the wind the turbine model
                dxyz=(3.0, 3.0, 3.0),
                seed=TF_seed,  # seed for random generator
            )
            tf_agent.scale_TI(ti=self.ti, U=self.ws)

        elif self.turbtype == "None":
            # Zero turbulence site.
            raise NotImplementedError(
                "This turbulence type doenst work with the current dynamiks version"
            )
            tf_agent = RandomTurbulence(ti=0, ws=self.ws)
        else:
            # Throw and error:
            raise ValueError("Invalid turbulence type specified")

        self.site = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_agent)

        if self.Baseline_comp:  # I am pretty sure we need to have 2 sites, as the flow simulation is run on the site, and the measurements are taken from the site.
            tf_base = copy.deepcopy(tf_agent)
            self.site_base = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_base)
            del tf_base
        tf_agent = None
        del tf_agent
        gc.collect()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. This is called at the start of every episode.
        - The wind conditions are sampled, and the site is set.
        - The flow simulation is run for the time it takes for the flow to develop.
        - The measurements are filled up with the initial values.

        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.timestep = 0
        # Sample global wind conditions and set the site
        self._set_windconditions()
        self._def_site()
        # Restart the measurement class. This is done to make sure that the measurements are not carried over from the last episode
        self._init_farm_mes()

        # This is the rated poweroutput of the turbine at the given ws. Used for reward scaling.
        self.rated_power = self.turbine.power(self.ws)

        self.fs = DWMFlowSimulation(
            site=self.site,
            windTurbines=self.wts,
            wind_direction=self.wd,
            particleDeficitGenerator=jDWMAinslieGenerator(),
            dt=self.dt,
            d_particle=self.d_particle,
            particleMotionModel=HillVortexParticleMotion(),
        )  # NOTE, we need this particlemotion to capture the yaw

        # Set the yaw angles of the farm
        # NOTE that I use yaw_start and not yaw_min/yaw_max. This is to make sure that the yaw angles are not too large at the start, but still not zero
        self.fs.windTurbines.yaw = self._yaw_init(
            min_val=-self.yaw_start,
            max_val=self.yaw_start,
            n=self.n_turb,
            yaws=self.yaw_initial,
        )

        # Calulate the time it takes for the flow to develop.
        turb_xpos = self.fs.windTurbines.rotor_positions_xyz[0, :]
        dist = turb_xpos.max() - turb_xpos.min()
        # turb_place = np.linalg.norm(self.fs.windTurbines.positions_xyz, axis=0)
        # dist = turb_place.max() - turb_place.min()

        # Time it takes for the flow to travel from one side of the farm to the other
        t_inflow = dist / self.ws
        # The time it takes for the flow to develop. Also a bit extra.
        t_developed = int(t_inflow * 3)

        # Max allowed timesteps
        self.time_max = int(t_inflow * self.n_passthrough)
        # first we run the simulation the time it takes the flow to develop
        self.fs.run(t_developed)

        # Just take one step
        self.fs.step()  # Take a step in the flow simulation
        # Save the power output of the farm
        self.farm_pow_deq.append(self.fs.windTurbines.power().sum())
        self._update_measurements()

        # Do the same for the baseline farm
        if self.Baseline_comp:
            self.fs_baseline = DWMFlowSimulation(
                site=self.site_base,
                windTurbines=self.wts_baseline,
                wind_direction=self.wd,
                particleDeficitGenerator=jDWMAinslieGenerator(),
                dt=self.dt,
                d_particle=self.d_particle,
                particleMotionModel=HillVortexParticleMotion(),
            )

            self.fs_baseline.windTurbines.yaw = self.fs.windTurbines.yaw
            self.fs_baseline.run(t_developed)

            self.fs_baseline.step()  # Take a step in the baseline flow simulation
            self.base_pow_deq.append(self.fs_baseline.windTurbines.power().sum())

        # Now we can start

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _action_penalty(self):
        """
        This function calculates a penalty for the actions. This is used to penalize the agent for taking actions, and try and make it more stable
        """
        if (
            self.action_penalty < 0.001
        ):  # If the penalty is very small, then we dont need to calculate it
            return 0

        elif self.action_penalty_type == "Change":
            # The penalty is dependent on the change in values
            pen_val = np.mean(np.abs(self.old_yaws - self.fs.windTurbines.yaw))
        elif self.action_penalty_type == "Total":
            # The penalty is dependent on the total values
            pen_val = np.mean(np.abs(self.fs.windTurbines.yaw)) / self.yaw_max

        return self.action_penalty * pen_val

    def _adjust_yaws(self, action):
        """
        Heavily inspired from https://github.com/AlgTUDelft/wind-farm-env
        This function adjusts the yaw angles of the turbines, based on the actions given, but we now have differnt methods for the actions
        """

        if self.ActionMethod == "yaw":
            # The new yaw angles are the old yaw angles + the action, scaled with the yaw_step
            # 0 action means no change
            # the new yaw angles are the old yaw angles + the action, scaled with the yaw_step
            self.fs.windTurbines.yaw += action * self.yaw_step
            # clip the yaw angles to be between -30 and 30
            self.fs.windTurbines.yaw = np.clip(
                self.fs.windTurbines.yaw, self.yaw_min, self.yaw_max
            )

        elif self.ActionMethod == "wind":
            # The new yaw angles are the action, scaled to be between the min and max yaw angles
            # 0 action means to move to 0 yaw angle, and 1 action means to move to the max yaw angle
            new_yaws = (action + 1.0) / 2.0 * (
                self.yaw_max - self.yaw_min
            ) + self.yaw_min

            # The bounds for the yaw angles are:
            yaw_max = self.fs.windTurbines.yaw + self.yaw_step
            yaw_min = self.fs.windTurbines.yaw - self.yaw_step

            # The new yaw angles are the new yaw angles, but clipped to be between the yaw_max and yaw_min
            self.fs.windTurbines.yaw = np.clip(
                np.clip(new_yaws, yaw_min, yaw_max), self.yaw_min, self.yaw_max
            )

        elif self.ActionMethod == "absolute":
            raise NotImplementedError("The absolute method is not implemented yet")

        else:
            raise ValueError("The ActionMethod must be yaw, wind or absolute")

    def track_rew_none(self):
        """If we are not using power tracking, then just return 0"""
        return 0.0

    def track_rew_avg(self):
        """
        The reward is the negative difference between the power output and the power setpoint squared
        The reward is: - (power_agent - power_setpoint)^2
        """
        power_agent = np.mean(self.farm_pow_deq)
        return -((power_agent - self.power_setpoint) ** 2)

    def power_rew_baseline(self):
        """Calculate reward based on baseline farm comparison using available history"""
        power_agent = self.fs.windTurbines.power().sum()
        power_baseline = self.fs_baseline.windTurbines.power().sum()

        # Add to histories
        self.farm_pow_deq.append(power_agent)
        self.base_pow_deq.append(power_baseline)

        # Use whatever history we have so far for averaging
        power_agent_avg = np.mean(self.farm_pow_deq)
        power_baseline_avg = np.mean(self.base_pow_deq)

        if power_baseline_avg == 0:
            print("The baseline power is zero. This is probably not good")
            print("self.farm_pow_deq: ", self.farm_pow_deq)
            print("self.base_pow_deq: ", self.base_pow_deq)
            0 / 0  # This will raise an error

        reward = power_agent_avg / power_baseline_avg - 1
        return reward

    def power_rew_avg(self):
        """Calculate power reward based on available history"""
        power_agent = np.mean(self.farm_pow_deq)
        reward = power_agent / self.n_turb / self.rated_power
        return reward

    def power_rew_none(self):
        """Return zero for the power reward"""
        return 0.0

    def power_rew_diff(self):
        """Calculate reward based on power difference over time"""
        power_latest = np.mean(
            list(
                itertools.islice(
                    self.farm_pow_deq,
                    self.power_len - self._power_wSize,
                    self.power_len,
                )
            )
        )
        power_oldest = np.mean(
            list(itertools.islice(self.farm_pow_deq, 0, self._power_wSize))
        )
        return (power_latest - power_oldest) / self.n_turb

    def step(self, action):
        """
        The step function
        1. Adjust the yaw angles of the turbines
        2. Take a step in the flow simulation
        3. Update the measurements
        4. Calculate the reward
        5. Return the observation, reward, terminated, truncated and info

        """

        # Save the old yaw angles, so we can calculate the change in yaw angles
        self.old_yaws = copy.copy(self.fs.windTurbines.yaw)

        self._adjust_yaws(action)  # Adjust the yaw angles of the agent farm
        # Run multiple simulation steps for each environment step

        # Initialize list to store observations
        observations = []
        powers = []
        for _ in range(self.sim_steps_per_env_step):
            # Step the flow simulation
            self.fs.step()

            # If we have baseline comparison, step it too
            if self.Baseline_comp:
                new_baseline_yaws = self._base_controller(
                    fs=self.fs_baseline, yaw_step=self.yaw_step
                )
                self.fs_baseline.windTurbines.yaw = new_baseline_yaws
                self.fs_baseline.step()

            self._update_measurements()
            observations.append(self._get_obs())
            powers.append(self.fs.windTurbines.power().sum())

        if self.Baseline_comp:
            self.base_pow_deq.append(self.fs_baseline.windTurbines.power().sum())
        if np.any(np.isnan(self.farm_pow_deq)):
            raise Exception("NaN Power")

        # Average observations and power over simulation steps
        observation = np.mean(observations, axis=0)
        self.farm_pow_deq.append(np.mean(powers))
        info = self._get_info()
        # Save the power output of the farm
        # self.farm_pow_deq.append(self.fs.windTurbines.power().sum())

        # Calculate the reward
        # The power production reward with the scaling
        power_rew = self._power_rew() * self.Power_scaling
        # track_rew = self._track_rew()  #The power tracking reward. This is just a placeholder so far.
        track_rew = 0.0

        action_penalty = self._action_penalty()  # The penalty for the actions

        # The reward is: power reward - action penalty. This makes it possible to add a reward for power tracking, and/or damage, easily.
        # The reward is the power reward minus the action penalty
        reward = power_rew + track_rew - action_penalty

        # If we are at the end of the simulation, we truncate the agents.
        # Note that this is not the same as terminating the agents.
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        if self.timestep >= self.time_max:
            # terminated = {a: True for a in self.agents}
            truncated = True
            # Clean up the flow simulation. This is to make sure that we dont have a memory leak.
            if self.Baseline_comp:
                self.fs_baseline = None
                self.site_base = None
                del self.fs_baseline
                del self.site_base
            self.fs = None
            self.site = None
            self.farm_measurements = None
            del self.fs
            del self.site
            del self.farm_measurements
            gc.collect()
        else:
            truncated = False

        self.timestep += 1

        terminated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        This is the rendering function.
        It renders the flow field and the wind turbines
        Can be much improved, but it is a start
        """

        plt.ion()
        ax1 = plt.gca()

        # uvw = self.fs.get_windspeed(self.view, include_wakes=True, xarray=False)
        uvw = self.fs.get_windspeed(self.view, include_wakes=True, xarray=True)

        wt = self.fs.windTurbines
        x_turb, y_turb = self.fs.windTurbines.positions_xyz[:2]
        yaw, tilt = wt.yaw_tilt()

        # [0] is the u component of the wind speed
        plt.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest")
        # plt.colorbar().set_label('Wind speed, u [m/s]')
        WindTurbinesPW.plot_xy(
            self.fs.windTurbines,
            x_turb,
            y_turb,
            types=self.fs.windTurbines.types,
            wd=self.fs.wind_direction,
            ax=ax1,
            yaw=yaw,
            tilt=tilt,
        )
        ax1.set_title("Flow field at {} s".format(self.fs.time))
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if self.render_mode == "human":
            pass

        else:
            # If we have the RGB mode.
            pass

    def close(self):
        plt.close()
        if self.Baseline_comp:
            self.fs_baseline = None
            self.site_base = None
            del self.fs_baseline
            del self.site_base
        self.fs = None
        self.site = None
        self.farm_measurements = None
        del self.fs
        del self.site
        del self.farm_measurements
        gc.collect()
