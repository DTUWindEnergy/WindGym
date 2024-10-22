from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dynamiks.utils.test_utils import tfp
import copy
import os
#For the site
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField

#For flow simulation
from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion

from dynamiks.wind_turbines import PyWakeWindTurbines

from dynamiks.views import XYView
from IPython import display
import time
from .WindEnv import WindEnv
from .MesClass import farm_mes

from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from py_wake.utils.plotting import setup_plot
from collections import deque
import itertools
import yaml

"""
This is the base for the wind farm environment. This is where the magic happens.
For now it only supports the PyWakeWindTurbines, but it should be easy to expand to other types of turbines.
"""
#TODO So some sources says that the rewards should be in the order of 10. Therefore we could try and do that.
#TODO Assert that the turbine is a subclass of the PyWakeWindTurbines
#TODO make it so that the turbines can be other then a square grid
#TODO user defined observed variables
#TODO thrust coefficient control
#TODO for now I have just hardcoded this scaling value (1 and 25 for the wind_speed min and max). This is beacuse the wind speed is chosen from the normal distribution, but becasue of the wakes and the turbulence, we canhave cases where we go above or below these values.
#TODO IMPORTANT: We need to make sure that the turbines are consistent with the returning of variables. So that independent of the self.wd
#TODO maybe we should take larger steps in the flow simulation. Or multiple. This coulde be done with fs.run() instead. 

class WindFarmEnv(WindEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, turbine, 
                 TI_min_mes:float = 0.0, TI_max_mes:float = 0.50, 
                 TurbBox = "Default",
                 yaml_path = None,
                 Baseline_comp = False,  
                 yaw_init = None,
                 render_mode=None, seed = None,
                 ):
        """
        This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
        Args: 
            TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
            TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
            TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
            yaml_path: str: The path to the yaml file that contains the configuration of the environment. TODO make a default value for this?
            Baseline_comp: bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
            yaw_init: str: The method for initializing the yaw angles of the turbines. If 'Random', then the yaw angles will be random. Else they will be zeros.
            render_mode: str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
            seed: int: The seed for the environment. If None, then the seed will be random.
        """
        
        #Predefined values
        self.power_setpoint = 0.0   #The power setpoint for the farm. This is used if the Track_power is True. (Not used yet)
        self.act_var = 1                 #number of actions pr. turbine. For now it is just the yaw angles
        self.dt = 1                 #time step for the flow simulation. 
        self.yaw_start = 15.0       #This is the limit for the initialization of the yaw angles. This is used to make sure that the yaw angles are not too large at the start, but still not zero
        self.maxturbpower = max(turbine.power(np.arange(10, 25, 1))) #Max power pr turbine. Used in the measurement class
        self.yaw_step = 1           #The step size for the yaw angles. How manny degress the yaw angles can change pr. step
        self.d_particle = 0.1       #The distance between the particles. This is used in the flow simulation.
        
        #Saves to self
        self.seed = seed
        self.TurbBox = TurbBox
        self.turbine = turbine
        
        self.TF_files = []
        self.yaw_initial = [0]      #The initial yaw of the turbines. This is used if the yaw_init is "Defined"
        

        # Load the configuration
        self.load_config(yaml_path) 

        self.n_turb = self.nx * self.ny   #The number of turbines
        
        #Deques that holds the power output of the farm and the baseline farm. This is used for the power reward
        self.farm_pow_deq = deque(maxlen=self.power_avg) 
        self.base_pow_deq = deque(maxlen=self.power_avg) 
        self.power_len = self.power_avg


        #Sets the yaw init method. If Random, then the yaw angles will be random. Else they will be zeros
        #If yaw_init is defined (it will be if we initialize from EnvEval) then set it like this. Else just use the value from the yaml 
        if yaw_init is not None:
            #We only ever have this, IF we have set the value from 
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

        #Define the power tracking reward function TODO Not implemented yet. Also make the power_setpoint an observable parameter
        if self.Track_power:
            self.power_setpoint = 42 #???
            self._track_rew = self.track_rew_avg
            raise NotImplementedError("The Track_power is not implemented yet")
        else:
            self._track_rew = self.track_rew_none

        #Define the power production reward function
        if self.power_reward =="Baseline":
            self._power_rew = self.power_rew_baseline   #The baseline power reward function
        elif self.power_reward == "Power_avg":
            self._power_rew = self.power_rew_avg        #The power_avg reward function
        elif self.power_reward == "None":
            self._power_rew = self.power_rew_none       #The no power reward function
        elif self.power_reward == "Power_diff":
            #TODO rethink this way of doing it.
            self._power_rew = self.power_rew_diff       #The power_diff reward function
            self._power_wSize = self.power_avg // 10 #We set this to 10, to have some space in the middle. 
            if self.power_avg < 40:
                raise ValueError("The Power_avg must be larger then 40 for the Power_diff reward. Also it should probably be way larger my guy") #Why 40? I just chose this as the minimum value. In reality 2 could have sufficed, but to save myself a headache, I set it to 10
        else:
            raise ValueError("The Power_reward must be either Baseline, Power_avg, None or Power_diff")
        
        #Read in the turb boxes
        if TurbBox == "Default":
            self.TF_files.append(tfp + "mann_turb/hipersim_mann_l29.4_ae1.0000_g3.9_h0_1024x128x32_3.200x3.20x3.20_s0001.nc")
        else:
            try:
                for f in os.listdir(TurbBox):
                    if f.split("_")[0] == "TF":
                        self.TF_files.append( os.path.join(TurbBox, f)   )
            except:
                print("Coudnt find the turbulence box file(s), so we just use the default one")
                self.TF_files = [tfp + "mann_turb/hipersim_mann_l29.4_ae1.0000_g3.9_h0_1024x128x32_3.200x3.20x3.20_s0001.nc"]

        #If we need to have a "baseline" farm, then we need to set up the baseline controller
        #This could be moved to the Power_reward check, but I have a feeling this will be expanded in the future, when we include damage. 
        if self.power_reward == "Baseline" or Baseline_comp:
            self.Baseline_comp = True
        else:
            self.Baseline_comp = False

        #Initializing the measurements class with the specified values.
        #TODO if history_length is 1, then we dont need to save the history, and we can just use the current values.
        #TODO is history_N is 1 or larger, then it is kinda implied that the rolling_mean is true.. Therefore we can change the if self.rolling_mean: check in the Mes() class, to be a if self.history_N >= 1 check... or something like that
        self.farm_measurements = farm_mes(self.n_turb, self.noise,  
                                          self.mes_level["turb_ws"], self.mes_level["turb_wd"], self.mes_level["turb_TI"], self.mes_level["turb_power"],
                                          self.mes_level["farm_ws"], self.mes_level["farm_wd"], self.mes_level["farm_TI"], self.mes_level["farm_power"],
                                          self.ws_mes["ws_current"], self.ws_mes["ws_rolling_mean"], self.ws_mes["ws_history_N"], self.ws_mes["ws_history_length"], self.ws_mes["ws_window_length"],
                                          self.wd_mes["wd_current"], self.wd_mes["wd_rolling_mean"], self.wd_mes["wd_history_N"], self.wd_mes["wd_history_length"], self.wd_mes["wd_window_length"],
                                          self.yaw_mes["yaw_current"], self.yaw_mes["yaw_rolling_mean"], self.yaw_mes["yaw_history_N"], self.yaw_mes["yaw_history_length"], self.yaw_mes["yaw_window_length"],
                                          self.power_mes["power_current"], self.power_mes["power_rolling_mean"], self.power_mes["power_history_N"], self.power_mes["power_history_length"], self.power_mes["power_window_length"],
                                          2.0, 25.0,                        #Max and min values for wind speed measuremenats
                                          self.wd_min-5, self.wd_max+5,     #Max and min values for wind direction measurements   NOTE i have added 5 for some slack in the measurements. so the scaling is better.
                                          self.yaw_min, self.yaw_max,       #Max and min values for yaw measurements
                                          TI_min_mes, TI_max_mes,           #Max and min values for the turbulence intensity measurements
                                          power_max=self.maxturbpower)   

        self.hist_max = self.farm_measurements.max_hist()  #The maximum history length of the measurements
         
        #Setting up the turbines:

        D = turbine.diameter()

        x = np.linspace(0, D*self.xDist*self.nx, self.nx)
        y = np.linspace(0, D*self.yDist*self.ny, self.ny)

        xv, yv = np.meshgrid(x, y, indexing='xy')

        self.x_pos = xv.flatten()
        self.y_pos = yv.flatten()
        self.y_pos += 200 #Note we move the farm 200 units up. This is done because I think I saw some weird behaviour with y = 0 :/ 

        
        self.wts = PyWakeWindTurbines(x=self.x_pos, y=self.y_pos,   # x and y position of two wind turbines
                                      windTurbine=self.turbine)

        #Setting up the baseline controller if we need it
        if self.Baseline_comp:
            #If we compare to some baseline performance, then we also need a controller for that
            if self.BaseController == "Local":
                self._base_controller = self._base_controller_local
            elif self.BaseController == "Global":
                self._base_controller = self._base_controller_global
            else:
                raise ValueError("The BaseController must be either Local or Global... For now")
            #Definde the turbines
            self.wts_baseline = PyWakeWindTurbines(x=self.x_pos, y=self.y_pos,   # x and y position of two wind turbines
                                        windTurbine=turbine)


        #Define the observation and action space
        self.obs_var = self.farm_measurements.observed_variables()

        self.reset(seed=seed)  #I dont hope anything breaks by not having this here.

        #TODO the render mode is not implemented yet. I think?
        #Asserting that the render_mode is valid.
        #If render_mode is None, we will not render anything, if it is human, we will render the environment in a window, if it is rgb_array, we will render the environment as an array
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.init_render()

    def load_config(self, config_path):
        """ 
        This loads in the yaml file, and sets a bunch of internal values. 
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)  # Load the YAML file
        
        # Set the attributes of the class based on the config file
        self.yaw_init = config.get('yaw_init')
        self.noise = config.get('noise')
        self.BaseController = config.get('BaseController')
        self.ActionMethod = config.get('ActionMethod')
        # self.Baseline_comp = config.get('Baseline_comp')
        self.Track_power = config.get('Track_power')
        
        #Unpack the farm params
        farm_params = config.get('farm')
        self.yaw_min = farm_params["yaw_min"]
        self.yaw_max = farm_params["yaw_max"]
        self.xDist = farm_params["xDist"]
        self.yDist = farm_params["yDist"]
        self.nx = farm_params["nx"]
        self.ny = farm_params["ny"]

        #Unpack the wind params
        wind_params = config.get('wind')
        self.ws_min = wind_params["ws_min"]
        self.ws_max = wind_params["ws_max"]
        self.TI_min = wind_params["TI_min"]
        self.TI_max = wind_params["TI_max"]
        self.wd_min = wind_params["wd_min"]
        self.wd_max = wind_params["wd_max"]

        self.act_pen = config.get('act_pen')
        self.power_def = config.get('power_def')
        self.mes_level = config.get('mes_level')
        self.ws_mes = config.get('ws_mes')
        self.wd_mes = config.get('wd_mes')
        self.yaw_mes = config.get('yaw_mes')
        self.power_mes = config.get('power_mes')

        # unpack some more, because we use these later. 
        self.action_penalty = self.act_pen["action_penalty"] 
        self.action_penalty_type = self.act_pen["action_penalty_type"] 
        self.Power_scaling = self.power_def["Power_scaling"]
        self.power_avg = self.power_def["Power_avg"]
        self.power_reward = self.power_def["Power_reward"]

    def _init_spaces(self):
        """
        This function initializes the observation and action spaces. 
        This is done in a seperate function, so we can replace it in the multi agent version of the environment
        """
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, 
                                                shape=((self.obs_var), ), dtype=np.float32) 
        self.action_space = gym.spaces.Box(low= -1, high=1, 
                                           shape=((self.n_turb * self.act_var), ), dtype=np.float32) 

    def init_render(self):
        plt.ion()

        x_turb, y_turb = self.fs.windTurbines.positions_xyz[:2]

        self.figure, self.ax = plt.subplots(figsize=(10,4))
        self.a = np.linspace(-200 + min(x_turb), 1000 + max(x_turb), 250)
        self.b = np.linspace(-200 + min(y_turb), 200 + max(y_turb), 250)

        self.view = XYView(z=self.turbine.hub_height(), x=self.a, y=self.b, ax=self.ax, adaptive=False)

        plt.close()

    def _update_measurements(self):
        """
        This function adds the current observations to the farm_measurements class
        """
        
        
        #Get the observation of the environment
        self.current_ws = np.linalg.norm(self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0) #The current ws is the norm of the three components
        #The current wd is the invtan of the u/v components of the wind speed. Remember to add the "global" wind direction to this measurement, as we are rotating the farm 
        u_speed = self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True)[1]
        v_speed = self.fs.windTurbines.rotor_avg_windspeed(include_wakes=True)[0]
        self.current_wd = np.rad2deg(np.arctan( u_speed / v_speed)) + self.wd

        self.current_yaw = self.fs.windTurbines.yaw
        powers = self.fs.windTurbines.power() #The Power pr turbine

        self.farm_measurements.add_measurements(self.current_ws, self.current_wd, self.current_yaw, powers)

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
            return_dict["Power pr turbine baseline"] = self.fs_baseline.windTurbines.power()
        
        return return_dict

    def _set_windconditions(self):
        """
        Sets the global windconditions for the environment
        """

        self.ws = self._random_uniform(self.ws_min, self.ws_max)  #The wind speed is a random number between ws_min and ws_max
        self.ti = self._random_uniform(self.TI_min, self.TI_max)  #The turbulence intensity is a random number between TI_min and TI_max
        self.wd = self._random_uniform(self.wd_min, self.wd_max)  #The wind direction is a random number between wd_min and wd_max

    def _def_site(self):
        """
        Defines the self.site. This is the site that the flow simulation is run on.
        We choose a random turbulence box, and scale it to the correct TI and wind speed
        It is repeated for the baseline is we have that.
        """

        tf_file = self.np_random.choice( self.TF_files ) 

        tf_agent = MannTurbulenceField.from_netcdf(filename = tf_file )
        tf_agent.scale_TI(ti=self.ti, U=self.ws)  
        self.site = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_agent)

        if self.Baseline_comp:
            tf_base = MannTurbulenceField.from_netcdf(filename = tf_file )
            tf_base.scale_TI(ti=self.ti, U=self.ws)  
            self.site_base = TurbulenceFieldSite(ws=self.ws, turbulenceField=tf_base)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. This is called at the start of every episode.
        - The wind conditions are sampled, and the site is set.
        - The flow simulation is run for the time it takes for the flow to develop.
        - The measurements are filled up with the initial values.
        
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Sample global wind conditions and set the site
        self._set_windconditions()  
        self._def_site()


        self.rated_power = self.turbine.power(self.ws) #This is the rated poweroutput of the turbine at the given ws. Used for reward scaling. 

        self.fs = DWMFlowSimulation(site=self.site, windTurbines=self.wts, 
                        wind_direction=self.wd,
                        particleDeficitGenerator=jDWMAinslieGenerator(), 
                        dt=self.dt,
                        d_particle = self.d_particle,
                        particleMotionModel=HillVortexParticleMotion())  #NOTE, we need this particlemotion to capture the yaw
        

        #Set the yaw angles of the farm
        #NOTE that I use yaw_start and not yaw_min/yaw_max. This is to make sure that the yaw angles are not too large at the start, but still not zero
        self.fs.windTurbines.yaw = self._yaw_init(min_val=-self.yaw_start, max_val=self.yaw_start, n=self.n_turb, yaws=self.yaw_initial) 
        
        #Calulate the time it takes for the flow to develop.
        
        turb_place = np.linalg.norm(self.fs.windTurbines.positions_xyz, axis=0)
        dist = turb_place.max() - turb_place.min()  

        t_developed = int( (dist / self.ws) * 1.1 ) #The time it takes for the flow to develop. Also a bit extra.

        #first we run the simulation the time it takes the flow to develop
        self.fs.run(t_developed)  

        #After the flow is fully developed, we fill up the measurements 
        for _ in range(int(  max( self.hist_max, self.power_len) )):
            self.fs.step()  #Take a step in the flow simulation
            self.farm_pow_deq.append( self.fs.windTurbines.power().sum() ) #Save the power output of the farm
            self._update_measurements()
            
        #Do the same for the baseline farm
        if self.Baseline_comp:
            self.fs_baseline = DWMFlowSimulation(site=self.site_base, 
                                                windTurbines=self.wts_baseline, 
                                                wind_direction=self.wd,
                                                particleDeficitGenerator=jDWMAinslieGenerator(), 
                                                dt=self.dt,
                                                d_particle = self.d_particle,
                                                particleMotionModel=HillVortexParticleMotion()) 

            self.fs_baseline.windTurbines.yaw = self.fs.windTurbines.yaw
            self.fs_baseline.run(t_developed)  

            for _ in range(int(  max( self.hist_max, self.power_len) )):
                self.fs_baseline.step()  #Take a step in the baseline flow simulation
                self.base_pow_deq.append( self.fs_baseline.windTurbines.power().sum() ) 

        
        #Now we can start

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _base_controller_local(self):
        """
        This is the logic for the base controller. It just wants to move back to have zero yaw angles.
        It works on the "local" wind conditions, and tries to move the yaw angles back to zero.
        Note that it doesnt filter the local winddirections in any way, so it just moves perfectly towards the winddirection at every step.
        """
        #Fist we get the current yaw offset, in relation to the "global wind"
        yaw_baseline = self.fs_baseline.windTurbines.yaw

        #Then by taking the inverse tan of the wind speed components, we get the LOCAL wind direction
        wind_dir_baseline = np.rad2deg(np.arctan(self.fs_baseline.windTurbines.rotor_avg_windspeed(include_wakes=True)[1] / self.fs_baseline.windTurbines.rotor_avg_windspeed(include_wakes=True)[0]))
        
        #The desired yaw offset is the difference between the baseline yaw and the baseline wind direction
        yaw_offset = wind_dir_baseline - yaw_baseline

        #We find the direction of the yaw_action, by taking the sign
        step_dir = np.sign(yaw_offset) 

        #Then we find the size of the steps. This is the minimum of the yaw_offset and the yaw_step. This is to make sure that we dont overshoot the target, and to kap the max step taken to be the yaw_step
        step_scale = np.abs(yaw_offset)                         #this is how large the steps would be, if it was unlimited
        step_scale[step_scale > self.yaw_step] = self.yaw_step  #here we replace all values that are larger then the max, with the max
        
        yaw_action = step_dir * step_scale 
        
        new_yaw = yaw_baseline + yaw_action

        self.fs_baseline.windTurbines.yaw = new_yaw #Update the yaw angles

    def _base_controller_global(self):
        """
        This is the logic for the base controller. But now it only sees the "global" wind direction. 
        """

        #The current yaw offset, in relation to the "global wind"
        yaw_offset = self.fs_baseline.windTurbines.yaw  

        #Frst we find the direction of the yaw_action, by taking the sign
        step_dir = np.sign(yaw_offset) 

        #Then we find the size of the steps. This is the minimum of the yaw_offset and the yaw_step. This is to make sure that we dont overshoot the target, and to kap the max step taken to be the yaw_step
        step_scale = np.abs(yaw_offset)                         #this is how large the steps would be, if it was unlimited
        step_scale[step_scale > self.yaw_step] = self.yaw_step  #here we replace all values that are larger then the max, with the max
        
        yaw_action = step_dir * step_scale 
        
        new_yaw = yaw_offset - yaw_action
        
        self.fs_baseline.windTurbines.yaw = new_yaw #Update the yaw angles

    def _action_penalty(self):
        """
        This function calculates a penalty for the actions. This is used to penalize the agent for taking actions, and try and make it more stable
        """
        if self.action_penalty < 0.001: #If the penalty is very small, then we dont need to calculate it
            return 0
        
        elif self.action_penalty_type == "Change":
            #The penalty is dependent on the change in values
            pen_val = np.mean(np.abs(self.old_yaws - self.fs.windTurbines.yaw))  
        elif self.action_penalty_type == "Total":
            #The penalty is dependent on the total values
            pen_val = np.mean(np.abs(self.fs.windTurbines.yaw)) / self.yaw_max

        return self.action_penalty * pen_val

    def _adjust_yaws(self, action):
        """
        Heavily inspired from https://github.com/AlgTUDelft/wind-farm-env
        This function adjusts the yaw angles of the turbines, based on the actions given, but we now have differnt methods for the actions
        """

        if self.ActionMethod == "yaw":
            #The new yaw angles are the old yaw angles + the action, scaled with the yaw_step
            # 0 action means no change
            self.fs.windTurbines.yaw += action * self.yaw_step  #the new yaw angles are the old yaw angles + the action, scaled with the yaw_step
            self.fs.windTurbines.yaw = np.clip(self.fs.windTurbines.yaw, self.yaw_min, self.yaw_max)  #clip the yaw angles to be between -30 and 30
        
        elif self.ActionMethod == "wind":
            #The new yaw angles are the action, scaled to be between the min and max yaw angles
            # 0 action means to move to 0 yaw angle, and 1 action means to move to the max yaw angle
            new_yaws = (action + 1.0) / 2.0 * (self.yaw_max - self.yaw_min) + self.yaw_min 

            #The bounds for the yaw angles are:
            yaw_max = self.fs.windTurbines.yaw + self.yaw_step
            yaw_min = self.fs.windTurbines.yaw - self.yaw_step

            #The new yaw angles are the new yaw angles, but clipped to be between the yaw_max and yaw_min
            self.fs.windTurbines.yaw = np.clip(new_yaws, yaw_min, yaw_max)
        
        elif self.ActionMethod == "absolute":
            raise NotImplementedError("The absolute method is not implemented yet")
        
        else:
            raise ValueError("The ActionMethod must be yaw, wind or absolute")

    def power_rew_baseline(self):
        """
        Calculate the power reward based on the baseline farm
        The reward is: (power_agent / power_baseline - 1)
        """

        power_agent = np.mean(self.farm_pow_deq)
        power_baseline = np.mean(self.base_pow_deq)

        reward = (power_agent / power_baseline - 1)    

        return reward

    def power_rew_avg(self):
        """
        Calculate the power reward based on the average power output
        The reward is: power_agent / n_turbines / rated_power
        NOTE I have found the reward to be somewhat sensitive to the number of values in the deque. Larger deque gives lower reward, but it might make it more stable? 
        """

        power_agent = np.mean(self.farm_pow_deq)
        reward = power_agent / self.n_turb / self.rated_power 
        
        return reward

    def power_rew_none(self):
        """
        Return zero for the power reward
        This is used if we dont care about the power output directly. E.g. if we do power tracking or something like that
        """

        return 0.0

    def power_rew_diff(self):
        """
        This reward is based on the current power putput, compared to the average power output over the last Power_avg steps
        NOTE if using this: a good starting point for Power_scaling is 0.0001. Atleast that lookes to be somewhat decent. 
        More experimentation is needed
        """
        
        #Latest power measurements:
        power_latest = np.mean(   list(itertools.islice(self.farm_pow_deq, self.power_len-self._power_wSize, self.power_len ))   )

        #Oldest power measurements:
        power_oldest =  np.mean(   list(itertools.islice(self.farm_pow_deq, 0, self._power_wSize ))   )

        return (power_latest - power_oldest) / self.n_turb 

    def track_rew_none(self):
        """
        If we are not using power tracking, then just return 0
        """
        return 0.0

    def track_rew_avg(self):
        """
        The reward is the negative difference between the power output and the power setpoint squared
        The reward is: - (power_agent - power_setpoint)^2
        The further we are from the desired power output, the larger the penalty 
        """
        #
        power_agent = np.mean(self.farm_pow_deq)
        return - (power_agent - self.power_setpoint)**2   

    def step(self, action):
        """
        The step function
        1. Adjust the yaw angles of the turbines
        2. Take a step in the flow simulation
        3. Update the measurements
        4. Calculate the reward
        5. Return the observation, reward, terminated, truncated and info
    
        """


        self.old_yaws = copy.copy(self.fs.windTurbines.yaw)  #Save the old yaw angles, so we can calculate the change in yaw angles

        self._adjust_yaws(action)       #Adjust the yaw angles of the agent farm
        self.fs.step()                  #Take a step in the flow simulation
        self._update_measurements()     #Update the measurements
        observation = self._get_obs()
        info = self._get_info()
        self.farm_pow_deq.append( self.fs.windTurbines.power().sum() ) #Save the power output of the farm

        if self.Baseline_comp:
            #If we have the baseline farm, then we do mostly the same as above, but for the baseline farm
            self._base_controller()   #Run the base controller step also.
            self.fs_baseline.step()   #Take a step in the baseline flow simulation
            self.base_pow_deq.append( self.fs_baseline.windTurbines.power().sum() ) #Save the power output of the baseline farm
        
        #Calculate the reward
        power_rew = self._power_rew() * self.Power_scaling  #The power production reward with the scaling
        # track_rew = self._track_rew()  #The power tracking reward. This is just a placeholder so far. 
        track_rew = 0.0

        action_penalty = self._action_penalty()  #The penalty for the actions


        #The reward is: power reward - action penalty. This makes it possible to add a reward for power tracking, and/or damage, easily. 
        reward = power_rew + track_rew - action_penalty  #The reward is the power reward minus the action penalty

        terminated = False
        truncated = False

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


        plt.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest")  #[0] is the u component of the wind speed
        # plt.colorbar().set_label('Wind speed, u [m/s]')
        WindTurbinesPW.plot_xy(self.fs.windTurbines, x_turb, y_turb, types=self.fs.windTurbines.types, 
                               wd=self.fs.wind_direction, ax=ax1, yaw=yaw, tilt=tilt)
        ax1.set_title('Flow field at {} s'.format(self.fs.time))
        display.display(plt.gcf())
        display.clear_output(wait=True)


        if self.render_mode == "human":
                pass

        else:
            # If we have the RGB mode.
            pass

    def close(self):
        plt.close()