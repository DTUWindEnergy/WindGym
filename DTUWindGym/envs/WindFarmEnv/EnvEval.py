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
from .Wind_Farm_Env import WindFarmEnv


"""
This class can be used to evaluate the performance of an agent. We can directly specify the wind values, and the yaw values.
"""

class EnvEval(WindFarmEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, turbine,
                # xDist, yDist, nx, ny,  
                # ws_min:float = 7.0, ws_max:float = 20.0, 
                # TI_min:float = 0.03, TI_max:float = 0.15, 
                # wd_min:float = 270.0, wd_max:float = 360.0,
                TI_min_mes:float = 0.0, TI_max_mes:float = 0.50, #Max and min values for the turbulence intensity measurements. Used for internal scaling
                # yaw_min = -45, yaw_max = 45,
                yaw_init = "Zeros", #Like with the baseline_comp = True, we also always start with zero yaw angles at initilization
                # noise = "None",
                TurbBox = "Default",
                yaml_path = None,
                # action_penalty = 0.0,
                # action_penalty_type = "Change", #Can be "Change" or "Total"
                # BaseController = "Local",  #Can be "Local" or "Global
                # ActionMethod = "yaw", #Can be: "yaw", "wind", "absolute"
                # Power_reward = "Baseline", #Method for calculating the power reward. Can be "Baseline", "Power_avg" or "Power_diff"
                # Power_avg = 1, #Size of the average window for the power reward. If 1 then we only save the current power output.
                # Power_scaling = 1.0,  #Scaling of the power reward
                # turb_ws = True, turb_wd = True, turb_TI = True,    # Do we want wind speed and wind direction measurements from the turbines
                # farm_ws = True, farm_wd = True, farm_TI = False,   # Do we want wind speed and wind direction measuremwnts from the farm
                # ws_current=False, ws_rolling_mean=True, ws_history_N=1, ws_history_length=10, ws_window_length=10,     #Values to specify wind speed measuremrents
                # wd_current=False, wd_rolling_mean=True, wd_history_N=1, wd_history_length=10, wd_window_length=10,     #Values to specify wind direction measurements
                # yaw_current=False, yaw_rolling_mean=True, yaw_history_N=2, yaw_history_length=30, yaw_window_length=1, #Values to specify yaw measurements
                render_mode=None, seed = None,
                # Track_power = False,
                ):
        """
        Initialize the environment
        xDist: float, the rotor distance in the x direction
        yDist: float, the rotor distance in the y direction
        nx: int, the number of turbines in the x direction
        ny: int, the number of turbines in the y direction
        turbine: PyWaketurbine, the turbine to be used
        yaw_init: str, the way to initialize the yaw. Options are "Zeros", "Random"
        noise: str, the noise to be added to the observations. Options are...
        ws: float, the wind speed to be used
        ti: float, the turbulence intensity to be used
        wd: float, the wind direction to be used
        """
        
        #TODO There must be a better way to set all these values ):   maybe **kwargs???
        #Run the Env with these values, to make sure that the oberservartion space is the same.
        super().__init__(turbine=turbine, 
                        #  xDist=xDist, yDist=yDist, nx=nx, ny=ny,
                        #  ws_min=ws_min, ws_max=ws_max,
                        #  TI_min=TI_min, TI_max=TI_max,
                        #  wd_min=wd_min, wd_max=wd_max,
                         TI_min_mes=TI_min_mes, TI_max_mes=TI_max_mes,
                        #  yaw_min=yaw_min, yaw_max=yaw_max,
                         yaml_path=yaml_path,
                         yaw_init=yaw_init, 
                        #  noise=noise, 
                         TurbBox=TurbBox,
                        #  action_penalty=action_penalty,
                        #  action_penalty_type=action_penalty_type,
                        #  BaseController = BaseController, 
                        #  ActionMethod=ActionMethod,
                         Baseline_comp = True, #We always want to compare to the baseline, so this is true
                        #  Power_reward=Power_reward,
                        #  Power_avg=Power_avg,
                        #  Power_scaling=Power_scaling,
                         #Baseline_comp = Baseline_comp,  
                        #  turb_ws = turb_ws, turb_wd = turb_wd,
                        #  farm_ws = farm_ws, farm_wd = farm_wd,    # Do we want wind speed and wind direction measuremwnts from the farm
                        #  turb_TI=turb_TI, farm_TI=farm_TI,
                        #  ws_current=ws_current, ws_rolling_mean=ws_rolling_mean, 
                        #  ws_history_N=ws_history_N, ws_history_length=ws_history_length, ws_window_length=ws_window_length,     
                        #  wd_current=wd_current, wd_rolling_mean=wd_rolling_mean, 
                        #  wd_history_N=wd_history_N, wd_history_length=wd_history_length, wd_window_length=wd_window_length,     
                        #  yaw_current=yaw_current, yaw_rolling_mean=yaw_rolling_mean, 
                        #  yaw_history_N=yaw_history_N, yaw_history_length=yaw_history_length, yaw_window_length=yaw_window_length, 
                         seed=seed,
                        #  Track_power=Track_power,
                         )
        
        #Maybe add a call to the _set_wind_vals here, so that we can set the wind values here.
        #But for now you need to initialize it and then run the function.
        

    def set_wind_vals(self, ws=None, ti=None, wd=None):
        #print("Running the set_wind_vals method")
        #I should be able to set the wind values here. Then when we call the reset method, it has just used these values. 
        if ws is not None:
            self.ws = ws
            self.ws_min = ws
            self.ws_max = ws
        if ti is not None:
            self.ti = ti
            self.ti_min = ti
            self.ti_max = ti
        if wd is not None:
            self.wd = wd
            self.wd_min = wd
            self.wd_max = wd

    def set_yaw_vals(self, yaw_vals):
        #Defines the initial way values, that we want to use for the evaluation
        self.yaw_initial = yaw_vals

    def update_tf(self, path):
        #Here we overwrite the _def_site method to set the turbulence field to the path given
        #Sets the turbbox to the path given
        self.TF_files = [path]
    
