from typing import Optional
import numpy as np
import gymnasium as gym
from abc import abstractmethod

"""This is the base class for the wind environment
All the other wind environments will inherit from this class.

This just contains some helper functions that are used in the other classes
"""

class WindEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def _scale_val(self, val, min_val, max_val):
        #Scale the value from -1 to 1
        return 2 * (val - min_val) / (max_val - min_val) - 1 
    

    def _random_choice(self, min_val, max_val, n):
        #Return a random choice between min_val and max_val
        return self.np_random.choice(np.linspace(min_val, max_val, n))
    
    def _random_uniform(self, min_val, max_val, n=None):
        #Return a random value between min_val and max_val
        #Note that we ignore the n parameter
        return self.np_random.uniform(low=min_val, high=max_val)
    
    def _return_zeros(self, **kwargs):
        #Return a numpy array of zeros with the specified shape
        return np.zeros(kwargs["n"])
    
    def _randoms_uniform(self, **kwargs):
        #Return a numpy array of n random values between min_val and max_val
        return self.np_random.uniform(low=kwargs["min_val"], high=kwargs["max_val"], size=kwargs["n"])
    
    def _random_normal(self, **kwargs):
        #Return a numpy array of n random values with a normal distribution
        # print("Distributing with mean: ", kwargs["mean"], " and std: ", kwargs["std"])
        return self.np_random.normal(loc=kwargs["mean"], scale=kwargs["std"], size=kwargs["n"])
    
    def _defined_yaw(self, **kwargs):
        #Set the yaw values to a specified value
        #If the length of the yaw values is equal to the number of turbines, return the yaw values
        #If the length of the yaw values is 1, we assume that all turbines should have that yaw value
        yaw_vals = kwargs["yaws"]
        if len(yaw_vals) == self.n_turb:
            return yaw_vals
        elif len(yaw_vals) == 1:
            return np.ones(self.n_turb) * yaw_vals[0]
        else:
            raise ValueError("So I am pretty sure something has gone wrong here. The specified yaw values are not the right length.")


