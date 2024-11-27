import numpy as np

import matplotlib.pyplot as plt

from DTUWindGym.envs.WindFarmEnv.BasicControllers import local_yaw_controller, global_yaw_controller
from .BaseAgent import BaseAgent
"""
This is the basic agent class. It is used to create a simple agent that can be used in the AgentEval class.
The agent will always try and get to zero yaw offset
"""


class GreedyAgent(BaseAgent):
    def __init__(self, type='local', yaw_max=45, yaw_min=-45, yaw_step=1, env=None):
        # This is used in a hasattr in the AgentEval class/function.
        self.GreedyAgent = True
        self.env = None

        # these are initial values, but they should be overwritten in the eval call
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        # This should be 1, as the action is scaled to be between -1 and 1.
        self.yaw_step = yaw_step

        if type == 'local':
            self.controller = local_yaw_controller
        elif type == 'global':
            self.controller = global_yaw_controller

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """
        yaw_goal = self.controller(fs=self.env.fs, yaw_step=self.yaw_step)

        action = self.scale_yaw(yaw_goal)

        return action, None
