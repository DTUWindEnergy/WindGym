import numpy as np
from .BaseAgent import BaseAgent

"""
The ConstantAgent is a class used for evaluating the performance, under predefinned constant yaw angles.
"""


class ConstantAgent(BaseAgent):
    def __init__(
        self,
        yaw_angles,
        yaw_max=45,
        yaw_min=-45,
    ):
        self.UseEnv = True
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min

        # Turn the yaw_angles into a numpy array if it is a list
        if isinstance(yaw_angles, list):
            yaw_angles = np.array(yaw_angles)

        self.yaw_angles = yaw_angles

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        action = self.scale_yaw(self.yaw_angles)

        return action, None
