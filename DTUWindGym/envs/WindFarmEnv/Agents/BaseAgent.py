"""
Base container for all "basic" agents.
We MUST have a predict function, that returns the action and the state.

The scale_yaw function is used to scale the yaw angles to be between -1 and 1.
"""


class BaseAgent:
    def __init__(self, yaw_max=45, yaw_min=-45):
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min

    def predict(self, *args, **kwargs):
        pass

    def scale_yaw(self, yaws):
        """
        Scale the yaw angles to be between -1 and 1.
        """
        action = (yaws - self.yaw_min) / (self.yaw_max - self.yaw_min) * 2 - 1

        return action
