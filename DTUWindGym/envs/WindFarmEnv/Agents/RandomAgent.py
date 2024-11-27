
from .BaseAgent import BaseAgent
"""
This agent takes random actions, and is used to test the environment.
"""


class RandomAgent(BaseAgent):
    def __init__(self, env=None):
        # This is used in a hasattr in the AgentEval class/function.
        self.UseEnv = True
        self.env = env

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        action = self.env.action_space.sample()

        return action, None
