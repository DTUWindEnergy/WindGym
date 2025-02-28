from gymnasium.envs.registration import register
from .Wind_Farm_Env import WindFarmEnv
from .AgentEval import AgentEval
from .Agents import PyWakeAgent
from .FarmEval import FarmEval
from .WindEnvMulti import WindFarmEnvMulti
from .AgentEval import eval_single_fast as AgentEvalFast

register(
    id="WindGym/WindFarmEnv-v0",
    entry_point="WindGym.envs:WindFarmEnv",
)
