from gymnasium.envs.registration import register
from .wind_farm_env import WindFarmEnv
from .agent_eval import AgentEval
from .Agents import PyWakeAgent
from .farm_eval import FarmEval
from .wind_env_multi import WindFarmEnvMulti
from .agent_eval import eval_single_fast as AgentEvalFast

register(
    id="WindGym/WindFarmEnv-v0",
    entry_point="WindGym.envs:WindFarmEnv",
)
