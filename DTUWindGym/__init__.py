from gymnasium.envs.registration import register

register(
    id="DTUWindGym/WindFarmEnv-v0",
    entry_point="DTUWindGym.envs:WindFarmEnv",
)
