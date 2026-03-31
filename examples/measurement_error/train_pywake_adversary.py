import os
import tyro
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from dataclasses import dataclass
import yaml
import wandb
from wandb.integration.sb3 import WandbCallback

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from WindGym.Measurement_Manager import MeasurementManager
from WindGym.Agents import NoisyPyWakeAgent  # Use the noise-aware PyWakeAgent

# --- Adversarial Environment Wrapper ---


class AdversarialEnvWrapper(gym.Wrapper):
    """
    Wraps a WindFarmEnv to create an adversarial training scenario.
    - The 'adversary' is the agent being trained.
    - The 'protagonist' is a fixed, pre-trained agent (in this case, NoisyPyWakeAgent).
    The adversary's action is to apply noise to the observations.
    The adversary's reward is the negative of the protagonist's reward.
    """

    def __init__(
        self, env: WindFarmEnv, protagonist_agent: NoisyPyWakeAgent, constraints: dict
    ):
        super().__init__(env)
        self.protagonist = protagonist_agent
        self.constraints = constraints

        # The adversary's action space: one action per turbine for ws bias, one for wd bias
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.n_turb * 2,), dtype=np.float32
        )
        # The adversary sees the same clean observations as the protagonist would
        self.observation_space = self.env.observation_space

        self.mm = MeasurementManager(self.env)
        self.current_bias_state = {}  # Stores evolving physical biases

    def reset(self, **kwargs):
        self.current_bias_state = {}
        return self.env.reset(**kwargs)

    def step(self, adv_action: np.ndarray):
        clean_obs = self.env._get_obs()
        noisy_obs, _ = self._apply_adversarial_noise(clean_obs, adv_action)

        # The NoisyPyWakeAgent will estimate wind conditions from the noisy_obs
        protagonist_action, _ = self.protagonist.predict(noisy_obs, deterministic=True)

        next_clean_obs, prot_reward, terminated, truncated, info = self.env.step(
            protagonist_action
        )

        adversary_reward = -prot_reward

        return next_clean_obs, adversary_reward, terminated, truncated, info

    def _apply_adversarial_noise(
        self, clean_observations: np.ndarray, adv_action: np.ndarray
    ):
        """Applies noise based on the adversary's action."""
        noisy_obs = clean_observations.copy()
        action_idx = 0

        for spec in self.mm.specs:
            is_ws = "ws_current" in spec.name and "turb" in spec.name
            is_wd = "wd_current" in spec.name and "turb" in spec.name

            if (is_ws or is_wd) and action_idx < len(adv_action):
                max_bias = (
                    self.constraints["max_bias_ws"]
                    if is_ws
                    else self.constraints["max_bias_wd"]
                )

                max_physical_change_per_step = max_bias * 0.1
                bias_change_physical_delta = (
                    adv_action[action_idx] * max_physical_change_per_step
                )

                current_physical_bias = self.current_bias_state.get(spec.name, 0.0)
                new_physical_bias = current_physical_bias + bias_change_physical_delta
                new_physical_bias = np.clip(new_physical_bias, -max_bias, max_bias)

                self.current_bias_state[spec.name] = new_physical_bias
                action_idx += 1

                span = spec.max_val - spec.min_val
                if span > 0:
                    scaled_bias = (new_physical_bias * 2.0) / span
                    noisy_obs[spec.index_range[0] : spec.index_range[1]] += scaled_bias

        noisy_obs = np.clip(noisy_obs, -1.0, 1.0)
        return noisy_obs, self.current_bias_state


# --- Command-line Arguments ---
@dataclass
class Args:
    project_name: str = "WindGym_Adversary_vs_PyWake"
    run_name_prefix: str = "PPO_Adversary_vs_PyWake"
    seed: int = 42
    total_timesteps: int = 250000
    n_envs: int = 4

    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"

    max_bias_ws: float = 2.0
    max_bias_wd: float = 20.0

    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: str = "64,64"

    models_dir: str = "models/adversary_vs_pywake"


def make_env_factory(args: Args) -> callable:
    """Creates a factory for the adversarial environment against PyWake."""
    with open(args.yaml_config_path, "r") as f:
        base_env_config = yaml.safe_load(f)

    base_env_config["turbtype"] = "None"

    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine=turbine_obj,
        nx=base_env_config["farm"]["nx"],
        ny=base_env_config["farm"]["ny"],
        xDist=base_env_config["farm"]["xDist"],
        yDist=base_env_config["farm"]["yDist"],
    )

    constraints = {"max_bias_ws": args.max_bias_ws, "max_bias_wd": args.max_bias_wd}

    def env_thunk():
        base_env = WindFarmEnv(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            config=args.yaml_config_path,
            turbtype=base_env_config["turbtype"],
            seed=np.random.randint(0, 100000),
        )

        # Create the protagonist agent for this environment instance
        mm = MeasurementManager(base_env)
        pywake_protagonist = NoisyPyWakeAgent(
            measurement_manager=mm,
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            # REMOVED: `env=base_env` is no longer needed here
            look_up=True,  # Use the pre-computed lookup table for speed
        )

        return AdversarialEnvWrapper(base_env, pywake_protagonist, constraints)

    return env_thunk


# --- Main Training Loop ---
def main(args: Args):
    run_name = f"{args.run_name_prefix}_{wandb.util.generate_id()}"
    run = wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
        save_code=True,
    )

    models_save_path = os.path.join(args.models_dir, run.id)
    os.makedirs(models_save_path, exist_ok=True)

    env_factory = make_env_factory(args)
    vec_env = SubprocVecEnv([env_factory for _ in range(args.n_envs)])

    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=net_arch_list)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/adversary_vs_pywake/{run.id}",
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=os.path.join(models_save_path, "checkpoints"),
                verbose=2,
            ),
        )
        model.save(os.path.join(models_save_path, "final_adversary_model"))
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
