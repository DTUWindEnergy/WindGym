# filename: examples/measurement_error/train_protagonist.py (Updated with Self-Play)

import os
import tyro
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass
import yaml
import wandb
import time
import glob
from typing import List

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager, NoisyWindFarmEnv, NoiseModel
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from noise_definitions import create_procedural_noise_model, AdversarialNoiseModel


# ============================================================================
# 1. NEW: Helper Classes for Synthetic Self-Play
# ============================================================================
class NoiseModelPool:
    """Holds a collection of noise models and allows for seeded random sampling."""

    def __init__(self, noise_models: List[NoiseModel]):
        if not noise_models:
            raise ValueError(
                "NoiseModelPool must be initialized with at least one noise model."
            )
        self.models = noise_models

    def sample(self, rng: np.random.Generator) -> NoiseModel:
        """Selects a noise model from the pool using the provided RNG."""
        return rng.choice(self.models)


class SelfPlayEnvWrapper(gym.Wrapper):
    """
    A wrapper that dynamically samples a new noise model from a pool at the start of each episode.
    """

    def __init__(self, env: WindFarmEnv, noise_model_pool: NoiseModelPool):
        super().__init__(env)
        self.noise_model_pool = noise_model_pool
        self.mm = MeasurementManager(env, seed=env.seed)

        # The observation and action spaces are those of the underlying env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        # The base env's reset is called first, which correctly seeds its RNG
        clean_obs, info = self.env.reset(**kwargs)

        # Use the env's seeded RNG to sample a noise model for this episode
        active_noise_model = self.noise_model_pool.sample(rng=self.env.np_random)
        self.mm.set_noise_model(active_noise_model)

        # Reset the chosen noise model (e.g., sample a new bias)
        self.mm.reset_noise()

        # Apply the initial noise
        noisy_obs, noise_info = self.mm.apply_noise(clean_obs)
        info.update(noise_info)
        info["clean_obs"] = clean_obs

        return noisy_obs, info

    def step(self, action: np.ndarray):
        clean_obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply noise using the model selected during the last reset
        noisy_obs, noise_info = self.mm.apply_noise(clean_obs)

        info.update(noise_info)
        info["clean_obs"] = clean_obs

        return noisy_obs, reward, terminated, truncated, info


# ============================================================================
# 2. Command-line Arguments
# ============================================================================
@dataclass
class Args:
    """Script arguments"""

    project_name: str = "WindGym_Protagonist_Training"
    run_name_prefix: str = "PPO_Protagonist"
    seed: int = 42
    total_timesteps: int = 2000000
    n_envs: int = 8

    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"

    noise_type: str = (
        "procedural"  # 'none', 'procedural', 'adversarial', or 'synthetic_self_play'
    )

    antagonist_path: str = ""  # Path to a single antagonist model
    adversary_pool_path: str = (
        "models/adversaries_stateful/"  # <-- NEW: Path to a directory of antagonists
    )

    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: str = "128,128"
    models_dir: str = "models/protagonist_training"
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01


# ============================================================================
# 3. Environment Factory
# ============================================================================
def make_env_factory(args: Args, rank: int) -> callable:
    """Creates a factory function for a single environment instance."""

    def _init() -> gym.Env:
        with open(args.yaml_config_path, "r") as f:
            config = yaml.safe_load(f)

        turbine_obj = V80()
        x_pos, y_pos = generate_square_grid(
            turbine=turbine_obj,
            nx=config["farm"]["nx"],
            ny=config["farm"]["ny"],
            xDist=config["farm"]["xDist"],
            yDist=config["farm"]["yDist"],
        )

        base_env_kwargs = {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "turbine": turbine_obj,
            "config": args.yaml_config_path,
            "turbtype": config.get("turbtype", "None"),
            "dt_sim": 5,
            "dt_env": 10,
            "reset_init": True,
            "Baseline_comp": True,
        }

        env_seed = args.seed + rank
        env_instance = WindFarmEnv(**base_env_kwargs, seed=env_seed)

        if args.noise_type == "none":
            return Monitor(env_instance)

        if args.noise_type == "synthetic_self_play":
            print(f"Rank {rank}: Setting up Synthetic Self-Play pool...")
            if not args.adversary_pool_path:
                raise ValueError(
                    "--adversary-pool-path is required for synthetic_self_play."
                )

            noise_models = []
            # 1. Add the procedural noise model to the pool
            noise_models.append(create_procedural_noise_model())

            # 2. Find and load all adversary models
            adversary_paths = glob.glob(
                os.path.join(args.adversary_pool_path, "**/final_adversary_model.zip"),
                recursive=True,
            )
            print(
                f"Found {len(adversary_paths)} adversaries in '{args.adversary_pool_path}'."
            )

            for path in adversary_paths:
                antagonist = PPO.load(path, device="cpu")
                noise_models.append(
                    AdversarialNoiseModel(antagonist_agent=antagonist, device="cpu")
                )

            # 3. Create the pool and the final wrapped environment
            model_pool = NoiseModelPool(noise_models)
            wrapped_env = SelfPlayEnvWrapper(env_instance, model_pool)

        else:  # Handle procedural and single adversarial cases
            mm = MeasurementManager(env_instance, seed=env_seed)
            noise_model = None

            if args.noise_type == "procedural":
                noise_model = create_procedural_noise_model()
            elif args.noise_type == "adversarial":
                if not args.antagonist_path:
                    raise ValueError(
                        "--antagonist-path is required for adversarial training."
                    )
                antagonist = PPO.load(args.antagonist_path, device="cpu")
                noise_model = AdversarialNoiseModel(
                    antagonist_agent=antagonist, device="cpu"
                )
            else:
                raise ValueError(
                    f"Unsupported noise_type for this block: {args.noise_type}"
                )

            mm.set_noise_model(noise_model)
            wrapped_env = NoisyWindFarmEnv(
                type(env_instance), mm, **env_instance.kwargs
            )

        return Monitor(wrapped_env)

    return _init


# ============================================================================
# 4. Main Training Loop
# ============================================================================
def main(args: Args):
    if not os.path.exists(args.yaml_config_path):
        raise FileNotFoundError(
            f"YAML configuration file not found at: {args.yaml_config_path}"
        )

    run_name = f"{args.run_name_prefix}_vs_{args.noise_type}_{int(time.time())}"
    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
        save_code=True,
    )

    models_save_path = os.path.join(args.models_dir, wandb.run.id)
    os.makedirs(models_save_path, exist_ok=True)

    env_factories = [make_env_factory(args, i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_factories)

    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=dict(pi=net_arch_list, vf=net_arch_list))

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/{wandb.run.id}",
        verbose=1,
    )

    callbacks = CallbackList(
        [
            WandbCallback(
                model_save_path=os.path.join(models_save_path, "checkpoints"), verbose=2
            )
        ]
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        model.save(os.path.join(models_save_path, "final_model"))
        print(f"\n✅ Training complete. Final model saved to '{models_save_path}'")
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    args = tyro.cli(Args)
    main(args)
