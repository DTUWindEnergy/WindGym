# filename: examples/measurement_error/train_adversary.py (Rewritten)

import os
import time
import tyro
import yaml
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from dataclasses import dataclass

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from noise_definitions import AdversarialNoiseModel, get_adversarial_constraints
from WindGym.Measurement_Manager import MeasurementManager


# ============================================================================
# 1. New, Simplified Adversarial Environment Wrapper
# ============================================================================


class AdversarialTrainingWrapper(gym.Wrapper):
    """
    Orchestrates the turn-based interaction for training an adversary.
    This is the environment the PPO algorithm will train on.
    """

    def __init__(
        self, env: WindFarmEnv, protagonist_agent, noise_model: AdversarialNoiseModel
    ):
        super().__init__(env)
        self.protagonist = protagonist_agent
        self.noise_model = noise_model

        # The MeasurementManager holds the specs and orchestrates noise.
        self.mm = MeasurementManager(env, seed=env.seed)
        self.mm.set_noise_model(noise_model)

        # The adversary's action space and observation space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.n_turb * 4,), dtype=np.float32
        )
        self.observation_space = self.env.observation_space

    def set_antagonist(self, antagonist_agent):
        """Injects the antagonist agent into the noise model after initialization."""
        self.noise_model.set_antagonist_agent(antagonist_agent)
        self.action_space = antagonist_agent.action_space

    def reset(self, **kwargs):
        # Reset the underlying environment first
        clean_obs, info = self.env.reset(**kwargs)

        # --- FIX IS HERE ---
        # Pass the MeasurementManager's specs, not the farm_measurements' specs
        self.noise_model.reset_noise(self.mm.specs, self.env.np_random)

        return clean_obs, info

    def step(self, adv_action: np.ndarray):
        # Get the current clean observation from the underlying environment
        current_clean_obs = self.env._get_obs()

        # Use the noise model to apply the adversary's action and get a noisy observation
        noisy_obs = self.noise_model.apply_noise(
            current_clean_obs,
            self.mm.specs,  # <-- Pass the correct specs here as well
            self.env.np_random,
            # adv_action=adv_action
        )

        # The fixed protagonist acts on the noisy observation
        protagonist_action, _ = self.protagonist.predict(noisy_obs, deterministic=True)

        # The protagonist's action is executed in the real environment
        next_clean_obs, prot_reward, terminated, truncated, info = self.env.step(
            protagonist_action
        )

        # The adversary's reward is the negative of the protagonist's reward
        adversary_reward = -prot_reward

        # Add noise info to the log
        info.update(self.noise_model.get_info())

        return next_clean_obs, adversary_reward, terminated, truncated, info

    def close(self):
        self.env.close()


# ============================================================================
# 2. Command-line Arguments & Configuration
# ============================================================================
@dataclass
class Args:
    project_name: str = "WindGym_Adversary_Training"
    run_name_prefix: str = "PPO_StatefulAdv_vs_PPO_Protagonist"
    protagonist_path: str = "models/protagonist_training/fazozee3/final_model.zip"
    seed: int = 42
    total_timesteps: int = 500000
    n_envs: int = 8
    yaml_config_path: str = "env_config/two_turbine_yaw.yaml"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    net_arch: str = "64,64"
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    models_dir: str = "models/adversaries_stateful"


# ============================================================================
# 3. Environment Factory
# ============================================================================
def make_env_factory(args: Args, rank: int, protagonist_model: PPO) -> callable:
    """Creates a factory for a single adversarial environment instance."""

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

        base_env = WindFarmEnv(
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            config=args.yaml_config_path,
            turbtype=config.get("turbtype", "None"),
            dt_sim=5,
            dt_env=10,
            reset_init=True,
            Baseline_comp=True,
            seed=args.seed + rank,
        )

        constraints = get_adversarial_constraints()

        # The noise model holds a reference to the antagonist that will be trained
        noise_model = AdversarialNoiseModel(constraints=constraints)

        # The wrapper orchestrates the turn-based logic
        adv_env = AdversarialTrainingWrapper(base_env, protagonist_model, noise_model)

        return RecordEpisodeStatistics(adv_env)

    return _init


# ============================================================================
# 4. Main Training Loop
# ============================================================================
def main(args: Args):
    import wandb

    run_name = f"{args.run_name_prefix}_{int(time.time())}"
    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=run_name,
        sync_tensorboard=True,
        save_code=True,
    )
    models_save_path = os.path.join(args.models_dir, wandb.run.id)
    os.makedirs(models_save_path, exist_ok=True)

    # --- Load the fixed protagonist model (the victim) ---
    print(f"Loading protagonist from: {args.protagonist_path}")
    protagonist_model = PPO.load(args.protagonist_path)

    # --- Create the vectorized environment ---
    env_factories = [
        make_env_factory(args, i, protagonist_model) for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_factories)

    # --- Create and train the adversary agent ---
    net_arch_list = [int(x) for x in args.net_arch.split(",")]
    policy_kwargs = dict(net_arch=dict(pi=net_arch_list, vf=net_arch_list))

    adversary_model = PPO(
        "MlpPolicy",
        vec_env,  # The model is created directly with the correct vectorized environment
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=f"logs/adversary_training/{wandb.run.id}",
    )

    # --- Callbacks ---
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(10000 // args.n_envs, 1),
                save_path=os.path.join(models_save_path, "checkpoints"),
                name_prefix="adversary_model",
            ),
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=os.path.join(models_save_path, "wandb_models"),
                verbose=2,
            ),
        ]
    )

    print("\n--- Starting Stateful Adversary Training ---")
    try:
        adversary_model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )
        adversary_model.save(os.path.join(models_save_path, "final_adversary_model"))
        print(
            f"\n✅ Training complete. Final adversary model saved to '{models_save_path}'"
        )
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    args = tyro.cli(Args)
    main(args)
