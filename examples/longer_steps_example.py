import gymnasium as gym
import numpy as np
import os
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80 as wind_turbine

import wandb
from wandb.integration.sb3 import WandbCallback

import torch
import argparse
from pathlib import Path

# Fixed yaw rate in degrees per second
YAW_RATE = 1.0


class WindFarmMonitor(BaseCallback):
    """Custom callback for plotting wind farm performance metrics"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = None
        self.episode_count = 0

        # For tracking moving averages
        self.window_size = 100
        self.agent_powers = []
        self.base_powers = []

    def _on_step(self) -> bool:
        if self.current_rewards is None:
            self.current_rewards = np.zeros(self.training_env.num_envs)

        # Get info from environments
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        # Update current rewards
        self.current_rewards += rewards

        # Process each environment
        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            agent_power = info["Power agent"]
            base_power = info["Power baseline"]

            self.agent_powers.append(agent_power)
            self.base_powers.append(base_power)

            # Calculate moving averages
            agent_power_avg = np.mean(self.agent_powers[-self.window_size :])
            base_power_avg = np.mean(self.base_powers[-self.window_size :])
            power_ratio = agent_power / base_power if base_power != 0 else 1.0

            wandb.log(
                {
                    f"charts/agent_power-{env_idx}": agent_power,
                    f"charts/base_power-{env_idx}": base_power,
                    f"charts/agent_power_avg-{env_idx}": agent_power_avg,
                    f"charts/base_power_avg-{env_idx}": base_power_avg,
                    f"charts/power_ratio-{env_idx}": power_ratio,
                    f"charts/step_reward-{env_idx}": rewards[env_idx],
                    # Add wind farm state tracking
                    f"states/yaw_angle_t1-{env_idx}": info["yaw angles agent"][
                        0
                    ],  # First turbine
                    f"states/yaw_angle_t2-{env_idx}": info["yaw angles agent"][
                        1
                    ],  # Second turbine
                    f"states/wind_speed_t1-{env_idx}": info["Wind speed at turbines"][
                        0
                    ],
                    f"states/wind_speed_t2-{env_idx}": info["Wind speed at turbines"][
                        1
                    ],
                    f"states/wind_dir_t1-{env_idx}": info["Wind direction at turbines"][
                        0
                    ],
                    f"states/wind_dir_t2-{env_idx}": info["Wind direction at turbines"][
                        1
                    ],
                    f"states/global_wind_speed-{env_idx}": info["Wind speed Global"],
                    f"states/global_wind_dir-{env_idx}": info["Wind direction Global"],
                    f"global_step-{env_idx}": self.num_timesteps,
                }
            )

            # If episode is done, log episode metrics
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_rewards[env_idx])

                # Calculate episode stats
                recent_rewards = self.episode_rewards[-self.window_size :]

                wandb.log(
                    {
                        f"charts/episode_reward-{env_idx}": self.current_rewards[
                            env_idx
                        ],
                        f"charts/episode_reward_mean-{env_idx}": np.mean(
                            recent_rewards
                        ),
                        f"charts/episode_reward_std-{env_idx}": np.std(recent_rewards)
                        if len(recent_rewards) > 1
                        else 0,
                        f"charts/episodes-{env_idx}": self.episode_count,
                        f"global_step-{env_idx}": self.num_timesteps,
                    }
                )

                # Reset current reward for this environment
                self.current_rewards[env_idx] = 0

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for timestep and power averaging window study"
    )

    parser.add_argument(
        "--dt_env", required=True, type=int, help="Environment timestep size"
    )
    parser.add_argument(
        "--power_avg",
        required=True,
        type=int,
        help="Number of steps for power averaging",
    )
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument(
        "--n_env", default=8, type=int, help="Number of parallel environments"
    )
    parser.add_argument(
        "--train_steps", default=100000, type=int, help="Total training timesteps"
    )
    parser.add_argument(
        "--yaml_path", required=True, type=str, help="Path to environment config YAML"
    )
    parser.add_argument(
        "--turbbox_path",
        default="Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc",
        help="Path to turbulence box",
    )

    args = parser.parse_args()

    # Calculate yaw step size based on environment timestep
    yaw_step = YAW_RATE * args.dt_env

    # Create unique run name
    run_name = (
        f"dt_nodir2_pow_study_dt{args.dt_env}_pow{args.power_avg}_seed{args.seed}"
    )

    print(
        f"Running with dt_env={args.dt_env}, power_avg={args.power_avg} steps, yaw_step={yaw_step} deg/step"
    )

    # Initialize wandb with chart configurations
    run = wandb.init(
        project="WindFarm_PowerStudy",
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        config={
            "dt_env": args.dt_env,
            "power_avg": args.power_avg,
            "seed": args.seed,
            "yaw_step": yaw_step,
            "n_env": args.n_env,
            "train_steps": args.train_steps,
            "gamma": 0.1,
            # "gae_lambda": 0.98,
            "n_steps": 2048,
            "learning_rate": 1e-5,
            # "batch_size": 64,
            "ent_coef": 0.01,
        },
    )

    # Create vectorized environment
    env = make_vec_env(
        lambda: WindFarmEnv(
            turbine=wind_turbine(),
            yaml_path=args.yaml_path,
            TurbBox=args.turbbox_path,
            seed=args.seed,
            dt_sim=1,
            dt_env=args.dt_env,
            yaw_step=yaw_step,
            observation_window_size=args.power_avg,
            n_passthrough=5,
        ),
        n_envs=args.n_env,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # Initialize model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=args.seed,  # gamma=0.2,
        n_steps=2048,
    )

    # Setup directories for model saving
    models_dir = f"models_pow3/dt_{args.dt_env}_pow_{args.power_avg}"
    os.makedirs(models_dir, exist_ok=True)

    # Create callbacks
    callbacks = CallbackList(
        [
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
                model_save_path=models_dir,
                model_save_freq=10000,
            ),
            WindFarmMonitor(),
        ]
    )

    # Train model
    print(f"Starting training for dt_env={args.dt_env}, power_avg={args.power_avg}")
    model.learn(total_timesteps=args.train_steps, callback=callbacks)

    # Save final model
    final_model_path = os.path.join(
        models_dir, f"final_model_dt{args.dt_env}_pow{args.power_avg}"
    )
    model.save(final_model_path)
    wandb.save(final_model_path)

    # Cleanup
    wandb.finish()
    del model
    env.close()

    print(f"Study completed for dt_env={args.dt_env}, power_avg={args.power_avg}")
