import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import wandb
from wandb.integration.sb3 import WandbCallback
from WindGym import WindFarmEnv

# from WindGym.Simple_Wind_Farm_Env import WindFarmEnv
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80 as wind_turbine
import argparse

# Device selection
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class WindFarmMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = None
        self.episode_count = 0
        self.window_size = 100
        self.agent_powers = []
        self.base_powers = []

    # def _on_training_start(self):
    #    self.window_size = self.training_env.get_attr("lookback_window")[0]

    def _on_step(self) -> bool:
        if self.current_rewards is None:
            self.current_rewards = np.zeros(self.training_env.num_envs)

        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_rewards += rewards

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            agent_power = info["Power agent"]
            base_power = info["Power baseline"]

            self.agent_powers.append(agent_power)
            self.base_powers.append(base_power)

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
                    f"global_step-{env_idx}": self.num_timesteps,
                    f"reward_weight-{env_idx}": info.get("curriculum_weight", 0.0),
                    f"yaw_diff-{env_idx}": info.get("yaw_diff", 0.0),
                    f"states/yaw_angle_t1-{env_idx}": info["yaw angles agent"][0],
                    f"states/yaw_angle_t2-{env_idx}": info["yaw angles agent"][1],
                    f"states/global_wind_speed-{env_idx}": info["Wind speed Global"],
                    f"states/global_wind_dir-{env_idx}": info["Wind direction Global"],
                    f"states/pywake_yaw_t1-{env_idx}": info["pywake_yaws"][0],
                    f"states/pywake_yaw_t2-{env_idx}": info["pywake_yaws"][1],
                }
            )

            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_rewards[env_idx])
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

                self.current_rewards[env_idx] = 0

        return True


class WindFarmFeatureExtractor(nn.Module):
    def __init__(self, n_features, window_size):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Keep the statistical features as they're valuable
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.window_size, self.n_features)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]  # Take last hidden state

        # Statistical features (these are still valuable)
        x_trans = x.transpose(1, 2)  # [batch, features, window]
        mean = torch.mean(x_trans, dim=2)
        features = [lstm_features, mean]

        if self.window_size > 1:
            std = torch.std(x_trans, dim=2, unbiased=False)
            max_features = torch.max(x_trans, dim=2)[0]
            min_features = torch.min(x_trans, dim=2)[0]
            features.extend([std, max_features, min_features])

        return torch.cat(features, dim=1)


class WindFarmMLPExtractor(nn.Module):
    def __init__(self, n_features, window_size):
        super().__init__()
        self.feature_extractor = WindFarmFeatureExtractor(n_features, window_size)

        # Calculate feature dimension
        if window_size >= 3:
            feature_dim = 32 + n_features * 4
        elif window_size > 1:
            feature_dim = n_features * 4
        else:
            feature_dim = n_features

        # Projection layers to get to attention dimension
        self.policy_proj = nn.Linear(feature_dim, 256)
        self.value_proj = nn.Linear(feature_dim, 256)

        # Policy network with skip connections
        self.policy_layers = nn.ModuleList(
            [
                nn.Linear(256, 256),
                nn.Linear(256, 128),
                nn.Linear(128 + 256, 64),  # Skip connection from first layer
                nn.Linear(64 + 128, 32),  # Skip connection from second layer
                nn.Linear(32, 16),
            ]
        )

        # Value network with skip connections
        self.value_layers = nn.ModuleList(
            [
                nn.Linear(256, 256),
                nn.Linear(256, 128),
                nn.Linear(128 + 256, 64),  # Skip connection from first layer
                nn.Linear(64 + 128, 32),  # Skip connection from second layer
                nn.Linear(32, 8),
            ]
        )

        self.policy_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True
        )
        self.value_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True
        )

        self.policy_layer_norm = nn.LayerNorm(256)
        self.value_layer_norm = nn.LayerNorm(256)

        self.activation = nn.ReLU()

        # Initialize networks
        self._init_weights()

        self.latent_dim_pi = 16
        self.latent_dim_vf = 8

    def _init_weights(self):
        # Initialize all linear layers
        for layer in (
            [self.policy_proj, self.value_proj]
            + list(self.policy_layers)
            + list(self.value_layers)
        ):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _forward_with_skip(self, x, layers):
        # Store intermediate activations for skip connections
        activations = []
        out = x

        for i, layer in enumerate(layers):
            if i == 0:
                out = self.activation(layer(out))
                activations.append(out)
            elif i == 1:
                out = self.activation(layer(out))
                activations.append(out)
            elif i == 2:
                # Concatenate with first layer output
                skip_connection = torch.cat([out, activations[0]], dim=1)
                out = self.activation(layer(skip_connection))
            elif i == 3:
                # Concatenate with second layer output
                skip_connection = torch.cat([out, activations[1]], dim=1)
                out = self.activation(layer(skip_connection))
            else:
                out = layer(out)

        return out

    def forward(self, obs):
        features = self.feature_extractor(obs)
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Invalid feature values detected")

        batch_size = features.shape[0]

        # Project to attention dimension first
        policy_features = self.policy_proj(features)
        value_features = self.value_proj(features)

        # Reshape for attention (batch, 1, dim)
        policy_features = policy_features.unsqueeze(1)
        value_features = value_features.unsqueeze(1)

        # Apply layer norm
        policy_features = self.policy_layer_norm(policy_features)
        value_features = self.value_layer_norm(value_features)

        # Self-attention
        policy_features, _ = self.policy_attention(
            policy_features, policy_features, policy_features
        )
        value_features, _ = self.value_attention(
            value_features, value_features, value_features
        )

        # Squeeze back to (batch, dim)
        policy_features = policy_features.squeeze(1)
        value_features = value_features.squeeze(1)

        # Forward through networks with skip connections
        policy_output = self._forward_with_skip(policy_features, self.policy_layers)
        value_output = self._forward_with_skip(value_features, self.value_layers)

        return policy_output, value_output

    def forward_actor(self, features):
        """Extract actor features"""
        if isinstance(features, tuple):
            features = features[0]
        features = self.feature_extractor(features)

        # Project and prepare for attention
        features = self.policy_proj(features).unsqueeze(1)
        features = self.policy_layer_norm(features)

        # Apply attention
        features, _ = self.policy_attention(features, features, features)

        # Process through policy network
        features = features.squeeze(1)
        return self._forward_with_skip(features, self.policy_layers)

    def forward_critic(self, features):
        """Extract critic features"""
        if isinstance(features, tuple):
            features = features[0]
        features = self.feature_extractor(features)

        # Project and prepare for attention
        features = self.value_proj(features).unsqueeze(1)
        features = self.value_layer_norm(features)

        # Apply attention
        features, _ = self.value_attention(features, features, features)

        # Process through value network
        features = features.squeeze(1)
        return self._forward_with_skip(features, self.value_layers)


class WindFarmPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features,
        window_size,
        **kwargs,
    ):
        self.n_features = n_features
        self.window_size = window_size
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = WindFarmMLPExtractor(self.n_features, self.window_size)
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, curriculum_steps, pure_similarity_steps):
        super().__init__(env)
        self.curriculum_steps = curriculum_steps
        self.pure_similarity_steps = pure_similarity_steps
        self.current_step = 0
        self.pywake_agent = PyWakeAgent(
            x_pos=self.env.fs.windTurbines.positions_xyz[0],
            y_pos=self.env.fs.windTurbines.positions_xyz[1],
        )
        self.env_reward_weight = 0.0
        self.previous_yaws = None
        self.yaw_change_history = []
        self.reward_momentum = 0.9
        self.last_reward = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pywake_agent.update_wind(self.env.ws, self.env.wd, self.env.ti)
        self.pywake_agent.optimize()
        self.pywake_yaws = self.pywake_agent.optimized_yaws
        info["pywake_yaws"] = self.pywake_yaws
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ppo_yaws = info["yaw angles agent"]
        pywake_yaws = self.pywake_yaws

        yaw_diff = np.abs(np.array(ppo_yaws) - np.array(pywake_yaws)).mean()
        similarity_reward = 1 / (1 + yaw_diff)
        # similarity_reward = -np.log(yaw_diff)

        # Penalize large changes between steps
        movement_penalties = 0
        if self.previous_yaws is not None:
            yaw_changes = np.abs(np.array(ppo_yaws) - self.previous_yaws)
            self.yaw_change_history.append(yaw_changes)

            # Immediate change penalty
            change_penalty = np.mean(yaw_changes) / self.env.yaw_max
            movement_penalties += 0.3 * change_penalty

            # Oscillation penalty - check if changes keep reversing direction
            if len(self.yaw_change_history) >= 2:
                recent_changes = np.array(list(self.yaw_change_history))
                direction_changes = np.diff(np.sign(recent_changes), axis=0)
                oscillation_penalty = np.mean(np.abs(direction_changes)) * 0.2
                movement_penalties += oscillation_penalty

            # Cumulative movement penalty - discourage constant adjustments
            if len(self.yaw_change_history) >= 5:
                cumulative_changes = np.sum(self.yaw_change_history, axis=0)
                cumulative_penalty = (
                    np.mean(cumulative_changes) / self.env.yaw_max * 0.1
                )
                movement_penalties += cumulative_penalty

        self.previous_yaws = np.array(ppo_yaws)

        current_reward = (1 - self.env_reward_weight) * (
            similarity_reward - movement_penalties / 600
        ) + self.env_reward_weight * reward
        smoothed_reward = (
            self.reward_momentum * self.last_reward
            + (1 - self.reward_momentum) * current_reward
        )
        self.last_reward = smoothed_reward

        info["curriculum_weight"] = self.env_reward_weight
        info["yaw_diff"] = yaw_diff
        info["pywake_yaws"] = self.pywake_yaws
        return obs, smoothed_reward, terminated, truncated, info

    def update_curriculum(self, step):
        self.current_step = step
        self.env_reward_weight = min(
            1.0,
            max(
                0.0,
                (step - self.pure_similarity_steps)
                / (self.curriculum_steps - self.pure_similarity_steps),
            ),
        )
        # self.env_reward_weight = min(1.0, step / self.curriculum_steps)


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps

    def _on_step(self):
        self.training_env.env_method("update_curriculum", self.num_timesteps)
        return True


def make_env(
    seed,
    yaml_path,
    turbbox_path,
    dt_env,
    power_avg,
    curriculum_steps,
    pure_similarity_steps,
):
    def _init():
        env = WindFarmEnv(
            turbine=wind_turbine(),
            yaml_path=yaml_path,
            TurbBox=turbbox_path,
            seed=seed,
            dt_env=dt_env,
            # observation_window_size=args.lookback_window,
            n_passthrough=50,
        )
        return CurriculumWrapper(env, curriculum_steps, pure_similarity_steps)

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Wind farm control with PPO")
    parser.add_argument("--dt_env", type=int, required=True)
    parser.add_argument("--power_avg", type=int, required=True)
    parser.add_argument("--lookback_window", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_env", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument(
        "--turbbox_path",
        type=str,
        default="Hipersim_mann_l5.0_ae1.0000_g0.0_h0_128x128x128_3.000x3.00x3.00_s0001.nc",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # curriculum_steps = 200
    curriculum_steps = args.train_steps // 2
    pure_similarity_steps = args.train_steps // 100

    wandb.init(
        project="WindFarm_Curriculum",
        name=f"curriculum_dt{args.dt_env}_pow{args.power_avg}_seed{args.seed}",
        config=vars(args),
    )

    wandb.run.summary.update(
        {
            "views/training_progress": {
                "version": "v2",
                "panels": [
                    {
                        "panel_type": "line",
                        "title": f"Yaw Angles - Env {i}",
                        "source": "glob",
                        "series": [
                            f"states/yaw_angle_t1-{i}",
                            f"states/yaw_angle_t2-{i}",
                        ],
                    }
                    for i in range(args.n_env)
                ]
                + [
                    {
                        "panel_type": "line",
                        "title": f"Power - Env {i}",
                        "source": "glob",
                        "series": [f"charts/agent_power-{i}", f"charts/base_power-{i}"],
                    }
                    for i in range(args.n_env)
                ],
            }
        }
    )

    # fake env for processing purposes
    temp_env = WindFarmEnv(
        turbine=wind_turbine(),
        yaml_path=args.yaml_path,
        TurbBox=args.turbbox_path,
        seed=args.seed,
        dt_env=args.dt_env,
        # observation_window_size=args.lookback_window
    )
    n_features = temp_env._get_num_raw_features()

    # real env
    env = DummyVecEnv(
        [
            make_env(
                args.seed,
                args.yaml_path,
                args.turbbox_path,
                args.dt_env,
                args.power_avg,
                curriculum_steps,
                pure_similarity_steps,
            )
            for _ in range(args.n_env)
        ]
    )

    model = PPO(
        WindFarmPolicy,
        env,
        n_steps=256,  # Frequent updates
        batch_size=256,  # Smaller updates per batch
        n_epochs=4,  # Reuse data more
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        policy_kwargs={
            "n_features": temp_env._get_num_raw_features(),
            "window_size": args.lookback_window,
        },
        verbose=1,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )
    # model = PPO.load('model/model.zip', env=env)
    callbacks = CallbackList(
        [
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
                model_save_path="model2",
                model_save_freq=10000,
            ),
            WindFarmMonitor(),
            CurriculumCallback(args.train_steps),
        ]
    )

    model.learn(total_timesteps=args.train_steps, callback=callbacks)
    model.save("windfarm_curriculum_model")
