# filename: utils/model_loader.py

import torch
from stable_baselines3 import PPO
import os

# Your custom Agent class from train_self_play.py
# For this to work, you might need to move the Agent class to its own file
# or ensure it's accessible from this utility. Let's assume it's imported.
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes a linear layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """A PPO Agent class for the self-play environment."""

    def __init__(self, obs_space, act_space, net_arch):
        super().__init__()
        obs_shape = np.array(obs_space.shape).prod()
        act_shape = np.prod(act_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, net_arch[0])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[0], net_arch[1])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[1], 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, net_arch[0])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[0], net_arch[1])),
            nn.Tanh(),
            layer_init(nn.Linear(net_arch[1], act_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def load_sb3_weights_into_custom_agent(
    sb3_model_path: str, custom_agent: Agent, device
):
    """
    Loads weights from a saved Stable Baselines 3 PPO model (.zip)
    into our custom PyTorch Agent class.
    """
    print(f"INFO: Loading SB3 policy from '{os.path.basename(sb3_model_path)}'...")
    sb3_model = PPO.load(sb3_model_path, device=device)

    sb3_state_dict = sb3_model.policy.state_dict()
    custom_state_dict = custom_agent.state_dict()

    mapping = {
        "mlp_extractor.policy_net.": "actor_mean.",
        "mlp_extractor.value_net.": "critic.",
        "action_net.": "actor_mean.4.",
        "value_net.": "critic.4.",
    }

    new_state_dict = {}
    for sb3_key, sb3_tensor in sb3_state_dict.items():
        found_match = False
        for sb3_prefix, custom_prefix in mapping.items():
            if sb3_key.startswith(sb3_prefix):
                custom_key = sb3_key.replace(sb3_prefix, custom_prefix)
                if (
                    custom_key in custom_state_dict
                    and custom_state_dict[custom_key].shape == sb3_tensor.shape
                ):
                    new_state_dict[custom_key] = sb3_tensor
                    found_match = True
                    break

    custom_agent.load_state_dict(new_state_dict, strict=False)

    if "log_std" in sb3_state_dict:
        custom_agent.actor_logstd.data.copy_(sb3_state_dict["log_std"])

    print("INFO: Successfully transferred weights from SB3 model to custom agent.")
    return custom_agent
