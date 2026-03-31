import os
import glob
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import tyro
from dataclasses import dataclass
from tqdm import tqdm
import gymnasium as gym

# --- Project Imports ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager, NoisyWindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# Custom Utils
from utils import Agent
from noise_definitions import (
    create_adversarial_noise_model,
    create_procedural_noise_model,
    get_adversarial_constraints,
)


@dataclass
class Args:
    models_dir: str = "models/self_play"
    config_path: str = "env_config/two_turbine_yaw.yaml"
    output_filename: str = "figure_noise_comparison.png"
    target_iteration: int = 9  # Specifically looking for Antagonist 9
    sim_time: int = 600
    seed: int = 42


def find_specific_antagonist(models_dir, iteration):
    """Finds the antagonist model for a specific iteration."""
    # Pattern: iter_009_antagonist_*.pt
    pattern = os.path.join(models_dir, f"iter_{iteration:03d}_antagonist_*.pt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No antagonist found for iteration {iteration} in {models_dir}"
        )

    # If multiple (different runs), pick the first one or the latest one
    # Sorting ensures deterministic selection
    files.sort()
    selected = files[-1]
    print(f"Selected Antagonist: {os.path.basename(selected)}")
    return selected


def run_simulation_with_noise(args, noise_model_factory):
    """Runs a sim with a specific noise model and returns the bias history."""

    # 1. Setup Env
    with open(args.config_path, "r") as f:
        config_str = f.read()
        config_data = yaml.safe_load(config_str)

    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine=turbine_obj,
        nx=config_data["farm"]["nx"],
        ny=config_data["farm"]["ny"],
        xDist=config_data["farm"]["xDist"],
        yDist=config_data["farm"]["yDist"],
    )

    base_env_kwargs = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "turbine": turbine_obj,
        "config": config_str,
        "reset_init": True,
        "Baseline_comp": True,
        "dt_sim": 1,
        "dt_env": 10,
        "turbtype": "None",
        "seed": args.seed,
    }

    mm_env = WindFarmEnv(**base_env_kwargs)
    mm = MeasurementManager(mm_env, seed=args.seed)

    # 2. Inject Noise Model
    mm.set_noise_model(noise_model_factory())
    env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)

    # 3. Dummy Agent (Fixed Yaw) - The agent doesn't matter for noise visualization
    # We just want to see what the noise model does to the observations.
    obs, info = env.reset()

    wd_biases = []
    ws_biases = []

    for _ in range(args.sim_time):
        # Action: 0 change (keep yaw steady)
        action = np.zeros(env.action_space.shape)
        obs, _, term, trunc, info = env.step(action)

        # Calculate Bias (Sensed - True) for Turbine 0
        wd_bias = (
            info["obs_sensed/turb_0/wd_current"] - info["obs_true/turb_0/wd_current"]
        )
        ws_bias = (
            info["obs_sensed/turb_0/ws_current"] - info["obs_true/turb_0/ws_current"]
        )

        wd_biases.append(wd_bias)
        ws_biases.append(ws_bias)

        if term or trunc:
            break

    env.close()
    return np.array(wd_biases), np.array(ws_biases)


def main(args: Args):
    device = torch.device("cpu")

    # --- Run 1: Adversarial Noise (Antagonist 9) ---
    ant_path = find_specific_antagonist(args.models_dir, args.target_iteration)

    # Reconstruct Antagonist Network
    # We need to temporarily create an env to get spaces, or hardcode based on config
    # Let's do it properly:
    with open(args.config_path, "r") as f:
        config_data = yaml.safe_load(f)
    n_turb = config_data["farm"]["nx"] * config_data["farm"]["ny"]

    # Create dummy env just to get spaces
    # Note: This is slightly inefficient but safe
    temp_env = WindFarmEnv(
        turbine=V80(),
        x_pos=np.zeros(n_turb),
        y_pos=np.zeros(n_turb),
        config=args.config_path,
    )
    constraints = get_adversarial_constraints()
    ant_act_space = gym.spaces.Box(low=-1, high=1, shape=(n_turb * len(constraints),))

    antagonist_agent = Agent(temp_env.observation_space, ant_act_space, [128, 128]).to(
        device
    )
    antagonist_agent.load_state_dict(torch.load(ant_path, map_location=device))

    def adversarial_factory():
        return create_adversarial_noise_model(antagonist_agent, device)

    print("Running Adversarial Simulation...")
    adv_wd_bias, adv_ws_bias = run_simulation_with_noise(args, adversarial_factory)

    # --- Run 2: Procedural Noise ---
    def procedural_factory():
        return (
            create_procedural_noise_model()
        )  # Uses defaults from noise_definitions.py

    print("Running Procedural Simulation...")
    proc_wd_bias, proc_ws_bias = run_simulation_with_noise(args, procedural_factory)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    time = np.arange(len(adv_wd_bias)) * 10  # 10s steps

    # Plot WD Bias
    axs[0].plot(
        time,
        adv_wd_bias,
        label=f"Adversary {args.target_iteration}",
        color="#D32F2F",
        linewidth=2,
    )
    axs[0].plot(
        time,
        proc_wd_bias,
        label="Procedural Noise",
        color="gray",
        alpha=0.6,
        linewidth=1,
    )
    axs[0].set_ylabel("Wind Dir. Bias ($^\circ$)")
    axs[0].set_title("Comparison of Wind Direction Signal Corruption")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Plot WS Bias
    axs[1].plot(
        time,
        adv_ws_bias,
        label=f"Adversary {args.target_iteration}",
        color="#1976D2",
        linewidth=2,
    )
    axs[1].plot(
        time,
        proc_ws_bias,
        label="Procedural Noise",
        color="gray",
        alpha=0.6,
        linewidth=1,
    )
    axs[1].set_ylabel("Wind Speed Bias (m/s)")
    axs[1].set_title("Comparison of Wind Speed Signal Corruption")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_filename, dpi=300)
    print(f"Saved comparison figure to {args.output_filename}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
