import os
import yaml
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tyro
from stable_baselines3 import PPO
from tqdm import tqdm

# --- Important Imports from your project ---
from WindGym import WindFarmEnv
from WindGym.Measurement_Manager import MeasurementManager, NoisyWindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from WindGym.Agents import PyWakeAgent, NoisyPyWakeAgent

# Import the custom Agent and noise definitions
from utils import Agent, load_sb3_weights_into_custom_agent
from noise_definitions import (
    create_procedural_noise_model,
    AdversarialNoiseModel,
    get_adversarial_constraints,
)


@dataclass
class Args:
    """Arguments for self-play model evaluation"""

    scenario: str
    output_path: str
    protagonist_path: str  # Always required for this script
    antagonist_path: str = ""  # Required for 'adversarial' scenario
    protagonist_is_sb3: bool = False  # Load protagonist from SB3 .zip format
    antagonist_is_sb3: bool = False  # Load antagonist from SB3 .zip format
    random_antagonist: bool = False  # Use random-initialized antagonist
    sim_time: int = 2000
    seed: int = 42
    config_path: str = "env_config/two_turbine_yaw.yaml"


def load_custom_agent(model_path: str, obs_space, act_space, device) -> Agent:
    """Helper function to load a .pt model into the Agent class."""
    net_arch = [128, 128]
    agent = Agent(obs_space, act_space, net_arch).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()  # Set model to evaluation mode
    return agent


def main(args: Args):
    print(
        f"--- Evaluating Self-Play Protagonist in Scenario: {args.scenario.upper()} ---"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Base Environment & Helper
    with open(args.config_path, "r") as f:
        config_data = yaml.safe_load(f)
    YAML_CONFIG_STR = open(args.config_path, "r").read()

    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine=turbine_obj,
        nx=config_data["farm"]["nx"],
        ny=config_data["farm"]["ny"],
        xDist=config_data["farm"].get("xDist", 7),
        yDist=config_data["farm"].get("yDist", 7),
    )
    base_env_kwargs = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "turbine": turbine_obj,
        "config": YAML_CONFIG_STR,
        "reset_init": True,
        "Baseline_comp": True,
        "dt_sim": 5,
        "dt_env": 10,
        "turbtype": "None",
        "seed": args.seed,
    }

    mm_env = WindFarmEnv(**base_env_kwargs)
    mm = MeasurementManager(mm_env, seed=args.seed)

    # Get spaces for loading Neural Networks (if needed)
    protagonist_obs_space = mm_env.observation_space
    protagonist_act_space = mm_env.action_space

    # 2. SETUP SCENARIO & NOISE (Must happen BEFORE Agent Init)
    env = None

    if args.scenario == "clean":
        env = mm_env

    elif args.scenario == "procedural":
        print("Applying procedural noise...")
        mm.set_noise_model(create_procedural_noise_model())
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)

    elif args.scenario == "adversarial":
        # Antagonist needs to be loaded now to create the noise model
        n_constraints = len(get_adversarial_constraints())
        antagonist_act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(mm_env.n_turb * n_constraints,), dtype=np.float32
        )

        if args.random_antagonist:
            print("Using random-initialized antagonist...")
            net_arch = [128, 128]
            antagonist_agent = Agent(
                protagonist_obs_space, antagonist_act_space, net_arch
            ).to(device)
            antagonist_agent.eval()
        else:
            if not args.antagonist_path:
                raise ValueError(
                    "--antagonist-path is required for adversarial scenario (unless --random-antagonist is set)."
                )
            print(f"Loading antagonist from: {args.antagonist_path}")
            if args.antagonist_is_sb3:
                antagonist_agent = PPO.load(args.antagonist_path, device=device)
            else:
                antagonist_agent = load_custom_agent(
                    args.antagonist_path,
                    protagonist_obs_space,
                    antagonist_act_space,
                    device,
                )

        noise_model = AdversarialNoiseModel(
            antagonist_agent=antagonist_agent,
            constraints=get_adversarial_constraints(),
            device=device,
        )
        mm.set_noise_model(noise_model)
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)

    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    # 3. LOAD PROTAGONIST AGENT (Now safe to init PyWake)
    if args.protagonist_path.lower() == "pywake":
        print("Initializing PyWake Agent (Baseline)...")
        if args.scenario == "clean":
            protagonist_agent = PyWakeAgent(
                x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj, env=mm_env
            )
        else:
            # NOW this works correctly because mm.set_noise_model() has already run!
            protagonist_agent = NoisyPyWakeAgent(
                measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj
            )
    elif args.protagonist_is_sb3:
        print(f"Loading SB3 protagonist from: {args.protagonist_path}")
        net_arch = [128, 128]
        protagonist_agent = Agent(
            protagonist_obs_space, protagonist_act_space, net_arch
        ).to(device)
        protagonist_agent = load_sb3_weights_into_custom_agent(
            args.protagonist_path, protagonist_agent, device
        )
        protagonist_agent.eval()
    else:
        print(f"Loading protagonist from: {args.protagonist_path}")
        protagonist_agent = load_custom_agent(
            args.protagonist_path, protagonist_obs_space, protagonist_act_space, device
        )

    # 4. SIMULATION LOOP
    log = []
    obs, info = env.reset()
    terminated = truncated = False

    # The environment randomizes wind conditions on reset.
    # We must update the PyWakeAgent (Clean version) AFTER reset so it sees the correct wind.
    if isinstance(protagonist_agent, PyWakeAgent) and not isinstance(
        protagonist_agent, NoisyPyWakeAgent
    ):
        # For 'Clean' scenario only:
        protagonist_agent.update_wind(env.ws, env.wd, env.ti)

    with tqdm(total=args.sim_time, desc="Simulating") as pbar:
        last_time = 0
        while not (terminated or truncated):
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)

            if isinstance(protagonist_agent, (PyWakeAgent, NoisyPyWakeAgent)):
                # PyWake expects numpy array and returns (action, state)
                action, _ = protagonist_agent.predict(obs, deterministic=True)
            else:
                # Neural Network expects Tensor and we grab the actor_mean
                obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)

                with torch.no_grad():
                    action_tensor = protagonist_agent.actor_mean(obs_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

            obs, _, terminated, truncated, info = env.step(action)

            current_time = info["time_array"][-1]
            for i in range(len(info["time_array"])):
                log.append(
                    {
                        "time": info["time_array"][i],
                        "power_agent": info["powers"][i].sum(),
                        "power_baseline": info["baseline_powers"][i].sum(),
                    }
                )

            pbar.update(current_time - last_time)
            last_time = current_time
            if current_time >= args.sim_time:
                break

    env.close()
    mm_env.close()

    log_df = pd.DataFrame(log)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    log_df.to_csv(args.output_path, index=False)
    print(f"\nTime series data saved to '{args.output_path}'")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
