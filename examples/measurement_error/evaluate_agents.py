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

from WindGym import WindFarmEnv
from WindGym.Agents import PyWakeAgent, NoisyPyWakeAgent
from WindGym.Measurement_Manager import MeasurementManager, NoisyWindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from noise_definitions import (
    create_procedural_noise_model,
    create_adversarial_noise_model,
)

DETERMINISTIC = True


@dataclass
class Args:
    agent_type: str
    scenario: str
    output_path: str
    protagonist_path: str = ""
    antagonist_path: str = ""
    sim_time: int = 1000
    seed: int = 42
    config_path: str = "env_config/two_turbine_yaw.yaml"


def main(args: Args):
    print(
        f"--- Evaluating Agent '{args.agent_type.upper()}' in Scenario: {args.scenario.upper()} ---"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "n_passthrough": 10,
        "burn_in_passthroughs": 2.0,
        "fill_window": True,
        "dt_sim": 1,
        "dt_env": 10,
        "turbtype": "None",
        "seed": args.seed,
    }

    mm_env = WindFarmEnv(**base_env_kwargs)
    mm = MeasurementManager(mm_env, seed=args.seed)
    env = None
    protagonist_agent = None

    if args.scenario == "clean":
        env = mm_env
        if args.agent_type.lower() == "pywake":
            protagonist_agent = PyWakeAgent(
                x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj, env=env
            )

    elif args.scenario == "procedural":
        print("Applying procedural noise...")
        mm.set_noise_model(create_procedural_noise_model())
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)
        if args.agent_type.lower() == "pywake":
            protagonist_agent = NoisyPyWakeAgent(
                measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj
            )

    elif args.scenario == "adversarial":
        print("Applying adversarial noise...")
        if not args.antagonist_path:
            raise ValueError("--antagonist-path is required for adversarial scenario.")

        antagonist_model = PPO.load(args.antagonist_path, device=device)
        noise_model = create_adversarial_noise_model(antagonist_model, device)
        mm.set_noise_model(noise_model)

        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)
        if args.agent_type.lower() == "pywake":
            protagonist_agent = NoisyPyWakeAgent(
                measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine_obj
            )

    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    if args.agent_type.lower() == "ppo":
        protagonist_agent = PPO.load(args.protagonist_path, device=device)

    if protagonist_agent is None:
        raise ValueError(
            f"Protagonist agent could not be created for agent_type: {args.agent_type}"
        )

    log = []
    obs, info = env.reset()
    if isinstance(protagonist_agent, PyWakeAgent) and not isinstance(
        protagonist_agent, NoisyPyWakeAgent
    ):
        protagonist_agent.update_wind(env.ws, env.wd, env.ti)

    terminated = truncated = False
    with tqdm(total=args.sim_time, desc="Simulating") as pbar:
        last_time = 0
        while not (terminated or truncated):
            action, _ = protagonist_agent.predict(obs, deterministic=True)
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
    print(f"\n✅ Time series data saved to '{args.output_path}'")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
