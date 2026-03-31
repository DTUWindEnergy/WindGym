import os
import glob
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
from WindGym.Agents import NoisyPyWakeAgent
from py_wake.examples.data.hornsrev1 import V80

# --- Custom Imports ---
from utils import Agent
from noise_definitions import (
    create_adversarial_noise_model,
    create_procedural_noise_model,
    get_adversarial_constraints,
)


@dataclass
class Args:
    # --- Mode Selection ---
    mode: str

    # --- Paths ---
    protagonist_path: str = ""
    antagonist_path: str = ""

    # --- Configuration ---
    config_path: str = "env_config/two_turbine_yaw.yaml"
    output_filename: str = "scenario_plot.png"

    # Simulation Params (MATCHING GAUNTLET)
    sim_time: int = 200  # Increased to 2000 to match Gauntlet
    seed: int = 70  # Seed 69 matches the Gauntlet start seed

    # Fixed Evaluation Conditions - REMOVED to allow Env to handle consistency


def load_rl_agent(path, obs_space, act_space, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    if path.endswith(".zip"):
        from stable_baselines3 import PPO

        print(f"Loading SB3 PPO Agent from: {os.path.basename(path)}")
        return PPO.load(path, device=device)
    else:
        print(f"Loading Custom RL Agent from: {os.path.basename(path)}")
        agent = Agent(obs_space, act_space, [128, 128]).to(device)
        agent.load_state_dict(torch.load(path, map_location=device))
        agent.eval()
        return agent


def main(args: Args):
    device = torch.device("cpu")

    # --- 1. Initialize Base Environment ---
    with open(args.config_path, "r") as f:
        config_str = f.read()
    config_data = yaml.safe_load(config_str)

    # DO NOT FORCE WIND CONDITIONS HERE. Let the seed handle it.

    if "BaseController" not in config_data:
        config_data["BaseController"] = "Local"

    turbine_obj = V80()
    x_pos, y_pos = generate_square_grid(
        turbine_obj,
        config_data["farm"]["nx"],
        config_data["farm"]["ny"],
        config_data["farm"]["xDist"],
        config_data["farm"]["yDist"],
    )

    base_env_kwargs = {
        "x_pos": x_pos,
        "y_pos": y_pos,
        "turbine": turbine_obj,
        "config": config_data,
        "reset_init": True,
        "Baseline_comp": True,
        "dt_sim": 5,
        "dt_env": 10,
        "turbtype": "None",
        "seed": args.seed,
        "n_passthrough": 500,
        "burn_in_passthroughs": 2,
    }

    mm_env = WindFarmEnv(**base_env_kwargs)
    mm = MeasurementManager(mm_env, seed=args.seed)

    # --- 2. Configure Noise Model ---
    is_adversarial = "adv" in args.mode

    is_clean = "clean" in args.mode

    if is_clean:
        print("--- Scenario: Clean (No Noise) ---")
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)
    elif is_adversarial:
        if not args.antagonist_path:
            raise ValueError(f"Mode '{args.mode}' requires --antagonist-path")
        print(
            f"--- Scenario: Adversarial ({os.path.basename(args.antagonist_path)}) ---"
        )

        constraints = get_adversarial_constraints()
        n_act_ant = len(x_pos) * len(constraints)
        ant_act_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_act_ant,), dtype=np.float32
        )

        antagonist_agent = load_rl_agent(
            args.antagonist_path, mm_env.observation_space, ant_act_space, device
        )
        noise_model = create_adversarial_noise_model(antagonist_agent, device)
        mm.set_noise_model(noise_model)
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)
    else:
        print("--- Scenario: Procedural Noise ---")
        noise_model = create_procedural_noise_model()
        mm.set_noise_model(noise_model)
        env = NoisyWindFarmEnv(WindFarmEnv, mm, **base_env_kwargs)

    # --- 3. Configure Protagonist ---
    is_pywake = "pywake" in args.mode

    if is_pywake:
        print("--- Agent: PyWake (Baseline) ---")
        # Initialize NoisyPyWakeAgent. It will use the env's noise implicitly.
        protagonist_agent = NoisyPyWakeAgent(
            measurement_manager=mm,
            x_pos=x_pos,
            y_pos=y_pos,
            turbine=turbine_obj,
            look_up=False,
        )
    else:
        if not args.protagonist_path:
            raise ValueError(f"Mode '{args.mode}' requires --protagonist-path")
        print(f"--- Agent: Trained RL ({os.path.basename(args.protagonist_path)}) ---")
        protagonist_agent = load_rl_agent(
            args.protagonist_path, env.observation_space, env.action_space, device
        )

    # --- 4. Run Simulation ---
    obs, info = env.reset()

    # Log initial true conditions
    print(f"--- Simulation Conditions (Seed {args.seed}) ---")
    print(f"Wind Speed: {env.unwrapped.ws:.2f} m/s")
    print(f"Wind Dir:   {env.unwrapped.wd:.2f} deg")

    # CRITICAL: Do NOT manually force wind on the agent or env here.
    # NoisyPyWakeAgent automatically estimates wind from 'obs' in its predict() method.

    # Data logging
    history = {
        "time": [],
        "power_agent": [],
        "power_baseline": [],
        "t0": {
            "wd_t": [],
            "wd_s": [],
            "ws_t": [],
            "ws_s": [],
            "pow_t": [],
            "pow_s": [],
            "yaw_t": [],
            "yaw_s": [],
        },
        "t1": {
            "wd_t": [],
            "wd_s": [],
            "ws_t": [],
            "ws_s": [],
            "pow_t": [],
            "pow_s": [],
            "yaw_t": [],
            "yaw_s": [],
        },
    }

    # Debug Check
    wd_true_start = info["obs_true/turb_0/wd_current"]
    wd_sensed_start = info["obs_sensed/turb_0/wd_current"]
    print(f"DEBUG CHECK: Bias at Step 0: {wd_sensed_start - wd_true_start:.2f}")

    for _ in tqdm(range(args.sim_time), desc="Simulating"):
        if is_pywake:
            action, _ = protagonist_agent.predict(obs, deterministic=True)
        else:
            if hasattr(protagonist_agent, "predict"):
                action, _ = protagonist_agent.predict(obs, deterministic=True)
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = protagonist_agent.actor_mean(obs_tensor).squeeze().numpy()

        phys_yaw = info.get("yaw angles agent", np.zeros(2))

        obs, _, term, trunc, info = env.step(action)

        # --- LOGGING ---
        history["time"].append(info["time_array"][-1])
        history["power_agent"].append(info["powers"][-1].sum())
        history["power_baseline"].append(info["baseline_powers"][-1].sum())

        for i, key in enumerate(["t0", "t1"]):
            history[key]["wd_t"].append(info[f"obs_true/turb_{i}/wd_current"])
            history[key]["wd_s"].append(info[f"obs_sensed/turb_{i}/wd_current"])
            history[key]["ws_t"].append(info[f"obs_true/turb_{i}/ws_current"])
            history[key]["ws_s"].append(info[f"obs_sensed/turb_{i}/ws_current"])
            history[key]["pow_t"].append(info["Power pr turbine agent"][i])
            history[key]["pow_s"].append(info[f"obs_sensed/turb_{i}/power_current"])
            history[key]["yaw_t"].append(phys_yaw[i])
            history[key]["yaw_s"].append(info[f"obs_sensed/turb_{i}/yaw_current"])

        if term or trunc:
            break
    env.close()

    # --- 5. Analysis ---
    mean_power_agent_mw = np.mean(history["power_agent"]) / 1e6
    mean_power_base_mw = np.mean(history["power_baseline"]) / 1e6

    gain_percent = 0.0
    if mean_power_base_mw > 0:
        gain_percent = ((mean_power_agent_mw / mean_power_base_mw) - 1) * 100

    print("\n" + "=" * 40)
    print(f"EVALUATION RESULTS ({args.sim_time} steps)")
    print(f"   Mean Agent Power:    {mean_power_agent_mw:.4f} MW")
    print(f"   Mean Baseline Power: {mean_power_base_mw:.4f} MW")
    print(f"   Power Gain vs Base: {gain_percent:+.2f}%")
    print("=" * 40 + "\n")

    # --- 6. Plotting (4x2 Matrix Style) ---
    constraints = get_adversarial_constraints()
    fig, axs = plt.subplots(4, 2, figsize=(7, 8), sharex=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.3)

    time = np.array(history["time"])
    cols = ["t0", "t1"]
    titles = ["Turbine 0 (Upstream)", "Turbine 1 (Downstream)"]
    SOLID, DASHED = "black", "#D32F2F"

    for c, t_key in enumerate(cols):
        d = history[t_key]

        # Row 0: WD
        limit = constraints["max_bias_wd"]
        true_sig = np.array(d["wd_t"])
        axs[0, c].fill_between(
            time,
            true_sig - limit,
            true_sig + limit,
            color="gray",
            alpha=0.2,
            label="Allowed Bounds",
        )
        axs[0, c].plot(time, true_sig, color=SOLID, label="Physical Reality")
        axs[0, c].plot(time, d["wd_s"], color=DASHED, ls="--", label="Sensed (Noisy)")
        axs[0, c].set_ylabel("Wind Direction ($^\circ$)")

        # Row 1: WS
        limit = constraints["max_bias_ws"]
        true_sig = np.array(d["ws_t"])
        axs[1, c].fill_between(
            time, true_sig - limit, true_sig + limit, color="gray", alpha=0.2
        )
        axs[1, c].plot(time, true_sig, color=SOLID)
        axs[1, c].plot(time, d["ws_s"], color=DASHED, ls="--")
        axs[1, c].set_ylabel("Wind Speed (m/s)")

        # Row 2: Power
        limit = constraints["max_bias_power"] / 1e6
        true_sig = np.array(d["pow_t"]) / 1e6
        axs[2, c].fill_between(
            time, true_sig - limit, true_sig + limit, color="gray", alpha=0.2
        )
        axs[2, c].plot(time, true_sig, color=SOLID)
        axs[2, c].plot(time, np.array(d["pow_s"]) / 1e6, color=DASHED, ls="--")
        axs[2, c].set_ylabel("Power (MW)")

        # Row 3: Yaw
        true_sig = np.array(d["yaw_t"])
        limit_combined = constraints["max_bias_yaw"] + constraints["max_bias_wd"]
        axs[3, c].fill_between(
            time,
            true_sig - limit_combined,
            true_sig + limit_combined,
            color="gray",
            alpha=0.2,
            label="Coupled Limit",
        )

        axs[3, c].plot(time, true_sig, color=SOLID, label="Physical")
        axs[3, c].plot(time, d["yaw_s"], color=DASHED, ls="--", label="Sensed")
        axs[3, c].set_ylabel("Yaw ($^\circ$)")
        axs[3, c].set_xlabel("Time (s)")

        axs[0, c].set_title(titles[c], fontsize=11, fontweight="bold")
        for r in range(4):
            axs[r, c].grid(True, alpha=0.3)

    # Legends
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles, labels, loc="upper right", fontsize="x-small", frameon=True
    )
    h2, l2 = axs[3, 0].get_legend_handles_labels()
    axs[3, 0].legend(h2, l2, loc="lower right", fontsize="x-small", frameon=True)

    title_text = (
        f"\nGain: {gain_percent:+.2f}% | Mean Power: {mean_power_agent_mw:.2f} MW"
    )

    plt.suptitle(title_text, y=0.96, fontsize=13)
    plt.savefig(args.output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved detailed plot to {args.output_filename}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
