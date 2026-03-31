# two_turbs/create_animation.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import shutil
from pathlib import Path

# --- WindGym Imports ---
from WindGym import WindFarmEnv
from WindGym.Agents import NoisyPyWakeAgent
from WindGym.Measurement_Manager import (
    MeasurementManager,
    NoisyWindFarmEnv,
    MeasurementType,
    EpisodicBiasNoiseModel,
)
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from dynamiks.views import XYView
import matplotlib.patheffects as path_effects

# --- Configuration ---
WIND_DIRECTION = 270.0
BIAS_DEGREES = 0.0
SIM_TIME = 600
OUTPUT_DIR = "biased_pywake_flow_animation"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
ANIMATION_FILENAME = os.path.join(OUTPUT_DIR, "pywake_biased_flow.mp4")
PLOT_FILENAME = os.path.join(OUTPUT_DIR, "pywake_biased_summary_plot.png")

YAML_CONFIG = f"""
yaw_init: "Zeros"
noise: "None"
BaseController: "Local"
ActionMethod: "wind"
farm:
  yaw_min: -1000
  yaw_max: 1000
  nx: 2
  ny: 1
  xDist: 7
  yDist: 7
wind:
  ws_min: 8
  ws_max: 8
  TI_min: 0.07
  TI_max: 0.07
  wd_min: {WIND_DIRECTION}
  wd_max: {WIND_DIRECTION}
act_pen:
  action_penalty: 0.0
power_def:
  Power_reward: "Baseline"
  Power_avg: 10
  Power_scaling: 1.0
mes_level:
  turb_ws: True
  turb_wd: True
  farm_ws: True
  farm_wd: True
  turb_TI: False
  turb_power: True
  farm_TI: False
  farm_power: True
ws_mes: {{ws_current: True, ws_rolling_mean: False, ws_history_N: 0, ws_history_length: 1, ws_window_length: 1}}
wd_mes: {{wd_current: True, wd_rolling_mean: False, wd_history_N: 0, wd_history_length: 1, wd_window_length: 1}}
yaw_mes: {{yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 0, yaw_history_length: 1, yaw_window_length: 1}}
power_mes: {{power_current: True, power_rolling_mean: False, power_history_N: 0, power_history_length: 1, power_window_length: 1}}
"""


def plot_summary_chart(log, output_path):
    """Generates the 4-panel time-series summary plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"PyWakeAgent with {BIAS_DEGREES}° Wind Direction Bias",
        fontsize=16,
        fontweight="bold",
    )

    axs[0].plot(
        log["time"],
        np.array(log["power_agent"]) / 1e6,
        "b-",
        label="PyWake Agent",
        linewidth=2,
    )
    axs[0].plot(
        log["time"],
        np.array(log["power_baseline"]) / 1e6,
        "k:",
        label="Baseline (No Control)",
        linewidth=2,
    )
    axs[0].set_ylabel("Total Power (MW)")
    axs[0].set_title("Power Production")
    axs[0].legend(loc="lower right")

    axs[1].plot(
        log["time"], log["true_wd"], "k-", label="True Wind Direction", linewidth=2
    )
    axs[1].plot(
        log["time"],
        log["sensed_wd"],
        "r--",
        label=f"Sensed by Agent (True + {BIAS_DEGREES}°)",
        linewidth=2,
    )
    axs[1].set_ylabel("Direction (°)")
    axs[1].set_title("Perception: Sensed vs. True Wind Direction")
    axs[1].legend(loc="lower right")

    axs[2].plot(
        log["time"], log["yaw_t0"], "b-", label="Agent Commanded Yaw", linewidth=2
    )
    axs[2].set_ylabel("Yaw Angle (°)")
    axs[2].set_title("Decision: Upstream Turbine (T0) Yaw Angle")

    axs[3].plot(
        log["time"], log["yaw_t1"], "b-", label="Agent Commanded Yaw", linewidth=2
    )
    axs[3].set_ylabel("Yaw Angle (°)")
    axs[3].set_title("Decision: Downstream Turbine (T1) Yaw Angle")
    axs[3].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.label_outer()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """Main function to run the simulation and generate outputs."""
    os.makedirs(FRAME_DIR, exist_ok=True)

    turbine = V80()
    farm_layout = yaml.safe_load(YAML_CONFIG)["farm"]
    x_pos, y_pos = generate_square_grid(
        turbine=turbine,
        nx=farm_layout["nx"],
        ny=farm_layout["ny"],
        xDist=farm_layout["xDist"],
        yDist=farm_layout["yDist"],
    )

    base_env_kwargs = {
        "turbine": turbine,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "config": YAML_CONFIG,
        "Baseline_comp": True,
        "reset_init": False,
        "turbtype": "None",
        "render_mode": "rgb_array",  # Important for capturing frames
    }

    temp_env = WindFarmEnv(**base_env_kwargs, seed=42)
    mm = MeasurementManager(env=temp_env, seed=42)

    bias_model = EpisodicBiasNoiseModel(
        {MeasurementType.WIND_DIRECTION: (BIAS_DEGREES, BIAS_DEGREES)}
    )
    mm.set_noise_model(bias_model)

    env = NoisyWindFarmEnv(
        base_env_class=WindFarmEnv, measurement_manager=mm, **base_env_kwargs, seed=42
    )

    agent = NoisyPyWakeAgent(
        measurement_manager=mm, x_pos=x_pos, y_pos=y_pos, turbine=turbine
    )

    temp_env.close()

    log = {
        "time": [],
        "power_agent": [],
        "power_baseline": [],
        "yaw_t0": [],
        "yaw_t1": [],
        "true_wd": [],
        "sensed_wd": [],
    }

    obs, info = env.reset()
    terminated = truncated = False
    frame_filenames = []

    pbar = tqdm(total=SIM_TIME, desc="Simulating Episode")
    while not (terminated or truncated):
        # Use the environment's render method
        frame = env.render()
        frame_path = os.path.join(FRAME_DIR, f"frame_{len(frame_filenames):04d}.png")

        # imageio expects a path, so we save the frame from memory
        plt.imsave(frame_path, frame)
        frame_filenames.append(frame_path)

        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        current_time = info["time_array"][-1]
        log["time"].append(current_time)
        log["power_agent"].append(info["powers"][-1].sum())
        log["power_baseline"].append(info["baseline_powers"][-1].sum())
        log["yaw_t0"].append(info["yaws"][-1][0])
        log["yaw_t1"].append(info["yaws"][-1][1])
        log["true_wd"].append(info["obs_true/farm/wd_current"])
        log["sensed_wd"].append(info["obs_sensed/farm/wd_current"])

        pbar.update(current_time - (log["time"][-2] if len(log["time"]) > 1 else 0))
        if current_time >= SIM_TIME:
            break

    pbar.close()
    env.close()

    print("Generating summary plot...")
    plot_summary_chart(log, PLOT_FILENAME)

    print("Assembling MP4 animation...")
    with imageio.get_writer(
        ANIMATION_FILENAME, mode="I", fps=10, macro_block_size=8
    ) as writer:
        for filename in tqdm(frame_filenames, desc="Building MP4"):
            image = imageio.imread(filename)
            writer.append_data(image)

    print("Cleaning up temporary frames...")
    shutil.rmtree(FRAME_DIR)

    print(f"\n✅ Success! Outputs saved in '{OUTPUT_DIR}' directory:")
    print(f"   - Animation: {ANIMATION_FILENAME}")
    print(f"   - Summary Plot: {PLOT_FILENAME}")


if __name__ == "__main__":
    main()
