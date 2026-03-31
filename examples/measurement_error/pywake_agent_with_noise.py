"""
===================================================================================
WindGym Coliseum Example: Telling the Full Story of Noisy Control (Unscaled Noise)
===================================================================================

This script demonstrates a robust method for evaluating agent performance by
comparing an "Oracle" agent (with perfect information) to a "Realistic" agent
that must contend with noisy sensor data. The noise characteristics (std. dev.
and bias) are specified in their physical units (m/s, degrees) for clarity.

The final plot is structured as a narrative to show:
1. The noisy signals the agent perceives vs. ground truth.
2. The resulting sub-optimal decisions (yaw angles) the agent makes.
3. The final impact on power production.
"""

import tempfile
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# --- Imports from the WindGym project ---
# Note: Ensure WindGym is correctly installed in your environment.
from WindGym import WindFarmEnv
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.Agents.PyWakeAgent import PyWakeAgent, NoisyPyWakeAgent
from WindGym.Agents.BaseAgent import BaseAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Correctly import all necessary components from the measurement_manager module ---
from WindGym.Measurement_Manager import (
    MeasurementManager,
    MeasurementType,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    HybridNoiseModel,
    NoisyWindFarmEnv,
)

# --- Helper Functions ---


def unscale(scaled_val, min_val, max_val):
    """Converts a value scaled to [-1, 1] back to its original physical range."""
    if (max_val - min_val) == 0:
        return scaled_val
    return (scaled_val + 1) / 2 * (max_val - min_val) + min_val


# --- Environment Wrapper for Plotting ---


class ObsToInfoWrapper(gym.Wrapper):
    """A simple wrapper to add the current observation to the info dict for plotting."""

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["observation"] = obs
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["observation"] = obs
        return obs, info


# --- Main Comparison Script ---


def run_comparison():
    """Main function to run the comparison and generate plots."""

    # 1. --- Configuration ---
    YAML_CONFIG = """
    yaw_init: "Zeros"
    noise: "None"
    BaseController: "Local"
    ActionMethod: "yaw"
    farm:
      yaw_min: -30
      yaw_max: 30
      nx: 2
      ny: 1
      xDist: 7
      yDist: 7
    wind: {ws_min: 6, ws_max: 10, TI_min: 0.02, TI_max: 0.07, wd_min: 250, wd_max: 290}
    act_pen: {action_penalty: 0.001, action_penalty_type: "Change"}
    power_def: {Power_reward: "Baseline", Power_avg: 5, Power_scaling: 1.0}
    mes_level:
      turb_ws: True
      turb_wd: True
      turb_TI: True
      turb_power: True
      farm_ws: True
      farm_wd: True
      farm_TI: False
      farm_power: False
    ws_mes: {ws_current: True, ws_rolling_mean: False, ws_history_N: 1, ws_history_length: 1, ws_window_length: 1}
    wd_mes: {wd_current: True, wd_rolling_mean: False, wd_history_N: 1, wd_history_length: 1, wd_window_length: 1}
    yaw_mes: {yaw_current: True, yaw_rolling_mean: False, yaw_history_N: 1, yaw_history_length: 1, yaw_window_length: 1}
    power_mes: {power_current: True, power_rolling_mean: False, power_history_N: 1, power_history_length: 1, power_window_length: 1}
    """

    config = yaml.safe_load(YAML_CONFIG)
    # Create a temporary YAML file for the environment to load
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(YAML_CONFIG)
        yaml_filepath = tmp_file.name

    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    env_kwargs = {
        "turbine": V80(),
        "x_pos": x_pos,
        "y_pos": y_pos,
        "config": yaml_filepath,
        "Baseline_comp": True,
        "reset_init": False,  # Important: Delay reset until Coliseum does it
    }

    # --- Setup Measurement Manager and Agents ---
    # Create a temporary environment just to initialize the MeasurementManager
    temp_env_kwargs = env_kwargs.copy()
    if "reset_init" in temp_env_kwargs:
        del temp_env_kwargs["reset_init"]
    temp_env_for_manager = WindFarmEnv(reset_init=False, **temp_env_kwargs)
    mm = MeasurementManager(env=temp_env_for_manager)
    temp_env_for_manager.close()  # Clean up the temporary environment

    agents = {
        "NoisyPyWakeAgent": NoisyPyWakeAgent(
            measurement_manager=mm, x_pos=x_pos, y_pos=y_pos
        ),
        "OraclePyWakeAgent": PyWakeAgent(x_pos=x_pos, y_pos=y_pos),
    }

    # --- Run Evaluation in a Noisy Environment ---
    print("\n--- Running Evaluation on NOISY Environment ---")

    # Define noise models with physical units
    white_noise = WhiteNoiseModel(
        {
            MeasurementType.WIND_SPEED: 0.5,  # 0.5 m/s standard deviation
            MeasurementType.WIND_DIRECTION: 2.0,  # 2.0 degrees standard deviation
        }
    )
    episodic_bias = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_SPEED: (-1.0, 1.0),  # Bias between -1.0 and 1.0 m/s
            MeasurementType.WIND_DIRECTION: (
                -10.0,
                10.0,
            ),  # Bias between -10.0 and 10.0 degrees
        }
    )
    mm.set_noise_model(HybridNoiseModel([white_noise, episodic_bias]))

    # Factory to create the wrapped, noisy environment for evaluation
    def env_factory_noisy():
        return ObsToInfoWrapper(
            NoisyWindFarmEnv(
                base_env_class=WindFarmEnv, measurement_manager=mm, **env_kwargs
            )
        )

    coliseum = Coliseum(env_factory=env_factory_noisy, agents=agents, n_passthrough=6)

    num_episodes_to_run = 2
    coliseum.run_time_series_evaluation(
        num_episodes=num_episodes_to_run, seed=69, save_detailed_history=True
    )
    print("Evaluation complete.")

    print(f"\n--- Generating {num_episodes_to_run} Comparison Plots ---")
    for episode_idx in range(num_episodes_to_run):
        print(f"Plotting results for episode {episode_idx}...")

        # Access the detailed history for the current episode
        history_noisy_agent = coliseum.time_series_results["NoisyPyWakeAgent"][
            episode_idx
        ]["history"]
        history_oracle_agent = coliseum.time_series_results["OraclePyWakeAgent"][
            episode_idx
        ]["history"]

        # The bias is constant for the episode, so we get it from the first step's info dict.
        first_step_info = history_noisy_agent["info"][0]
        noise_info = first_step_info.get("noise_info", {})

        # Find the episodic bias component from the hybrid model's list of components.
        bias_component = {}
        for component in noise_info.get("component_models", []):
            if component.get("noise_type") == "episodic_bias":
                bias_component = component
                break

        applied_biases = bias_component.get("applied_bias (physical_units)", {})
        ws_bias_str = f"{applied_biases.get('turb_0/ws_current', 0.0):.2f}"
        wd_bias_str = f"{applied_biases.get('turb_0/wd_current', 0.0):.2f}"

        time_steps = history_noisy_agent["step"]

        # --- Extract data for all plots ---
        power_noisy_agent = [
            info["Power agent"] for info in history_noisy_agent["info"]
        ]
        power_oracle_agent = [
            info["Power agent"] for info in history_oracle_agent["info"]
        ]
        yaw_noisy_agent = [
            info["yaw angles agent"][0] for info in history_noisy_agent["info"]
        ]
        yaw_oracle_agent = [
            info["yaw angles agent"][0] for info in history_oracle_agent["info"]
        ]

        # Ground truth physical values from the oracle's run
        gt_ws_ms = [
            info["Wind speed at turbines"][0] for info in history_oracle_agent["info"]
        ]
        gt_wd_deg = [
            info["Wind direction at turbines"][0]
            for info in history_oracle_agent["info"]
        ]

        # --- ROBUST METHOD for extracting and unscaling noisy values ---
        noisy_obs_vector = np.array(
            [info.get("observation") for info in history_noisy_agent["info"]]
        )

        # Find the full specification for the measurements we want to plot
        ws_spec = next((s for s in mm.specs if s.name == "turb_0/ws_current"), None)
        wd_spec = next((s for s in mm.specs if s.name == "turb_0/wd_current"), None)

        if ws_spec is None or wd_spec is None:
            raise ValueError(
                "Could not find required measurement specifications in MeasurementManager."
            )

        # Extract the index and scaling parameters from the spec objects
        ws_idx = ws_spec.index_range[0]
        wd_idx = wd_spec.index_range[0]

        # Use the min/max values from the spec for robust unscaling
        noisy_ws_ms = unscale(
            noisy_obs_vector[:, ws_idx], ws_spec.min_val, ws_spec.max_val
        )
        noisy_wd_deg = unscale(
            noisy_obs_vector[:, wd_idx], wd_spec.min_val, wd_spec.max_val
        )
        # --- End of robust unscaling ---

        # --- Create the 4-panel plot ---
        fig, axs = plt.subplots(4, 1, figsize=(14, 22), sharex=True)

        actual_wind_speed = history_oracle_agent["info"][0]["Wind speed Global"]
        actual_wind_direction = history_oracle_agent["info"][0]["Wind direction Global"]
        title_text = (
            f"Episode {episode_idx} - Global Inflow: {actual_wind_speed:.2f} m/s, {actual_wind_direction:.2f}°\n"
            f"Episodic Bias: Wind Speed = {ws_bias_str} m/s, Wind Direction = {wd_bias_str}°"
        )
        fig.suptitle(title_text, fontsize=16)

        # Panel 1: Wind Speed Perception
        axs[0].plot(time_steps, gt_ws_ms, "b-", label="Ground Truth", lw=2)
        axs[0].plot(
            time_steps,
            noisy_ws_ms,
            "r--",
            label="Agent's Noisy Perception",
            lw=1.5,
            alpha=0.9,
        )
        axs[0].set_ylabel("Wind Speed (m/s)")
        axs[0].set_title("1. Perception: Wind Speed Measurement at Turbine 0")
        axs[0].legend()
        axs[0].grid(True, alpha=0.5)

        # Panel 2: Wind Direction Perception
        axs[1].plot(time_steps, gt_wd_deg, "b-", label="Ground Truth", lw=2)
        axs[1].plot(
            time_steps,
            noisy_wd_deg,
            "r--",
            label="Agent's Noisy Perception",
            lw=1.5,
            alpha=0.9,
        )
        axs[1].set_ylabel("Wind Direction (deg)")
        axs[1].set_title("2. Perception: Wind Direction Measurement at Turbine 0")
        axs[1].legend()
        axs[1].grid(True, alpha=0.5)

        # Panel 3: Control Decision (Yaw)
        axs[2].plot(
            time_steps,
            yaw_oracle_agent,
            "g-",
            label="Decision with Perfect Info (Oracle)",
            lw=2,
        )
        axs[2].plot(
            time_steps,
            yaw_noisy_agent,
            "r--",
            label="Decision with Noisy Info (Noisy Agent)",
            lw=2,
        )
        axs[2].set_ylabel("Turbine 0 Yaw Angle (degrees)")
        axs[2].set_title("3. Decision: Control Action Taken by Agents")
        axs[2].legend()
        axs[2].grid(True, alpha=0.5)

        # Panel 4: Final Outcome (Power)
        axs[3].plot(
            time_steps,
            power_oracle_agent,
            "g-",
            label="Power from Oracle's Actions",
            lw=2,
        )
        axs[3].plot(
            time_steps,
            power_noisy_agent,
            "r--",
            label="Power from Noisy Agent's Actions",
            lw=2,
        )
        axs[3].set_xlabel("Time Step")
        axs[3].set_ylabel("Total Farm Power (W)")
        axs[3].set_title("4. Outcome: Resulting Power Production")
        axs[3].legend()
        axs[3].grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_filename = (
            f"full_story_comparison_episode_{episode_idx}_unscaled_noise.png"
        )
        plt.savefig(output_filename)
        print(f"Comparison plot saved to '{output_filename}'")
        plt.close(fig)

    # Clean up the temporary YAML file
    os.remove(yaml_filepath)


if __name__ == "__main__":
    # Note: To run this script, you must have the WindGym library and its
    # dependencies (gymnasium, numpy, py_wake, etc.) installed in your
    # Python environment. The provided library files must also be in the
    # correct package structure (e.g., WindGym/Agents/, WindGym/utils/, etc.).
    run_comparison()
