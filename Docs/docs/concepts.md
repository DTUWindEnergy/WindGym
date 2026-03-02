# Core Concepts

This section introduces the fundamental concepts and components of the WindGym environment. Understanding these concepts is crucial for effectively configuring simulations, developing agents, and interpreting results.

---

## 1. The Wind Farm Environment (`WindFarmEnv`)

The `WindFarmEnv` is the primary Gymnasium (formerly Gym) environment in WindGym. It simulates a wind farm and provides the interface for agents to interact with it. Key aspects include:

- **Turbines**: WindGym supports various wind turbine models, from simplified PyWake models to high-fidelity HAWC2 models for more detailed load and power calculations.
- **Farm Layout**: You define the `x_pos` and `y_pos` for each turbine, allowing for flexible wind farm configurations (e.g., inline, staggered, circular).
- **Wind Conditions**:
  - **Wind Speed (`ws`)**: The ambient wind speed entering the farm.
  - **Wind Direction (`wd`)**: The direction from which the wind is blowing (meteorological convention, 0° is North, 270° is West).
  - **Turbulence Intensity (`TI`)**: A measure of the wind's turbulence.
  - **Turbulence Box (`TurbBox`, `turbtype`)**: WindGym can use pre-generated or dynamically generated Mann turbulence boxes, or simpler random turbulence models, to simulate realistic wind flow.
  - **`sample_site`**: An optional PyWake `Site` object that allows for sampling realistic wind conditions (wind speed, direction) based on actual wind resource distributions (e.g., Weibull for speed, frequency for direction). If not provided, uniform random sampling within defined bounds is used.
- **Simulation Timesteps (`dt_sim`, `dt_env`)**:
  - `dt_sim`: The internal timestep of the underlying flow simulation (Dynamiks DWM).
  - `dt_env`: The environment timestep, which is the frequency at which the agent receives observations and takes actions. `dt_env` must be a multiple of `dt_sim`.
- **Flow Passthroughs (`n_passthrough`, `burn_in_passthroughs`)**:
  - `n_passthrough`: Determines the total duration of an episode by specifying how many times the wind flow is expected to "pass through" the wind farm. This helps in defining a relevant episode length based on the physical scale of the farm and wind speed.
  - `burn_in_passthroughs`: Specifies an initial period, in terms of flow passthroughs, during which the simulation runs to allow the flow field to develop and stabilize before the actual episode starts. This helps to ensure that initial transient effects do not heavily influence agent learning.
- **Baseline Comparison (`Baseline_comp`)**: Enables a parallel "baseline" wind farm simulation (e.g., with fixed yaw angles) for direct comparison with the agent's performance.

---

## 2. Observations and Measurement System (`MesClass`)

The `MesClass` module handles all measurements within the WindGym environment. It provides a flexible way to configure what data an agent observes.

- **`Mes`**: A base class for individual sensor types, managing a `deque` (double-ended queue) to store historical measurements. It allows configuring:
  - `current`: Whether to include the most recent measurement.
  - `rolling_mean`: Whether to include rolling averages.
  - `history_N`: Number of rolling windows to use for the mean.
  - `history_length`: Total number of measurements to save in the history.
  - `window_length`: Size of the rolling average window.
- **`turb_mes`**: Collects measurements for a **single turbine**, including wind speed (`ws`), wind direction (`wd`), yaw angle (`yaw`), and power (`power`). It can also calculate Turbulence Intensity (`TI`) from high-frequency wind speed data.
- **`farm_mes`**: Aggregates measurements for the **entire wind farm**. It contains a list of `turb_mes` objects (one for each turbine) and an additional `Mes` object for farm-level aggregated data (e.g., average wind speed, total power).
- **Configurable Observations**: The `yaml_path` in the environment's initialization points to a YAML file where you define which measurements (turbine-level, farm-level, current, rolling mean, TI) are included in the observation space. This allows you to tailor the observation space to your agent's needs. All observations are scaled to be between -1 and 1.

---

## 3. Actions and Control

Agents interact with the environment by providing actions that modify the wind turbine yaw angles.

![Action Methods Visualization](actions.png)

- **Yaw Angle Control**: The primary action an agent takes is to adjust the yaw angle of each turbine.
- **`ActionMethod`**: Configurable in the YAML file, this determines how the agent's action translates into a yaw change:
  - `"yaw"`: The action represents a desired _change_ in yaw angle. The environment limits the rate of change per `dt_sim`.
  - `"wind"`: The action represents a _target yaw angle_ (relative to the global wind direction). The environment then adjusts the turbine's yaw towards this target, respecting the maximum yaw rate.
  - `"absolute"`: (Currently not implemented) The yaw angles are direclt based on the global coordinate system.
- **`yaw_step_sim` / `yaw_step_env`**: Defines the maximum rate at which turbine yaw angles can change.
  - `yaw_step_sim`: The maximum change in yaw angle allowed per internal simulation step (`dt_sim`).
  - `yaw_step_env`: The maximum change in yaw angle allowed per environment step (`dt_env`). If `yaw_step_env` is `None`, it's calculated based on `yaw_step_sim` and the ratio of `dt_env` to `dt_sim`.
- **`yaw_min` / `yaw_max`**: Define the permissible range for yaw angles (e.g., -45° to 45°).

---

## 4. Rewards

The reward function guides the agent's learning process. WindGym computes rewards as:

`total_reward = (power_reward × Power_scaling) − action_penalty`

WindGym supports five power reward types: **Baseline**, **Wake_recovery**, **Power_avg**, **Power_diff**, and **None**. Each uses a different formula to quantify how well the agent is controlling the farm. An optional action penalty can be added to discourage excessive yaw adjustments.

For detailed formulas, configuration options, and guidance on choosing a reward function, see the dedicated **[Rewards](rewards.md)** page.

---

## 5. Agents

Agents are the intelligent entities that interact with the WindGym environment.

- **`BaseAgent`**: Provides a common interface for all agents, requiring a `predict()` method to generate actions and helper functions for scaling yaw angles.
- **`PyWakeAgent`**: A baseline agent that uses the PyWake library's internal optimization capabilities to calculate optimal static yaw angles for a given wind condition and applies them. This is often used as a strong baseline for comparison.
- **`GreedyAgent`**: A simple rule-based agent that attempts to align turbines to the local or global wind direction.
- **`RandomAgent`**: An agent that takes random actions, useful for sanity checks and exploring the environment's dynamics.
- **`ConstantAgent`**: An agent that maintains fixed, predefined yaw angles.
- **`model_type` (for RL agents)**: Agents that come from specific RL libraries (like CleanRL) can have a `model_type` attribute to indicate how their `predict` method should be called (e.g., requiring a PyTorch tensor input).

---

## 6. Evaluation Framework (`AgentEval`, `Coliseum`)

WindGym offers robust tools for evaluating agent performance.

- **`AgentEval`**: A legacy class for evaluating a single agent across different fixed wind conditions and generating plots and data.
- **`Coliseum` (`WindGym/utils/evaluate_PPO.py`)**: The recommended and more advanced evaluation framework.
  - **Compares Multiple Agents**: Allows simultaneous evaluation of several agents.
  - **Time Series Evaluation**: Runs episodes with stochastic wind conditions sampled from a `sample_site`, providing insights into long-term performance and robustness.
  - **Wind Grid Evaluation**: Evaluates agents across a predefined grid of fixed wind speed, direction, and turbulence intensity combinations, generating an `xarray.Dataset` for easy analysis.
  - **Plotting and Reporting**: Includes methods for plotting time series of rewards, summarizing agent performance, and visualizing wind grid results.

---

By understanding these core concepts, you'll be well-equipped to navigate the WindGym framework and apply it to your wind farm control research!
