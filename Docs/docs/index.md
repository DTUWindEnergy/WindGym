# WindGym Documentation

Welcome to the WindGym documentation! This site provides comprehensive information about the WindGym project, its environments, agents, and evaluation tools.

---

## What is WindGym?

WindGym is a powerful and flexible framework for developing and evaluating **reinforcement learning (RL) agents** for **wind farm control**. It provides a simulated environment where agents can learn to optimize various aspects of wind farm operation, such as:

- **Maximizing power production**
- **Minimizing structural loads**
- **Reducing wake effects** between turbines

The core of WindGym is built upon **Dynamiks** and **PyWake**, leveraging their capabilities for high-fidelity and engineering-level wind farm simulations, respectively.

---

## Key Features

- **Modular Environment Design**: Easily configure wind farm layouts, wind conditions (steady, turbulent, or sampled from real-world sites), and turbine models (PyWake or high-fidelity HAWC2).
- **Flexible Agent Integration**: Integrate and test various control strategies, from simple rule-based controllers to complex deep reinforcement learning agents.
- **Comprehensive Evaluation Tools**: Evaluate agent performance across a range of wind conditions, compare against baselines, and visualize results with built-in plotting utilities.
- **Multi-Agent Support**: Explore cooperative or competitive control strategies with the multi-agent environment.
- **Realistic Wind Conditions**: Utilize turbulence models (e.g., Mann turbulence) and real-world wind site data for more realistic simulations.

---

## Getting Started

To begin using WindGym, we recommend the following steps:

1.  **Installation**: Follow the instructions in the [Installation Guide](installation.md) to set up your WindGym environment.
2.  **Basic Concepts**: Familiarize yourself with the fundamental concepts of the WindGym environment in the [Core Concepts](concepts.md) section.
3.  **Running Simulations**: Learn how to run your first simulation with different environments and agents in the [Simulation Guide](simulations.md).
4.  **Developing Agents**: Dive into creating and training your own custom agents using our [Agent Development](agents.md) guide.
5.  **Evaluation**: Understand how to rigorously evaluate and compare the performance of different agents using the [Evaluation Framework](evaluations.md).

---

## Project Structure

The WindGym project is organized into several key directories:

- `docs/`: Contains all the documentation files for this website.
- `WindGym/envs/`: Defines the core Gymnasium environments, including `WindFarmEnv` and `FarmEval`.
- `WindGym/wrappers/`: Provides Gymnasium wrappers to modify environment behavior, such as `RecordEpisodeVals`, `CurriculumWrapper`, and `PowerWrapper`.
- `WindGym/Agents/`: Implements various agent types, from basic `GreedyAgent` and `PyWakeAgent` to `RandomAgent` and `ConstantAgent`.
- `WindGym/MesClass.py`: Manages the collection and processing of wind farm measurements.
- `WindGym/utils/`: Contains utility functions, including the `Coliseum` class for advanced evaluation workflows and `generate_layouts` for creating wind farm configurations.
- `scripts/`: Holds scripts for pre-building evaluation data and other tasks (e.g., `prebuild.sh`).

---

## Examples

Check out our [Examples folder on GitLab](https://gitlab.windenergy.dtu.dk/sys/windgym/-/tree/main/examples?ref_type=heads) for practical demonstrations of how to use WindGym for various wind farm control scenarios.

---

We hope you find the WindGym documentation helpful and enjoy using the framework for your wind energy research and development!
