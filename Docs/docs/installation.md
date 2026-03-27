# Installation Guide

This guide will help you set up your environment to run the **WindGym** simulation environment and its agents.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: You'll need Git to clone the WindGym and its dependency repositories.
  - [Download Git](https://git-scm.com/downloads)
- **Pixi**: This is a modern package manager that WindGym uses to manage its Python and other dependencies in an isolated environment.
  - [Install Pixi](https://pixi.sh/latest/#installation): `curl -fsSL https://pixi.sh/install.sh | bash`
- **Python (3.10 to 3.13)**: Pixi will handle the specific Python version for you, but it's good to be aware of the compatible range.

## 2. Core WindGym Installation

First, clone the main WindGym repository. Navigate to your desired working directory and run:

```bash
git clone https://gitlab.windenergy.dtu.dk/sys/windgym.git
cd windgym
```

This command creates a new directory named `windgym`, which contains the core Python package.

Next, use `pixi` to create and activate an isolated Python environment and install all of **WindGym's** core dependencies:

```bash
pixi install
```

This process reads the `pyproject.toml` file, resolves all Python dependencies (including `WindGym` itself in editable mode, `dynamiks` from GitLab, `gymnasium`, `stable_baselines3`, `pytest`, `xarray`, etc.), and sets up a dedicated virtual environment. This might take a few minutes on the first run.

## 3. Activate the Environment

After `pixi install` completes, you need to activate the environment to use WindGym's tools.

To activate the WindGym Python environment in your current terminal session:

```bash
pixi shell
```

You should see `(WindGym)` (or a similar prefix) appear in your terminal prompt, indicating that the environment is active and all of WindGym's Python dependencies are available.

**Remember:** You'll need to run `pixi shell` whenever you open a new terminal session and want to execute WindGym code. Alternatively, you can directly run tasks defined in `pyproject.toml` using `pixi run <task_name>` (e.g., `pixi run test` to run tests).

## 4. Verify Your Installation

Once the environment is set up, you can quickly verify the installation by importing a core WindGym component. From within the `WindGym` directory (after running `pixi shell`), try this:

```bash
python -c "from WindGym.Wind_Farm_Env import WindFarmEnv; print('WindGym installed successfully!')"
```

If you see "WindGym installed successfully!", your basic WindGym environment is ready for you to use!

## 5. Pixi free installation

If you do not wish to use pixi, you can also install the package locally with pip using

```bash
git clone https://gitlab.windenergy.dtu.dk/sys/windgym.git
cd windgym
pip install -e .
```

## Next Steps

Now that you have WindGym installed, you can:

- Follow the [Quick Start](./quick-start.md) guide to create your first environment in a few lines of code
- Explore the [Examples](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/README.md) to see WindGym in action
- Learn about [Core Concepts](./concepts.md) to understand how WindGym works
- Start with [Example 1](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/Example%201%20Make%20environment.ipynb) to create your first environment
