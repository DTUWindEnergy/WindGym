# Installation Guide

This guide will help you set up your environment to run the **WindGym** simulation environment and its agents.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: You'll need Git to clone the WindGym repository.
  - [Download Git](https://git-scm.com/downloads)
- **Python (3.10 to 3.13)**: Required for running WindGym.

## 2. Clone the Repository

```bash
git clone https://gitlab.windenergy.dtu.dk/sys/windgym.git
cd windgym
```

## 3. Install WindGym

### Option A: pip

The simplest way to install WindGym is with pip:

```bash
pip install -e .
```

### Option B: pixi

[Pixi](https://pixi.sh) is a modern package manager that automatically manages Python and all dependencies in an isolated environment. To install pixi:

**Linux / macOS:**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://pixi.sh/install.ps1 | iex"
```

For more options, see the [pixi installation docs](https://pixi.sh/latest/#installation).

Then install and activate WindGym:

```bash
pixi install
pixi shell
```

This reads the `pyproject.toml` file, resolves all dependencies (including `WindGym` itself in editable mode, `dynamiks`, `gymnasium`, `stable_baselines3`, `xarray`, etc.), and sets up a dedicated environment. This might take a few minutes on the first run.

You should see `(WindGym)` (or a similar prefix) appear in your terminal prompt, indicating the environment is active. You'll need to run `pixi shell` whenever you open a new terminal session. Alternatively, use `pixi run <task_name>` (e.g., `pixi run test`) to run tasks directly.

## 4. Verify Your Installation

```bash
python -c "from WindGym import WindFarmEnv; print('WindGym installed successfully!')"
```

If you see "WindGym installed successfully!", your WindGym environment is ready!

## Next Steps

Now that you have WindGym installed, you can:

- Follow the [Quick Start](./quick-start.md) guide to create your first environment in a few lines of code
- Explore the [Examples](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/README.md) to see WindGym in action
- Learn about [Core Concepts](./concepts.md) to understand how WindGym works
- Start with [Example 1](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/Example%201%20Make%20environment.ipynb) to create your first environment
