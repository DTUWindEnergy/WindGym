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

## 5. dynamiks Version and DWM Setup

> **This is the `dev_dynamiks` branch.** Its `pyproject.toml` pins the dev
> dynamiks build (DYNAMIKS/dynamiks `3913cd9`), plus the jDWM (`c455592`,
> `sys_branch`) and `hipersim>=0.1.21` it needs. All features listed below work
> here. The code is the same as on `main`; to update, merge `main` into this
> branch. Only the dependency pins, the lock files and `tests/golden/G4.npz`
> differ. G4 runs added turbulence, which the two dynamiks versions compute
> differently.

The `main` branch runs on the PyPI release `dynamiks==0.0.7`. By default,
WindGym builds a plain dynamiks DWM flow simulation, i.e. dynamiks' own model
defaults.

WindGym also ships an LES-calibrated DWM setup. It is opt-in:

```python
from WindGym.core.calibrated_values import LES_CALIBRATED

env = WindFarmEnv(..., dwm_setup=LES_CALIBRATED)
```

or `dwm_setup: les_calibrated` in the env config. You can also build your own
`DWMSetup` (`WindGym/core/dwm_defaults.py`), e.g.
`dataclasses.replace(LES_CALIBRATED, superposition="linear")`.

Parts of the calibrated setup, and some other features, need the development
build of dynamiks. The `dev_dynamiks` branch pins that build from git:

- `dwm_setup=LES_CALIBRATED` (TI sensor, Branlard scaling, mixed superposition)
- `turbtype="Precursor"` (LES-precursor inflow)
- `turb_memmap=True` (memory-mapped Mann boxes)
- wind veer (`wind.veer_min` / `wind.veer_max` ≠ 0 in the config)
- the DWM speed-ups `interpolation` and `lateral_cutoff`

On `main`, using any of these raises an `ImportError` that points to the
`dev_dynamiks` branch.

## Next Steps

Now that you have WindGym installed, you can:

- Follow the [Quick Start](./quick-start.md) guide to create your first environment in a few lines of code
- Explore the [Examples](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/README.md) to see WindGym in action
- Learn about [Core Concepts](./concepts.md) to understand how WindGym works
- Start with [Example 1](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/Example%201%20Make%20environment.ipynb) to create your first environment
