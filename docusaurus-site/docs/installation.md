# Installation Guide

This guide will help you set up your environment to run the **WindGym** simulation environment and its agents.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: You'll need Git to clone the WindGym and its dependency repositories.
  - [Download Git](https://git-scm.com/downloads)
- **Pixi**: This is a modern package manager that WindGym uses to manage its Python and other dependencies in an isolated environment.
  - [Install Pixi](https://pixi.sh/latest/#installation)
- **Python (3.7 to 3.11)**: Pixi will handle the specific Python version for you, but it's good to be aware of the compatible range.

### System Requirements

- **Operating System**: Linux (64-bit) or macOS (ARM64/Intel)
- **Disk Space**: At least 5 GB free space for dependencies and simulation data
- **RAM**: Minimum 8 GB (16 GB recommended for large-scale simulations)

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

## 5. Troubleshooting

### Common Installation Issues

#### Pixi Installation Fails

- **Issue**: Pixi fails to resolve dependencies or times out
- **Solution**: Try using a different mirror or clearing the pixi cache:
  ```bash
  pixi clean cache-dir
  pixi install
  ```

#### Import Errors After Installation

- **Issue**: `ModuleNotFoundError` when importing WindGym
- **Solution**: Ensure you've activated the environment with `pixi shell` and that the installation completed successfully:
  ```bash
  pixi shell
  pip list | grep WindGym
  ```

#### Git Clone Authentication Issues

- **Issue**: Access denied when cloning from GitLab
- **Solution**: Ensure you have access to the repository. You may need to set up SSH keys or use HTTPS with credentials. See [GitLab documentation](https://docs.gitlab.com/ee/user/ssh.html) for help.

#### Platform-Specific Issues

- **macOS**: If you encounter issues with HDF5 or NetCDF, ensure Xcode Command Line Tools are installed:
  ```bash
  xcode-select --install
  ```
- **Linux**: Some systems may require additional system libraries. Install them via your package manager:

  ```bash
  # Ubuntu/Debian
  sudo apt-get install libhdf5-dev libnetcdf-dev

  # Fedora/RHEL
  sudo dnf install hdf5-devel netcdf-devel
  ```

### Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting & FAQ](troubleshooting.md) page
2. Search [existing issues](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues) on GitLab
3. Open a [new issue](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues/new) with details about your problem

## Next Steps

Now that you have WindGym installed, you can:

- Explore the [Examples](../examples/README.md) to see WindGym in action
- Learn about [Core Concepts](concepts.md) to understand how WindGym works
- Start with [Example 1](../examples/Example%201%20Make%20environment.ipynb) to create your first environment
