# WindGym Examples

This folder contains example scripts and notebooks to help you get started with WindGym.

## Main Examples

The primary examples are provided as Jupyter notebooks:

| Example | Description | Complexity |
|---------|-------------|------------|
| **Quick Start** | Fastest introduction to WindGym basics | Beginner |
| **Example 1: Make Environment** | Learn how to create and configure a WindGym environment with different settings | Beginner |
| **Example 2: Evaluate Pretrained Agent** | Evaluate a pre-trained reinforcement learning agent on wind farm scenarios | Intermediate |
| **Example 3: Load Results from Pre-Evaluated Model** | Load and analyze results from previously evaluated models | Intermediate |
| **Example 4: Change Wind Directions** | Changes the wind direction over time | Intermediate |
| **Example Custom Rendering** | Custom visualization and rendering of wind farm simulations | Advanced |

## Additional Examples

### Agent Comparison
- `compare_agents_grid.py` - Compare multiple agents across a grid of wind conditions
- `compare_agents_sampling.py` - Compare agents using sampled wind conditions from a site

### Measurement Error and Uncertainty
The `measurement_error/` directory contains a full pipeline for training and evaluating agents under measurement uncertainty. See its [README](measurement_error/README.md) for details.

### Multi-Agent Environment
The `WindEnvMulti-example/` directory demonstrates how to use the PettingZoo-based multi-agent environment (`WindFarmEnvMulti`) with multiple turbines.

## Need Help?

- Check the [full documentation](https://sys.pages.windenergy.dtu.dk/windgym/)
- Review the [troubleshooting guide](https://sys.pages.windenergy.dtu.dk/windgym/troubleshooting)
- Open an issue on [GitLab](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues)
