# WindGym Examples

This folder contains example scripts and notebooks to help you get started with WindGym.

## Main Examples

The primary examples are provided as Jupyter notebooks:

| Example | Description | Complexity |
|---------|-------------|------------|
| **Example 1: Make Environment** | Learn how to create and configure a WindGym environment with different settings | Beginner |
| **Example 2: Evaluate Pretrained Agent** | Evaluate a pre-trained reinforcement learning agent on wind farm scenarios | Intermediate |
| **Example 3: Load Results from Pre-Evaluated Model** | Load and analyze results from previously evaluated models | Intermediate |

## Additional Examples

### Agent Comparison
- `compare_agents_grid.py` - Compare multiple agents across a grid of wind conditions
- `compare_agents_sampling.py` - Compare agents using sampled wind conditions from a site

### Noise and Uncertainty
The `noise_examples/` directory contains examples for working with noisy observations:
- `evaluate_agents.py` - Evaluate agents under measurement uncertainty
- `train_protagonist.py` - Train an agent to be robust to noise
- `train_adversary.py` - Train an adversarial noise model
- `pywake_adversary.py` - Use PyWake agent with adversarial noise
- `train_self_play.py` - Self-play training with noise
- `noise_definitions.py` - Different noise model configurations

### Training Examples
- `ppo_curriculum_example.py` - Train a PPO agent with curriculum learning
- `curriculum.py` - Curriculum learning setup and configuration

### Advanced Examples
- `longer_steps_example.py` - Using longer timesteps for faster simulation
- `example_with_noise.py` - Basic example with measurement noise
- `pywake_agent_with_noise.py` - PyWake agent operating with noisy measurements
- `WindEnvMulti-example/` - Multi-agent environment examples

## Getting Started

1. Ensure you have activated the WindGym environment:
   ```bash
   pixi shell
   ```

2. Start with Example 1 to understand the basics:
   ```bash
   jupyter notebook "Example 1 Make environment.ipynb"
   ```

3. Explore other examples based on your interests

## Requirements

All examples assume you have completed the [installation guide](../docusaurus-site/docs/installation.md) and have access to:
- A working WindGym installation
- Jupyter notebook (included in the Pixi environment)
- Pre-trained models (where applicable)

## Need Help?

- Check the [full documentation](https://sys.pages.windenergy.dtu.dk/windgym/)
- Review the [troubleshooting guide](../docusaurus-site/docs/troubleshooting.md)
- Open an issue on [GitLab](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues)
