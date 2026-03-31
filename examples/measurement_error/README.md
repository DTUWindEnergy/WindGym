# Measurement Error Examples

This directory contains scripts for training and evaluating wind farm control agents under measurement uncertainty. The pipeline covers multiple training strategies, evaluation, and visualization.

## Training Scripts

| Script | Description |
|--------|-------------|
| `train_protagonist.py` | Train a protagonist agent robust to procedural, adversarial, or synthetic self-play noise |
| `train_adversary.py` | Train an adversarial noise model against a learned protagonist |
| `train_pywake_adversary.py` | Train an adversarial noise model against a fixed PyWake baseline agent |
| `train_self_play.py` | Simultaneous protagonist/adversary training via PettingZoo self-play |

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `evaluate_agents.py` | Evaluate trained agents under procedural and adversarial noise scenarios |
| `evaluate_self_play.py` | Evaluate agents from self-play training against various noise models |
| `pywake_agent_with_noise.py` | Compare oracle vs. realistic agent performance under noisy observations |

## Analysis and Plotting

| Script | Description |
|--------|-------------|
| `plot_training.py` | Parse Stable-Baselines3 log files and plot training progress |
| `plot_noise_comparison.py` | Compare different noise models and their impact on agent performance |
| `plot_scenarios.py` | Visualize agent behavior across wind scenarios with and without noise |
| `create_animation.py` | Generate animations from evaluation results |
| `create_matrix_from_csvs.py` | Aggregate evaluation CSVs into comparison matrices |

## Shared Modules

| File | Description |
|------|-------------|
| `noise_definitions.py` | Noise model configurations (white noise, episodic bias, adversarial) and constraint functions |
| `utils.py` | Custom PyTorch Agent class, layer initialization, and SB3 weight loading utilities |

## Shell Scripts

| Script | Description |
|--------|-------------|
| `arms_race.sh` | Orchestrate iterative arms-race or synthetic self-play training |
| `run_self_play.sh` | Run the self-play training pipeline |
| `run_gauntlet.sh` | Run a comprehensive gauntlet of evaluations |
| `evaluate_agents.sh` | Agent evaluation pipeline |
| `evaluate_arms_race.sh` | Evaluate arms-race training results |

## Configuration

- `env_config/two_turbine_yaw.yaml` - Two-turbine yaw control environment configuration
