# Evaluation Framework

Evaluating the performance of wind farm control agents is crucial for understanding their effectiveness and comparing different control strategies. WindGym provides a flexible and comprehensive evaluation framework, centered around the `AgentEval` and `Coliseum` classes.

## 1. Objectives of Evaluation

Effective evaluation in WindGym aims to:

- **Quantify Performance**: Measure how well an agent achieves its objectives (e.g., maximizing power, minimizing loads).
- **Compare Agents**: Benchmark the performance of different agents against each other and against baseline controllers.
- **Assess Robustness**: Determine how an agent performs under varying wind conditions, turbulence levels, and farm configurations.
- **Identify Strengths and Weaknesses**: Pinpoint scenarios where an agent excels or struggles.
- **Track Progress**: Monitor the learning progress of reinforcement learning agents over time.

## 2. The AgentEval Class (Legacy)

The `AgentEval` class (and its associated `eval_single_fast` function) provides a way to evaluate a single agent under specific, fixed wind conditions.

- **Fixed Conditions**: You can explicitly set parameters like wind speed, wind direction, and turbulence intensity for an evaluation run.
- **Time Series Data**: It records detailed time series data for power output, yaw angles, and wind speeds for both the agent-controlled farm and a baseline farm.
- **Basic Plotting**: It includes basic plotting functionalities to visualize the flow field and time series results.
- **Xarray Dataset Output**: Results are stored in an `xarray.Dataset`, a powerful data structure for multi-dimensional data, facilitating post-processing and analysis.

While `AgentEval` is functional, the `Coliseum` class offers a more advanced and scalable approach for multi-agent comparisons and broader evaluation scenarios.

## 3. The Coliseum Class (Recommended)

The `Coliseum` class, located in `WindGym/utils/evaluate_PPO.py`, is the recommended and most powerful tool for evaluating multiple agents in WindGym. It provides a robust framework for systematic performance assessment.

### Key Features of Coliseum:

- **Multi-Agent Comparison**: Easily compare the performance of several different agents simultaneously.
- **Flexible Environment Factory**: It takes an "environment factory" (a function that returns a new environment instance) allowing it to create fresh environments for each evaluation run, ensuring isolation and reproducibility.

### Time Series Evaluation (`run_time_series_evaluation`):

- Designed for realistic, long-term performance assessment.
- Utilizes the `sample_site` feature of `WindFarmEnv` to draw wind conditions from realistic statistical distributions (e.g., Weibull for wind speed, frequency for wind direction).
- Runs multiple episodes (e.g., 10, 50, 100) and aggregates results to provide average performance and standard deviations.
- Can save detailed episode histories for in-depth analysis.

### Wind Grid Evaluation (`run_wind_grid_evaluation`):

- Systematically evaluates agents across a predefined grid of fixed wind conditions (e.g., varying wind speeds, directions, and turbulence intensities).
- Generates an `xarray.Dataset` containing the mean reward for each agent at each point in the wind grid.
- Ideal for creating performance maps and identifying optimal operating points or challenging conditions for agents.

### Comprehensive Plotting:

Includes built-in methods for visualizing:

- Mean cumulative rewards over time for multiple agents.
- Summary bar charts comparing average performance across agents.
- Heatmaps or line plots of agent performance across the wind condition grid.

### Detailed Statistics:

Provides summary statistics (mean, std, min, max, median) of agent performance across all evaluation episodes.

### Workflow with Coliseum:

1. **Define Agent(s)**: Instantiate the `BaseAgent` subclasses you want to evaluate (e.g., `PyWakeAgent`, `RandomAgent`, your custom RL agent).
2. **Create Environment Factory**: Define a Python function (or lambda) that returns a new instance of your `WindFarmEnv` (or `FarmEval`) with the desired base parameters. This ensures each evaluation run starts with a clean environment.
3. **Initialize Coliseum**: Pass your environment factory and a dictionary of agents (or a list of agents with labels) to the `Coliseum` constructor.
4. **Run Evaluations**: Call `run_time_series_evaluation` for stochastic, long-term tests, or `run_wind_grid_evaluation` for systematic fixed-condition analysis.
5. **Analyze Results**: Use Coliseum's plotting methods or access the returned `pandas.DataFrame` or `xarray.Dataset` for further custom analysis and visualization.

## 4. Considerations for Evaluation

- **Baseline Selection**: Always compare your agents against a sensible baseline (e.g., a `PyWakeAgent`, a simple `GreedyAgent`, or a traditional control strategy).
- **Wind Condition Representativeness**: Choose evaluation wind conditions (fixed points or a `sample_site` for stochastic runs) that are representative of the real-world scenarios your agent will encounter.
- **Episode Length**: Ensure evaluation episodes are long enough for wake effects to develop and for agents to demonstrate stable behavior.
- **Reproducibility**: Use fixed random seeds for consistent evaluation runs.
- **Metrics**: Consider a variety of metrics beyond just total power, such as power efficiency, yaw activity, and potentially load indicators if using high-fidelity models.

By leveraging the `Coliseum` framework, you can rigorously test and validate your WindGym agents, ensuring their robustness and effectiveness in diverse wind farm scenarios.

---

## Complete Code Examples

### Basic Evaluation with AgentEval

```python
from WindGym.FarmEval import FarmEval
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80
import numpy as np

# Turbine positions
x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

# Create evaluation environment
env = FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    config="EnvConfigs/Env1.yaml",
    Baseline_comp=True  # Compare against baseline
)

# Test PyWake agent
pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

obs, info = env.reset()
rewards = []

for step in range(100):
    action, _ = pywake_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)

    if terminated or truncated:
        break

# Get detailed results
results = env.get_results()

print(f"Average reward: {np.mean(rewards):.3f}")
print(f"Total power: {results['power'].mean():.2f} W")
```

---

### Using Coliseum for Multi-Agent Comparison

#### Step 1: Create Environment Factory

```python
from WindGym.FarmEval import FarmEval
from py_wake.examples.data.hornsrev1 import V80
from py_wake.site import UniformSite

def create_env():
    """Factory function that creates a new environment instance."""
    # Create a site with realistic wind distribution
    site = UniformSite(
        p_wd=[0.1, 0.2, 0.3, 0.4],  # Wind direction probabilities
        a=[9.0],                     # Weibull A parameter
        k=[2.0],                     # Weibull k parameter
        ti=0.06
    )

    return FarmEval(
        turbine=V80(),
        x_pos=[0, 500, 1000],
        y_pos=[0, 0, 0],
        config="EnvConfigs/Env1.yaml",
        sample_site=site,  # Sample from site distribution
        n_passthrough=3,
        Baseline_comp=True
    )
```

#### Step 2: Initialize Agents

```python
from WindGym.Agents import PyWakeAgent
from WindGym.Agents.GreedyAgent import GreedyAgent
from WindGym.Agents.RandomAgent import RandomAgent
from py_wake.examples.data.hornsrev1 import V80
from stable_baselines3 import PPO

# Turbine positions
x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

# Create temporary environment for agent initialization
temp_env = create_env()

# Initialize agents
agents = {
    'PyWake': PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80()),
    'Greedy': GreedyAgent(temp_env, use_global_wind=True),
    'Random': RandomAgent(temp_env),
}

# Optionally add trained RL agent
# agents['PPO'] = PPO.load("trained_ppo_model")
```

#### Step 3: Create Coliseum and Run Evaluation

```python
from WindGym.utils.evaluate_PPO import Coliseum

# Initialize Coliseum
coliseum = Coliseum(
    env_factory=create_env,
    agents=agents
)

# Run time series evaluation (stochastic conditions)
ts_results = coliseum.run_time_series_evaluation(
    n_episodes=50,           # Number of episodes per agent
    save_histories=True,     # Save detailed episode data
    verbose=True
)

# Display summary statistics
print("\n=== Time Series Evaluation Results ===")
print(ts_results[['agent', 'mean_reward', 'std_reward', 'mean_power']])
```

#### Step 4: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot cumulative rewards
coliseum.plot_time_series_rewards(
    ts_results,
    save_path='time_series_rewards.png'
)

# Plot summary comparison
coliseum.plot_summary_bar_chart(
    ts_results,
    metric='mean_reward',
    save_path='agent_comparison.png'
)

plt.show()
```

---

### Wind Grid Evaluation

Evaluate agents across a grid of wind conditions:

```python
import numpy as np
from WindGym.utils.evaluate_PPO import Coliseum

# Define wind condition grid
wind_speeds = np.arange(6, 14, 2)      # 6, 8, 10, 12 m/s
wind_directions = np.arange(250, 290, 10)  # 250°, 260°, 270°, 280°
turbulence_intensities = [0.06, 0.10]

# Environment factory for grid evaluation
def grid_env_factory():
    from py_wake.examples.data.hornsrev1 import V80
    return FarmEval(
        turbine=V80(),
        x_pos=[0, 500, 1000],
        y_pos=[0, 0, 0],
        config="EnvConfigs/Env1.yaml",
        n_passthrough=2,
        Baseline_comp=True
    )

# Initialize Coliseum
coliseum = Coliseum(
    env_factory=grid_env_factory,
    agents=agents
)

# Run grid evaluation
grid_results = coliseum.run_wind_grid_evaluation(
    wind_speeds=wind_speeds,
    wind_directions=wind_directions,
    turbulence_intensities=turbulence_intensities,
    n_episodes_per_condition=5,  # Average over 5 episodes
    verbose=True
)

# grid_results is an xarray.Dataset
print(grid_results)

# Access specific results
pywake_performance = grid_results.sel(agent='PyWake')
print(f"PyWake mean reward: {pywake_performance['mean_reward'].values}")
```

#### Visualize Grid Results

```python
import matplotlib.pyplot as plt

# Plot heatmap for each agent
for agent_name in agents.keys():
    agent_data = grid_results.sel(agent=agent_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(
        wind_directions,
        wind_speeds,
        agent_data['mean_reward'].mean(dim='ti'),  # Average over TI
        levels=20,
        cmap='viridis'
    )
    plt.colorbar(im, label='Mean Reward')
    ax.set_xlabel('Wind Direction (°)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title(f'{agent_name} Performance Across Wind Conditions')
    plt.savefig(f'{agent_name}_heatmap.png')
    plt.show()
```

---

### Advanced: Custom Evaluation Metrics

```python
from WindGym.FarmEval import FarmEval
from WindGym.Agents.PyWakeAgent import PyWakeAgent
import numpy as np

def evaluate_agent_custom(agent, env, n_episodes=10):
    """
    Custom evaluation function with additional metrics.

    Returns:
        dict: Dictionary of evaluation metrics
    """
    all_rewards = []
    all_powers = []
    all_yaw_changes = []
    all_episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []
        episode_powers = []
        prev_yaw = None

        step_count = 0
        while step_count < 200:  # Max steps per episode
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_rewards.append(reward)

            # Track power
            if 'power' in info:
                episode_powers.append(info['power'])

            # Track yaw changes
            if prev_yaw is not None and 'yaw' in info:
                yaw_change = np.abs(info['yaw'] - prev_yaw)
                all_yaw_changes.append(np.mean(yaw_change))

            if 'yaw' in info:
                prev_yaw = info['yaw']

            step_count += 1
            if terminated or truncated:
                break

        all_rewards.extend(episode_rewards)
        all_powers.extend(episode_powers)
        all_episode_lengths.append(step_count)

    # Compute statistics
    metrics = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_power': np.mean(all_powers),
        'std_power': np.std(all_powers),
        'mean_yaw_change': np.mean(all_yaw_changes),
        'mean_episode_length': np.mean(all_episode_lengths),
    }

    return metrics

# Use custom evaluation
from py_wake.examples.data.hornsrev1 import V80

x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

env = FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    config="EnvConfigs/Env1.yaml",
)
agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

metrics = evaluate_agent_custom(agent, env, n_episodes=20)

print("\n=== Custom Evaluation Metrics ===")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

---

### Saving and Loading Results

```python
import pandas as pd
import xarray as xr

# Save time series results
ts_results.to_csv('time_series_results.csv', index=False)

# Save grid results
grid_results.to_netcdf('wind_grid_results.nc')

# Load results later
loaded_ts = pd.read_csv('time_series_results.csv')
loaded_grid = xr.open_dataset('wind_grid_results.nc')

print("Loaded time series results:")
print(loaded_ts.head())
```

---

### Comparing Against Baseline

```python
from WindGym.FarmEval import FarmEval
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80

# Turbine positions
x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

# Environment with baseline comparison enabled
env = FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    config="EnvConfigs/Env1.yaml",
    Baseline_comp=True  # Run parallel baseline simulation
)

agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Get results with baseline comparison
results = env.get_results()

# Calculate performance metrics
agent_power = results['power'].mean()
baseline_power = results['baseline_power'].mean()
improvement = (agent_power - baseline_power) / baseline_power * 100

print(f"Agent power: {agent_power:.2f} W")
print(f"Baseline power: {baseline_power:.2f} W")
print(f"Improvement: {improvement:.2f}%")
```

---

## Best Practices for Evaluation

### 1. Statistical Significance

Run multiple episodes to ensure statistical significance:

```python
# Run 50+ episodes for robust statistics
ts_results = coliseum.run_time_series_evaluation(
    n_episodes=50,
    verbose=True
)

# Check confidence intervals
for agent_name in agents.keys():
    agent_data = ts_results[ts_results['agent'] == agent_name]
    mean = agent_data['mean_reward'].mean()
    std = agent_data['mean_reward'].std()
    ci_95 = 1.96 * std / np.sqrt(len(agent_data))

    print(f"{agent_name}: {mean:.3f} ± {ci_95:.3f}")
```

### 2. Representative Conditions

Test across diverse conditions:

```python
# Include edge cases
wind_speeds = [4, 6, 8, 10, 12, 14, 16]  # Include cut-in and high winds
wind_directions = np.arange(0, 360, 30)  # Full circle
turbulence_intensities = [0.04, 0.06, 0.08, 0.12]  # Low to high TI
```

### 3. Performance Profiling

Track computational cost:

```python
import time

def evaluate_with_timing(agent, env, n_episodes=10):
    """Evaluate agent and track computation time."""
    start_time = time.time()

    total_reward = 0
    total_steps = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        while total_steps < 1000:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if terminated or truncated:
                break

    elapsed_time = time.time() - start_time
    time_per_step = elapsed_time / total_steps

    return {
        'total_reward': total_reward,
        'elapsed_time': elapsed_time,
        'time_per_step': time_per_step,
        'steps_per_second': total_steps / elapsed_time
    }

# Compare computational efficiency
from py_wake.examples.data.hornsrev1 import V80

x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

env = FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    config="EnvConfigs/Env1.yaml",
)

for agent_name, agent in agents.items():
    results = evaluate_with_timing(agent, env, n_episodes=5)
    print(f"\n{agent_name}:")
    print(f"  Steps per second: {results['steps_per_second']:.1f}")
    print(f"  Time per step: {results['time_per_step']*1000:.2f} ms")
```

---

## Troubleshooting Evaluation

### Issue: Inconsistent Results

**Solution**: Increase number of episodes and use fixed random seeds:

```python
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Run more episodes
ts_results = coliseum.run_time_series_evaluation(n_episodes=100)
```

### Issue: Long Evaluation Times

**Solution**: Reduce episode length or use parallel evaluation:

```python
# Shorter episodes
def fast_env_factory():
    from py_wake.examples.data.hornsrev1 import V80
    return FarmEval(
        turbine=V80(),
        x_pos=[0, 500, 1000],
        y_pos=[0, 0, 0],
        config="EnvConfigs/Env1.yaml",
        n_passthrough=1,  # Shorter episodes
        dt_env=2.0,        # Larger timesteps
    )

# Note: Parallel evaluation would require custom implementation
```

---

## Next Steps

- Learn about [creating custom agents](agents.md)
- Explore [simulation configuration](simulations.md)
- Understand [noise and uncertainty](noise-and-uncertainty.md)
- Check out [evaluation examples](../examples/compare_agents_grid.py)
