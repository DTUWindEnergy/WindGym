# API Reference

This page provides a reference for the main classes and functions in WindGym.

---

## Environment Classes

### `WindFarmEnv`

The base wind farm environment class.

```python
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnv(
    turbine=V80(),           # PyWake turbine model (REQUIRED)
    x_pos=[0, 500, 1000],    # Turbine x positions in meters (REQUIRED)
    y_pos=[0, 0, 0],         # Turbine y positions in meters (REQUIRED)
    config="path/to/config.yaml",  # Path to YAML configuration file
    n_passthrough=5,         # Number of flow passthroughs (default: 5)
    dt_sim=1,                # Simulation timestep in seconds (default: 1)
    dt_env=1,                # Environment timestep in seconds (default: 1)
    burn_in_passthroughs=2,  # Burn-in passthroughs (default: 2)
    turbtype="Random",       # Turbulence type: "Random" or "MannGenerate" (default: "Random")
    TurbBox="Default",       # Path to turbulence box file (default: "Default")
    sample_site=None,        # PyWake Site for sampling wind conditions
    Baseline_comp=False,     # Enable baseline comparison
    seed=None,               # Random seed for reproducibility
    yaw_step_sim=1,          # Max yaw change per sim step (degrees)
    yaw_step_env=None,       # Max yaw change per env step (degrees)
    backend="dynamiks",      # Simulation backend (default: "dynamiks")
    render_mode=None,        # Render mode: None, "human", or "rgb_array"
)
```

**Note**: Wind conditions (wind speed, direction, turbulence intensity) are sampled at each `reset()` based on the `config` YAML file settings, NOT specified as constructor parameters.

**Key Methods:**

- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return next state
- `render()`: Visualize the environment (optional)
- `close()`: Clean up resources

**Attributes:**

- `observation_space`: Gymnasium Space object defining observations
- `action_space`: Gymnasium Space object defining actions
- `n_wt`: Number of turbines
- `dt_env`: Environment timestep

---

### `FarmEval`

Evaluation wrapper for detailed performance tracking.

```python
from WindGym.FarmEval import FarmEval

env = FarmEval(
    n_wt=3,
    ws=10.0,
    wd=270.0,
    TI=0.06,
    Baseline_comp=True,  # Enable baseline comparison
    # ... other WindFarmEnv parameters
)
```

**Additional Methods:**

- `get_results()`: Return xarray.Dataset with episode results
- `plot_flow_field(time_idx=-1, save_path=None)`: Visualize flow field
- `get_power_time_series()`: Get power production over time

---

### `WindFarmEnvMulti`

Multi-agent wind farm environment (PettingZoo compatible).

```python
from WindGym import WindFarmEnvMulti
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnvMulti(
    turbine=V80(),
    x_pos=[0, 500, 1000, 0, 500, 1000],
    y_pos=[0, 0, 0, 500, 500, 500],
    config="path/to/config.yaml",
    n_passthrough=5,
    # ... other WindFarmEnv parameters
)
```

**Key Differences:**

- Returns dict of observations (one per agent)
- Requires dict of actions (one per agent)
- Compatible with PettingZoo interface

---

## Wrappers

### `NoisyWindFarmEnv`

Wrapper that adds measurement noise to observations.

```python
from WindGym.core import NoisyWindFarmEnv, MeasurementManager, WhiteNoiseModel

manager = MeasurementManager(base_env)
manager.set_noise_model('wd', WhiteNoiseModel({MeasurementType.WIND_DIRECTION: 2.0}))
noisy_env = NoisyWindFarmEnv(base_env, manager)
```

**Info Dictionary Additions:**

- `info['clean_obs']`: Clean (ground truth) observations
- `info['noise_info']`: Dictionary of applied noise for each measurement

---

### `RecordEpisodeVals`

Records episode statistics.

```python
from WindGym.wrappers import RecordEpisodeVals

env = RecordEpisodeVals(base_env)
```

---

### `CurriculumWrapper`

Implements curriculum learning by gradually increasing difficulty.

```python
from WindGym.wrappers import CurriculumWrapper

env = CurriculumWrapper(
    base_env,
    initial_difficulty=0.5,
    max_difficulty=1.0,
    increase_rate=0.01
)
```

---

## Agent Classes

### `BaseAgent`

Base class for all agents.

```python
from WindGym.Agents.BaseAgent import BaseAgent

class MyAgent(BaseAgent):
    def predict(self, obs):
        """
        Generate action from observation.

        Args:
            obs (np.ndarray): Current observation

        Returns:
            action (np.ndarray): Action to take
            state: Optional agent state
        """
        # Your control logic here
        action = ...
        return action, None
```

**Helper Methods:**

- `scale_yaw(yaw_angles)`: Scale yaw angles to [-1, 1] action space
- `unscale_yaw(scaled_actions)`: Convert scaled actions back to degrees

---

### `PyWakeAgent`

Optimal static yaw control using PyWake optimization.

```python
from WindGym.Agents import PyWakeAgent
from py_wake.examples.data.hornsrev1 import V80

agent = PyWakeAgent(
    x_pos=[0, 500, 1000],    # Turbine x positions (REQUIRED)
    y_pos=[0, 0, 0],         # Turbine y positions (REQUIRED)
    turbine=V80(),           # PyWake turbine model (default: V80())
    wind_speed=8,            # Default wind speed for optimization (default: 8)
    wind_dir=270,            # Default wind direction (default: 270)
    TI=0.07,                 # Turbulence intensity (default: 0.07)
    yaw_max=45,              # Max yaw angle (default: 45)
    yaw_min=-45,             # Min yaw angle (default: -45)
    env=None,                # Optional environment reference
)

action, _ = agent.predict(obs)
```

**Behavior:**

- Computes optimal yaw angles for current wind conditions
- Uses PyWake's internal optimization
- Does not adapt to observation changes during episode

---

### `NoisyPyWakeAgent`

Robust variant of PyWakeAgent for noisy observations.

```python
from WindGym.Agents import NoisyPyWakeAgent
from py_wake.examples.data.hornsrev1 import V80

agent = NoisyPyWakeAgent(
    x_pos=[0, 500, 1000],    # Turbine x positions (REQUIRED)
    y_pos=[0, 0, 0],         # Turbine y positions (REQUIRED)
    turbine=V80(),           # PyWake turbine model
    # ... other PyWakeAgent parameters
)

action, _ = agent.predict(obs)
```

**Behavior:**

- Averages multiple wind measurements to estimate true conditions
- More robust to measurement noise than standard PyWakeAgent

---

### `GreedyAgent`

Simple reactive agent that aligns turbines with wind.

```python
from WindGym.Agents.GreedyAgent import GreedyAgent

agent = GreedyAgent(env, use_global_wind=True)
action, _ = agent.predict(obs)
```

**Parameters:**

- `use_global_wind` (bool): Use global wind direction vs local measurements

---

### `RandomAgent`

Takes random actions within action space.

```python
from WindGym.Agents.RandomAgent import RandomAgent

agent = RandomAgent(env)
action, _ = agent.predict(obs)
```

---

### `ConstantAgent`

Maintains fixed yaw angles.

```python
from WindGym.Agents.ConstantAgent import ConstantAgent
import numpy as np

yaw_angles = np.array([0.0, 5.0, -5.0])  # degrees
agent = ConstantAgent(env, yaw_angles=yaw_angles)
action, _ = agent.predict(obs)
```

---

## Noise Models

### `WhiteNoiseModel`

Adds independent Gaussian noise at each timestep.

```python
from WindGym.core import WhiteNoiseModel, MeasurementType

noise_model = WhiteNoiseModel({
    MeasurementType.WIND_DIRECTION: 2.0,  # 2 degrees std dev
    MeasurementType.WIND_SPEED: 0.5,      # 0.5 m/s std dev
})
```

---

### `EpisodicBiasNoiseModel`

Adds consistent bias throughout an episode.

```python
from WindGym.core import EpisodicBiasNoiseModel, MeasurementType

noise_model = EpisodicBiasNoiseModel({
    MeasurementType.WIND_DIRECTION: 5.0  # 5 degrees bias std dev
})
```

---

### `HybridNoiseModel`

Combines white noise and episodic bias.

```python
from WindGym.core import HybridNoiseModel, MeasurementType

noise_model = HybridNoiseModel(
    white_noise_std={MeasurementType.WIND_DIRECTION: 2.0},
    episodic_bias_std={MeasurementType.WIND_DIRECTION: 5.0}
)
```

---

### `MeasurementManager`

Manages noise application to observations.

```python
from WindGym.core import MeasurementManager, WhiteNoiseModel, MeasurementType

manager = MeasurementManager(env)

# Configure noise for specific measurement types
wd_noise = WhiteNoiseModel({MeasurementType.WIND_DIRECTION: 2.0})
ws_noise = WhiteNoiseModel({MeasurementType.WIND_SPEED: 0.5})

manager.set_noise_model(MeasurementType.WIND_DIRECTION, wd_noise)
manager.set_noise_model(MeasurementType.WIND_SPEED, ws_noise)
```

**Methods:**

- `set_noise_model(measurement_type, noise_model)`: Configure noise for a measurement
- `apply_noise(clean_obs, reset=False)`: Apply noise to clean observations
- `get_measurement_spec()`: Get specification of all measurements

---

## Evaluation Tools

### `Coliseum`

Multi-agent evaluation framework.

```python
from WindGym.utils.evaluate_PPO import Coliseum

coliseum = Coliseum(
    env_factory=create_env,  # Function that returns new environment
    agents=agent_dict         # Dictionary of {name: agent}
)
```

**Methods:**

- `run_time_series_evaluation(n_episodes, save_histories=False)`: Stochastic evaluation
- `run_wind_grid_evaluation(wind_speeds, wind_directions, turbulence_intensities)`: Grid evaluation
- `plot_time_series_rewards(results, save_path=None)`: Plot time series
- `plot_summary_bar_chart(results, metric='mean_reward')`: Compare agents

**Returns:**

- Time series: pandas.DataFrame with episode statistics
- Grid: xarray.Dataset with results across wind conditions

---

## Utility Functions

### `generate_layouts`

Generate wind farm turbine layouts.

```python
from WindGym.utils import generate_layouts

# Generate grid layout
x_pos, y_pos = generate_layouts.grid_layout(
    n_rows=2,
    n_cols=3,
    spacing_x=500,  # meters
    spacing_y=500
)

# Generate circular layout
x_pos, y_pos = generate_layouts.circular_layout(
    n_turbines=6,
    radius=1000  # meters
)
```

---

## Configuration

### YAML Configuration File

WindGym uses YAML files to configure observations, actions, and rewards.

**Example `config.yaml`:**

```yaml
# Observation configuration
observations:
  turbine_level:
    - ws
    - wd
    - yaw
    - power
  farm_level:
    - ws_mean
    - total_power
  include_history: true
  history_length: 10

# Action configuration
ActionMethod: "yaw" # or "wind"
yaw_min: -30
yaw_max: 30
yaw_step_env: 3.0

# Reward configuration
power_reward: "Baseline" # or "Power_avg", "Power_diff", "None"
Power_scaling: 1.0
action_penalty: "Change" # or "Total", "None"
penalty_scaling: 0.01
```

---

## Data Structures

### Observation Space

Observations are returned as flattened numpy arrays with values scaled to [-1, 1].

**Structure depends on YAML configuration:**

- Turbine-level measurements: repeated for each turbine
- Farm-level measurements: single values
- History: past N observations concatenated

**Example:**

```
obs = [ws_t1, wd_t1, yaw_t1, power_t1, ws_t2, wd_t2, yaw_t2, power_t2, ..., ws_farm, total_power]
```

---

### Action Space

Actions are numpy arrays of length `n_wt`, with values in [-1, 1] representing:

- **"yaw" method**: Change in yaw angle (scaled)
- **"wind" method**: Target yaw offset from wind direction (scaled)

---

### Info Dictionary

The `info` dict returned by `step()` contains:

**Standard keys:**

- `'episode'`: Episode statistics (if terminated)
- `'TimeLimit.truncated'`: Whether episode was truncated by time limit

**FarmEval additions:**

- `'power'`: Current power output per turbine
- `'yaw'`: Current yaw angles
- `'ws'`, `'wd'`: Current wind conditions

**NoisyWindFarmEnv additions:**

- `'clean_obs'`: Ground truth observations
- `'noise_info'`: Applied noise details

---

## Type Definitions

```python
# Observation
ObsType = np.ndarray  # Shape: (obs_dim,)

# Action
ActType = np.ndarray  # Shape: (n_wt,)

# Info
InfoDict = Dict[str, Any]

# Step return
StepReturn = Tuple[ObsType, float, bool, bool, InfoDict]

# Reset return
ResetReturn = Tuple[ObsType, InfoDict]
```

---

## Related Pages

- [Core Concepts](concepts.md) - Detailed explanations of key concepts
- [Agents](agents.md) - Agent development guide
- [Simulations](simulations.md) - Running simulations
- [Evaluations](evaluations.md) - Evaluation tools and methods
