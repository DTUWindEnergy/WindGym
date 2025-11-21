# Agents

In WindGym, an **agent** is the intelligent entity responsible for making decisions within the wind farm environment. These decisions typically involve adjusting the yaw angles of wind turbines to achieve specific objectives, such as maximizing power production or minimizing structural loads.

WindGym provides a modular framework for developing and integrating various types of agents, from simple rule-based controllers to sophisticated reinforcement learning policies.

---

## The `BaseAgent`

All agents in WindGym are built upon the `BaseAgent` class. This foundational class defines a common interface that all agents must adhere to, primarily requiring a `predict()` method. This ensures consistency and allows different agents to be seamlessly interchanged within the simulation and evaluation frameworks.

The `BaseAgent` also includes utility functions for scaling yaw angles to the environment's expected action space (typically between -1 and 1) and inversely unscaling them back to degrees.

---

## Types of Agents

WindGym comes with several pre-built agent implementations that serve as examples, baselines, or simple controllers for various testing and comparison purposes.

### 1. `PyWakeAgent`

The `PyWakeAgent` is a powerful baseline agent that leverages the **PyWake** library's advanced wind farm modeling and optimization capabilities.

- **Behavior**: For a given set of static wind conditions (wind speed, direction, turbulence intensity), the `PyWakeAgent` calculates the theoretically optimal yaw angles for all turbines in the farm that would maximize the total power production according to its internal wake models. It then applies these optimized, static yaw angles throughout the simulation.
- **Use Case**: Ideal as a high-performing benchmark to compare against dynamic, learning-based agents. It represents a "perfect" steady-state controller.

### 2. `GreedyAgent`

The `GreedyAgent` is a simple, rule-based controller.

- **Behavior**: This agent attempts to align each turbine's yaw angle with the perceived wind direction. It can operate based on either the _local_ wind direction at each turbine (influenced by wakes) or the _global_ incoming wind direction. It tries to move towards a zero yaw offset (aligned with the wind) or a fixed, global setpoint.
- **Use Case**: Useful for quick sanity checks, demonstrating basic control logic, or as a very simple baseline against which more complex agents can show improvement.

### 3. `RandomAgent`

As its name suggests, the `RandomAgent` takes completely random actions within the permissible action space of the environment.

- **Behavior**: In each step, the `RandomAgent` generates a random set of yaw adjustments (or target yaw angles, depending on the environment's `ActionMethod`) for all turbines.
- **Use Case**: Primarily used for testing the environment's stability, ensuring that it can handle arbitrary inputs without crashing, and establishing a "worst-case" performance baseline. Any agent should ideally perform better than a `RandomAgent`.

### 4. `ConstantAgent`

The `ConstantAgent` applies a predefined, fixed set of yaw angles to all turbines throughout the simulation.

- **Behavior**: This agent is initialized with a specific array of yaw angles (e.g., all zeros, or a fixed offset for specific turbines), and it maintains these angles without change.
- **Use Case**: Useful for evaluating the wind farm's performance under static yaw settings, verifying specific design conditions, or as a simple non-reactive baseline.

---

## Agent-Environment Interaction

Regardless of their internal logic, all agents interact with the WindGym environment through the standard Gymnasium `step()` method. They receive an observation (sensor data from the wind farm), process it with their `predict()` method to determine an action, and send that action back to the environment. The environment then applies the action, simulates the next timestep, and returns a new observation, reward, and episode status.

When developing custom agents, you will typically create a new Python class that inherits from `BaseAgent` and implements your desired control or learning algorithm within its `predict()` method.

---

## Creating Custom Agents

### Basic Custom Agent Example

Here's how to create a simple custom agent:

```python
from WindGym.Agents.BaseAgent import BaseAgent
import numpy as np

class MyCustomAgent(BaseAgent):
    """A simple agent that aligns turbines with a target yaw angle."""

    def __init__(self, env, target_yaw=0.0):
        super().__init__(env)
        self.target_yaw = target_yaw

    def predict(self, obs):
        """
        Generate actions based on observation.

        Args:
            obs: Current observation from environment

        Returns:
            action: Array of yaw adjustments for each turbine
            state: Optional state (not used here)
        """
        n_turbines = self.env.n_wt

        # Simple strategy: move towards target yaw
        action = np.full(n_turbines, self.target_yaw)

        # Scale action to environment's expected range [-1, 1]
        action = self.scale_yaw(action)

        return action, None
```

### Using Your Custom Agent

```python
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

# Create environment
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)

# Initialize your agent
agent = MyCustomAgent(env, target_yaw=5.0)

# Run simulation
obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

---

## Using Built-in Agents

### PyWakeAgent

Optimal static yaw control using PyWake optimization:

```python
from WindGym.Agents import PyWakeAgent
from WindGym.FarmEval import FarmEval
from py_wake.examples.data.hornsrev1 import V80

# Turbine positions
x_pos = [0, 500, 1000]
y_pos = [0, 0, 0]

# Create environment
env = FarmEval(
    turbine=V80(),
    x_pos=x_pos,
    y_pos=y_pos,
    config="EnvConfigs/Env1.yaml",
    Baseline_comp=True,
)

# PyWake agent automatically optimizes yaw angles
agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())

# Run evaluation
obs, info = env.reset()
total_reward = 0

for step in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"PyWake agent total reward: {total_reward:.2f}")
```

### GreedyAgent

Simple alignment with wind direction:

```python
from WindGym import WindFarmEnv
from WindGym.Agents.GreedyAgent import GreedyAgent
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)

# Agent tries to align with wind
agent = GreedyAgent(env, use_global_wind=True)

obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### RandomAgent

For baseline comparison:

```python
from WindGym import WindFarmEnv
from WindGym.Agents.RandomAgent import RandomAgent
from py_wake.examples.data.hornsrev1 import V80

env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
agent = RandomAgent(env)

obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### ConstantAgent

Fixed yaw angles:

```python
from WindGym import WindFarmEnv
from WindGym.Agents.ConstantAgent import ConstantAgent
from py_wake.examples.data.hornsrev1 import V80
import numpy as np

env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)

# Set specific yaw angles for each turbine
yaw_angles = np.array([0.0, 5.0, -5.0])  # degrees
agent = ConstantAgent(env, yaw_angles=yaw_angles)

obs, info = env.reset()
for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Advanced Custom Agent

### Observation-Based Control

Here's a more sophisticated agent that uses observations:

```python
from WindGym.Agents.BaseAgent import BaseAgent
import numpy as np

class ObservationBasedAgent(BaseAgent):
    """Agent that adjusts yaw based on local wind measurements."""

    def __init__(self, env, aggressiveness=1.0):
        super().__init__(env)
        self.aggressiveness = aggressiveness

    def predict(self, obs):
        """
        Adjust yaw angles based on measured wind direction at each turbine.

        Args:
            obs: Observation vector from environment

        Returns:
            action: Yaw adjustments
            state: None
        """
        # Parse observation (depends on your YAML config)
        # Assuming obs contains: [ws_t1, wd_t1, yaw_t1, power_t1, ws_t2, ...]
        n_turbines = self.env.n_wt
        obs_per_turbine = len(obs) // n_turbines

        actions = []
        for i in range(n_turbines):
            # Extract turbine i's observations
            idx = i * obs_per_turbine

            # Unscale observations (they're normalized -1 to 1)
            wd = self.unscale_measurement(obs[idx + 1], 'wd')
            current_yaw = self.unscale_measurement(obs[idx + 2], 'yaw')

            # Calculate desired yaw change to align with local wind
            yaw_error = wd - current_yaw
            desired_action = self.aggressiveness * yaw_error

            # Clip to reasonable limits
            desired_action = np.clip(desired_action, -10, 10)
            actions.append(desired_action)

        actions = np.array(actions)

        # Scale to environment action space
        actions = self.scale_yaw(actions)

        return actions, None

    def unscale_measurement(self, scaled_value, measurement_type):
        """Convert scaled observation back to physical units."""
        # This depends on your environment's scaling
        # Example implementation:
        if measurement_type == 'wd':
            return scaled_value * 180  # Assuming wd scaled to [-1, 1] from [-180, 180]
        elif measurement_type == 'yaw':
            return scaled_value * 30   # Assuming yaw range of [-30, 30]
        return scaled_value
```

---

## Reinforcement Learning Agents

### Using Stable Baselines3

Train a PPO agent:

```python
from stable_baselines3 import PPO
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

# Create training environment
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
    n_passthrough=2
)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_windfarm_agent")

# Test the trained agent
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Loading a Trained Model

```python
from stable_baselines3 import PPO
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80

# Load pre-trained model
model = PPO.load("ppo_windfarm_agent")

# Use in environment
env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
obs, info = env.reset()

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Agent Comparison Table

| Agent Type          | Use Case                | Requires Training | Computational Cost |
| ------------------- | ----------------------- | ----------------- | ------------------ |
| **PyWakeAgent**     | Optimal static baseline | No                | Medium             |
| **GreedyAgent**     | Simple reactive control | No                | Low                |
| **RandomAgent**     | Testing & baseline      | No                | Low                |
| **ConstantAgent**   | Fixed strategies        | No                | Very Low           |
| **Custom RL Agent** | Learning-based control  | Yes               | High               |

---

## Best Practices

1. **Start Simple**: Begin with built-in agents like `GreedyAgent` to understand environment dynamics
2. **Use PyWake as Baseline**: Always compare custom agents against `PyWakeAgent` performance
3. **Test on Multiple Conditions**: Evaluate across various wind speeds, directions, and turbulence levels
4. **Monitor Training**: Use tools like TensorBoard or Weights & Biases for RL agents
5. **Implement Curriculum Learning**: Gradually increase task difficulty for better convergence

---

## Related Examples

- [Example 2: Evaluate Pretrained Agent](../examples/Example Evaluate pretrained agent.ipynb)
- [PPO Training Example](../examples/ppo_curriculum_example.py)
- [Agent Comparison Scripts](../examples/compare_agents_grid.py)

---

## Next Steps

- Learn about [evaluation tools](evaluations.md)
- Explore [running simulations](simulations.md)
- Understand [noise and uncertainty](noise-and-uncertainty.md)
