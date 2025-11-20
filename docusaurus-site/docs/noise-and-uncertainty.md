# Noise, Uncertainty, and Advanced Agents

Real-world wind farm control systems must operate with imperfect information from noisy sensors. WindGym models this challenge through a modular system designed to introduce and manage measurement uncertainty. This allows for the development and testing of more robust, real-world-ready agents.

---

## 1. The Measurement & Noise System

The system is composed of two key components: the `MeasurementManager` and the `NoisyWindFarmEnv` wrapper.

### The `MeasurementManager`

The `MeasurementManager` is the brain behind the noise system. It is initialized with a clean environment instance and builds a detailed specification (`MeasurementSpec`) for every variable in the observation space. Its primary roles are:

- **Defining Noise Models**: It allows you to set a `NoiseModel` (e.g., `WhiteNoiseModel`, `EpisodicBiasNoiseModel`, or a `HybridNoiseModel`) that defines the statistical properties of the noise in physical units (e.g., a standard deviation of 2.0 degrees for wind direction).
- **Applying Noise**: At each step, it takes the clean observation from the base environment and applies the defined noise model to generate a noisy observation.

### The `NoisyWindFarmEnv` Wrapper

This is a Gymnasium `Wrapper` that orchestrates the noise application. Its workflow is:

1.  It contains a clean, underlying `WindFarmEnv` instance.
2.  On `reset()` or `step()`, it first calls the underlying environment to get a "ground truth" clean observation.
3.  It passes this clean observation to the `MeasurementManager`.
4.  The `MeasurementManager` applies the configured noise and returns a noisy observation.
5.  The wrapper then passes this noisy observation to the agent.
6.  Crucially, the wrapper adds both the `clean_obs` and detailed `noise_info` (including any biases applied) to the `info` dictionary, making it possible to analyze the difference between ground truth and the agent's perception.

---

## 2. The Nature of Yaw in Simulation

It's critical to understand how "yaw" is defined within the DYNAMIKS simulation backend versus how it might be measured in the real world.

- **Simulation State (Ground Truth):** In DYNAMIKS, a turbine's orientation is defined by its **yaw offset** relative to the **true, average wind direction**. A yaw offset of 0° means the turbine is perfectly aligned with the true average incoming flow.

- **Agent's Perception:** Any agent designed for deployment should not have access to this perfect ground truth direction information. This requires introducing synthetic errors when performing a desired action and when recording measured information.

---

## 3. How Measurement Errors Impact Control

Because the simulation's yaw is relative to the _true_ wind, a noisy global wind direction measurement does not directly affect the physics. However, it critically affects the agent's _decision-making process_.

- **`ActionMethod: "yaw"`**: The agent's action is a **change** in yaw offset (e.g., `+1°`). The execution of this action is unaffected by wind direction error. However, the agent's policy (especially a model-based one like `PyWakeAgent`) uses the sensed wind direction to calculate the _optimal_ yaw offset.

- **`ActionMethod: "wind"`**: The agent's action is a **target** yaw offset. The environment controller then works to move the turbine to this target.

---

## 4. How Measurement Errors Impact the Recorded Measurements

Several measurements are available during wind farm simulation. The turbine- and farm-level measured direction is obviously corrupted by uncertianty in direction.

WindGym currently only records the yaw position relative to the true wind direction. So, when there is uncertainty in wind direction, the yaw positions should also be confounded.

---

## 5. PyWake Agent under uncertainty

The `NoisyPyWakeAgent` is specifically designed to handle this challenge. It does not trust any single measurement. Instead, it:

1.  Receives the full, noisy observation vector.
2.  Identifies all measurements related to wind speed and wind direction (both turbine-level and farm-level).
3.  Un-scales and **averages** these values to create a single, more robust estimate of the true wind conditions.
4.  Uses this **estimated** wind condition to run its internal PyWake optimization to determine the best yaw angles to command.

- **`ActionMethod: "wind"`**: The agent's action is a **target** yaw offset relative to the sensed wind direction. The environment controller then works to move the turbine to this target. Because of the explicit dependancy on the sensed wind direction, the noisy pywake agent uses the sensed wind direction when determining the appropriate offset to make, leading to potentially large missalignments with the incoming flow.

- **`ActionMethod: "yaw"`**: Since the agent is meant to predict changes in the yaw postion, we assume this is relative to the turbine controller, which generally is assumed to not be vulnerable to errors in wind direction

---

## Practical Code Examples

### Basic Noisy Environment Setup

```python
from WindGym import WindFarmEnv
from WindGym.core import NoisyWindFarmEnv, MeasurementManager, WhiteNoiseModel
from py_wake.examples.data.hornsrev1 import V80

# Create a clean base environment
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)

# Create measurement manager with noise configuration
manager = MeasurementManager(base_env)

# Add white noise to wind direction measurements
manager.set_noise_model(
    measurement_type='wd',
    noise_model=WhiteNoiseModel(std=2.0)  # 2 degree standard deviation
)

# Add noise to wind speed measurements
manager.set_noise_model(
    measurement_type='ws',
    noise_model=WhiteNoiseModel(std=0.5)  # 0.5 m/s standard deviation
)

# Wrap the environment
noisy_env = NoisyWindFarmEnv(base_env, manager)

# Now use the noisy environment
obs, info = noisy_env.reset()
print(f"Noisy observation: {obs}")
print(f"Clean observation available in info: {info['clean_obs']}")
print(f"Applied noise: {info['noise_info']}")
```

### Episodic Bias Noise

```python
from WindGym.core import EpisodicBiasNoiseModel

# Create environment
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)

# Add episodic bias - same bias throughout one episode
manager.set_noise_model(
    measurement_type='wd',
    noise_model=EpisodicBiasNoiseModel(
        bias_std=5.0  # Bias drawn from N(0, 5°) at episode start
    )
)

noisy_env = NoisyWindFarmEnv(base_env, manager)

# First episode
obs, info = noisy_env.reset()
bias_1 = info['noise_info']['wd']['bias']

# Run episode - bias stays constant
for _ in range(10):
    obs, _, _, _, info = noisy_env.step(noisy_env.action_space.sample())
    assert info['noise_info']['wd']['bias'] == bias_1

# New episode - new bias
obs, info = noisy_env.reset()
bias_2 = info['noise_info']['wd']['bias']
assert bias_2 != bias_1  # Different bias
```

### Hybrid Noise Model

```python
from WindGym.noise.noise_models import HybridNoiseModel

base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)

# Combine white noise and episodic bias
hybrid_noise = HybridNoiseModel(
    white_noise_std=1.0,    # Random noise each step
    episodic_bias_std=3.0   # Consistent bias per episode
)

manager.set_noise_model(
    measurement_type='wd',
    noise_model=hybrid_noise
)

noisy_env = NoisyWindFarmEnv(base_env, manager)

obs, info = noisy_env.reset()
print(f"Episodic bias: {info['noise_info']['wd']['bias']:.2f}°")
print(f"Total noise: {info['noise_info']['wd']['total_noise']:.2f}°")
```

### Using NoisyPyWakeAgent

```python
from WindGym.Agents.NoisyPyWakeAgent import NoisyPyWakeAgent
from WindGym.wrappers.NoisyWindFarmEnv import NoisyWindFarmEnv
from WindGym.noise.measurement_manager import MeasurementManager
from WindGym.noise.noise_models import HybridNoiseModel

# Setup noisy environment
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)

# Add significant noise
manager.set_noise_model(
    'wd',
    HybridNoiseModel(white_noise_std=2.0, episodic_bias_std=5.0)
)
manager.set_noise_model(
    'ws',
    HybridNoiseModel(white_noise_std=0.3, episodic_bias_std=0.5)
)

noisy_env = NoisyWindFarmEnv(base_env, manager)

# NoisyPyWakeAgent averages multiple wind measurements
agent = NoisyPyWakeAgent(noisy_env)

# Run evaluation
obs, info = noisy_env.reset()
total_reward = 0

for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = noisy_env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Total reward with noisy observations: {total_reward:.2f}")
```

### Comparing Clean vs Noisy Performance

```python
import numpy as np
from WindGym.Agents.PyWakeAgent import PyWakeAgent
from WindGym.Agents.NoisyPyWakeAgent import NoisyPyWakeAgent

def evaluate_agent(env, agent, n_episodes=10):
    """Evaluate agent over multiple episodes."""
    episode_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0

        for _ in range(100):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)

# Clean environment
clean_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
clean_agent = PyWakeAgent(clean_env)

clean_mean, clean_std = evaluate_agent(clean_env, clean_agent, n_episodes=20)
print(f"Clean environment: {clean_mean:.2f} ± {clean_std:.2f}")

# Noisy environment
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)
manager.set_noise_model('wd', HybridNoiseModel(2.0, 5.0))
noisy_env = NoisyWindFarmEnv(base_env, manager)

# Regular PyWake agent (not robust to noise)
regular_agent = PyWakeAgent(noisy_env)
regular_mean, regular_std = evaluate_agent(noisy_env, regular_agent, n_episodes=20)
print(f"Regular agent with noise: {regular_mean:.2f} ± {regular_std:.2f}")

# Noisy PyWake agent (robust to noise)
robust_agent = NoisyPyWakeAgent(noisy_env)
robust_mean, robust_std = evaluate_agent(noisy_env, robust_agent, n_episodes=20)
print(f"Robust agent with noise: {robust_mean:.2f} ± {robust_std:.2f}")
```

### Custom Noise-Robust Agent

```python
from WindGym.Agents.BaseAgent import BaseAgent
import numpy as np

class MovingAverageAgent(BaseAgent):
    """Agent that uses moving average to filter noisy observations."""

    def __init__(self, env, window_size=5):
        super().__init__(env)
        self.window_size = window_size
        self.observation_history = []

    def predict(self, obs):
        """
        Predict action using filtered observations.

        Args:
            obs: Current (noisy) observation

        Returns:
            action: Control action
            state: None
        """
        # Add to history
        self.observation_history.append(obs)

        # Keep only recent observations
        if len(self.observation_history) > self.window_size:
            self.observation_history.pop(0)

        # Compute moving average
        if len(self.observation_history) > 1:
            smoothed_obs = np.mean(self.observation_history, axis=0)
        else:
            smoothed_obs = obs

        # Use smoothed observations for control decision
        # Example: simple proportional control based on wind direction
        n_turbines = self.env.n_wt

        # Extract wind directions (depends on observation structure)
        # This is a simplified example
        actions = np.zeros(n_turbines)

        # Scale actions
        actions = self.scale_yaw(actions)

        return actions, None

    def reset(self):
        """Clear observation history at episode start."""
        self.observation_history = []

# Use the custom agent
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)
manager.set_noise_model('wd', WhiteNoiseModel(std=3.0))
noisy_env = NoisyWindFarmEnv(base_env, manager)

agent = MovingAverageAgent(noisy_env, window_size=5)

obs, info = noisy_env.reset()
agent.reset()

for _ in range(100):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = noisy_env.step(action)
    if terminated or truncated:
        break
```

### Analyzing Noise Impact

```python
import matplotlib.pyplot as plt

def analyze_noise_impact(base_env, noise_levels, n_episodes=20):
    """
    Analyze how different noise levels affect agent performance.

    Args:
        base_env: Clean environment
        noise_levels: List of noise standard deviations to test
        n_episodes: Number of episodes per noise level

    Returns:
        results: Dictionary of performance metrics
    """
    agent = PyWakeAgent(base_env)
    results = {'noise_levels': [], 'mean_rewards': [], 'std_rewards': []}

    for noise_std in noise_levels:
        # Create noisy environment
        manager = MeasurementManager(base_env)
        manager.set_noise_model(
            'wd',
            WhiteNoiseModel(std=noise_std)
        )
        noisy_env = NoisyWindFarmEnv(base_env, manager)

        # Evaluate
        episode_rewards = []
        for _ in range(n_episodes):
            obs, info = noisy_env.reset()
            episode_reward = 0

            for _ in range(100):
                action, _ = agent.predict(obs)
                obs, reward, terminated, truncated, info = noisy_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)

        results['noise_levels'].append(noise_std)
        results['mean_rewards'].append(np.mean(episode_rewards))
        results['std_rewards'].append(np.std(episode_rewards))

    return results

# Run analysis
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
noise_levels = [0, 1, 2, 3, 5, 7, 10]  # degrees

results = analyze_noise_impact(base_env, noise_levels, n_episodes=10)

# Plot results
plt.figure(figsize=(10, 6))
plt.errorbar(
    results['noise_levels'],
    results['mean_rewards'],
    yerr=results['std_rewards'],
    marker='o',
    capsize=5
)
plt.xlabel('Wind Direction Noise (degrees)')
plt.ylabel('Mean Reward')
plt.title('Agent Performance vs Measurement Noise')
plt.grid(True, alpha=0.3)
plt.savefig('noise_impact_analysis.png')
plt.show()
```

### Training RL Agents with Noise

```python
from stable_baselines3 import PPO
from WindGym.wrappers.NoisyWindFarmEnv import NoisyWindFarmEnv
from WindGym.noise.measurement_manager import MeasurementManager
from WindGym.noise.noise_models import HybridNoiseModel

# Create noisy training environment
base_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
manager = MeasurementManager(base_env)

# Add realistic measurement uncertainty
manager.set_noise_model(
    'wd',
    HybridNoiseModel(white_noise_std=1.5, episodic_bias_std=3.0)
)
manager.set_noise_model(
    'ws',
    HybridNoiseModel(white_noise_std=0.3, episodic_bias_std=0.5)
)

train_env = NoisyWindFarmEnv(base_env, manager)

# Train PPO agent
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
)

print("Training agent with noisy observations...")
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_noise_robust_agent")

# Test on clean environment
clean_env = WindFarmEnv(
    turbine=V80(),
    x_pos=[0, 500, 1000],
    y_pos=[0, 0, 0],
    config="EnvConfigs/Env1.yaml",
)
obs, info = clean_env.reset()
clean_reward = 0

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = clean_env.step(action)
    clean_reward += reward
    if terminated or truncated:
        break

print(f"Performance on clean environment: {clean_reward:.2f}")

# Test on noisy environment
test_noisy_env = NoisyWindFarmEnv(
    WindFarmEnv(n_wt=3, ws=10.0, wd=270.0, TI=0.06),
    manager
)
obs, info = test_noisy_env.reset()
noisy_reward = 0

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_noisy_env.step(action)
    noisy_reward += reward
    if terminated or truncated:
        break

print(f"Performance on noisy environment: {noisy_reward:.2f}")
```

---

## Best Practices

### 1. Start with Realistic Noise Levels

Base noise levels on real sensor specifications:

```python
# Realistic noise levels for wind farm sensors
realistic_noise = {
    'wd': WhiteNoiseModel(std=2.0),      # ±2° typical for nacelle anemometers
    'ws': WhiteNoiseModel(std=0.3),      # ±0.3 m/s typical uncertainty
    'yaw': WhiteNoiseModel(std=0.5),     # ±0.5° yaw angle measurement
    'power': WhiteNoiseModel(std=50.0),  # ±50 W power measurement noise
}
```

### 2. Test Both White Noise and Bias

```python
# Test agent under different noise conditions
noise_configs = [
    ('white_only', WhiteNoiseModel(std=2.0)),
    ('bias_only', EpisodicBiasNoiseModel(bias_std=5.0)),
    ('hybrid', HybridNoiseModel(white_noise_std=2.0, episodic_bias_std=5.0)),
]

for name, noise_model in noise_configs:
    manager = MeasurementManager(base_env)
    manager.set_noise_model('wd', noise_model)
    noisy_env = NoisyWindFarmEnv(base_env, manager)

    # Evaluate agent
    mean_reward, _ = evaluate_agent(noisy_env, agent, n_episodes=10)
    print(f"{name}: {mean_reward:.2f}")
```

### 3. Validate Noise Models

```python
# Check that noise has correct statistical properties
measurements = []
for _ in range(1000):
    obs, info = noisy_env.reset()
    measurements.append(info['noise_info']['wd']['total_noise'])

print(f"Noise mean: {np.mean(measurements):.4f} (should be ~0)")
print(f"Noise std: {np.std(measurements):.4f}")
```

---

## Troubleshooting

### Issue: Agent Performance Degrades Significantly with Noise

**Solution**: Use robust agents or filtering:

```python
# Option 1: Use NoisyPyWakeAgent
agent = NoisyPyWakeAgent(noisy_env)

# Option 2: Implement observation filtering
# See MovingAverageAgent example above

# Option 3: Train with noise during development
# See Training RL Agents with Noise example above
```

### Issue: Episodic Bias Not Changing Between Episodes

**Solution**: Ensure environment is properly reset:

```python
# Check that reset() is called
obs, info = noisy_env.reset()  # This samples new bias
print(f"New bias: {info['noise_info']['wd']['bias']}")
```

---

## Related Examples

- [Noise Examples Directory](../examples/noise_examples/)
- [PyWake Agent with Noise](../examples/pywake_agent_with_noise.py)
- [Training with Adversarial Noise](../examples/noise_examples/train_adversary.py)

---

## Next Steps

- Learn about [agent development](agents.md)
- Explore [evaluation methods](evaluations.md)
- Review [simulation configuration](simulations.md)
