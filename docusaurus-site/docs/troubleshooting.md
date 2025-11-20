# Troubleshooting & FAQ

This page covers common issues and frequently asked questions about WindGym.

---

## Installation Issues

### Pixi Installation Fails

**Problem**: `pixi install` fails with dependency resolution errors

**Solutions**:

1. **Clear the cache and retry:**

   ```bash
   pixi clean cache-dir
   pixi install
   ```

2. **Check your pixi version:**

   ```bash
   pixi --version
   # Should be >= 0.10.0
   ```

3. **Update pixi:**

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

4. **Try installing with verbose output:**
   ```bash
   pixi install -vv
   ```

---

### Import Errors After Installation

**Problem**: `ModuleNotFoundError: No module named 'WindGym'`

**Solutions**:

1. **Ensure the environment is activated:**

   ```bash
   pixi shell
   python -c "import WindGym; print('Success!')"
   ```

2. **Check if WindGym is installed in editable mode:**

   ```bash
   pip list | grep WindGym
   # Should show: WindGym 0.0.2 /path/to/windgym
   ```

3. **Reinstall in editable mode:**
   ```bash
   pixi run install
   # or manually:
   pip install -e .
   ```

---

### Pixi Platform Not Supported

**Problem**: `Platform win-64 is not supported` when using pixi

**Solution**: The pixi configuration currently only includes Linux and macOS platforms. However, **WindGym itself is OS-independent** and can run on Windows. You have two options:

**Option 1 - Use pip directly on Windows:**

```bash
git clone https://gitlab.windenergy.dtu.dk/sys/windgym.git
cd windgym
pip install -e .
```

**Option 2 - Use WSL2 on Windows:**

```bash
# In WSL2
git clone https://gitlab.windenergy.dtu.dk/sys/windgym.git
cd windgym
pixi install
```

---

### Missing System Libraries

**Problem**: Errors related to HDF5, NetCDF, or other system libraries

**Solutions**:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install libhdf5-dev libnetcdf-dev build-essential
```

**macOS:**

```bash
brew install hdf5 netcdf
# If you get Xcode errors:
xcode-select --install
```

**Fedora/RHEL:**

```bash
sudo dnf install hdf5-devel netcdf-devel gcc-c++
```

---

## Runtime Issues

### Simulation Runs Very Slowly

**Problem**: Episodes take too long to complete

**Solutions**:

1. **Increase timesteps:**

   ```python
   env = WindFarmEnv(
       dt_sim=0.2,    # Increase from 0.1
       dt_env=2.0,    # Increase from 1.0
   )
   ```

2. **Reduce episode length:**

   ```python
   env = WindFarmEnv(
       n_passthrough=1,  # Reduce from 3
   )
   ```

3. **Use simpler turbulence:**

   ```python
   env = WindFarmEnv(
       turbtype='random',  # Instead of 'mann'
   )
   ```

4. **Reduce number of turbines:**
   ```python
   env = WindFarmEnv(n_wt=3)  # Start small
   ```

---

### Memory Issues / Out of Memory

**Problem**: Python crashes with memory errors

**Solutions**:

1. **Reduce history length in observations:**

   ```yaml
   # In config.yaml
   observations:
     include_history: true
     history_length: 5 # Reduce from 10
   ```

2. **Limit turbulence box size:**
   - Use pre-generated smaller turbulence boxes
   - Reduce simulation duration

3. **Close environments properly:**

   ```python
   env.close()  # Always close when done
   ```

4. **Process episodes in batches:**
   ```python
   # Instead of collecting all episodes at once
   for episode in range(100):
       results = run_episode()
       save_results(results)  # Save incrementally
       del results  # Free memory
   ```

---

### Reward is Always Zero or Negative

**Problem**: Agent receives no positive reward

**Solutions**:

1. **Check reward configuration:**

   ```yaml
   power_reward: "Baseline" # Make sure this is set
   Power_scaling: 1.0 # Adjust scaling
   ```

2. **Verify baseline is enabled for "Baseline" reward:**

   ```python
   env = FarmEval(
       Baseline_comp=True  # Required for Baseline reward
   )
   ```

3. **Check action penalties:**

   ```yaml
   penalty_scaling: 0.01 # If too high, reduces reward significantly
   ```

4. **Verify turbines are producing power:**
   ```python
   obs, info = env.reset()
   obs, reward, _, _, info = env.step(env.action_space.sample())
   print(f"Power: {info.get('power', 'N/A')}")
   print(f"Reward: {reward}")
   ```

---

### Actions Have No Effect

**Problem**: Changing actions doesn't affect the simulation

**Solutions**:

1. **Check action method configuration:**

   ```yaml
   ActionMethod: "yaw" # or "wind"
   yaw_step_env: 3.0 # Must be > 0
   ```

2. **Verify action scaling:**

   ```python
   # Actions should be in [-1, 1]
   action = np.array([1.0, -1.0, 0.0])  # Valid
   action = np.array([10.0, -5.0, 0.0])  # Invalid - will be clipped
   ```

3. **Check if actions are within limits:**
   ```yaml
   yaw_min: -30
   yaw_max: 30
   # Actions outside this range have no additional effect
   ```

---

### Observations Contain NaN or Inf

**Problem**: `obs` contains invalid values

**Solutions**:

1. **Check wind conditions are valid:**

   ```python
   env = WindFarmEnv(
       ws=10.0,   # Must be > 0
       wd=270.0,  # Should be 0-360
       TI=0.06,   # Must be > 0
   )
   ```

2. **Verify turbine positions don't overlap:**

   ```python
   x_pos = np.array([0, 500, 1000])  # Good spacing
   y_pos = np.array([0, 0, 0])
   ```

3. **Check for simulation instabilities:**
   - Reduce `dt_sim`
   - Increase `burn_in_passthroughs`

---

## Agent Issues

### PyWakeAgent Performs Poorly

**Problem**: PyWakeAgent doesn't improve over baseline

**Possible Causes**:

1. **PyWake optimization failed silently:**

   ```python
   agent = PyWakeAgent(env)
   action, _ = agent.predict(obs)
   print(f"Action: {action}")  # Check if all zeros
   ```

2. **Wind conditions changed but agent didn't adapt:**
   - PyWakeAgent computes static optimal yaw angles
   - Doesn't adapt to changing wind during episode
   - Use stochastic evaluation to test across conditions

3. **Action method incompatibility:**
   - PyWakeAgent works best with `ActionMethod: "wind"`

---

### Trained RL Agent Not Learning

**Problem**: RL agent's performance doesn't improve during training

**Solutions**:

1. **Check reward signal:**

   ```python
   obs, _ = env.reset()
   for _ in range(100):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       print(f"Step reward: {reward}")
   ```

2. **Verify observation/action spaces:**

   ```python
   print(f"Observation space: {env.observation_space}")
   print(f"Action space: {env.action_space}")
   # Should be Box spaces
   ```

3. **Adjust hyperparameters:**

   ```python
   model = PPO(
       "MlpPolicy",
       env,
       learning_rate=3e-4,  # Try different values
       n_steps=2048,         # Increase for longer episodes
       batch_size=64,
       gamma=0.99,
   )
   ```

4. **Use curriculum learning:**

   ```python
   from WindGym.wrappers import CurriculumWrapper
   env = CurriculumWrapper(base_env, initial_difficulty=0.3)
   ```

5. **Check for observation scaling issues:**
   - Observations should be in [-1, 1]
   - Verify in YAML configuration

---

## Noise and Uncertainty Issues

### Noise Not Being Applied

**Problem**: Observations appear clean despite noise configuration

**Solutions**:

1. **Check wrapper order:**

   ```python
   # Correct order
   base_env = WindFarmEnv(...)
   manager = MeasurementManager(base_env)
   noisy_env = NoisyWindFarmEnv(base_env, manager)

   # Incorrect - noise configured but not used
   base_env = WindFarmEnv(...)
   manager = MeasurementManager(base_env)
   # Missing: noisy_env = NoisyWindFarmEnv(base_env, manager)
   ```

2. **Verify noise model is set:**
   ```python
   manager.set_noise_model('wd', WhiteNoiseModel(std=2.0))
   # Check what noise is applied:
   obs, info = noisy_env.reset()
   print(f"Noise info: {info['noise_info']}")
   ```

---

### Episodic Bias Not Changing

**Problem**: Same bias value across multiple episodes

**Solution**: Ensure `reset()` is called:

```python
# Correct
for episode in range(10):
    obs, info = env.reset()  # Samples new bias
    print(f"Episode {episode} bias: {info['noise_info']['wd']['bias']}")

# Incorrect - no reset between episodes
obs, info = env.reset()
for episode in range(10):
    # Missing reset() - bias stays same
    ...
```

---

## Evaluation Issues

### Coliseum Evaluation Crashes

**Problem**: `Coliseum.run_time_series_evaluation()` crashes

**Solutions**:

1. **Check environment factory returns fresh environment:**

   ```python
   def create_env():
       return FarmEval(n_wt=3, ws=10.0, wd=270.0)  # New instance

   # Don't do this:
   def create_env():
       return my_env  # Reuses same instance - bad!
   ```

2. **Verify agents are initialized correctly:**

   ```python
   temp_env = create_env()
   agents = {
       'PyWake': PyWakeAgent(temp_env),  # Use temp env
   }
   ```

3. **Check for memory issues:**
   - Reduce `n_episodes`
   - Set `save_histories=False`

---

### Grid Evaluation Takes Too Long

**Problem**: `run_wind_grid_evaluation()` runs for hours

**Solutions**:

1. **Reduce grid resolution:**

   ```python
   wind_speeds = np.arange(8, 12, 2)  # Fewer points
   wind_directions = [270]             # Single direction
   ```

2. **Reduce episodes per condition:**

   ```python
   coliseum.run_wind_grid_evaluation(
       ...,
       n_episodes_per_condition=1  # Reduce from 5
   )
   ```

3. **Use shorter episodes:**
   ```python
   def fast_env_factory():
       return FarmEval(n_wt=3, n_passthrough=1)
   ```

---

## Frequently Asked Questions

### Q: What Python versions are supported?

**A:** Python 3.7 through 3.11. Python 3.12+ is not yet supported due to dependency constraints.

---

### Q: Can I use WindGym with other RL libraries besides Stable Baselines3?

**A:** Yes! WindGym follows the standard Gymnasium interface, so it's compatible with:

- CleanRL
- RLlib (Ray)
- TorchRL
- Any library that supports Gymnasium environments

---

### Q: How do I save and load trained agents?

**A:** Depends on the library:

**Stable Baselines3:**

```python
# Save
model.save("my_agent")

# Load
from stable_baselines3 import PPO
model = PPO.load("my_agent")
```

**PyTorch models:**

```python
# Save
torch.save(agent.state_dict(), "agent.pth")

# Load
agent.load_state_dict(torch.load("agent.pth"))
```

---

### Q: Can I use real wind data?

**A:** Yes! Use PyWake's `Site` objects:

```python
from py_wake.site import UniformSite

site = UniformSite(
    p_wd=[...],  # Your wind direction frequencies
    a=[...],     # Your Weibull parameters
    k=[...],
)

env = WindFarmEnv(sample_site=site)
```

---

### Q: How do I visualize the flow field?

**A:**

```python
env = FarmEval(...)

# Run episode
obs, _ = env.reset()
for _ in range(50):
    obs, _, _, _, _ = env.step(agent.predict(obs)[0])

# Plot flow field
env.plot_flow_field(time_idx=-1, save_path='flowfield.png')
```

---

### Q: What's the difference between `WindFarmEnv` and `FarmEval`?

**A:**

- **`WindFarmEnv`**: Base environment for training
- **`FarmEval`**: Wrapper that adds evaluation features:
  - Detailed results tracking
  - Baseline comparison
  - Flow field visualization
  - Power time series

Use `FarmEval` for evaluation, `WindFarmEnv` for training.

---

### Q: How many turbines can WindGym handle?

**A:** Depends on your hardware and configuration:

- **3-6 turbines**: Fast, suitable for development and testing
- **10-20 turbines**: Moderate, realistic farm sizes
- **50+ turbines**: Slow, requires significant computational resources

For large farms, consider:

- Using larger `dt_sim` and `dt_env`
- Simplifying turbulence (`turbtype='random'`)
- Reducing observation history

---

### Q: Can I customize the reward function?

**A:** Yes, by:

1. **Using built-in options in YAML:**

   ```yaml
   power_reward: "Power_avg" # or "Baseline", "Power_diff"
   Power_scaling: 1.0
   action_penalty: "Change"
   penalty_scaling: 0.01
   ```

2. **Subclassing the environment:**
   ```python
   class CustomRewardEnv(WindFarmEnv):
       def _compute_reward(self, ...):
           # Your custom logic
           return custom_reward
   ```

---

### Q: How do I cite WindGym in my paper?

**A:**

```bibtex
@software{windgym2023,
  title = {WindGym: Reinforcement Learning Environment for Wind Farm Control},
  author = {Nilsen, Marcus and Quick, Julian and Simutis, Ernestas and Åstrand, Teodor},
  year = {2023},
  url = {https://gitlab.windenergy.dtu.dk/sys/windgym}
}
```

---

## Getting More Help

If your issue isn't covered here:

1. **Search existing issues:** [GitLab Issues](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues)
2. **Check the examples:** [Examples directory](../examples/README.md)
3. **Review the documentation:** [Full documentation](https://sys.pages.windenergy.dtu.dk/windgym/)
4. **Open a new issue:** [New issue](https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues/new)

When reporting issues, please include:

- WindGym version (`python -c "import WindGym; print(WindGym.__version__)"`)
- Python version
- Operating system
- Minimal code example that reproduces the issue
- Complete error message and traceback

---

## Related Pages

- [Installation Guide](installation.md)
- [Core Concepts](concepts.md)
- [API Reference](api-reference.md)
- [Developer Guidelines](developer-guidelines.md)
