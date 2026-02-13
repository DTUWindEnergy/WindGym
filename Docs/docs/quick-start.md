# Quick Start

WindGym ships with **presets** — factory functions that create fully configured wind farm environments in a single call, so you can skip the configuration boilerplate and start experimenting immediately.

---

## Minimal Example

```python
from WindGym.presets import three_turbine_row

env = three_turbine_row()
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

This creates a 3-turbine row of V80 turbines with 5D spacing — the same layout used in the paper examples.

---

## Available Presets

| Preset | Layout | Turbines | Use case |
|---|---|---|---|
| `three_turbine_row` | 3 × 1 row, 5D spacing | 3 | Tutorials, sanity checks |
| `two_by_two_grid` | 2 × 2 grid, 4D spacing | 4 | Small-scale wake steering research |
| `nine_turbine_grid` | 3 × 3 grid, 5D spacing | 9 | Larger-scale wake experiments |

All presets use Vestas V80 turbines and the dynamic wake meandering backend by default.

---

## Customisation

Every preset accepts `**kwargs` that are forwarded directly to `WindFarmEnv`, so you can override any constructor parameter:

```python
from WindGym.presets import three_turbine_row

# Use the PyWake steady-state backend with rendering enabled
env = three_turbine_row(backend="pywake", render_mode="human")

# Longer episodes via more flow passthroughs
env = three_turbine_row(n_passthrough=10)
```

---

## Next Steps

- Read the [Core Concepts](./concepts.md) page to understand observations, actions, and rewards
- Try the [Quick Start notebook](https://gitlab.windenergy.dtu.dk/sys/windgym/-/blob/main/examples/Quick%20Start.ipynb) for an interactive walkthrough
