# Single-Agent vs Multi-Agent Environments

WindGym provides two environment flavors for the same underlying wind farm: a **single-agent**
environment, `WindFarmEnv`, and a **multi-agent** environment, `WindFarmEnvMulti`. They simulate
identical physics and accept the same configuration. What changes is *who is in control*. The
single-agent env exposes one controller for the whole farm through the standard
[Gymnasium](https://gymnasium.farama.org/) API, while the multi-agent env exposes one controller
per turbine through the [PettingZoo](https://pettingzoo.farama.org/) Parallel API.

This page explains what each one is, where they are the same, where they differ, and how to run
many copies of either in parallel for training.

---

## A shared foundation

Both environments are built on the same simulation core, so most of what you know about one
carries over to the other:

- **Same simulator and physics.** Both run on the `dynamiks` (dynamic wake meandering) or
  `pywake` (steady-state) backend, with the same turbulence modeling and measurement system.
- **Same configuration.** Both take the same `config` (a YAML file or dict) describing the farm
  layout, wind conditions, and measurements.
- **Same control quantity.** In both, an action is a normalized **yaw** command per turbine.
- **Literally shared code.** `WindFarmEnvMulti` subclasses `WindFarmEnv` and reuses the
  single-agent simulation core and only re-shapes the inputs and outputs to match the PettingZoo
  Parallel API. Anything you change in the core environment is reflected in both.

The difference is purely in the *interface*: how observations and actions are grouped, and which
API the environment speaks.

---

## The single-agent environment (`WindFarmEnv`)

`WindFarmEnv` is a Gymnasium environment (`gymnasium.Env`). A single agent observes the whole
farm and controls **every** turbine at once: the action is one flat vector of yaw values, one per
turbine, and the observation is one flat vector of all turbine- and farm-level measurements.

```python
from WindGym import WindFarmEnv

env = WindFarmEnv(config="config.yaml")
obs, info = env.reset()

# Action: a single flat vector with one yaw command per turbine.
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

This is the right choice when you want a **single policy** that decides for the entire farm, and
it plugs directly into the standard single-agent RL ecosystem (e.g. Stable-Baselines3).

---

## The multi-agent environment (`WindFarmEnvMulti`)

`WindFarmEnvMulti` implements the PettingZoo **`ParallelEnv`** API, where every agent acts at the
same time. There is **one agent per turbine** (`"turbine_0"`, `"turbine_1"`, …). Each agent
observes its own turbine plus the shared farm-level measurements and controls only its own yaw.
Observations, actions, rewards, and dones are therefore **dictionaries keyed by agent**.

```python
from WindGym import WindFarmEnvMulti

env = WindFarmEnvMulti(config="config.yaml")
observations, infos = env.reset()   # dicts: {"turbine_0": obs, "turbine_1": obs, ...}

# Each agent supplies its own action.
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

Use this when you want **per-turbine policies**: cooperative or competitive multi-agent control
where each turbine reasons locally. For a worked, end-to-end example, see the Multi-Agent
Simulations section of the [Simulations guide](simulations.md).

---

## Similarities and differences

| Aspect | `WindFarmEnv` (single-agent) | `WindFarmEnvMulti` (multi-agent) |
|---|---|---|
| API / base class | Gymnasium `gymnasium.Env` | PettingZoo `ParallelEnv` (subclasses `WindFarmEnv`) |
| Unit of control | One agent controls the whole farm | One agent per turbine |
| Observation | One flat vector for the farm | A dict `{agent: vector}` (own turbine + farm-level obs) |
| Action | One flat vector (one yaw per turbine) | A dict `{agent: yaw}` (one scalar per turbine) |
| `step()` returns | `obs, reward, terminated, truncated, info` | dicts of `obs, rewards, terminations, truncations, infos` |
| Simulator, physics, config | identical | identical |
| Parallelization | Gymnasium / SB3 vectorization | Custom PettingZoo wrapper (see below) |

In short: same farm, same physics, same yaw control; the multi-agent env just splits the single
flat observation/action into per-turbine pieces and speaks the PettingZoo dialect.

---

## Running many environments in parallel

Training is much faster when many environment copies step at once. The pattern is the same in
both cases: you build a **list of factory callables**, each of which constructs a fresh
environment. Only the runner differs, because Gymnasium and PettingZoo offer different tooling.

### Single-agent: Gymnasium / SB3 vectorization

Because `WindFarmEnv` is a standard Gymnasium env, you can use off-the-shelf vectorization:
Gymnasium's `SyncVectorEnv` / `AsyncVectorEnv`, or, as the training examples in this repo do,
Stable-Baselines3's `SubprocVecEnv`, which runs each env in its own subprocess.

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from WindGym import WindFarmEnv

def make_env_factory(rank):
    def _init():
        return WindFarmEnv(config="config.yaml", seed=rank)
    return _init

env_factories = [make_env_factory(i) for i in range(n_envs)]
vec_env = SubprocVecEnv(env_factories)
```

### Multi-agent: the PettingZoo parallel wrapper

PettingZoo does **not** ship a built-in vectorized/subprocess runner equivalent to Gymnasium's
vector envs, so WindGym provides its own: `ParallelPettingZooMultiprocessingWrapper`. It runs N
`WindFarmEnvMulti` instances, **one per subprocess**, communicating over multiprocessing pipes,
and combines each per-agent output across all environments. This is essentially a vectorized env,
but for multi-agent PettingZoo environments.

```python
from WindGym import WindFarmEnvMulti
from WindGym.wrappers.parallel_PettingZoo_wrapper import (
    ParallelPettingZooMultiprocessingWrapper,
)

def make_env_factory(rank):
    def _init():
        return WindFarmEnvMulti(config="config.yaml", seed=rank)
    return _init

env_fns = [make_env_factory(i) for i in range(n_envs)]
vec_env = ParallelPettingZooMultiprocessingWrapper(env_fns)

observations, infos = vec_env.reset()
# actions[agent] is a list with one action per environment.
# ...training loop...
vec_env.close()
```

Note the symmetry: both approaches take a list of env-factory callables. The single-agent path
reuses standard Gymnasium tooling; the multi-agent path uses WindGym's wrapper to fill the gap
PettingZoo leaves.

---

## When to use which

- **Use `WindFarmEnv`** when a single policy should control the whole farm, or when you want to
  reuse the mature single-agent RL ecosystem (Stable-Baselines3, Gymnasium vector envs, the
  standard wrappers).
- **Use `WindFarmEnvMulti`** when you want per-turbine agents (cooperative or competitive
  multi-agent control) and need the PettingZoo Parallel API and its tooling.

Because both share the same core, you can prototype the physics and configuration with one and
switch to the other without changing your farm, wind, or measurement setup.

---

See also: [Core Concepts](concepts.md), [Simulations](simulations.md), and [Agents](agents.md).
