# Power tracking

Power tracking turns WindGym from a power-*maximization* problem into a
power-*following* problem: the farm receives a reference power signal (like a
grid or AGC dispatch command) and is rewarded for matching it. Derating
provides the natural actuator, but any action set works — exceeding the greedy
farm power via wake steering is a legitimate way to reach a high reference.

---

## Enabling it

```yaml
Track_power: true

power_def:
  Power_reward: "None"   # required: tracking replaces the power reward
  Power_avg: 10          # shared averaging window (see reward below)
```

`Track_power: true` together with any power reward other than `"None"` raises
a `ValueError` at construction — tracking and maximization are mutually
exclusive objectives. Tracking does **not** require `Baseline_comp`: no
baseline farm is simulated unless you ask for one.

Multi-agent note: `WindFarmEnvMulti` raises `NotImplementedError` when
`Track_power` is set — its per-agent farm observation block is built through a
different path and would silently drop the tracking observations.

## The reference signal

The reference is resolved per env step (every `delay` seconds) and precomputed
for the whole episode at reset.

**Default sampler** (no callable given): a constant setpoint per episode,
drawn uniformly from `track_ref_range` (default `[0.2, 0.8]`) times the
episode *freestream* farm power `turbine.power(ws) * n_turb`. Because it is a
fraction of what the farm could produce in this episode's wind, the setpoint
is feasible by construction.

**Custom references** via the `power_ref_function` constructor kwarg,
mirroring `wd_function`. It receives the episode time in seconds and the env,
and returns watts:

```python
# Constant 4 MW
env = WindFarmEnv(..., power_ref_function=lambda t, env: 4e6)

# Ramp: 30% -> 70% of the episode freestream farm power over 300 s
def ramp(t, env):
    frac = 0.3 + 0.4 * min(t / 300.0, 1.0)
    return frac * env.rated_power * env.n_turb

# Stochastic steps, reproducible via the env seed
def steps(t, env):
    if t % 100 == 0:
        env._track_level = env.np_random.uniform(0.3, 0.7)
    return getattr(env, "_track_level", 0.5) * env.rated_power * env.n_turb

env = WindFarmEnv(..., power_ref_function=ramp)
```

The callable may use `env.np_random`, `env.ws`, `env.n_turb` and
`env.rated_power` (the freestream power of one turbine at the episode wind
speed, set per reset). It is evaluated in step order, once per step index, so
RNG draws are reproducible from the env seed.

`FarmEval` also accepts `power_ref_function`, so custom references are
**evaluable** (via `AgentEval`/`AgentEvalFast`), not just trainable — pass the
same callable you trained with to evaluate an agent against a time-varying
command.

Seeding caveat: enabling tracking adds one RNG draw at reset (the default
sampler), so a tracking env is seed-deterministic internally but its episodes
are not observation-identical to a non-tracking env with the same seed.

## Reward

Both forms compare **window means**: the farm side is the mean of the power
deque (window `Power_avg`), and the reference side is a parallel deque with
the same window. This keeps a reference step-change from causing an
unavoidable penalty spike — the error the agent is judged on moves as fast as
the power measurement itself.

With `P_norm = maxturbpower * n_turb` (rated/nameplate farm power, fixed per
farm — *not* the per-episode freestream power):

- `Track_reward: "abs"` (default):
  `r = -|mean(P_farm) - mean(P_ref)| / P_norm` — always ≤ 0, linear in the
  error, bounded by roughly −1.
- `Track_reward: "gaussian"`:
  `r = exp(-((mean(P_farm) - mean(P_ref)) / (sigma * P_norm))^2)` — in (0, 1],
  peaked at zero error; `track_sigma` (default 0.1) sets the width as a
  fraction of `P_norm`.

Like every power reward, the tracking reward is multiplied by `Power_scaling`
before the penalties are applied. Both forms above are already normalized by
`P_norm`, so the default `Power_scaling: 1.0` is the natural choice — raise it
only if you deliberately want to reweight tracking against the penalties.

The existing yaw `action_penalty` and `derate_penalty` are still subtracted,
so tracking composes with derating exactly like the maximization rewards do.
The reward breakdown gains a `tracking_error` entry (window-mean error in W).

### Config reference

```yaml
track_def:                       # all optional
  Track_reward: "abs"            # "abs" or "gaussian"
  track_sigma: 0.1               # gaussian width (fraction of rated farm power)
  track_ref_range: [0.2, 0.8]    # default sampler range (fraction of episode
                                 # freestream farm power)
  track_obs_setpoint: true       # observe the current reference
  track_obs_error: true          # observe farm power - reference
  track_obs_preview: 0           # observe the next k reference steps
```

## Observations

Three independently toggleable farm-level entries, appended after the farm
block of the observation vector:

- **setpoint** — current reference, scaled over `[0, maxturbpower * n_turb]`
  (same scaling as the farm power observation);
- **error** — instantaneous `P_farm - P_ref`, scaled over
  `[-maxturbpower * n_turb, +maxturbpower * n_turb]`;
- **preview** — the next `track_obs_preview` reference values, scaled like the
  setpoint. After step *i* the preview covers steps *i+1 … i+k*, so an agent
  can anticipate ramps instead of reacting to them.

These are commands, not sensor readings — measurement noise models pass them
through untouched.

## Info dict and evaluation

When tracking is on, `info` gains:

- `"Power reference"` — the current setpoint (W),
- `"Tracking error"` — instantaneous `P_farm - P_ref` (W) at the current step,
- `"Tracking error window mean"` — the window-mean error
  `mean(P_farm) - mean(P_ref)` (W), i.e. the exact quantity the reward
  normalizes (matches the `tracking_error` reward-breakdown entry),
- `"Power reference preview"` — the next `track_obs_preview` references.

`AgentEval`/`AgentEvalFast` datasets gain `power_ref` and `track_err` (same
time resolution as `powerF_a`) and a per-condition scalar `track_mae`
(mean absolute tracking error), which merges across conditions in
`eval_multiple` like any other variable.

## Where things live

- `examples/Example 6 Power tracking.ipynb` — end to end: derate-capable env,
  default sampler, custom ramp reference, tracking plots.
- `WindGym/core/power_tracking.py` — `PowerTrackingManager` (reference
  generation, preview, reference window deque).
- `WindGym/core/reward_calculator.py` — tracking reward forms + mutual
  exclusion validation.
- `WindGym/core/mes_class.py` — tracking observation entries (`FarmMes`).
- `WindGym/wind_farm_env.py` — config parsing (`track_def`), per-step wiring
  (`_push_tracking`), info dict.
