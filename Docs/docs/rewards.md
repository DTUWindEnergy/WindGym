# Rewards

The reward function is the primary signal that guides an RL agent's learning. In WindGym, the total reward at each environment step is composed of two parts:

`total_reward = (power_reward × Power_scaling) − action_penalty − derate_penalty`

The **power reward** quantifies how well the agent is controlling the farm's power output, and the optional **action penalty** discourages excessive yaw adjustments. When the [derating action](derating.md) is enabled, an optional **derate penalty** does the same for derate levels.

---

## 1. Power Reward Types

WindGym supports five power reward types, configured via the `Power_reward` key inside the `power_def` section of the YAML config file.

### 1.1 Baseline

Compares the agent-controlled farm to a parallel baseline farm (e.g., greedy yaw control).

- **Config value:** `Power_reward: "Baseline"`
- **Requires:** `Baseline_comp: True` in the environment config

**Formula:**

`reward = (P_agent / P_baseline) − 1`

where `P_agent` and `P_baseline` are the windowed average power of the agent and baseline farms respectively.

**Behavior:**
- `0` means the agent matches the baseline exactly
- Positive values mean the agent outperforms the baseline
- Negative values mean the agent underperforms

**When to use:** This is the default choice for wake steering research. It directly measures how much the agent improves over a conventional controller.

**Config example:**
```yaml
power_def:
  Power_reward: "Baseline"
  Power_scaling: 1.0
  Power_avg: 10
```

### 1.2 Wake_recovery

Measures how much of the available wake loss the agent recovers, normalized by the theoretical headroom.

- **Config value:** `Power_reward: "Wake_recovery"`
- **Requires:** `Baseline_comp: True` in the environment config (which provides both greedy baseline and no-wake freestream power)

**Formula:**

`gain = P_agent − P_greedy`

`headroom = max(P_freestream − P_greedy, tau × P_freestream)`

`reward = gain / headroom`

where:
- `P_agent` is the average power of the agent-controlled farm
- `P_greedy` is the average power of the greedy baseline farm
- `P_freestream` is the average power of the baseline turbines with wakes disabled (each turbine sees undisturbed flow)
- `tau` is a floor parameter (default `0.02`) that prevents division by near-zero when wake losses are small

**Behavior:**
- `0` means the agent matches the greedy baseline
- `1` means the agent recovers all available wake loss
- Values above `1` are possible if the agent exceeds the theoretical no-wake power (rare)

**Why the tau floor?** In cross-wind conditions or at low wind speeds, the difference `P_freestream − P_greedy` can be very small, making the reward noisy and unstable. The `tau × P_freestream` floor ensures a minimum denominator, keeping the reward signal well-behaved.

**When to use:** When you need a reward signal that is comparable across different wind speeds and directions. The normalization by headroom makes the reward condition-invariant, which is valuable for training agents that generalize well.

**Config example:**
```yaml
power_def:
  Power_reward: "Wake_recovery"
  Power_scaling: 1.0
  Power_avg: 10
  tau: 0.02
```

### 1.3 Power_avg

A simple absolute measure of farm power output as a fraction of rated capacity.

- **Config value:** `Power_reward: "Power_avg"`
- **Requires:** No baseline needed

**Formula:**

`reward = P_agent / (n_turbines × P_rated)`

where `P_agent` is the average farm power, `n_turbines` is the number of turbines, and `P_rated` is the rated power of a single turbine.

**Behavior:**
- Range is `[0, 1]` under normal conditions
- `1` means every turbine is producing at rated capacity

**When to use:** When you want a simple, absolute power metric that does not depend on a baseline simulation. Useful as this is significantly faster to run, as there is now no baseline farm.

**Config example:**
```yaml
power_def:
  Power_reward: "Power_avg"
  Power_scaling: 1.0
  Power_avg: 10
```

### 1.4 Power_diff

Rewards the agent for improving power over the course of an episode by comparing recent power to older power.

- **Config value:** `Power_reward: "Power_diff"`
- **Requires:** `Power_avg >= 40` (the power window must be large enough to split into comparison windows)

**Formula:**

`reward = (P_recent − P_older) / n_turbines`

where `P_recent` is the mean power over the most recent `Power_avg / 10` steps, and `P_older` is the mean power over the oldest `Power_avg / 10` steps of the power window.

**Behavior:**
- Positive when power is improving
- Zero when power is stable
- Negative when power is declining

**When to use:** For self-improvement or curriculum learning scenarios where you want the agent to learn to increase power over time, regardless of the absolute level.

**Config example:**
```yaml
power_def:
  Power_reward: "Power_diff"
  Power_scaling: 1.0
  Power_avg: 50
```

### 1.5 None

Returns `0.0` (no power reward is applied).

- **Config value:** `Power_reward: "None"`

**When to use:** When you are using a custom reward wrapper around the environment, when you want a penalty-only reward signal, or when [power tracking](power_tracking.md) is enabled (which requires it).

**Config example:**
```yaml
power_def:
  Power_reward: "None"
  Power_scaling: 1.0
  Power_avg: 50
```

### 1.6 Power tracking

When `Track_power: true` is set, the power reward is replaced by a *tracking*
reward that rewards matching a farm power reference instead of maximizing
power. It is **mutually exclusive** with every power reward type above:
`Track_power: true` together with a `Power_reward` other than `"None"` raises
a `ValueError` at construction.

Both forms compare the `Power_avg`-window mean of the farm power against the
same-window mean of the reference, with `P_norm` the rated (nameplate) farm
power:

- `Track_reward: "abs"` (default): `r = -|P̄_farm − P̄_ref| / P_norm`
- `Track_reward: "gaussian"`: `r = exp(-((P̄_farm − P̄_ref) / (sigma · P_norm))²)`

Like the maximization rewards, the tracking reward is multiplied by
`Power_scaling` (§2) before the penalties are subtracted; both forms are
already normalized by `P_norm`, so keep it at `1.0` unless you deliberately
want to reweight tracking against the penalties. The reward breakdown gains a
`tracking_error` entry. See [Power tracking](power_tracking.md) for the
reference signal, observations, and the full `track_def` config reference.

**Config example:**
```yaml
Track_power: true
power_def:
  Power_reward: "None"
  Power_avg: 10
track_def:
  Track_reward: "abs"   # or "gaussian" (+ track_sigma)
```

---

## 2. Power Scaling

The raw power reward is multiplied by a scaling factor before being combined with the action penalty:

`scaled_power_reward = power_reward × Power_scaling`

- **Config key:** `Power_scaling` (inside `power_def`)
- **Typical values:** All reward types are scaled to be resonable values, so no real scaling should be needed, besides `1.0`.

---

## 3. Power Averaging Window

The `Power_avg` parameter controls how many environment steps are averaged when computing power values for the reward.

- **Config key:** `Power_avg` (inside `power_def`)
- **Trade-off:** A larger window smooths out turbulence-induced fluctuations, giving a more stable reward signal but slower responsiveness to the agent's actions. A smaller window is more responsive but noisier.
- **Constraint:** Must be `>= 40` when using `Power_diff` reward type.

---

## 4. Action Penalty

The action penalty discourages excessive or erratic yaw adjustments. It is subtracted from the scaled power reward.

### 4.1 Change-based penalty

Penalizes the magnitude of yaw changes between steps:

`penalty = action_penalty × mean(|yaw_old − yaw_new|)`

This encourages stable control: the agent is penalized for large yaw adjustments.

### 4.2 Total-based penalty

Penalizes the absolute yaw magnitude, normalized by the maximum yaw:

`penalty = action_penalty × mean(|yaw_new|) / yaw_max`

This encourages alignment with the wind direction: the agent is penalized for maintaining large yaw offsets.

**Config example:**
```yaml
act_pen:
  action_penalty: 0.1
  action_penalty_type: "Change"  # or "Total"
```

:::note
If `action_penalty` is less than `0.001`, the penalty calculation is skipped entirely.
:::

### 4.3 Derate penalty

When the [derating action](derating.md) is enabled, an analogous optional penalty applies to the derate levels, with the same two types:

`penalty = derate_penalty × mean(|derate_old − derate_new|)` ("change")

`penalty = derate_penalty × mean(derate_new) / derate_max` ("total")

**Config example:**
```yaml
derate_penalty: 0.1
derate_penalty_type: "change"  # or "total"
```

---

## 5. Total Reward Composition

The final reward returned by the environment at each step is:

`total_reward = (power_reward × Power_scaling) − action_penalty − derate_penalty`

The `calculate_total_reward()` method also returns a breakdown dictionary with keys `power_reward`, `scaled_power_reward`, `action_penalty`, `derate_penalty`, and `total_reward`, which is useful for logging and debugging during training.

---

## 6. Choosing a Reward Function

| Reward Type | Needs Baseline? | Typical Range | Best For |
|:--|:--|:--|:--|
| **Baseline** | Yes | −0.5 to +0.5 (Layout dependent) | Wake steering research, comparing to greedy |
| **Wake_recovery** | Yes | 0 to 1 | Condition-invariant training across wind speeds |
| **Power_avg** | No | 0 to 1 | Simple experiments, no baseline available |
| **Power_diff** | No | Varies | Curriculum learning, self-improvement |
| **None** | No | 0 | Custom reward wrappers, penalty-only |

**Recommendations:**
- Start with **Baseline** if you have a baseline comparison enabled; it is the most intuitive and widely used.
- Switch to **Wake_recovery** if you are training across a wide range of wind conditions and want consistent reward scaling.
- Use **Power_avg** for quick prototyping when you don't need a baseline farm.
- Use **Power_diff** for curriculum learning setups where absolute performance matters less than improvement.

---

## 7. Complete Configuration Example

Below is a full configuration showing both `power_def` and `act_pen` sections:

```yaml
# --- Power Definition & Reward Settings ---
power_def:
  Power_scaling: 1.0           # Scale factor
  Power_avg: 50                # Steps to average for reward calculation
  Power_reward: "Baseline"     # "Baseline", "Wake_recovery", "Power_avg", "Power_diff", "None"
  # tau: 0.02                  # Headroom floor for Wake_recovery (default: 0.02)

# --- Action Penalty Settings ---
act_pen:
  action_penalty: 0.1          # Penalty weight (0 = no penalty)
  action_penalty_type: "Change" # "Change" (penalize yaw changes) or "Total" (penalize yaw magnitude)
```

---

## Next Steps

- Learn about the environment architecture and other concepts in [Core Concepts](concepts.md)
- Configure and run simulations in [Simulations](simulations.md)
- Evaluate agent performance in [Evaluation Framework](evaluations.md)
