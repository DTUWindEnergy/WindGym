# Derating

Derating (downregulation) is an optional per-turbine action in WindGym,
alongside yaw control. This page explains the physics, the surrogate-based
turbine model, and how to configure it.

---

## What derating does, physically

Derating makes a turbine produce a fraction `1 − derate` of its available
power. The controller achieves this by moving the operating point
(pitch + TSR), and this **also reduces the thrust coefficient** — so a
derated turbine leaves a weaker wake. That is the whole reason derating is
interesting as a *farm* control action: an upstream turbine can give up power
so that downstream turbines recover more than was sacrificed. At 5D spacing
and 9 m/s the tradeoff is nearly flat (farm-optimal upstream derate ≈ 0.25
buys only ~1% farm power), which makes it a genuinely hard control/RL problem
rather than a trivial one.

## The surrogate turbine

The DTU 10MW power/ct behaviour under derating is table of (power + ct only, float32, ~4.6 MB) ships with
WindGym at `examples/data/dtu10mw_derating_yaw_surrogate.nc`:

- **Dimensions**: wind speed 0–25 m/s, yaw ±40°, derating 0–0.8.
- **Variables**: power, ct (the full table also holds the pitch/TSR operating
  points and cp — regenerate it with hawcpowerctcurvegenerator if you need
  those or a different turbine).
- **Key property**: for every (ws, yaw, derating) the generator picks the
  pitch/TSR combination that meets the power target with **minimum ct**
  (HAWCStab2 grid + spline optimisation). This is what a well-designed derating
  controller would do — power fraction is exactly `1 − derate`, while ct falls
  much faster (0.87 → 0.63 already at 10% derating @ 9 m/s).

The table is wrapped in a PyWake `PowerCtNDTabular`:

```python
pctf = PowerCtNDTabular(
    input_keys=["ws", "yaw", "derate"],
    value_lst=[ws_grid, yaw_grid, derating_grid],
    power_arr=power, power_unit="W", ct_arr=ct,
    default_value_dict={"yaw": 0.0, "derate": 0.0},   # -> optional inputs
    additional_models=[DensityScale(1.225)],           # NO SimpleYawModel!
)
```

Three deliberate choices:

1. `default_value_dict` makes `yaw`/`derate` **optional** inputs with default
   0, so plain `turbine.power(ws)` calls keep working everywhere.
2. `SimpleYawModel` (PyWake's default cos³ yaw loss) is **excluded** — the
   surrogate's own yaw dimension handles misalignment, and keeping both would
   double-count yaw losses. Note the surrogate's yaw losses are much milder
   than cos³ (P/P₀ ≈ 0.85 at 30° vs 0.65), because the operating point
   re-optimises under misalignment.
3. Interpolator bounds are set to `"limit"` so out-of-table queries clamp to
   the edge instead of raising.

See Section 1 of `examples/Example 5 Derating PyWake surrogate.ipynb` for the
complete factory function.

## How WindGym integrates it

WindGym's contract is minimal: **the env only forwards per-turbine derate
values; the turbine must accept a `derate` input.** `WindFarmEnv` validates
this at construction and raises otherwise. (Earlier versions shipped a
power-curve-inversion retrofit, `add_derating_to_turbine`, for plain tabular
turbines; it was removed because sliding down the *normal* operating curve
barely reduces ct — no wake-control value — so turbines must accept `derate`
natively.)

Both backends are supported without turbine-specific code:

- **dynamiks (DWM)**: a `derate` sensor is registered; dynamiks'
  `PyWakeWindTurbines.get_kwargs()` automatically passes any sensor matching an
  optional input of the powerCtFunction, so the reduced ct flows into the DWM
  particles → weaker deficits downstream.
- **pywake (steady-state adapter)**: the adapter stores `_derate` and passes it
  as a kwarg to the wind-farm-model call.

### Config keys

```yaml
derate_action: true        # enable derating as an action (default false)
yaw_action:    true        # false -> derate-only agents (yaw fixed at init)
derate_min:    0.0
derate_max:    0.8         # surrogate table limit (no full shutdown)

derate_method: "absolute"  # or "step"
derate_reference: "available"  # or "rated" (absolute power cap)
derate_step_env: 0.1       # max |Δderate| per env step (step mode)
derate_step_sim: null      # max |Δderate| per sim substep toward the
                           # setpoint (both modes); null = instant

derate_penalty: 0.0        # weight of the derate reward penalty
derate_penalty_type: "change"  # "change" or "total"

derate_mes:                # observation; defaults to current-only when
  derate_current: true     # derate_action is set
  derate_rolling_mean: false
```

### Action space

Per-turbine action values in [-1, 1]:

- `yaw_action: true` (default): `[yaw_0..yaw_n | derate_0..derate_n]`
- `yaw_action: false`: `[derate_0..derate_n]`
- In the PettingZoo `WindFarmEnvMulti`, each agent outputs `[yaw, derate]`
  (or just `[derate]`).

Two application modes:

- **absolute** (default): the value is the setpoint, affine-mapped onto
  `[derate_min, derate_max]` (full action range stays useful even when
  `derate_max < 1`). Physically honest — derating is a power-reference command
  executed within seconds.
- **step**: the value is a delta, at most `derate_step_env` per env step,
  integrated on top of the current derate. Matches the incremental yaw action
  semantics and enforces smooth trajectories, at the cost of an artificial
  rate limit and slower exploration of extremes.

Independently of the mode, `derate_step_sim` limits how fast the plant tracks
the setpoint: the derate slews by at most that amount per simulation substep
(mirroring `yaw_step_sim` in the "wind" yaw method). Left at the default
`null`, the setpoint applies instantly.

### Derate reference: available vs rated

`derate_reference` sets what the commanded fraction *means*. It is orthogonal
to `derate_method` (which sets how the command evolves over time):

- **available** (default): fraction of locally available power. The turbine
  produces `(1 − derate) · P_avail(ws, yaw)`, so `derate = 0.5` at 8 MW
  available yields 4 MW. The output floats with the local wind.
- **rated**: fraction of rated power, i.e. an absolute power cap — the
  industry-typical setpoint semantics. On a 10 MW turbine, `derate = 0.5`
  targets 5 MW regardless of local wind. Rated power is the maximum of the
  turbine's power curve, computed once at init.

In rated mode the env converts the cap to the equivalent available-power
fraction each substep, estimating available power from the surrogate's own
invariant (`P_avail = current_power / (1 − applied_derate)`) — exact,
backend-agnostic, and one substep behind, like a real power-reference
controller. Consequences:

- **Dead zone**: a cap above locally available power is a no-op — a waked
  turbine already producing 3 MW ignores a 5 MW cap, exactly as a physical
  controller would. The action is *not* rescaled to hide this.
- The observation stays the **applied** available-power fraction (what the
  turbine is physically doing); the **commanded** fraction is reported in the
  info dict as `"derate command"`.

### Observation

Each turbine's current derate (scaled from [0, 1] to [-1, 1]) is appended to
its per-turbine measurements, mirroring the yaw measurement machinery
(`derate_mes` supports the same current/rolling-mean/history options). It is
on by default whenever `derate_action` is set — mandatory for step mode,
useful for absolute mode because wake feedback is delayed.

Info dicts expose `"derate agent"` (applied values), `"derate command"`
(commanded values — differs from applied only in rated reference mode), and
`"derate measured"`.

### Reward

No special term is needed: derating reduces farm power directly, so the
existing power reward already penalises pointless derating. The agent must
learn that upstream derating is only worth it if the downstream gain exceeds
the upstream loss.

An optional `derate_penalty` is subtracted from the reward, mirroring the yaw
`action_penalty`:

- `"change"`: penalises the mean |Δderate| per env step (discourages
  oscillation)
- `"total"`: penalises the mean derate level, normalised by `derate_max`
  (discourages standing derating)

## HAWC2 backend (level 3)

Derating also works through the high-fidelity HAWC2 backend (`HTC_path=...`):
instead of a surrogate lookup, each turbine runs a full aeroelastic HAWC2
simulation whose DTUWEC controller derates natively. The same env code path,
config keys, action layout, observations and info dict apply — only the
turbine model behind `wts.sensors.derate` changes.

### Contract

`derate_action: true` + `HTC_path` requires an htc whose `dll` section loads
the **DTUWEC derate controller** (a `type2_dll` with `derate` in its
filename). The env validates the controller's init constants at construction:

| constant | meaning | required |
|---------:|---------|----------|
| 79 | derate strategy (1 = const rotation, 2 = max rotation, 3 = min ct) | ≠ 0 |
| 80 | derate percentage; negative = runtime derating via input 18 | < 0 |
| 100–103 | derate pitch/speed reference shaping (rate/filter limits) | — |
| 104 | dr reference mode: 0 = % of rated, 1 = % of available power | match `derate_reference` (absent = 0) |
| 105 | effective TSR for the available-power estimate | — |

The runtime channel is HAWC2's `general variable 2` (controller input 18,
yaw uses variable 1): the env maps its derate fraction `d ∈ [0, 1]` to the
controller's `dr% = (1 − d) · 100` inside the derate sensor, so `d = 0`
(no derating) is `dr% = 100`. The htc initialises the channel to `100.0`,
covering the instant before the first write.

### Shipped model

`examples/HawcFiles/htc/DTU10mw_derate.htc` is the validated DTU 10 MW derate
model: strategy 2 (max-rotation) default, runtime derating enabled, avail
mode (104 = 1). The controller and servo binaries in
`examples/HawcFiles/control/` are **Linux-only** `.so` files (fork build of
DTU's BasicDTUController, branch `runtime-derating`) — there is no derate
`.dll` for Windows. For `derate_reference: "rated"` supply your own htc with
constant 104 = 0 (or absent).

### Rated mode is controller-native

With a 104 = 0 htc, the DTUWEC controller applies the rated-power cap and its
dead zone itself, so the env skips the surrogate-invariant conversion and
passes the commanded fraction straight through: `"derate agent"` /
`current_derate` then report the **commanded cap fraction**, not the applied
available-power fraction.

### Baselines, multi-agent, timing

- Baseline turbines stay greedy: nothing writes their `general variable 2`,
  and the htc init value 100 means no derating.
- `WindFarmEnvMulti` inherits everything — no extra wiring.
- Wake response is *slow* at this fidelity: at 5D and 10 m/s a derate change
  at the upstream turbine needs ~100 s advection + ~200 s DWM settling before
  the downstream turbine's power is meaningful again. Hold each setpoint
  ≥ 350 s (and judge only the tail) when validating; RL agents get this
  delayed credit assignment for free as part of the problem.

Runnable end-to-end check (2 × DTU 10 MW inline, ~10–30 min):
`examples/hawc2_derating_2wt.py`.

## Sanity numbers (DTU 10MW surrogate @ 9 m/s)

| derate | P/P₀ | ct   |
|-------:|------|------|
| 0.0    | 1.00 | 0.87 |
| 0.1    | 0.90 | 0.63 |
| 0.4    | 0.60 | 0.36 |
| 0.8    | 0.20 | 0.13 |

3×1 row, 5D, 9 m/s: derate T0 by 0.4 → T0 4.88 → 2.93 MW, waked T1
0.45 → 2.16 MW, farm ≈ net-neutral. Farm-optimal T0 derate ≈ 0.25 (+0.05 MW).

## Where things live

- `examples/Example 5 Derating PyWake surrogate.ipynb` — the surrogate
  approach, end to end (turbine factory, physics plots, env usage, rendered
  gifs in `examples/images/`).
- `examples/Example 5c Derating fidelity comparison.ipynb` — pywake
  steady-state / dynamiks DWM / HAWC2 side by side on the 2-WT scenario
  (cross-fidelity sanity check; reuses the `hawc2_derating_2wt.npz` cache).
- `examples/data/dtu10mw_derating_yaw_surrogate.nc` — reduced surrogate table.
- `WindGym/wind_farm_env.py` — action/observation plumbing
  (`_apply_derating`, `derate_mes`).
- `WindGym/core/mes_class.py` — derate measurement channel.
- `WindGym/core/reward_calculator.py` — derate penalty.
- `WindGym/core/derating.py` — derating validation (turbine + htc), HAWC2
  derate-sensor wiring.
- `examples/hawc2_derating_2wt.py` + `examples/HawcFiles/` — HAWC2-backend
  derating example and the shipped DTUWEC derate model (Linux-only binaries).
