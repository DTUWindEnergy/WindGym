# WindGym.Agents Module

This module contains the agent implementations for WindGym.

<a id="module-WindGym.Agents"></a>

Base container for all “basic” agents.

## BaseAgent

Base container for all “basic” agents.
We MUST have a predict function, that returns the action and the state.

The scale_yaw function is used to scale the yaw angles to be between -1 and 1.

### *class* WindGym.Agents.BaseAgent.BaseAgent(yaw_max=45, yaw_min=-45)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### \_\_init_\_(yaw_max=45, yaw_min=-45)

#### predict(\*args, \*\*kwargs)

#### scale_yaw(yaws)

Scale the yaw angles to be between -1 and 1.

#### unscale_yaw(action)

Unscale the action to the yaw range.

## PyWakeAgent

### *class* WindGym.Agents.PyWakeAgent.PyWakeAgent(x_pos, y_pos, wind_speed=8, wind_dir=270, TI=0.07, yaw_max=45, yaw_min=-45, refine_pass_n=6, yaw_n=7, look_up=False, turbine=py_wake.examples.data.hornsrev1.V80, env=None)

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(x_pos, y_pos, wind_speed=8, wind_dir=270, TI=0.07, yaw_max=45, yaw_min=-45, refine_pass_n=6, yaw_n=7, look_up=False, turbine=py_wake.examples.data.hornsrev1.V80, env=None)

#### update_wind(wind_speed, wind_direction, TI)

Update the wind conditions for the agent.

#### make_lookup()

Create a lookup table for the yaw angles.
This is done as we can save time by doing it once and then use it later.

#### use_lookup()

Use the lookup table to get the yaw angles for the current wind conditions.

#### reset()

Reset the wind things for the objective.

#### optimize()

Optimizes the yaw angles of the wind farm.

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.
Note that the command yaw offset is \_\_always_\_ defined relative to the incoming wind direction

#### calc_power(yaws)

Calculates the power of the farm, given the yaw angles.
Inputs are the yaw angles in degrees.
Returns the total power of the farm.

#### plot_flow()

Plot the flowfield of the wind farm.

### WindGym.Agents.PyWakeAgent.yaw_optimizer_srf_vect(x, y, wffm, yaw_max, wd, ws, ti=0.04, refine_pass_n=4, yaw_n=5, nn_cpu=1, sort_reverse=False)

This is the Serial-Refine Method for yaw optimization, implemented in PyWake.
This was done by Deniz. I just copied the function into this file.

Optimizes turbine yaw angles over arrays of wind directions and wind speeds
using a Serial-Refine Method for PyWake.

This version vectorizes the wind direction (wd) dimension by evaluating the candidate
yaw configurations for all wind directions at once. The candidate offset dimension is looped
over (typically small), while wd is fed in vectorized.

* **Parameters:**
  * **x** (*array_like* *,* *shape* *(**n_wt* *,* *)*) – x coordinates of the turbines.
  * **y** (*array_like* *,* *shape* *(**n_wt* *,* *)*) – y coordinates of the turbines.
  * **wffm** (*EngineeringWindFarmModel Object*) – PyWake wind farm flow model.
  * **wd** (*array_like* *or* [*float*](https://docs.python.org/3/library/functions.html#float)) – Wind direction(s) (in meteorological convention, degrees).
  * **yaw_max** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Maximum yaw angle (degrees)
  * **ws** (*array_like* *or* [*float*](https://docs.python.org/3/library/functions.html#float)) – Wind speed(s) (m/s).
  * **ti** (*array_like* *or* [*float*](https://docs.python.org/3/library/functions.html#float)) – Turbulence intensity.
  * **refine_pass_n** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Number of refine passes.
  * **yaw_n** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Number of candidate yaw offsets to test at each update step.
  * **nn_cpu** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Number of CPUs to use.
  * **sort_reverse** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – Whether to reverse turbine sorting (downstream-to-upstream).
* **Returns:**
  **yaw_opt** – The optimized yaw angles for each turbine, wind direction, and wind speed.
* **Return type:**
  ndarray, shape (n_wt, n_wd, n_ws)

### *class* WindGym.Agents.PyWakeAgent.NoisyPyWakeAgent(measurement_manager: [MeasurementManager](core.md#WindGym.core.measurement_manager.MeasurementManager), \*\*kwargs)

Bases: [`PyWakeAgent`](#WindGym.Agents.PyWakeAgent.PyWakeAgent)

A version of the PyWakeAgent that makes decisions based on noisy observations.

Unlike the base PyWakeAgent which gets perfect global wind conditions, this
agent must estimate the wind conditions from the observation vector it
receives at each step. It then re-runs its optimization based on this
imperfect, noisy information.

#### \_\_init_\_(measurement_manager: [MeasurementManager](core.md#WindGym.core.measurement_manager.MeasurementManager), \*\*kwargs)

Initializes the agent.

* **Parameters:**
  * **measurement_manager** ([*MeasurementManager*](core.md#WindGym.core.MeasurementManager)) – The MeasurementManager instance
    from the environment. This is required for the agent to understand
    the structure of the observation vector.
  * **\*\*kwargs** – Keyword arguments to be passed to the parent PyWakeAgent,
    such as x_pos, y_pos, turbine, etc.

#### predict(obs, deterministic=None)

This method now uses the observation to make a decision.

## GreedyAgent

### *class* WindGym.Agents.GreedyAgent.GreedyAgent(type='local', yaw_max=45, yaw_min=-45, yaw_step=1, env=None)

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(type='local', yaw_max=45, yaw_min=-45, yaw_step=1, env=None)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.

## RandomAgent

### *class* WindGym.Agents.RandomAgent.RandomAgent(env=None)

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(env=None)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.

## ConstantAgent

### *class* WindGym.Agents.ConstantAgent.ConstantAgent(yaw_angles, yaw_max=45, yaw_min=-45)

Bases: [`BaseAgent`](#WindGym.Agents.BaseAgent.BaseAgent)

#### \_\_init_\_(yaw_angles, yaw_max=45, yaw_min=-45)

#### predict(\*args, \*\*kwargs)

This class pretends to be an agent, so we need to have a predict function.
If we havent called the optimize function, we do that now, and return the action
Note that we dont use the obs or the deterministic arguments.
