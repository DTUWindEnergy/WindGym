# WindGym.core Module

This module contains the core components of WindGym.

<a id="module-WindGym.core"></a>

Core modules for WindGym environment.

This package contains modular components that handle specific responsibilities
of the wind farm environment, promoting separation of concerns and maintainability.

### *class* WindGym.core.RewardCalculator(power_reward_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'Baseline', track_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_scaling: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, action_penalty: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, action_penalty_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, power_window_size: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Calculates rewards and penalties for wind farm control.

Supports multiple reward strategies:
- Baseline: Compare agent performance to baseline controller
- Power_avg: Reward based on average power production
- Power_diff: Reward based on power improvement over time
- None: No power reward

Also handles action penalties to encourage stable control.

#### \_\_init_\_(power_reward_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'Baseline', track_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_scaling: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, action_penalty: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, action_penalty_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, power_window_size: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the reward calculator.

* **Parameters:**
  * **power_reward_type** – Type of power reward (“Baseline”, “Power_avg”, “Power_diff”, “None”)
  * **track_power** – Whether to include power tracking reward (not yet implemented)
  * **power_scaling** – Scaling factor for power reward
  * **action_penalty** – Weight for action penalty (0 = no penalty)
  * **action_penalty_type** – Type of penalty (“change” or “total”)
  * **power_window_size** – Window size for Power_diff reward type

#### calculate_action_penalty(old_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), new_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), yaw_max: [float](https://docs.python.org/3/library/functions.html#float)) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate penalty for turbine actions.

Supports two penalty types:
- “change”: Penalize changes in yaw angle (encourages stability)
- “total”: Penalize absolute yaw magnitude (encourages alignment)

* **Parameters:**
  * **old_yaws** – Previous yaw angles (degrees)
  * **new_yaws** – Current yaw angles (degrees)
  * **yaw_max** – Maximum allowed yaw angle (degrees)
* **Returns:**
  Action penalty value
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### calculate_power_reward(farm_power_deque, baseline_power_deque: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None, rated_power: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, n_turbines: [int](https://docs.python.org/3/library/functions.html#int) = 1) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate the power production reward.

* **Parameters:**
  * **farm_power_deque** – Deque containing farm power history
  * **baseline_power_deque** – Deque containing baseline power history (for Baseline reward)
  * **rated_power** – Rated power of a single turbine (for Power_avg reward)
  * **n_turbines** – Number of turbines in the farm
* **Returns:**
  The calculated power reward
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### calculate_total_reward(farm_power_deque, old_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), new_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), yaw_max: [float](https://docs.python.org/3/library/functions.html#float), baseline_power_deque: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None, rated_power: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, n_turbines: [int](https://docs.python.org/3/library/functions.html#int) = 1) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Calculate total reward including power reward and action penalty.

This is a convenience method that combines power reward and action penalty
calculations, returning both the total reward and a breakdown.

* **Parameters:**
  * **farm_power_deque** – Agent farm power history
  * **old_yaws** – Previous yaw angles
  * **new_yaws** – Current yaw angles
  * **yaw_max** – Maximum yaw angle
  * **baseline_power_deque** – Baseline power history (if needed)
  * **rated_power** – Rated power per turbine (if needed)
  * **n_turbines** – Number of turbines
* **Returns:**
  (total_reward, reward_breakdown_dict)
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

### *class* WindGym.core.WindManager(ws_min: [float](https://docs.python.org/3/library/functions.html#float), ws_max: [float](https://docs.python.org/3/library/functions.html#float), wd_min: [float](https://docs.python.org/3/library/functions.html#float), wd_max: [float](https://docs.python.org/3/library/functions.html#float), ti_min: [float](https://docs.python.org/3/library/functions.html#float), ti_max: [float](https://docs.python.org/3/library/functions.html#float), sample_site: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages wind condition sampling and wind direction time series generation.

Supports two sampling modes:
1. Uniform sampling within specified ranges
2. Site-based sampling using PyWake site data (Weibull distributions)

Also handles generation of time-varying wind direction sequences.

#### \_\_init_\_(ws_min: [float](https://docs.python.org/3/library/functions.html#float), ws_max: [float](https://docs.python.org/3/library/functions.html#float), wd_min: [float](https://docs.python.org/3/library/functions.html#float), wd_max: [float](https://docs.python.org/3/library/functions.html#float), ti_min: [float](https://docs.python.org/3/library/functions.html#float), ti_max: [float](https://docs.python.org/3/library/functions.html#float), sample_site: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the wind manager.

* **Parameters:**
  * **ws_min** – Minimum wind speed (m/s)
  * **ws_max** – Maximum wind speed (m/s)
  * **wd_min** – Minimum wind direction (degrees)
  * **wd_max** – Maximum wind direction (degrees)
  * **ti_min** – Minimum turbulence intensity (fraction)
  * **ti_max** – Maximum turbulence intensity (fraction)
  * **sample_site** – Optional PyWake site for realistic wind sampling

#### make_wind_direction_list(base_wd: [float](https://docs.python.org/3/library/functions.html#float), time_max: [float](https://docs.python.org/3/library/functions.html#float), dt_sim: [float](https://docs.python.org/3/library/functions.html#float), t_developed: [float](https://docs.python.org/3/library/functions.html#float), steps_on_reset: [int](https://docs.python.org/3/library/functions.html#int), wd_function: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[float](https://docs.python.org/3/library/functions.html#float)], [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)

Generate a time series of wind directions for an episode.

The wind direction list has two phases:
1. Burn-in/steady-state period: Constant wind direction
2. Episode period: Either constant or time-varying (if wd_function provided)

* **Parameters:**
  * **base_wd** – Base wind direction to start with (degrees)
  * **time_max** – Maximum simulation time for the episode (seconds)
  * **dt_sim** – Simulation timestep (seconds)
  * **t_developed** – Time for flow to develop (seconds)
  * **steps_on_reset** – Number of environment steps during reset
  * **wd_function** – Optional function(time) -> wd for time-varying wind
* **Returns:**
  Wind direction for each simulation timestep
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)

#### sample_conditions() → [WindConditions](#WindGym.core.wind_manager.WindConditions)

Sample wind speed, direction, and turbulence intensity.

* **Returns:**
  Sampled wind conditions
* **Return type:**
  [WindConditions](#WindGym.core.WindConditions)

### *class* WindGym.core.WindConditions(wind_speed: [float](https://docs.python.org/3/library/functions.html#float), wind_direction: [float](https://docs.python.org/3/library/functions.html#float), turbulence_intensity: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for wind conditions.

#### \_\_init_\_(wind_speed: [float](https://docs.python.org/3/library/functions.html#float), wind_direction: [float](https://docs.python.org/3/library/functions.html#float), turbulence_intensity: [float](https://docs.python.org/3/library/functions.html#float)) → [None](https://docs.python.org/3/library/constants.html#None)

#### unpack()

Unpack wind conditions as tuple.

#### wind_speed *: [float](https://docs.python.org/3/library/functions.html#float)*

#### wind_direction *: [float](https://docs.python.org/3/library/functions.html#float)*

#### turbulence_intensity *: [float](https://docs.python.org/3/library/functions.html#float)*

### *class* WindGym.core.TurbulenceManager(turbulence_type: [str](https://docs.python.org/3/library/stdtypes.html#str), turbulence_box_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_turb_move: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages turbulence field generation and site creation for wind farm simulations.

Supports multiple turbulence generation strategies:
- MannLoad: Load pre-generated Mann turbulence boxes from files
- MannGenerate: Generate new Mann turbulence boxes on-the-fly
- MannFixed: Generate a fixed Mann turbulence box (reproducible)
- Random: Use random turbulence (faster, less realistic)
- None: Zero turbulence (fastest, for testing)

#### \_\_init_\_(turbulence_type: [str](https://docs.python.org/3/library/stdtypes.html#str), turbulence_box_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_turb_move: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Initialize the turbulence manager.

* **Parameters:**
  * **turbulence_type** – Type of turbulence (“MannLoad”, “MannGenerate”,
    “MannFixed”, “Random”, “None”)
  * **turbulence_box_path** – Path to turbulence box files (required for MannLoad)
  * **max_turb_move** – Maximum distance turbines can move in one timestep (m)
    Used to calculate wind direction change rate limits

#### create_sites(ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), ti: [float](https://docs.python.org/3/library/functions.html#float), wd_list: [list](https://docs.python.org/3/library/stdtypes.html#list), dt_sim: [float](https://docs.python.org/3/library/functions.html#float), turbine_positions: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), rotor_diameter: [float](https://docs.python.org/3/library/functions.html#float), n_passthrough: [int](https://docs.python.org/3/library/functions.html#int), burn_in_passthroughs: [int](https://docs.python.org/3/library/functions.html#int), create_baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

Create turbulence fields and sites for agent and optionally baseline.

This method:
1. Generates turbulence field based on turbulence_type
2. Calculates t_developed and time_max based on farm geometry
3. Creates MetmastSite with wind direction time series
4. Optionally creates baseline site (deep copy of turbulence field)

* **Parameters:**
  * **ws** – Wind speed (m/s)
  * **wd** – Wind direction (degrees)
  * **ti** – Turbulence intensity (fraction)
  * **wd_list** – Wind direction time series
  * **dt_sim** – Simulation timestep (seconds)
  * **turbine_positions** – Turbine positions [x, y] array (n_turb, 2)
  * **rotor_diameter** – Rotor diameter (m)
  * **n_passthrough** – Number of flow passthroughs for episode
  * **burn_in_passthroughs** – Number of passthroughs for flow development
  * **create_baseline** – Whether to create baseline site
* **Returns:**
  (site, site_baseline, t_developed, time_max, added_turbulence_model)
  : site_baseline is None if create_baseline=False
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

### *class* WindGym.core.WindFarmRenderer(render_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Handles rendering of wind farm environments.

Supports multiple render modes:
- “rgb_array”: Return RGB frames for recording/saving
- “human”: Display frames in a window for human viewing
- None: No rendering

Also provides utility methods for plotting farm layouts and frames.

#### \_\_init_\_(render_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the renderer.

* **Parameters:**
  **render_mode** – Rendering mode (“human”, “rgb_array”, or None)

#### close()

Close any open matplotlib figures.

#### init_render(fs, turbine)

Initialize rendering objects.

This creates the matplotlib figure, axis, and XYView for rendering.
Should be called after the flow simulation is created.

* **Parameters:**
  * **fs** – Flow simulation object
  * **turbine** – Turbine object (for hub_height)

#### plot_farm(fs, fs_baseline=None, turbine=None, baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fix_turbines: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Plot the entire farm layout (legacy method for IPython notebooks).

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **turbine** – Turbine object
  * **baseline** – Whether to plot baseline instead of agent

#### plot_frame(fs, fs_baseline=None, turbine=None, baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Plot a single frame of the flow field and turbines.

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **turbine** – Turbine object
  * **baseline** – Whether to plot baseline instead of agent

#### render(fs, fs_baseline=None, probes=None, turbine=None)

Main render method - routes to appropriate rendering function.

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **probes** – Optional list of wind probes
  * **turbine** – Turbine object for lazy initialization
* **Returns:**
  RGB array if render_mode is “rgb_array”, None otherwise

### *class* WindGym.core.BaselineManager(baseline_controller_type: [str](https://docs.python.org/3/library/stdtypes.html#str), x_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), y_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), turbine, yaw_max: [float](https://docs.python.org/3/library/functions.html#float), yaw_min: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_env: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_sim: [float](https://docs.python.org/3/library/functions.html#float), htc_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages baseline controller setup and execution.

Supports multiple baseline controller types:
- Local: Local yaw controller
- Global: Global yaw controller
- PyWake: PyWake optimization-based agent (oracle or local mode)

Also handles baseline turbine initialization for HAWC2 or PyWake turbines.

#### \_\_init_\_(baseline_controller_type: [str](https://docs.python.org/3/library/stdtypes.html#str), x_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), y_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), turbine, yaw_max: [float](https://docs.python.org/3/library/functions.html#float), yaw_min: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_env: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_sim: [float](https://docs.python.org/3/library/functions.html#float), htc_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the baseline manager.

* **Parameters:**
  * **baseline_controller_type** – Type of baseline controller
    (“Local”, “Global”, “PyWake_oracle”, “PyWake_local”)
  * **x_pos** – X positions of turbines
  * **y_pos** – Y positions of turbines
  * **turbine** – Turbine object
  * **yaw_max** – Maximum yaw angle (degrees)
  * **yaw_min** – Minimum yaw angle (degrees)
  * **yaw_step_env** – Yaw step per environment step (degrees)
  * **yaw_step_sim** – Yaw step per simulation step (degrees)
  * **htc_path** – Optional path to HAWC2 HTC file

#### compute_baseline_action(fs, yaw_step: [float](https://docs.python.org/3/library/functions.html#float) = 1.0) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Compute baseline controller action.

* **Parameters:**
  * **fs** – Flow simulation object (baseline)
  * **yaw_step** – Yaw step size (degrees)
* **Returns:**
  New yaw angles for baseline turbines
* **Return type:**
  np.ndarray

#### initialize_baseline_turbines(name_string: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize baseline turbines (HAWC2 or PyWake).

* **Parameters:**
  **name_string** – Optional name string for HAWC2 case (required if htc_path is set)
* **Returns:**
  Baseline turbine object

#### update_wind_conditions(ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), ti: [float](https://docs.python.org/3/library/functions.html#float))

Update wind conditions for baseline manager.

This is needed for PyWake agent in oracle mode and for tracking
current conditions.

* **Parameters:**
  * **ws** – Wind speed (m/s)
  * **wd** – Wind direction (degrees)
  * **ti** – Turbulence intensity (fraction)

### *class* WindGym.core.ProbeManager(probes_config: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages wind probes in the environment.

Supports two placement modes:
1. Free placement: Probes at fixed absolute positions
2. Turbine-relative: Probes positioned relative to turbines, rotating with yaw

#### probes

List of all WindProbe objects

#### turbine_probes

Dict mapping turbine_index to list of probes

#### \_\_init_\_(probes_config: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the probe manager.

* **Parameters:**
  **probes_config** – List of probe configuration dictionaries.
  Each dict can contain:
  - position: Absolute [x, y, z] position (free placement)
  - turbine_index: Index of turbine to attach to
  - relative_position: [x, y, z] relative to turbine
  - include_wakes: Whether to include wake effects
  - exclude_wake_from: List of turbine indices to exclude
  - time: Specific time for probe reading
  - probe_type: “WS” or “TI”
  - name: Optional probe name

#### count_probes_per_turbine() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

Count how many probes are assigned to each turbine index.

* **Returns:**
  Dict mapping turbine_index to probe count

#### get_probe_readings() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]

Get readings from all probes.

* **Returns:**
  List of probe readings (wind speed or turbulence intensity)

#### get_turbine_probe_readings(turbine_index: [int](https://docs.python.org/3/library/functions.html#int)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]

Get readings from probes attached to a specific turbine.

* **Parameters:**
  **turbine_index** – Index of turbine
* **Returns:**
  List of probe readings for that turbine

#### has_probes() → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if any probes are configured.

#### initialize_probes(fs, yaw_angles) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[WindProbe](#WindGym.core.wind_probe.WindProbe)]

Initialize probes with turbine-relative positioning.

Probes can be placed relative to turbines and will rotate with turbine yaw.

* **Parameters:**
  * **fs** – Flow simulation object
  * **yaw_angles** – Initial yaw angles (degrees), scalar or array
* **Returns:**
  List of initialized WindProbe objects

#### initialize_probes_free_placement(env) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[WindProbe](#WindGym.core.wind_probe.WindProbe)]

Initialize probes with free (absolute) placement.

This mode is for probes at fixed positions that don’t rotate with turbines.

* **Parameters:**
  **env** – Environment object (for WindProbe compatibility)
* **Returns:**
  List of initialized WindProbe objects

#### update_probe_positions(fs, yaw_angles)

Update probe positions when turbines yaw.

Only updates probes that are attached to turbines (turbine-relative).

* **Parameters:**
  * **fs** – Flow simulation object
  * **yaw_angles** – New yaw angles (degrees), array

### *class* WindGym.core.Mes(current: [bool](https://docs.python.org/3/library/functions.html#bool) = True, rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = False, history_N: [int](https://docs.python.org/3/library/functions.html#int) = 3, history_length: [int](https://docs.python.org/3/library/functions.html#int) = 100, window_length: [int](https://docs.python.org/3/library/functions.html#int) = 5)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Baseclass for the measurements,
we can decide how large a memory we need, and also how many measurements we want to get back
Current: bool, if true return the latest measurement
Rolling Mean: bool, if true return the rolling mean of the measurements
history_N: int, number of rolling windows to use for the rolling mean. If 1, only return the latest value, if 2 return the lates and oldest value, if more then do some inbetween values also
history_length: int, number of measurements to save
window_length: int, size of the rolling window

#### \_\_call_\_(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque via the call function

#### \_\_init_\_(current: [bool](https://docs.python.org/3/library/functions.html#bool) = True, rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = False, history_N: [int](https://docs.python.org/3/library/functions.html#int) = 3, history_length: [int](https://docs.python.org/3/library/functions.html#int) = 100, window_length: [int](https://docs.python.org/3/library/functions.html#int) = 5) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_measurement(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque

#### append(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque via the append function

#### get_measurements() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Get the desired measurements with graceful handling of startup period

### *class* WindGym.core.TurbMes(ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, include_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Class for all measurements.
Each turbine stores measurements for wind speed, wind direction and yaw angle…

#### \_\_init_\_(ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, include_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_hf_ws(measurement: [float](https://docs.python.org/3/library/functions.html#float)) → [None](https://docs.python.org/3/library/constants.html#None)

Appends a single wind speed measurement to the high-frequency buffer.

#### add_power(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_wd(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_ws(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_yaw(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### calc_TI(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Calcualte TI from the wind speed measurements

#### empty_np(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Return an empty array

#### get_measurements(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_ws(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_yaw(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### max_hist() → [int](https://docs.python.org/3/library/functions.html#int)

Return the maximum history length of the measurements

#### observed_variables() → [int](https://docs.python.org/3/library/functions.html#int)

Returns the number of observed variables

### *class* WindGym.core.FarmMes(n_turbines: [int](https://docs.python.org/3/library/functions.html#int), n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, turb_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_power: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = False, farm_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Class for the measurements of the farm.
The farm stores measurements from each turbine for wind speed, wind direction, yaw angle, power

#### \_\_init_\_(n_turbines: [int](https://docs.python.org/3/library/functions.html#int), n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, turb_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_power: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = False, farm_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_measurements(ws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], wd: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], powers: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_power(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_wd(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_ws(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_yaw(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### empty_np(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_TI_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [float](https://docs.python.org/3/library/functions.html#float)

#### get_TI_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_measurements(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_ws_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_ws_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_yaw_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### max_hist() → [int](https://docs.python.org/3/library/functions.html#int)

Return the maximum history length of the measurements

#### observed_variables() → [int](https://docs.python.org/3/library/functions.html#int)

Returns the number of observed variables

### *class* WindGym.core.WindProbe(fs, position, yaw_angle, turbine_position, include_wakes=True, exclude_wake_from=[], time=None, probe_type='WS')

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### \_\_init_\_(fs, position, yaw_angle, turbine_position, include_wakes=True, exclude_wake_from=[], time=None, probe_type='WS')

Initialize a wind speed or TI probe.

* **Parameters:**
  * **fs** – The wind farm environment (should have get_windspeed() and get_turbulence_intensity()).
  * **position** – (x, y, z) tuple for probe location.
  * **include_wakes** – Whether to include wake effects in the wind calculation.
  * **exclude_wake_from** – Turbine indices to exclude wakes from.
  * **time** – Specific time (optional).
  * **probe_type** – ‘WS’ for wind speed, ‘TI’ for turbulence intensity.

#### get_inflow_angle_to_turbine(degrees=False)

Returns the angle from the probe to the turbine (horizontal XY-plane),
counter-clockwise from the x-axis.

* **Parameters:**
  * **turbine_position** – (x, y, z) of the turbine.
  * **degrees** – If True, return angle in degrees.
* **Returns:**
  Angle in radians (or degrees if requested).

#### get_projected_wind_speed_toward_turbine()

Projects the wind speed vector onto the direction from the probe to the turbine.

* **Parameters:**
  **turbine_position** – (x, y, z) of the turbine.
* **Returns:**
  Scalar wind speed component in direction from probe to turbine.

#### read()

Read either wind speed (u, v, w) or turbulence intensity depending on probe_type.

#### read_speed_magnitude()

Return scalar wind speed magnitude.

#### update_position(new_position)

Move probe to a new (x, y, z) position.

### *class* WindGym.core.MeasurementType(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

#### WIND_SPEED *= 'wind_speed'*

#### WIND_DIRECTION *= 'wind_direction'*

#### YAW_ANGLE *= 'yaw_angle'*

#### TURBULENCE_INTENSITY *= 'turbulence_intensity'*

#### POWER *= 'power'*

#### GENERIC *= 'generic'*

### *class* WindGym.core.MeasurementSpec(name: [str](https://docs.python.org/3/library/stdtypes.html#str), measurement_type: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType), index_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)], min_val: [float](https://docs.python.org/3/library/functions.html#float), max_val: [float](https://docs.python.org/3/library/functions.html#float), turbine_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, noise_sensitivity: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, is_circular: [bool](https://docs.python.org/3/library/functions.html#bool) = False, circular_range: [float](https://docs.python.org/3/library/functions.html#float) = 360.0)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Specification for a single component of the observation vector.

#### name

A descriptive name for the measurement (e.g., ‘turb_0/ws_current’).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### measurement_type

The physical type of the measurement.

* **Type:**
  [MeasurementType](#WindGym.core.MeasurementType)

#### index_range

The start and end indices in the flat observation array.

* **Type:**
  Tuple[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

#### min_val

The minimum physical value for scaling.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### max_val

The maximum physical value for scaling.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### turbine_id

The turbine index, if applicable.

* **Type:**
  Optional[[int](https://docs.python.org/3/library/functions.html#int)]

#### noise_sensitivity

A multiplier for the applied noise level.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### \_\_init_\_(name: [str](https://docs.python.org/3/library/stdtypes.html#str), measurement_type: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType), index_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)], min_val: [float](https://docs.python.org/3/library/functions.html#float), max_val: [float](https://docs.python.org/3/library/functions.html#float), turbine_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, noise_sensitivity: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, is_circular: [bool](https://docs.python.org/3/library/functions.html#bool) = False, circular_range: [float](https://docs.python.org/3/library/functions.html#float) = 360.0) → [None](https://docs.python.org/3/library/constants.html#None)

#### circular_range *: [float](https://docs.python.org/3/library/functions.html#float)* *= 360.0*

#### is_circular *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

#### noise_sensitivity *: [float](https://docs.python.org/3/library/functions.html#float)* *= 1.0*

#### turbine_id *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### measurement_type *: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType)*

#### index_range *: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]*

#### min_val *: [float](https://docs.python.org/3/library/functions.html#float)*

#### max_val *: [float](https://docs.python.org/3/library/functions.html#float)*

### *class* WindGym.core.NoiseModel

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

#### *abstractmethod* apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### *abstractmethod* get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

### *class* WindGym.core.WhiteNoiseModel(noise_std_devs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [float](https://docs.python.org/3/library/functions.html#float)])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

Applies Gaussian white noise defined in physical units (e.g., m/s, degrees).

#### \_\_init_\_(noise_std_devs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [float](https://docs.python.org/3/library/functions.html#float)])

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

### *class* WindGym.core.EpisodicBiasNoiseModel(bias_ranges: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

Applies a consistent bias for an entire episode, defined in physical units.

#### \_\_init_\_(bias_ranges: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]])

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Applies the sampled episodic bias to the given observations.

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

### *class* WindGym.core.HybridNoiseModel(models: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NoiseModel](#WindGym.core.measurement_manager.NoiseModel)])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

#### \_\_init_\_(models: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NoiseModel](#WindGym.core.measurement_manager.NoiseModel)])

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

### *class* WindGym.core.AdversarialNoiseModel(antagonist_agent, constraints, device)

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

#### \_\_init_\_(antagonist_agent, constraints, device)

#### apply_noise(clean_observations, specs, rng)

#### get_info()

#### reset_noise(specs: [list](https://docs.python.org/3/library/stdtypes.html#list), rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

### *class* WindGym.core.MeasurementManager(env, seed=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Orchestrates measurement specifications and the application of noise.

#### \_\_init_\_(env, seed=None)

#### apply_noise(clean_observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)]

#### reset_noise()

#### seed(seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Reseeds the random number generator for the noise model.

#### set_noise_model(noise_model: [NoiseModel](#WindGym.core.measurement_manager.NoiseModel))

### *class* WindGym.core.NoisyWindFarmEnv(base_env_class, measurement_manager: [MeasurementManager](#WindGym.core.measurement_manager.MeasurementManager), \*\*env_kwargs)

Bases: [`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)

A Gym wrapper that applies measurement errors to a base WindFarm environment.

#### \_\_init_\_(base_env_class, measurement_manager: [MeasurementManager](#WindGym.core.measurement_manager.MeasurementManager), \*\*env_kwargs)

Wraps an environment to allow a modular transformation of the [`step()`](#WindGym.core.NoisyWindFarmEnv.step) and [`reset()`](#WindGym.core.NoisyWindFarmEnv.reset) methods.

* **Parameters:**
  **env** – The environment to wrap

#### close()

Closes the wrapper and `env`.

#### reset(, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Uses the [`reset()`](#WindGym.core.NoisyWindFarmEnv.reset) of the `env` that can be overwritten to change the returned data.

#### step(action: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [float](https://docs.python.org/3/library/functions.html#float), [bool](https://docs.python.org/3/library/functions.html#bool), [bool](https://docs.python.org/3/library/functions.html#bool), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Uses the [`step()`](#WindGym.core.NoisyWindFarmEnv.step) of the `env` that can be overwritten to change the returned data.

## Reward Calculator

Reward calculation module for WindGym environments.

This module handles all reward and penalty calculations, providing a clean
interface for different reward strategies.

### *class* WindGym.core.reward_calculator.RewardCalculator(power_reward_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'Baseline', track_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_scaling: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, action_penalty: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, action_penalty_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, power_window_size: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Calculates rewards and penalties for wind farm control.

Supports multiple reward strategies:
- Baseline: Compare agent performance to baseline controller
- Power_avg: Reward based on average power production
- Power_diff: Reward based on power improvement over time
- None: No power reward

Also handles action penalties to encourage stable control.

#### \_\_init_\_(power_reward_type: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'Baseline', track_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_scaling: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, action_penalty: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, action_penalty_type: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, power_window_size: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the reward calculator.

* **Parameters:**
  * **power_reward_type** – Type of power reward (“Baseline”, “Power_avg”, “Power_diff”, “None”)
  * **track_power** – Whether to include power tracking reward (not yet implemented)
  * **power_scaling** – Scaling factor for power reward
  * **action_penalty** – Weight for action penalty (0 = no penalty)
  * **action_penalty_type** – Type of penalty (“change” or “total”)
  * **power_window_size** – Window size for Power_diff reward type

#### calculate_power_reward(farm_power_deque, baseline_power_deque: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None, rated_power: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, n_turbines: [int](https://docs.python.org/3/library/functions.html#int) = 1) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate the power production reward.

* **Parameters:**
  * **farm_power_deque** – Deque containing farm power history
  * **baseline_power_deque** – Deque containing baseline power history (for Baseline reward)
  * **rated_power** – Rated power of a single turbine (for Power_avg reward)
  * **n_turbines** – Number of turbines in the farm
* **Returns:**
  The calculated power reward
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### calculate_action_penalty(old_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), new_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), yaw_max: [float](https://docs.python.org/3/library/functions.html#float)) → [float](https://docs.python.org/3/library/functions.html#float)

Calculate penalty for turbine actions.

Supports two penalty types:
- “change”: Penalize changes in yaw angle (encourages stability)
- “total”: Penalize absolute yaw magnitude (encourages alignment)

* **Parameters:**
  * **old_yaws** – Previous yaw angles (degrees)
  * **new_yaws** – Current yaw angles (degrees)
  * **yaw_max** – Maximum allowed yaw angle (degrees)
* **Returns:**
  Action penalty value
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### calculate_total_reward(farm_power_deque, old_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), new_yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), yaw_max: [float](https://docs.python.org/3/library/functions.html#float), baseline_power_deque: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None, rated_power: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, n_turbines: [int](https://docs.python.org/3/library/functions.html#int) = 1) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[float](https://docs.python.org/3/library/functions.html#float), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Calculate total reward including power reward and action penalty.

This is a convenience method that combines power reward and action penalty
calculations, returning both the total reward and a breakdown.

* **Parameters:**
  * **farm_power_deque** – Agent farm power history
  * **old_yaws** – Previous yaw angles
  * **new_yaws** – Current yaw angles
  * **yaw_max** – Maximum yaw angle
  * **baseline_power_deque** – Baseline power history (if needed)
  * **rated_power** – Rated power per turbine (if needed)
  * **n_turbines** – Number of turbines
* **Returns:**
  (total_reward, reward_breakdown_dict)
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

## Wind Manager

Wind condition management module for WindGym environments.

This module handles wind speed, wind direction, and turbulence intensity sampling,
including support for site-based sampling using PyWake sites.

### *class* WindGym.core.wind_manager.WindConditions(wind_speed: [float](https://docs.python.org/3/library/functions.html#float), wind_direction: [float](https://docs.python.org/3/library/functions.html#float), turbulence_intensity: [float](https://docs.python.org/3/library/functions.html#float))

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Container for wind conditions.

#### wind_speed *: [float](https://docs.python.org/3/library/functions.html#float)*

#### wind_direction *: [float](https://docs.python.org/3/library/functions.html#float)*

#### turbulence_intensity *: [float](https://docs.python.org/3/library/functions.html#float)*

#### unpack()

Unpack wind conditions as tuple.

#### \_\_init_\_(wind_speed: [float](https://docs.python.org/3/library/functions.html#float), wind_direction: [float](https://docs.python.org/3/library/functions.html#float), turbulence_intensity: [float](https://docs.python.org/3/library/functions.html#float)) → [None](https://docs.python.org/3/library/constants.html#None)

### *class* WindGym.core.wind_manager.WindManager(ws_min: [float](https://docs.python.org/3/library/functions.html#float), ws_max: [float](https://docs.python.org/3/library/functions.html#float), wd_min: [float](https://docs.python.org/3/library/functions.html#float), wd_max: [float](https://docs.python.org/3/library/functions.html#float), ti_min: [float](https://docs.python.org/3/library/functions.html#float), ti_max: [float](https://docs.python.org/3/library/functions.html#float), sample_site: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages wind condition sampling and wind direction time series generation.

Supports two sampling modes:
1. Uniform sampling within specified ranges
2. Site-based sampling using PyWake site data (Weibull distributions)

Also handles generation of time-varying wind direction sequences.

#### \_\_init_\_(ws_min: [float](https://docs.python.org/3/library/functions.html#float), ws_max: [float](https://docs.python.org/3/library/functions.html#float), wd_min: [float](https://docs.python.org/3/library/functions.html#float), wd_max: [float](https://docs.python.org/3/library/functions.html#float), ti_min: [float](https://docs.python.org/3/library/functions.html#float), ti_max: [float](https://docs.python.org/3/library/functions.html#float), sample_site: [object](https://docs.python.org/3/library/functions.html#object) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the wind manager.

* **Parameters:**
  * **ws_min** – Minimum wind speed (m/s)
  * **ws_max** – Maximum wind speed (m/s)
  * **wd_min** – Minimum wind direction (degrees)
  * **wd_max** – Maximum wind direction (degrees)
  * **ti_min** – Minimum turbulence intensity (fraction)
  * **ti_max** – Maximum turbulence intensity (fraction)
  * **sample_site** – Optional PyWake site for realistic wind sampling

#### sample_conditions() → [WindConditions](#WindGym.core.wind_manager.WindConditions)

Sample wind speed, direction, and turbulence intensity.

* **Returns:**
  Sampled wind conditions
* **Return type:**
  [WindConditions](#WindGym.core.wind_manager.WindConditions)

#### make_wind_direction_list(base_wd: [float](https://docs.python.org/3/library/functions.html#float), time_max: [float](https://docs.python.org/3/library/functions.html#float), dt_sim: [float](https://docs.python.org/3/library/functions.html#float), t_developed: [float](https://docs.python.org/3/library/functions.html#float), steps_on_reset: [int](https://docs.python.org/3/library/functions.html#int), wd_function: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[float](https://docs.python.org/3/library/functions.html#float)], [float](https://docs.python.org/3/library/functions.html#float)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)

Generate a time series of wind directions for an episode.

The wind direction list has two phases:
1. Burn-in/steady-state period: Constant wind direction
2. Episode period: Either constant or time-varying (if wd_function provided)

* **Parameters:**
  * **base_wd** – Base wind direction to start with (degrees)
  * **time_max** – Maximum simulation time for the episode (seconds)
  * **dt_sim** – Simulation timestep (seconds)
  * **t_developed** – Time for flow to develop (seconds)
  * **steps_on_reset** – Number of environment steps during reset
  * **wd_function** – Optional function(time) -> wd for time-varying wind
* **Returns:**
  Wind direction for each simulation timestep
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)

## Turbulence Manager

Turbulence field and site management module for WindGym environments.

This module handles turbulence field generation, site creation, and time calculations
for wind farm simulations. Supports multiple turbulence generation strategies.

### *class* WindGym.core.turbulence_manager.TurbulenceManager(turbulence_type: [str](https://docs.python.org/3/library/stdtypes.html#str), turbulence_box_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_turb_move: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages turbulence field generation and site creation for wind farm simulations.

Supports multiple turbulence generation strategies:
- MannLoad: Load pre-generated Mann turbulence boxes from files
- MannGenerate: Generate new Mann turbulence boxes on-the-fly
- MannFixed: Generate a fixed Mann turbulence box (reproducible)
- Random: Use random turbulence (faster, less realistic)
- None: Zero turbulence (fastest, for testing)

#### \_\_init_\_(turbulence_type: [str](https://docs.python.org/3/library/stdtypes.html#str), turbulence_box_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_turb_move: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Initialize the turbulence manager.

* **Parameters:**
  * **turbulence_type** – Type of turbulence (“MannLoad”, “MannGenerate”,
    “MannFixed”, “Random”, “None”)
  * **turbulence_box_path** – Path to turbulence box files (required for MannLoad)
  * **max_turb_move** – Maximum distance turbines can move in one timestep (m)
    Used to calculate wind direction change rate limits

#### create_sites(ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), ti: [float](https://docs.python.org/3/library/functions.html#float), wd_list: [list](https://docs.python.org/3/library/stdtypes.html#list), dt_sim: [float](https://docs.python.org/3/library/functions.html#float), turbine_positions: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), rotor_diameter: [float](https://docs.python.org/3/library/functions.html#float), n_passthrough: [int](https://docs.python.org/3/library/functions.html#int), burn_in_passthroughs: [int](https://docs.python.org/3/library/functions.html#int), create_baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

Create turbulence fields and sites for agent and optionally baseline.

This method:
1. Generates turbulence field based on turbulence_type
2. Calculates t_developed and time_max based on farm geometry
3. Creates MetmastSite with wind direction time series
4. Optionally creates baseline site (deep copy of turbulence field)

* **Parameters:**
  * **ws** – Wind speed (m/s)
  * **wd** – Wind direction (degrees)
  * **ti** – Turbulence intensity (fraction)
  * **wd_list** – Wind direction time series
  * **dt_sim** – Simulation timestep (seconds)
  * **turbine_positions** – Turbine positions [x, y] array (n_turb, 2)
  * **rotor_diameter** – Rotor diameter (m)
  * **n_passthrough** – Number of flow passthroughs for episode
  * **burn_in_passthroughs** – Number of passthroughs for flow development
  * **create_baseline** – Whether to create baseline site
* **Returns:**
  (site, site_baseline, t_developed, time_max, added_turbulence_model)
  : site_baseline is None if create_baseline=False
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

## Measurement Manager

Modular Measurement and Noise System for WindGym Environments.

This module provides a structured way to define, manage, and apply various
types of noise (e.g., white noise, episodic bias, adversarial) to the
observations from a WindGym environment. It is designed for modularity and extensibility.

### *class* WindGym.core.measurement_manager.MeasurementType(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

#### WIND_SPEED *= 'wind_speed'*

#### WIND_DIRECTION *= 'wind_direction'*

#### YAW_ANGLE *= 'yaw_angle'*

#### TURBULENCE_INTENSITY *= 'turbulence_intensity'*

#### POWER *= 'power'*

#### GENERIC *= 'generic'*

### *class* WindGym.core.measurement_manager.MeasurementSpec(name: [str](https://docs.python.org/3/library/stdtypes.html#str), measurement_type: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType), index_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)], min_val: [float](https://docs.python.org/3/library/functions.html#float), max_val: [float](https://docs.python.org/3/library/functions.html#float), turbine_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, noise_sensitivity: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, is_circular: [bool](https://docs.python.org/3/library/functions.html#bool) = False, circular_range: [float](https://docs.python.org/3/library/functions.html#float) = 360.0)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Specification for a single component of the observation vector.

#### name

A descriptive name for the measurement (e.g., ‘turb_0/ws_current’).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### measurement_type

The physical type of the measurement.

* **Type:**
  [MeasurementType](#WindGym.core.measurement_manager.MeasurementType)

#### index_range

The start and end indices in the flat observation array.

* **Type:**
  Tuple[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

#### min_val

The minimum physical value for scaling.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### max_val

The maximum physical value for scaling.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### turbine_id

The turbine index, if applicable.

* **Type:**
  Optional[[int](https://docs.python.org/3/library/functions.html#int)]

#### noise_sensitivity

A multiplier for the applied noise level.

* **Type:**
  [float](https://docs.python.org/3/library/functions.html#float)

#### name *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### measurement_type *: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType)*

#### index_range *: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]*

#### min_val *: [float](https://docs.python.org/3/library/functions.html#float)*

#### max_val *: [float](https://docs.python.org/3/library/functions.html#float)*

#### turbine_id *: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None)* *= None*

#### noise_sensitivity *: [float](https://docs.python.org/3/library/functions.html#float)* *= 1.0*

#### is_circular *: [bool](https://docs.python.org/3/library/functions.html#bool)* *= False*

#### circular_range *: [float](https://docs.python.org/3/library/functions.html#float)* *= 360.0*

#### \_\_init_\_(name: [str](https://docs.python.org/3/library/stdtypes.html#str), measurement_type: [MeasurementType](#WindGym.core.measurement_manager.MeasurementType), index_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)], min_val: [float](https://docs.python.org/3/library/functions.html#float), max_val: [float](https://docs.python.org/3/library/functions.html#float), turbine_id: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, noise_sensitivity: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, is_circular: [bool](https://docs.python.org/3/library/functions.html#bool) = False, circular_range: [float](https://docs.python.org/3/library/functions.html#float) = 360.0) → [None](https://docs.python.org/3/library/constants.html#None)

### *class* WindGym.core.measurement_manager.NoiseModel

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

#### *abstractmethod* apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### *abstractmethod* get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

### *class* WindGym.core.measurement_manager.WhiteNoiseModel(noise_std_devs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [float](https://docs.python.org/3/library/functions.html#float)])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

Applies Gaussian white noise defined in physical units (e.g., m/s, degrees).

#### \_\_init_\_(noise_std_devs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [float](https://docs.python.org/3/library/functions.html#float)])

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

### *class* WindGym.core.measurement_manager.EpisodicBiasNoiseModel(bias_ranges: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

Applies a consistent bias for an entire episode, defined in physical units.

#### \_\_init_\_(bias_ranges: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[MeasurementType](#WindGym.core.measurement_manager.MeasurementType), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]])

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Applies the sampled episodic bias to the given observations.

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

### *class* WindGym.core.measurement_manager.HybridNoiseModel(models: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NoiseModel](#WindGym.core.measurement_manager.NoiseModel)])

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

#### \_\_init_\_(models: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NoiseModel](#WindGym.core.measurement_manager.NoiseModel)])

#### reset_noise(specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

#### apply_noise(observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), specs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasurementSpec](#WindGym.core.measurement_manager.MeasurementSpec)], rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

#### get_info() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)

### *class* WindGym.core.measurement_manager.MeasurementManager(env, seed=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Orchestrates measurement specifications and the application of noise.

#### \_\_init_\_(env, seed=None)

#### seed(seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Reseeds the random number generator for the noise model.

#### set_noise_model(noise_model: [NoiseModel](#WindGym.core.measurement_manager.NoiseModel))

#### reset_noise()

#### apply_noise(clean_observations: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)]

### *class* WindGym.core.measurement_manager.NoisyWindFarmEnv(base_env_class, measurement_manager: [MeasurementManager](#WindGym.core.measurement_manager.MeasurementManager), \*\*env_kwargs)

Bases: [`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)

A Gym wrapper that applies measurement errors to a base WindFarm environment.

#### \_\_init_\_(base_env_class, measurement_manager: [MeasurementManager](#WindGym.core.measurement_manager.MeasurementManager), \*\*env_kwargs)

Wraps an environment to allow a modular transformation of the [`step()`](#WindGym.core.measurement_manager.NoisyWindFarmEnv.step) and [`reset()`](#WindGym.core.measurement_manager.NoisyWindFarmEnv.reset) methods.

* **Parameters:**
  **env** – The environment to wrap

#### reset(, seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Uses the [`reset()`](#WindGym.core.measurement_manager.NoisyWindFarmEnv.reset) of the `env` that can be overwritten to change the returned data.

#### step(action: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [float](https://docs.python.org/3/library/functions.html#float), [bool](https://docs.python.org/3/library/functions.html#bool), [bool](https://docs.python.org/3/library/functions.html#bool), [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Uses the [`step()`](#WindGym.core.measurement_manager.NoisyWindFarmEnv.step) of the `env` that can be overwritten to change the returned data.

#### close()

Closes the wrapper and `env`.

### *class* WindGym.core.measurement_manager.AdversarialNoiseModel(antagonist_agent, constraints, device)

Bases: [`NoiseModel`](#WindGym.core.measurement_manager.NoiseModel)

#### \_\_init_\_(antagonist_agent, constraints, device)

#### reset_noise(specs: [list](https://docs.python.org/3/library/stdtypes.html#list), rng: [Generator](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator))

#### apply_noise(clean_observations, specs, rng)

#### get_info()

## Measurement Classes

### *class* WindGym.core.mes_class.Mes(current: [bool](https://docs.python.org/3/library/functions.html#bool) = True, rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = False, history_N: [int](https://docs.python.org/3/library/functions.html#int) = 3, history_length: [int](https://docs.python.org/3/library/functions.html#int) = 100, window_length: [int](https://docs.python.org/3/library/functions.html#int) = 5)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Baseclass for the measurements,
we can decide how large a memory we need, and also how many measurements we want to get back
Current: bool, if true return the latest measurement
Rolling Mean: bool, if true return the rolling mean of the measurements
history_N: int, number of rolling windows to use for the rolling mean. If 1, only return the latest value, if 2 return the lates and oldest value, if more then do some inbetween values also
history_length: int, number of measurements to save
window_length: int, size of the rolling window

#### \_\_init_\_(current: [bool](https://docs.python.org/3/library/functions.html#bool) = True, rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = False, history_N: [int](https://docs.python.org/3/library/functions.html#int) = 3, history_length: [int](https://docs.python.org/3/library/functions.html#int) = 100, window_length: [int](https://docs.python.org/3/library/functions.html#int) = 5) → [None](https://docs.python.org/3/library/constants.html#None)

#### \_\_call_\_(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque via the call function

#### append(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque via the append function

#### add_measurement(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

Append the measurement to the deque

#### get_measurements() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Get the desired measurements with graceful handling of startup period

### *class* WindGym.core.mes_class.TurbMes(ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, include_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Class for all measurements.
Each turbine stores measurements for wind speed, wind direction and yaw angle…

#### \_\_init_\_(ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, include_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_hf_ws(measurement: [float](https://docs.python.org/3/library/functions.html#float)) → [None](https://docs.python.org/3/library/constants.html#None)

Appends a single wind speed measurement to the high-frequency buffer.

#### empty_np(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Return an empty array

#### calc_TI(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

Calcualte TI from the wind speed measurements

#### max_hist() → [int](https://docs.python.org/3/library/functions.html#int)

Return the maximum history length of the measurements

#### observed_variables() → [int](https://docs.python.org/3/library/functions.html#int)

Returns the number of observed variables

#### add_ws(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_wd(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_yaw(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_power(measurement: [float](https://docs.python.org/3/library/functions.html#float) | [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### get_ws(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_yaw(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_measurements(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

### *class* WindGym.core.mes_class.FarmMes(n_turbines: [int](https://docs.python.org/3/library/functions.html#int), n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, turb_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_power: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = False, farm_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Class for the measurements of the farm.
The farm stores measurements from each turbine for wind speed, wind direction, yaw angle, power

#### \_\_init_\_(n_turbines: [int](https://docs.python.org/3/library/functions.html#int), n_probes_per_turb: [dict](https://docs.python.org/3/library/stdtypes.html#dict) = {}, turb_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = True, turb_power: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_ws: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_wd: [bool](https://docs.python.org/3/library/functions.html#bool) = True, farm_TI: [bool](https://docs.python.org/3/library/functions.html#bool) = False, farm_power: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, ws_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, ws_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, wd_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, wd_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, wd_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, yaw_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, yaw_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, yaw_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 2, yaw_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 30, yaw_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_current: [bool](https://docs.python.org/3/library/functions.html#bool) = False, power_rolling_mean: [bool](https://docs.python.org/3/library/functions.html#bool) = True, power_history_N: [int](https://docs.python.org/3/library/functions.html#int) = 1, power_history_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, power_window_length: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_min: [float](https://docs.python.org/3/library/functions.html#float) = 7.0, ws_max: [float](https://docs.python.org/3/library/functions.html#float) = 20.0, wd_min: [float](https://docs.python.org/3/library/functions.html#float) = 270.0, wd_max: [float](https://docs.python.org/3/library/functions.html#float) = 360.0, yaw_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TI_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, TI_max: [float](https://docs.python.org/3/library/functions.html#float) = 0.5, power_max: [float](https://docs.python.org/3/library/functions.html#float) = 2000000, ti_sample_count: [int](https://docs.python.org/3/library/functions.html#int) = 30) → [None](https://docs.python.org/3/library/constants.html#None)

#### empty_np(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### add_ws(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_wd(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_yaw(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_power(measurement: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### add_measurements(ws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], wd: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], powers: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]) → [None](https://docs.python.org/3/library/constants.html#None)

#### max_hist() → [int](https://docs.python.org/3/library/functions.html#int)

Return the maximum history length of the measurements

#### observed_variables() → [int](https://docs.python.org/3/library/functions.html#int)

Returns the number of observed variables

#### get_ws_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_ws_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_power_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_wd_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_TI_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_TI_farm(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [float](https://docs.python.org/3/library/functions.html#float)

#### get_yaw_turb(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

#### get_measurements(scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[float32]]

## Baseline Manager

Baseline controller management module for WindGym environments.

This module handles baseline controller setup, management, and execution
for comparing agent performance against baseline control strategies.

### *class* WindGym.core.baseline_manager.BaselineManager(baseline_controller_type: [str](https://docs.python.org/3/library/stdtypes.html#str), x_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), y_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), turbine, yaw_max: [float](https://docs.python.org/3/library/functions.html#float), yaw_min: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_env: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_sim: [float](https://docs.python.org/3/library/functions.html#float), htc_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages baseline controller setup and execution.

Supports multiple baseline controller types:
- Local: Local yaw controller
- Global: Global yaw controller
- PyWake: PyWake optimization-based agent (oracle or local mode)

Also handles baseline turbine initialization for HAWC2 or PyWake turbines.

#### \_\_init_\_(baseline_controller_type: [str](https://docs.python.org/3/library/stdtypes.html#str), x_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), y_pos: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), turbine, yaw_max: [float](https://docs.python.org/3/library/functions.html#float), yaw_min: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_env: [float](https://docs.python.org/3/library/functions.html#float), yaw_step_sim: [float](https://docs.python.org/3/library/functions.html#float), htc_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the baseline manager.

* **Parameters:**
  * **baseline_controller_type** – Type of baseline controller
    (“Local”, “Global”, “PyWake_oracle”, “PyWake_local”)
  * **x_pos** – X positions of turbines
  * **y_pos** – Y positions of turbines
  * **turbine** – Turbine object
  * **yaw_max** – Maximum yaw angle (degrees)
  * **yaw_min** – Minimum yaw angle (degrees)
  * **yaw_step_env** – Yaw step per environment step (degrees)
  * **yaw_step_sim** – Yaw step per simulation step (degrees)
  * **htc_path** – Optional path to HAWC2 HTC file

#### initialize_baseline_turbines(name_string: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize baseline turbines (HAWC2 or PyWake).

* **Parameters:**
  **name_string** – Optional name string for HAWC2 case (required if htc_path is set)
* **Returns:**
  Baseline turbine object

#### compute_baseline_action(fs, yaw_step: [float](https://docs.python.org/3/library/functions.html#float) = 1.0) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Compute baseline controller action.

* **Parameters:**
  * **fs** – Flow simulation object (baseline)
  * **yaw_step** – Yaw step size (degrees)
* **Returns:**
  New yaw angles for baseline turbines
* **Return type:**
  np.ndarray

#### update_wind_conditions(ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), ti: [float](https://docs.python.org/3/library/functions.html#float))

Update wind conditions for baseline manager.

This is needed for PyWake agent in oracle mode and for tracking
current conditions.

* **Parameters:**
  * **ws** – Wind speed (m/s)
  * **wd** – Wind direction (degrees)
  * **ti** – Turbulence intensity (fraction)

## Probe Manager

Probe management module for WindGym environments.

This module handles wind probe initialization, positioning, and rotation
to track wind conditions at specific locations in the wind farm.

### *class* WindGym.core.probe_manager.ProbeManager(probes_config: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Manages wind probes in the environment.

Supports two placement modes:
1. Free placement: Probes at fixed absolute positions
2. Turbine-relative: Probes positioned relative to turbines, rotating with yaw

#### probes

List of all WindProbe objects

#### turbine_probes

Dict mapping turbine_index to list of probes

#### \_\_init_\_(probes_config: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)] | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the probe manager.

* **Parameters:**
  **probes_config** – List of probe configuration dictionaries.
  Each dict can contain:
  - position: Absolute [x, y, z] position (free placement)
  - turbine_index: Index of turbine to attach to
  - relative_position: [x, y, z] relative to turbine
  - include_wakes: Whether to include wake effects
  - exclude_wake_from: List of turbine indices to exclude
  - time: Specific time for probe reading
  - probe_type: “WS” or “TI”
  - name: Optional probe name

#### count_probes_per_turbine() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

Count how many probes are assigned to each turbine index.

* **Returns:**
  Dict mapping turbine_index to probe count

#### initialize_probes_free_placement(env) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[WindProbe](#WindGym.core.wind_probe.WindProbe)]

Initialize probes with free (absolute) placement.

This mode is for probes at fixed positions that don’t rotate with turbines.

* **Parameters:**
  **env** – Environment object (for WindProbe compatibility)
* **Returns:**
  List of initialized WindProbe objects

#### initialize_probes(fs, yaw_angles) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[WindProbe](#WindGym.core.wind_probe.WindProbe)]

Initialize probes with turbine-relative positioning.

Probes can be placed relative to turbines and will rotate with turbine yaw.

* **Parameters:**
  * **fs** – Flow simulation object
  * **yaw_angles** – Initial yaw angles (degrees), scalar or array
* **Returns:**
  List of initialized WindProbe objects

#### update_probe_positions(fs, yaw_angles)

Update probe positions when turbines yaw.

Only updates probes that are attached to turbines (turbine-relative).

* **Parameters:**
  * **fs** – Flow simulation object
  * **yaw_angles** – New yaw angles (degrees), array

#### get_probe_readings() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]

Get readings from all probes.

* **Returns:**
  List of probe readings (wind speed or turbulence intensity)

#### get_turbine_probe_readings(turbine_index: [int](https://docs.python.org/3/library/functions.html#int)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]

Get readings from probes attached to a specific turbine.

* **Parameters:**
  **turbine_index** – Index of turbine
* **Returns:**
  List of probe readings for that turbine

#### has_probes() → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if any probes are configured.

## Wind Probe

### *class* WindGym.core.wind_probe.WindProbe(fs, position, yaw_angle, turbine_position, include_wakes=True, exclude_wake_from=[], time=None, probe_type='WS')

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### \_\_init_\_(fs, position, yaw_angle, turbine_position, include_wakes=True, exclude_wake_from=[], time=None, probe_type='WS')

Initialize a wind speed or TI probe.

* **Parameters:**
  * **fs** – The wind farm environment (should have get_windspeed() and get_turbulence_intensity()).
  * **position** – (x, y, z) tuple for probe location.
  * **include_wakes** – Whether to include wake effects in the wind calculation.
  * **exclude_wake_from** – Turbine indices to exclude wakes from.
  * **time** – Specific time (optional).
  * **probe_type** – ‘WS’ for wind speed, ‘TI’ for turbulence intensity.

#### read()

Read either wind speed (u, v, w) or turbulence intensity depending on probe_type.

#### read_speed_magnitude()

Return scalar wind speed magnitude.

#### update_position(new_position)

Move probe to a new (x, y, z) position.

#### get_projected_wind_speed_toward_turbine()

Projects the wind speed vector onto the direction from the probe to the turbine.

* **Parameters:**
  **turbine_position** – (x, y, z) of the turbine.
* **Returns:**
  Scalar wind speed component in direction from probe to turbine.

#### get_inflow_angle_to_turbine(degrees=False)

Returns the angle from the probe to the turbine (horizontal XY-plane),
counter-clockwise from the x-axis.

* **Parameters:**
  * **turbine_position** – (x, y, z) of the turbine.
  * **degrees** – If True, return angle in degrees.
* **Returns:**
  Angle in radians (or degrees if requested).

## Renderer

Rendering module for WindGym wind farm environments.

This module handles all visualization and rendering functionality,
separating it from the core environment logic.

### *class* WindGym.core.renderer.WindFarmRenderer(render_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Handles rendering of wind farm environments.

Supports multiple render modes:
- “rgb_array”: Return RGB frames for recording/saving
- “human”: Display frames in a window for human viewing
- None: No rendering

Also provides utility methods for plotting farm layouts and frames.

#### \_\_init_\_(render_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Initialize the renderer.

* **Parameters:**
  **render_mode** – Rendering mode (“human”, “rgb_array”, or None)

#### init_render(fs, turbine)

Initialize rendering objects.

This creates the matplotlib figure, axis, and XYView for rendering.
Should be called after the flow simulation is created.

* **Parameters:**
  * **fs** – Flow simulation object
  * **turbine** – Turbine object (for hub_height)

#### render(fs, fs_baseline=None, probes=None, turbine=None)

Main render method - routes to appropriate rendering function.

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **probes** – Optional list of wind probes
  * **turbine** – Turbine object for lazy initialization
* **Returns:**
  RGB array if render_mode is “rgb_array”, None otherwise

#### plot_farm(fs, fs_baseline=None, turbine=None, baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False, fix_turbines: [bool](https://docs.python.org/3/library/functions.html#bool) = True)

Plot the entire farm layout (legacy method for IPython notebooks).

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **turbine** – Turbine object
  * **baseline** – Whether to plot baseline instead of agent

#### plot_frame(fs, fs_baseline=None, turbine=None, baseline: [bool](https://docs.python.org/3/library/functions.html#bool) = False)

Plot a single frame of the flow field and turbines.

* **Parameters:**
  * **fs** – Flow simulation object
  * **fs_baseline** – Optional baseline flow simulation
  * **turbine** – Turbine object
  * **baseline** – Whether to plot baseline instead of agent

#### close()

Close any open matplotlib figures.
