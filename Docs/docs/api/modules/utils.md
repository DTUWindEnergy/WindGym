# WindGym.utils Module

This module contains utility functions for WindGym.

<a id="module-WindGym.utils"></a>

Utility functions and tools for WindGym environments.

### WindGym.utils.scale_val(val: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], min_val: [float](https://docs.python.org/3/library/functions.html#float), max_val: [float](https://docs.python.org/3/library/functions.html#float)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]

Scale the value from -1 to 1.

### WindGym.utils.defined_yaw(yaws: [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]], n_turb: [int](https://docs.python.org/3/library/functions.html#int)) → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), ...], [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[floating](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.floating)]]

Set the yaw values to a specified value.

If the length of the yaw values is equal to the number of turbines, return the yaw values.
If the length of the yaw values is 1, we assume that all turbines should have that yaw value.

## Evaluate PPO

### Enhanced Coliseum Evaluation Framework

A flexible evaluation framework for comparing multiple agents in WindFarm environments.

Wind Sampling Behavior:
- Time Series Evaluation: Uses sample_site for realistic stochastic wind sampling
- Wind Grid Evaluation: Uses fixed wind conditions across the specified grid

Supports time series evaluation with mean cumulative rewards and wind condition grid evaluation.

### *class* WindGym.utils.evaluate_PPO.Coliseum(env_factory: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable), agents: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [object](https://docs.python.org/3/library/functions.html#object)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)], agent_labels: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, n_passthrough: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, burn_in_passthroughs: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Enhanced evaluation framework to compare multiple agents in WindFarm environments.

Features:
- Time series evaluation with detailed episode history
- Wind condition grid evaluation with NetCDF export
- Mean cumulative reward tracking
- Flexible agent management with custom labels
- Comprehensive plotting capabilities

#### \_\_init_\_(env_factory: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable), agents: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [object](https://docs.python.org/3/library/functions.html#object)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)], agent_labels: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, n_passthrough: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, burn_in_passthroughs: [float](https://docs.python.org/3/library/functions.html#float) = 2.0)

Initialize the Coliseum evaluation framework.

* **Parameters:**
  * **env_factory** (*Callable*) – Function that returns a new environment instance.
    Example: lambda: WindFarmEnv(…)
  * **agents** (*Union* *[**Dict* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*object*](https://docs.python.org/3/library/functions.html#object) *]* *,* *List* *[*[*object*](https://docs.python.org/3/library/functions.html#object) *]* *]*) – Either a dictionary {name: agent} or list of agent objects.
    All agents must have a .predict(obs, deterministic) method.
  * **agent_labels** (*Optional* *[**List* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]*) – Custom labels for agents when using list input.
    If None, defaults to “Agent_0”, “Agent_1”, etc.
  * **n_passthrough** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Number of flow passthroughs for episode length.
    Defaults to 1.0.
  * **burn_in_passthroughs** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Number of flow passthroughs before episode

#### run_time_series_evaluation(num_episodes: [int](https://docs.python.org/3/library/functions.html#int) = 10, seed: [int](https://docs.python.org/3/library/functions.html#int) = 42, deterministic: [bool](https://docs.python.org/3/library/functions.html#bool) = True, save_detailed_history: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → DataFrame

Run time series evaluation with stochastic wind conditions using sample_site.

This method relies on the environment’s sample_site for realistic wind sampling.
Each episode will have different wind conditions sampled from the site’s
wind resource distributions (Weibull for wind speed, frequency for direction).

* **Parameters:**
  * **num_episodes** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of episodes to run
  * **seed** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Master seed for reproducibility
  * **deterministic** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Whether to use deterministic agent policies
  * **save_detailed_history** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Whether to save detailed time series data
* **Returns:**
  Summary results with mean cumulative rewards
* **Return type:**
  pd.DataFrame

#### run_wind_grid_evaluation(wd_step: [int](https://docs.python.org/3/library/functions.html#int) = 10, ws_step: [int](https://docs.python.org/3/library/functions.html#int) = 2, ti_points: [int](https://docs.python.org/3/library/functions.html#int) = 3, wd_min: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, wd_max: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, ws_min: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, ws_max: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, ti_min: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, ti_max: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, deterministic: [bool](https://docs.python.org/3/library/functions.html#bool) = True, save_netcdf: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → Dataset

Run evaluation over a grid of wind conditions and return as xarray Dataset.

* **Parameters:**
  * **wd_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Wind direction step size in degrees
  * **ws_step** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Wind speed step size in m/s
  * **ti_points** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of turbulence intensity points
  * **wd_min** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Minimum wind direction. If None, uses env.wd_min
  * **wd_max** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Maximum wind direction. If None, uses env.wd_max
  * **ws_min** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Minimum wind speed. If None, uses env.ws_min
  * **ws_max** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Maximum wind speed. If None, uses env.ws_max
  * **ti_min** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Minimum turbulence intensity. If None, uses env.TI_min
  * **ti_max** (*Optional* *[*[*float*](https://docs.python.org/3/library/functions.html#float) *]*) – Maximum turbulence intensity. If None, uses env.TI_max
  * **deterministic** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Whether to use deterministic policies
  * **save_netcdf** (*Optional* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – Path to save NetCDF file
* **Returns:**
  Results with dimensions (wd, ws, ti) and variables for each agent
* **Return type:**
  xr.Dataset

#### plot_time_series_comparison(episodes_to_plot: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, save_path: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'time_series_comparison.png')

Plot time series comparison of mean cumulative rewards.

* **Parameters:**
  * **episodes_to_plot** (*Optional* *[**List* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]* *]*) – Specific episodes to plot.
    If None, plots first 3 episodes.
  * **save_path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Path to save the figure

#### plot_summary_comparison(save_path: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'summary_comparison.png')

Plot summary comparison showing average performance across all episodes.

* **Parameters:**
  **save_path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Path to save the figure

#### plot_wind_grid_results(dataset: Dataset, agent_name: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, save_path: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'wind_grid_results.png')

Plot wind grid evaluation results as heatmaps.

* **Parameters:**
  * **dataset** (*xr.Dataset*) – Results from wind grid evaluation
  * **agent_name** (*Optional* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – Specific agent to plot. If None, plots all agents.
  * **save_path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Path to save the figure

#### get_summary_statistics() → DataFrame

Get summary statistics for all agents across all episodes.

#### *static* create_env_factory_with_site(env_class, site, \*\*env_kwargs)

Helper method to create an environment factory with sample_site configured.

* **Parameters:**
  * **env_class** – Environment class (e.g., WindFarmEnv, EvaluationEnv)
  * **site** – PyWake site object for realistic wind sampling
  * **\*\*env_kwargs** – Additional environment parameters
* **Returns:**
  Environment factory function
* **Return type:**
  Callable

## Generate Layouts

### WindGym.utils.generate_layouts.generate_square_grid(turbine, nx, ny, xDist, yDist)

Create a square grid of turbines.

* **Parameters:**
  * **turbine** (*WindTurbine*) – The wind turbine object.
  * **nx** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of turbines in the x-direction.
  * **ny** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of turbines in the y-direction.
  * **xDist** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Diameter distance between turbines in the x-direction.
  * **yDist** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Diameter distance between turbines in the y-direction.
* **Returns:**
  Array of turbine positions.
* **Return type:**
  np.ndarray

### WindGym.utils.generate_layouts.generate_circle(n, r, angle_offset=0)

Generate a circular grid of n points with radius r.

### WindGym.utils.generate_layouts.generate_cirular_farm(n_list: [Buffer](https://docs.python.org/3/library/collections.abc.html#collections.abc.Buffer) | \_SupportsArray[[dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | \_NestedSequence[\_SupportsArray[[dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]] | [bool](https://docs.python.org/3/library/functions.html#bool) | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [complex](https://docs.python.org/3/library/functions.html#complex) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | \_NestedSequence[[bool](https://docs.python.org/3/library/functions.html#bool) | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [complex](https://docs.python.org/3/library/functions.html#complex) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)], turbine, r_dist: [float](https://docs.python.org/3/library/functions.html#float) = 5, angle_offset_list: [Buffer](https://docs.python.org/3/library/collections.abc.html#collections.abc.Buffer) | \_SupportsArray[[dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | \_NestedSequence[\_SupportsArray[[dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]] | [bool](https://docs.python.org/3/library/functions.html#bool) | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [complex](https://docs.python.org/3/library/functions.html#complex) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes) | \_NestedSequence[[bool](https://docs.python.org/3/library/functions.html#bool) | [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float) | [complex](https://docs.python.org/3/library/functions.html#complex) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)] = None)

Generate a circular farm of n circular grids with radius r and m points.

### WindGym.utils.generate_layouts.generate_staggered_grid(turbine, nx, ny, xDist, yDist, x_stagger_offset=None, y_stagger_offset=None)

Create a staggered grid of turbines with column- or row-based offsets.

* **Parameters:**
  * **turbine** (*WindTurbine*) – The wind turbine object.
  * **nx** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of turbines in the x-direction.
  * **ny** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of turbines in the y-direction.
  * **xDist** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Distance between turbines in the x-direction, in rotor diameters.
  * **yDist** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Distance between turbines in the y-direction, in rotor diameters.
  * **x_stagger_offset** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3/library/functions.html#float) *] or* *None*) – List of horizontal offsets (in rotor diameters) per column.
  * **y_stagger_offset** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3/library/functions.html#float) *] or* *None*) – List of vertical offsets (in rotor diameters) per column.
* **Returns:**
  Array of turbine positions.
* **Return type:**
  np.ndarray

### WindGym.utils.generate_layouts.plot_farm(x, y, turbine=None, D=None)

Plots the turbines in the farm layout, and their minimum distance to the closest turbine
