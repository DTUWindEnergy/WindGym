# WindGym.visualization Module

This module contains visualization utilities for WindGym.

<a id="module-WindGym.visualization"></a>

## Visualization Module

This module contains plotting utilities for wind farm evaluation results.

Modules:
: - farm_plots: Farm-level plotting functions
  - turbine_plots: Turbine-level plotting functions
  - plot_utils: Shared plotting utilities

### WindGym.visualization.plot_power_farm(data, WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power output for the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **WSS** – List of wind speeds to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.plot_farm_inc(data, WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the percentage increase in power output for the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **WSS** – List of wind speeds to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.plot_power_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power output for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.plot_yaw_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the yaw angle for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.plot_speed_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the rotor wind speed for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.plot_turb(data, ws, wd, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power, yaw and rotor wind speed for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **wd** – Wind direction to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.setup_wind_grid_axes(axs: [Any](https://docs.python.org/3/library/typing.html#typing.Any), j: [int](https://docs.python.org/3/library/functions.html#int), i: [int](https://docs.python.org/3/library/functions.html#int), WSS: [list](https://docs.python.org/3/library/stdtypes.html#list), WDS: [list](https://docs.python.org/3/library/stdtypes.html#list), WS: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), add_grid: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [None](https://docs.python.org/3/library/constants.html#None)

Configure axes for wind condition grid plots.

* **Parameters:**
  * **axs** – Matplotlib axes array
  * **j** – Row index
  * **i** – Column index
  * **WSS** – List of wind speeds
  * **WDS** – List of wind directions
  * **WS** – Current wind speed
  * **wd** – Current wind direction
  * **add_grid** – Whether to add grid lines

### WindGym.visualization.calculate_time_limits(data: [Any](https://docs.python.org/3/library/typing.html#typing.Any), ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), TI: [float](https://docs.python.org/3/library/functions.html#float), TURBBOX: [str](https://docs.python.org/3/library/stdtypes.html#str), variable: [str](https://docs.python.org/3/library/stdtypes.html#str), avg_n: [int](https://docs.python.org/3/library/functions.html#int) = 10, turb_idx: [int](https://docs.python.org/3/library/functions.html#int) = None) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]

Calculate x-axis time limits for a given data selection.

* **Parameters:**
  * **data** – xarray Dataset
  * **ws** – Wind speed
  * **wd** – Wind direction
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **variable** – Variable name to calculate limits for
  * **avg_n** – Rolling average window size
  * **turb_idx** – Turbine index (None for farm-level data)
* **Returns:**
  Tuple of (x_start, x_end) time limits

## Farm Plots

Farm-level plotting functions for wind farm evaluation results.

### WindGym.visualization.farm_plots.plot_power_farm(data, WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power output for the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **WSS** – List of wind speeds to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.farm_plots.plot_farm_inc(data, WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the percentage increase in power output for the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **WSS** – List of wind speeds to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

## Turbine Plots

Turbine-level plotting functions for wind farm evaluation results.

### WindGym.visualization.turbine_plots.plot_power_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power output for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.turbine_plots.plot_yaw_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the yaw angle for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.turbine_plots.plot_speed_turb(data, ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the rotor wind speed for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **WDS** – List of wind directions to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

### WindGym.visualization.turbine_plots.plot_turb(data, ws, wd, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False, save_path=None)

Plot the power, yaw and rotor wind speed for each turbine in the farm.

* **Parameters:**
  * **data** – xarray Dataset with evaluation results
  * **ws** – Wind speed to plot
  * **wd** – Wind direction to plot
  * **avg_n** – Rolling average window size
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **axs** – Matplotlib axes array (optional)
  * **save** – Whether to save the figure
  * **save_path** – Path to save figure (uses default if None)
* **Returns:**
  Tuple of (fig, axs)

## Plot Utilities

Shared plotting utilities for wind farm evaluation visualizations.

### WindGym.visualization.plot_utils.setup_wind_grid_axes(axs: [Any](https://docs.python.org/3/library/typing.html#typing.Any), j: [int](https://docs.python.org/3/library/functions.html#int), i: [int](https://docs.python.org/3/library/functions.html#int), WSS: [list](https://docs.python.org/3/library/stdtypes.html#list), WDS: [list](https://docs.python.org/3/library/stdtypes.html#list), WS: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), add_grid: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [None](https://docs.python.org/3/library/constants.html#None)

Configure axes for wind condition grid plots.

* **Parameters:**
  * **axs** – Matplotlib axes array
  * **j** – Row index
  * **i** – Column index
  * **WSS** – List of wind speeds
  * **WDS** – List of wind directions
  * **WS** – Current wind speed
  * **wd** – Current wind direction
  * **add_grid** – Whether to add grid lines

### WindGym.visualization.plot_utils.calculate_time_limits(data: [Any](https://docs.python.org/3/library/typing.html#typing.Any), ws: [float](https://docs.python.org/3/library/functions.html#float), wd: [float](https://docs.python.org/3/library/functions.html#float), TI: [float](https://docs.python.org/3/library/functions.html#float), TURBBOX: [str](https://docs.python.org/3/library/stdtypes.html#str), variable: [str](https://docs.python.org/3/library/stdtypes.html#str), avg_n: [int](https://docs.python.org/3/library/functions.html#int) = 10, turb_idx: [int](https://docs.python.org/3/library/functions.html#int) = None) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]

Calculate x-axis time limits for a given data selection.

* **Parameters:**
  * **data** – xarray Dataset
  * **ws** – Wind speed
  * **wd** – Wind direction
  * **TI** – Turbulence intensity
  * **TURBBOX** – Turbulence box identifier
  * **variable** – Variable name to calculate limits for
  * **avg_n** – Rolling average window size
  * **turb_idx** – Turbine index (None for farm-level data)
* **Returns:**
  Tuple of (x_start, x_end) time limits
