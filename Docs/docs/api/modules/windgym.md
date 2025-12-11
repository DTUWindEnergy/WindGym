# WindGym Package

## Main Environment Classes

### WindFarmEnv

### *class* WindGym.WindFarmEnv(turbine, x_pos, y_pos, n_passthrough=5, ws_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ws_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 30.0, wd_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0, wd_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 360, ti_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ti_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, yaw_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TurbBox='Default', turbtype='Random', backend: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'dynamiks', config=None, Baseline_comp=False, yaw_init=None, render_mode=None, seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, fill_window=True, sample_site=None, HTC_path=None, reset_init=True, burn_in_passthroughs=2, cleanup_on_time_limit: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_function=None, max_turb_move=2, \*\*kwargs)

Bases: [`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env)

#### metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]* *= {'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, n_passthrough=5, ws_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ws_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 30.0, wd_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0, wd_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 360, ti_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ti_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, yaw_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TurbBox='Default', turbtype='Random', backend: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'dynamiks', config=None, Baseline_comp=False, yaw_init=None, render_mode=None, seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, fill_window=True, sample_site=None, HTC_path=None, reset_init=True, burn_in_passthroughs=2, cleanup_on_time_limit: [bool](https://docs.python.org/3/library/functions.html#bool) = True, wd_function=None, max_turb_move=2, \*\*kwargs)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.

#### init_render()

Initialize rendering - delegates to renderer.

#### reset(seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

#### step(action)

The step function
1. Adjust the yaw angles of the turbines
2. Take a step in the flow simulation
3. Update the measurements
4. Calculate the reward
5. Return the observation, reward, terminated, truncated and info

#### render()

Render method required by Gymnasium API - delegates to renderer.

#### close()

Close the environment and clean up resources.

#### plot_farm(baseline=False, fix_turbines=False)

Plot the entire farm layout - delegates to renderer.

#### plot_frame(baseline=False)

Plot a single frame - delegates to renderer.

#### *property* pywake_agent

Expose pywake_agent from baseline_manager for backward compatibility.

#### *property* py_agent_mode

Expose py_agent_mode from baseline_manager for backward compatibility.

### WindFarmEnvMulti

### *class* WindGym.WindFarmEnvMulti(\*args: Any, \*\*kwargs: Any)

Bases: `ParallelEnv`, [`WindFarmEnv`](#WindGym.WindFarmEnv)

#### metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]* *= {'name': 'MultiFarm_environment_v0', 'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, n_passthrough=20, ws_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ws_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 30.0, wd_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0, wd_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 360, ti_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ti_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, yaw_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, TurbBox='Default', turbtype='MannGenerate', config=None, Baseline_comp=False, yaw_init=None, render_mode=None, seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=1, fill_window=True, sample_site=None, HTC_path=None, reset_init=False, burn_in_passthroughs=2)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.

#### render()

Render method required by Gymnasium API - delegates to renderer.

#### reset(seed=None, options=None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

#### step(actions)

The step function.
We unpack the actions, and call the step function of the parent class.

#### observation_space(agent)

#### action_space(agent)

### FarmEval

### *class* WindGym.FarmEval(turbine, x_pos, y_pos, finite_episode: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ws_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 30.0, wd_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0, wd_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 360, ti_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ti_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, yaw_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, yaw_init='Zeros', TurbBox='Default', config=None, Baseline_comp=False, render_mode=None, turbtype='MannGenerate', seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, n_passthrough=5, HTC_path=None, reset_init=True, fill_window=True, sample_site=None, burn_in_passthroughs=2)

Bases: [`WindFarmEnv`](#WindGym.WindFarmEnv)

#### metadata *: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]* *= {'render_modes': ['human', 'rgb_array']}*

#### \_\_init_\_(turbine, x_pos, y_pos, finite_episode: [bool](https://docs.python.org/3/library/functions.html#bool) = False, ws_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ws_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 30.0, wd_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0, wd_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 360, ti_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = 0.0, ti_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 1.0, yaw_scaling_min: [float](https://docs.python.org/3/library/functions.html#float) = -45, yaw_scaling_max: [float](https://docs.python.org/3/library/functions.html#float) = 45, yaw_init='Zeros', TurbBox='Default', config=None, Baseline_comp=False, render_mode=None, turbtype='MannGenerate', seed=None, dt_sim=1, dt_env=1, yaw_step_sim=1, yaw_step_env=None, n_passthrough=5, HTC_path=None, reset_init=True, fill_window=True, sample_site=None, burn_in_passthroughs=2)

This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
:param turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
:param n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
:param TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
:param TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
:param TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
:param turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
:param config: The environment configuration.

> - If dict: taken directly.
> - If str/Path to an existing file: loaded from file.
> - If str containing YAML (multi-line, not a file path): parsed as YAML.
* **Parameters:**
  * **Baseline_comp** – bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
  * **yaw_init** – str: The method for initializing the yaw angles of the turbines. If ‘Random’, then the yaw angles will be random. Else they will be zeros.
  * **render_mode** – str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
  * **seed** – int: The seed for the environment. If None, then the seed will be random.
  * **dt_sim** – float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
  * **dt_env** – float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
  * **yaw_step_sim** – float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
  * **fill_window** – bool: If True, then the measurements will be filled up at reset.
  * **sample_site** – pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
  * **HTC_path** – str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
  * **reset_init** – bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
  * **cleanup_on_time_limit** – bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.

#### reset(seed=None, options=None)

Reset the environment. This is called at the start of every episode.
- The wind conditions are sampled, and the site is set.
- The flow simulation is run for the time it takes for the flow to develop.
- The measurements are filled up with the initial values.

#### set_wind_vals(ws=None, ti=None, wd=None)

Set the wind values to be used in the evaluation

#### set_yaw_vals(yaw_vals)

Set the yaw values to be used in the evaluation

#### update_tf(path)

Overwrite the \_def_site method to set the turbulence field to the path given

### AgentEval

### *class* WindGym.AgentEval(env=None, model=None, name='NoName', t_sim=1000)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

#### \_\_init_\_(env=None, model=None, name='NoName', t_sim=1000)

#### set_conditions(winddirs: [list](https://docs.python.org/3/library/stdtypes.html#list) = [], windspeeds: [list](https://docs.python.org/3/library/stdtypes.html#list) = [], turbintensities: [list](https://docs.python.org/3/library/stdtypes.html#list) = [], turbboxes: [list](https://docs.python.org/3/library/stdtypes.html#list) = ['Default'])

#### set_condition(ws=None, ti=None, wd=None, yaw=None, turbbox=None)

#### set_env_vals()

#### update_env(env)

#### update_model(model)

#### eval_single(save_figs=False, scale_obs=None, debug=False, deterministic=False, return_loads=False)

Evaluate the agent on a single wind direction, wind speed, turbulence intensity and turbulence box.

#### eval_multiple(save_figs=False, scale_obs=None, debug=False, return_loads=False)

Evaluate the agent on multiple wind directions, wind speeds, turbulence intensities and turbulence boxes.

#### run_simulation(winddir, windspeed, TI, box, save_figs, scale_obs, debug)

Run a singel simulation.
This function might be used for the parallelization of the simulation.

#### plot_initial()

Plot the initial conditions of the simulation, alongside the turbines with their numbering.

#### plot_performance()

Plot the performance of the agent, and the baseline farm.
We could plot the power output, the wind speed, the wind direction, the yaw angles, the turbulence intensity, the wake losses, etc.
The return is a plot of the performance metrics.

#### save_performance()

Save the performance metrics to a file.
TODO: Maybe add the options for a specific path to save the file to.

#### load_performance(path)

Load the performance metrics from a file.
Can be used to see the results from a previous evaluation.

#### plot_power_farm(WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the power output for the farm.

#### plot_farm_inc(WSS, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the percentage increase in power output for the farm.

#### plot_power_turb(ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the power output for each turbine in the farm.

#### plot_yaw_turb(ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the yaw angle for each turbine in the farm.

#### plot_speed_turb(ws, WDS, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the rotor wind speed for each turbine in the farm.

#### plot_turb(ws, wd, avg_n=10, TI=0.07, TURBBOX='Default', axs=None, save=False)

Plot the power, yaw and rotor wind speed for each turbine in the farm.

## Functions

### WindGym.AgentEvalFast(env, model, model_step=1, ws=10.0, ti=0.05, wd=270, turbbox='Default', save_figs=False, scale_obs=None, t_sim=1000, name='NoName', debug=False, deterministic=False, return_loads=False, cleanup=True)

This function evaluates the agent for a single wind direction, and then saves the results in a xarray dataset.
The function can also save the figures, if save_figs is set to True.

Args:
env: The environment to evaluate the agent on.
model: The agent to evaluate.
model_step: The step of the model. This is used to keep track of the model step in the xarray dataset.
ws: The wind speed to simulate.
ti: The turbulence intensity to simulate.
wd: The wind direction to simulate.
turbbox: The turbulence box to simulate.
save_figs: If True, the function will save the figures.
scale_obs: If True, the function will scale the observations for the plots.
t_sim: The time to simulate.
name: The name of the evaluation.
debug: If True, the function will print debug information on the plots.
deterministic: If True, the agent will be deterministic.
