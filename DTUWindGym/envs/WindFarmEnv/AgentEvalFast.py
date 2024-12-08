import xarray as xr
import numpy as np
# import concurrent.futures
# import multiprocessing
from dynamiks.views import XYView, EastNorthView
from dynamiks.visualizers.flow_visualizers import Flow2DVisualizer
from py_wake.utils.plotting import setup_plot
import os
import matplotlib.pyplot as plt
from collections import deque
from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
# from pathos.pools import ProcessPool

"""
AgentEval is a class that is used to evaluate an agent on the EnvEval environment.
The class is made to evaluate the agent for multiple wind directions, and then save a xarray dataset with the results.

TODO: Finish the class so that it can plot the results, and save the results to a file. maybe? 
TODO: We could add in a check that the agent has already been evaluated on a given condition. if yes, then we dont need to simulate it again.
TODO: Add a function to animate the results.
TODO: parallelize the evaluation in eval_multiple()
TODO: Consolidate the plotting functions, so that they are more general.
"""

# def eval_single_fast(env, model, ws=10.0, ti=0.05, wd=270, yaw=0.0, turbbox="Default", t_sim=1000, save_figs=False, scale_obs=None, debug=False):


def eval_single_fast(env, model,
                     model_step,
                     ws=10.0, ti=0.05,
                     wd=270,
                     turbbox="Default",
                     save_figs=False,
                     scale_obs=None,
                     t_sim=1000,
                     name="NoName",
                     debug=False,
                     deterministic=True):
    """
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

    """

    env.set_wind_vals(ws=ws, ti=ti, wd=wd)

    if not isinstance(scale_obs, list):  # if not a list, make it one
        scaling = [scale_obs]
    if debug:  # If debug, do both.
        scaling = [True, False]
        save_figs = True

    if model is None:
        AssertionError("You need to specify a model to evaluate the agent.")

    # Unpack some variables, to make the code more readable
    time = t_sim  # Time to simulate
    n_turb = env.n_turb  # Number of turbines
    n_ws = 1  # Number of wind speeds to simulate
    n_wd = 1  # Number of wind direction simulate
    n_turbbox = 1  # Number of turbulence boxes to simulate
    n_TI = 1  # Number of turbulence intensities to simulate

    # Initialize the arrays to store the results
    # _a is the agent and _b is the baseline
    powerF_a = np.zeros((time))
    # powerF_b = np.zeros((time))
    powerT_a = np.zeros((time, n_turb))
    # powerT_b = np.zeros((time, n_turb))
    yaw_a = np.zeros((time, n_turb))
    # yaw_b = np.zeros((time, n_turb))
    ws_a = np.zeros((time, n_turb))
    # ws_b = np.zeros((time, n_turb))
    time_plot = np.zeros((time))
    pct_inc = np.zeros((time))
    rew_plot = np.zeros((time))

    # Initialize the environment
    obs, info = env.reset()

    # This checks if we are using a pywakeagent. If we are, then we do this:
    if hasattr(model, "pywakeagent") or hasattr(model, "florisagent"):
        model.update_wind(env.ws, env.wd, env.ti)
        model.predict(obs, deterministic=deterministic)[0]
    # This checks if we are using an agent that needs the environment. If we are, then we do this
    if hasattr(model, "UseEnv"):
        model.yaw_max = env.yaw_max
        model.yaw_min = env.yaw_min
        model.env = env

        # Put the initial values in the arrays
    powerF_a[0] = env.fs.windTurbines.power().sum()
    powerT_a[0] = env.fs.windTurbines.power()

    yaw_a[0] = info["yaw angles agent"]
    ws_a[0] = np.linalg.norm(
        env.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
    time_plot[0] = env.fs.time
    # There is no reward at the first time step, so we just set it to zero.
    rew_plot[0] = 0.0

    # If save_figs is True, initalize some parameters here.
    if save_figs:
        FOLDER = './Temp_Figs_{}_ws{}_wd{}/'.format(name, env.ws, wd)
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        max_deque = 70
        time_deq = deque(maxlen=max_deque)
        pow_deq = deque(maxlen=max_deque)
        yaw_deq = deque(maxlen=max_deque)
        ws_deq = deque(maxlen=max_deque)

        time_deq.append(time_plot[0])
        pow_deq.append(powerF_a[0])
        yaw_deq.append(yaw_a[0])
        ws_deq.append(ws_a[0])
        # These are used for y limits on the plot.
        pow_max = powerF_a[0]*1.2
        pow_min = powerF_a[0]*0.8
        yaw_max = 5
        yaw_min = -5
        ws_max = env.ws+2
        ws_min = 3

        # Define the x and y values for the flow field plot
        a = np.linspace(-200 + min(env.x_pos), 200 + max(env.x_pos), 200)
        b = np.linspace(-200 + min(env.y_pos), 200 + max(env.y_pos), 200)

    # Run the simulation
    for i in range(1, time):

        action = model.predict(obs, deterministic=deterministic)[0]
        obs, reward, terminated, truncated, info = env.step(action)

        # Put the values in the arrays
        powerF_a[i] = env.fs.windTurbines.power().sum()
        powerT_a[i] = env.fs.windTurbines.power()
        yaw_a[i] = info["yaw angles agent"]

        ws_a[i] = np.linalg.norm(
            env.fs.windTurbines.rotor_avg_windspeed(include_wakes=True), axis=0)
        time_plot[i] = env.fs.time
        rew_plot[i] = reward  #
        if save_figs:
            time_deq.append(time_plot[i])
            pow_deq.append(powerF_a[i])
            yaw_deq.append(yaw_a[i])
            ws_deq.append(ws_a[i])

            fig = plt.figure(figsize=(12, 7.5))
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

            view = XYView(z=70, x=a, y=b, ax=fig.gca(), adaptive=False)

            wt = env.fs.windTurbines
            # x_turb, y_turb = wt.positions_xyz(self.env.fs.wind_direction, self.env.fs.center_offset)[:2]
            x_turb, y_turb = wt.positions_xyz[:2]
            yaw, tilt = wt.yaw_tilt()

            # Plot the flowfield in ax1
            uvw = env.fs.get_windspeed(view, include_wakes=True, xarray=True)
            # [0] is the u component of the wind speed
            plt.pcolormesh(uvw.x.values, uvw.y.values,
                           uvw[0].T, shading="nearest", vmin=3, vmax=env.ws+2)
            plt.colorbar().set_label('Wind speed, u [m/s]')
            WindTurbinesPW.plot_xy(wt, x_turb, y_turb, types=wt.types,
                                   wd=env.fs.wind_direction, ax=ax1, yaw=yaw, tilt=tilt)

            ax1.set_title('Flow field at {} s'.format(env.fs.time))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            ax2 = plt.subplot2grid((3, 3), (0, 2), )
            ax3 = plt.subplot2grid((3, 3), (1, 2), )
            ax4 = plt.subplot2grid((3, 3), (2, 2), )

            # Plot the power in ax2
            ax2.plot(time_deq, pow_deq, color='orange')
            ax2.set_title('Farm power [W]')

            # Plot the yaws in ax3
            ax3.plot(time_deq, yaw_deq, label=np.arange(n_turb))
            ax3.set_title('Turbine yaws [deg]')
            ax3.legend(loc='upper left')

            # Plot the rotor windspeeds in ax4
            ax4.plot(time_deq, ws_deq, label=np.arange(n_turb))
            ax4.set_title('Rotor windspeeds [m/s]')
            ax4.set_xlabel('Time [s]')

            # Set the x limits for the plots
            ax2.set_xlim(time_deq[0], time_deq[-1])
            ax3.set_xlim(time_deq[0], time_deq[-1])
            ax4.set_xlim(time_deq[0], time_deq[-1])

            pow_max = max(pow_max, powerF_a[i]*1.2)
            pow_min = min(pow_min, powerF_a[i]*0.8)
            yaw_max = max(yaw_max, max(yaw_a[i])*1.2)
            # This value can be negative, so we multiply 1.2, instead of 0.8
            yaw_min = min(yaw_min, min(yaw_a[i])*1.2)
            ws_max = max(ws_max, max(ws_a[i])*1.2)
            ws_min = min(ws_min, min(ws_a[i])*0.8)

            # Set the y limits for the plots. If we go over/under the limits, the plot will adjust the limits.
            ax2.set_ylim(pow_min, pow_max)
            ax3.set_ylim(yaw_min, yaw_max)
            ax4.set_ylim(ws_min, ws_max)
            ax2.set_xticks([])
            ax3.set_xticks([])

            # Set the number of ticks on the x-axis to 5
            ax4.locator_params(axis='x', nbins=5)

            img_name = FOLDER + 'img_{:05d}.png'.format(i)

            # Add a text to the plot with the sensor values
            for scale in scaling:  # scaling can be a list with True and False. If True, we add the scaled observations to the plot. If False, we only add the unscaled observations.
                if scale is not None:
                    turb_ws = np.round(
                        env.farm_measurements.get_ws_turb(scale), 2)
                    turb_wd = np.round(
                        env.farm_measurements.get_wd_turb(scale), 2)
                    turb_TI = np.round(
                        env.farm_measurements.get_TI_turb(scale), 2)
                    turb_yaw = np.round(
                        env.farm_measurements.get_yaw_turb(scale), 2)
                    farm_ws = np.round(
                        env.farm_measurements.get_ws_farm(scale), 2)
                    farm_wd = np.round(
                        env.farm_measurements.get_wd_farm(scale), 2)
                    farm_TI = np.round(env.farm_measurements.get_TI(scale), 2)
                    if scale:
                        text_plot = f" Agent observations scaled: \n Turbine level wind speed: {turb_ws} \n Turbine level wind direction: {turb_wd} \n Turbine level yaw: {turb_yaw} \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} \n Farm level wind direction: {farm_wd} \n Farm level TI: {farm_TI} "
                        ax1.text(1.1, 1.3, text_plot, verticalalignment='top',
                                 horizontalalignment='left', transform=ax1.transAxes)
                    else:
                        text_plot = f" Agent observations: \n Turbine level wind speed: {turb_ws} [m/s] \n Turbine level wind direction: {turb_wd} [deg] \n Turbine level yaw: {turb_yaw} [deg] \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} [m/s] \n Farm level wind direction: {farm_wd} [deg] \n Farm level TI: {farm_TI} "
                        ax1.text(-0.1, 1.3, text_plot, verticalalignment='top',
                                 horizontalalignment='left', transform=ax1.transAxes)
            # So I coudnt figure out how to add some space to the left, so I added a white text, and then use that to stretch the plot. Whatever, it works
            ax1.text(1.95, 0.5, "Hey", verticalalignment='top',
                     horizontalalignment='left', transform=ax1.transAxes, color='white')

            plt.savefig(img_name, dpi=100, bbox_extra_artists=(
                ax1, ax2, ax3, ax4), bbox_inches='tight')
            plt.clf()
            plt.close('all')

    env.close()

    # Reshape the arrays and put them in a xarray dataset
    powerF_a = powerF_a.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)
    powerT_a = powerT_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    yaw_a = yaw_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    ws_a = ws_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    rew_plot = rew_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)

    # Then create a xarray dataset with the results
    ds = xr.Dataset(
        data_vars={
            # For agent:
            # Power for the farm: [time, turbine, ws, wd, TI, turbbox]
            "powerF_a": (("time", "ws", "wd", "TI", "turbbox", "model_step"), powerF_a),
            # Power pr turbine [time, ws, wd, TI, turbbox]
            "powerT_a": (("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"), powerT_a),
            # yaw is array of: [time, turbine, ws, wd, TI, turbbox]
            "yaw_a": (("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"), yaw_a),
            # Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]
            "ws_a": (("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"), ws_a),

            # For environment
            # Reward
            "reward": (("time", "ws", "wd", "TI", "turbbox", "model_step"), rew_plot),
        },
        coords={
            "ws": np.array([ws]),
            "wd": np.array([wd]),
            "turb": np.arange(env.n_turb),
            "time": time_plot,
            "TI": np.array([ti]),
            "turbbox": [turbbox],
            "model_step": np.array([model_step])
        },
    )
    return ds
