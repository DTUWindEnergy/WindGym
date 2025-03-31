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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

"""
This function was created such that we can evaluate the agent for a singe wind condtion, but as a function. It was done such becuase it made parallelization easier.
Wind turbine has a lambda function, so we must use the pathos library to parallelize the evaluation.
"""


def eval_single_fast(
    env,
    model,
    model_step=1,
    ws=10.0,
    ti=0.05,
    wd=270,
    turbbox="Default",
    save_figs=False,
    scale_obs=None,
    t_sim=1000,
    name="NoName",
    debug=False,
    deterministic=False,
):
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

    device = torch.device("cpu")

    if hasattr(env.unwrapped, "parent_pipes"):
        raise AssertionError(
            "The eval_single_fast function is not compatible with vectorized versions of the environment. Please use unvectorized envs instead."
        )

    env.set_wind_vals(ws=ws, ti=ti, wd=wd)
    baseline_comp = env.Baseline_comp

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
    powerT_a = np.zeros((time, n_turb))
    yaw_a = np.zeros((time, n_turb))
    ws_a = np.zeros((time, n_turb))
    time_plot = np.zeros((time))
    rew_plot = np.zeros((time))

    if baseline_comp:
        powerF_b = np.zeros((time))
        powerT_b = np.zeros((time, n_turb))
        yaw_b = np.zeros((time, n_turb))
        ws_b = np.zeros((time, n_turb))
        pct_inc = np.zeros((time))

    # Initialize the environment
    obs, info = env.reset()

    # This checks if we are using a pywakeagent. If we are, then we do this:
    if hasattr(model, "pywakeagent") or hasattr(model, "florisagent"):
        model.update_wind(ws, wd, ti)
        model.predict(obs, deterministic=deterministic)[0]
    # This checks if we are using an agent that needs the environment. If we are, then we do this
    if hasattr(model, "UseEnv"):
        model.yaw_max = env.yaw_max
        model.yaw_min = env.yaw_min
        model.env = env

    # Put the initial values in the arrays
    powerF_a[0] = env.fs.windTurbines.power().sum()
    powerT_a[0] = env.fs.windTurbines.power()
    yaw_a[0] = env.fs.windTurbines.yaw
    ws_a[0] = np.linalg.norm(env.fs.windTurbines.rotor_avg_windspeed, axis=1)
    time_plot[0] = env.fs.time
    # There is no reward at the first time step, so we just set it to zero.
    rew_plot[0] = 0.0

    if baseline_comp:
        powerF_b[0] = env.fs_baseline.windTurbines.power().sum()
        powerT_b[0] = env.fs_baseline.windTurbines.power()
        yaw_b[0] = env.fs_baseline.windTurbines.yaw
        ws_b[0] = np.linalg.norm(
            env.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
        )
        pct_inc[0] = (
            ((powerF_a[0] - powerF_b[0]) / powerF_b[0]) * 100
        )  # Percentage increase in power output. This should be zero (or close to zero) at the first time step.

    # If save_figs is True, initalize some parameters here.
    if save_figs:
        FOLDER = "./Temp_Figs_{}_ws{}_wd{}/".format(name, env.ws, wd)
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
        pow_max = powerF_a[0] * 1.2
        pow_min = powerF_a[0] * 0.8
        yaw_max = 5
        yaw_min = -5
        ws_max = env.ws + 2
        ws_min = 3

        # Define the x and y values for the flow field plot
        a = np.linspace(-200 + min(env.x_pos), 200 + max(env.x_pos), 200)
        b = np.linspace(-200 + min(env.y_pos), 200 + max(env.y_pos), 200)

    # Run the simulation
    for i in range(1, time):
        if hasattr(model, "model_type"):
            if model.model_type == "CleanRL":
                obs = np.expand_dims(obs, 0)
                action, _, _ = model.get_action(
                    torch.Tensor(obs).to(device), deterministic=deterministic
                )
                action = action.detach().cpu().numpy()
        else:  # I assume this will always be SB3 models.
            action = model.predict(obs, deterministic=deterministic)[0]

        obs, reward, terminated, truncated, info = env.step(action)

        # Put the values in the arrays
        powerF_a[i] = env.fs.windTurbines.power().sum()
        powerT_a[i] = env.fs.windTurbines.power()
        yaw_a[i] = env.fs.windTurbines.yaw
        ws_a[i] = np.linalg.norm(env.fs.windTurbines.rotor_avg_windspeed, axis=1)
        time_plot[i] = env.fs.time
        rew_plot[i] = reward  #

        if baseline_comp:
            powerF_b[i] = env.fs_baseline.windTurbines.power().sum()
            powerT_b[i] = env.fs_baseline.windTurbines.power()
            yaw_b[i] = env.fs_baseline.windTurbines.yaw
            ws_b[i] = np.linalg.norm(
                env.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
            )
            pct_inc[i] = (
                ((powerF_a[i] - powerF_b[i]) / powerF_b[i]) * 100
            )  # Percentage increase in power output. This should be zero (or close to zero) at the first time step.

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
            plt.pcolormesh(
                uvw.x.values,
                uvw.y.values,
                uvw[0].T,
                shading="nearest",
                vmin=3,
                vmax=env.ws + 2,
            )
            plt.colorbar().set_label("Wind speed, u [m/s]")
            WindTurbinesPW.plot_xy(
                wt,
                x_turb,
                y_turb,
                types=wt.types,
                wd=env.fs.wind_direction,
                ax=ax1,
                yaw=yaw,
                tilt=tilt,
            )

            ax1.set_title("Flow field at {} s".format(env.fs.time))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            ax2 = plt.subplot2grid(
                (3, 3),
                (0, 2),
            )
            ax3 = plt.subplot2grid(
                (3, 3),
                (1, 2),
            )
            ax4 = plt.subplot2grid(
                (3, 3),
                (2, 2),
            )

            # Plot the power in ax2
            ax2.plot(time_deq, pow_deq, color="orange")
            ax2.set_title("Farm power [W]")

            # Plot the yaws in ax3
            ax3.plot(time_deq, yaw_deq, label=np.arange(n_turb))
            ax3.set_title("Turbine yaws [deg]")
            ax3.legend(loc="upper left")

            # Plot the rotor windspeeds in ax4
            ax4.plot(time_deq, ws_deq, label=np.arange(n_turb))
            ax4.set_title("Rotor windspeeds [m/s]")
            ax4.set_xlabel("Time [s]")

            # Set the x limits for the plots
            ax2.set_xlim(time_deq[0], time_deq[-1])
            ax3.set_xlim(time_deq[0], time_deq[-1])
            ax4.set_xlim(time_deq[0], time_deq[-1])

            pow_max = max(pow_max, powerF_a[i] * 1.2)
            pow_min = min(pow_min, powerF_a[i] * 0.8)
            yaw_max = max(yaw_max, max(yaw_a[i]) * 1.2)
            # This value can be negative, so we multiply 1.2, instead of 0.8
            yaw_min = min(yaw_min, min(yaw_a[i]) * 1.2)
            ws_max = max(ws_max, max(ws_a[i]) * 1.2)
            ws_min = min(ws_min, min(ws_a[i]) * 0.8)

            # Set the y limits for the plots. If we go over/under the limits, the plot will adjust the limits.
            ax2.set_ylim(pow_min, pow_max)
            ax3.set_ylim(yaw_min, yaw_max)
            ax4.set_ylim(ws_min, ws_max)
            ax2.set_xticks([])
            ax3.set_xticks([])

            # Set the number of ticks on the x-axis to 5
            ax4.locator_params(axis="x", nbins=5)

            img_name = FOLDER + "img_{:05d}.png".format(i)

            # Add a text to the plot with the sensor values
            for scale in scaling:  # scaling can be a list with True and False. If True, we add the scaled observations to the plot. If False, we only add the unscaled observations.
                if scale is not None:
                    turb_ws = np.round(env.farm_measurements.get_ws_turb(scale), 2)
                    turb_wd = np.round(env.farm_measurements.get_wd_turb(scale), 2)
                    turb_TI = np.round(env.farm_measurements.get_TI_turb(scale), 2)
                    turb_yaw = np.round(env.farm_measurements.get_yaw_turb(scale), 2)
                    farm_ws = np.round(env.farm_measurements.get_ws_farm(scale), 2)
                    farm_wd = np.round(env.farm_measurements.get_wd_farm(scale), 2)
                    farm_TI = np.round(env.farm_measurements.get_TI(scale), 2)
                    if scale:
                        text_plot = f" Agent observations scaled: \n Turbine level wind speed: {turb_ws} \n Turbine level wind direction: {turb_wd} \n Turbine level yaw: {turb_yaw} \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} \n Farm level wind direction: {farm_wd} \n Farm level TI: {farm_TI} "
                        ax1.text(
                            1.1,
                            1.3,
                            text_plot,
                            verticalalignment="top",
                            horizontalalignment="left",
                            transform=ax1.transAxes,
                        )
                    else:
                        text_plot = f" Agent observations: \n Turbine level wind speed: {turb_ws} [m/s] \n Turbine level wind direction: {turb_wd} [deg] \n Turbine level yaw: {turb_yaw} [deg] \n Turbine level TI: {turb_TI} \n Farm level wind speed: {farm_ws} [m/s] \n Farm level wind direction: {farm_wd} [deg] \n Farm level TI: {farm_TI} "
                        ax1.text(
                            -0.1,
                            1.3,
                            text_plot,
                            verticalalignment="top",
                            horizontalalignment="left",
                            transform=ax1.transAxes,
                        )
            # So I coudnt figure out how to add some space to the left, so I added a white text, and then use that to stretch the plot. Whatever, it works
            ax1.text(
                1.95,
                0.5,
                "Hey",
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax1.transAxes,
                color="white",
            )

            plt.savefig(
                img_name,
                dpi=100,
                bbox_extra_artists=(ax1, ax2, ax3, ax4),
                bbox_inches="tight",
            )
            plt.clf()
            plt.close("all")

    env.close()

    # Reshape the arrays and put them in a xarray dataset
    powerF_a = powerF_a.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)
    powerT_a = powerT_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    yaw_a = yaw_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    ws_a = ws_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
    rew_plot = rew_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)

    if baseline_comp:
        powerF_b = powerF_b.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)
        powerT_b = powerT_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
        yaw_b = yaw_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
        ws_b = ws_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)
        pct_inc = pct_inc.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1)

    # Then create a xarray dataset with the results
    if not baseline_comp:
        ds = xr.Dataset(
            data_vars={
                # For agent:
                # Power for the farm: [time, turbine, ws, wd, TI, turbbox]
                "powerF_a": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerF_a,
                ),
                # Power pr turbine [time, ws, wd, TI, turbbox]
                "powerT_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerT_a,
                ),
                # yaw is array of: [time, turbine, ws, wd, TI, turbbox]
                "yaw_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    yaw_a,
                ),
                # Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]
                "ws_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    ws_a,
                ),
                # For environment
                # Reward
                "reward": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    rew_plot,
                ),
            },
            coords={
                "ws": np.array([ws]),
                "wd": np.array([wd]),
                "turb": np.arange(env.n_turb),
                "time": time_plot,
                "TI": np.array([ti]),
                "turbbox": [turbbox],
                "model_step": np.array([model_step]),
            },
        )

    else:
        ds = xr.Dataset(
            data_vars={
                # For agent:
                "powerF_a": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerF_a,
                ),  # Power for the farm: [time, turbine, ws, wd, TI, turbbox]
                "powerT_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerT_a,
                ),  # Power pr turbine [time, ws, wd, TI, turbbox]
                "yaw_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    yaw_a,
                ),  # yaw is array of: [time, turbine, ws, wd, TI, turbbox]
                "ws_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    ws_a,
                ),  # Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]
                # #For baseline
                "powerF_b": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerF_b,
                ),  # Power for the farm: [time, turbine, ws, wd, TI, turbbox]
                "powerT_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    powerT_b,
                ),  # Power pr turbine [time, ws, wd, TI, turbbox]
                "yaw_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    yaw_b,
                ),  # yaw is array of: [time, turbine, ws, wd, TI, turbbox]
                "ws_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    ws_b,
                ),  # Ws at each turbine: [time, turbine, ws, wd, TI, turbbox]
                # For environment
                "reward": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    rew_plot,
                ),  # Reward
                "pct_inc": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step"),
                    pct_inc,
                ),  # Percentage increase in power output
            },
            coords={
                "ws": np.array([ws]),
                "wd": np.array([wd]),
                "turb": np.arange(n_turb),
                "time": time_plot,
                "TI": np.array([ti]),
                "turbbox": [turbbox],
                "model_step": np.array([model_step]),
            },
        )

    return ds


class AgentEval:
    def __init__(self, env=None, model=None, name="NoName", t_sim=1000):
        # Initialize the evaluater with some default values.
        self.ws = 10.0
        self.ti = 0.05
        self.wd = 270
        self.yaw = 0.0
        self.turbbox = "Default"

        self.t_sim = t_sim

        self.winddirs = [270]
        self.windspeeds = [10]
        self.turbintensities = [0.05]
        self.turbboxes = ["Default"]

        self.multiple_eval = False  # Flag if multiple_eval has been called.
        self.env = env
        self.model = model
        self.name = name

    def set_conditions(
        self,
        winddirs: list = [],
        windspeeds: list = [],
        turbintensities: list = [],
        turbboxes: list = ["Default"],
    ):
        # Update the conditions for the evaluation.
        if winddirs:
            self.winddirs = winddirs
        if windspeeds:
            self.windspeeds = windspeeds
        if turbintensities:
            self.turbintensities = turbintensities
        if turbboxes:
            self.turbboxes = turbboxes

    def set_condition(self, ws=None, ti=None, wd=None, yaw=None, turbbox=None):
        # Set the conditions for the individual evaluation, and then update the env with these values.
        if ws is not None:
            self.ws = ws
        if ti is not None:
            self.ti = ti
        if wd is not None:
            self.wd = wd
        if yaw is not None:
            self.yaw = yaw
        if turbbox is not None:
            self.turbbox = turbbox

        self.set_env_vals()

    def set_env_vals(self):
        # Update the environment with the new conditions
        # First we initialize the environment with the specified conditions
        self.env.set_yaw_vals(self.yaw)  # Specified yaw vals
        # Set the wind values, used for initialization
        self.env.set_wind_vals(ws=self.ws, ti=self.ti, wd=self.wd)
        if self.turbbox != "Default":
            # NOTE you must make sure that the self.turbbox is set to a path with a turbulence box file.
            # Also it must point to a specific file, and not a folder.
            # Here we can specify a path for the turbulence box to be used.
            self.env.update_tf(self.turbbox)

    def update_env(self, env):
        # Update the environment with the new conditions
        self.env = env

    def update_model(self, model):
        # Update the model with the new conditions
        # Can be used if model=None in the inital call.
        self.model = model

    def eval_single(
        self, save_figs=False, scale_obs=None, debug=False, deterministic=False
    ):
        """
        Evaluate the agent on a single wind direction, wind speed, turbulence intensity and turbulence box.
        """

        ds = eval_single_fast(
            env=self.env,
            model=self.model,
            ws=self.ws,
            ti=self.ti,
            wd=self.wd,
            turbbox=self.turbbox,
            save_figs=save_figs,
            scale_obs=scale_obs,
            t_sim=self.t_sim,
            name=self.name,
            debug=debug,
            deterministic=deterministic,
        )

        return ds

    def eval_multiple(self, save_figs=False, scale_obs=None, debug=False):
        """
        Evaluate the agent on multiple wind directions, wind speeds, turbulence intensities and turbulence boxes.

        """

        print(
            "Running for a total of ",
            len(self.winddirs)
            * len(self.windspeeds)
            * len(self.turbintensities)
            * len(self.turbboxes),
            " simulations.",
        )
        # Flag that we are running multiple evaluations.
        self.multiple_eval = True

        # TODO this should be parallelized.
        ds_list = []
        for winddir in self.winddirs:
            for windspeed in self.windspeeds:
                for TI in self.turbintensities:
                    for box in self.turbboxes:
                        # For all these in the loop...
                        # Set the conditions
                        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
                        # Run the simulation
                        ds = self.eval_single(
                            save_figs=save_figs, scale_obs=scale_obs, debug=debug
                        )
                        # Save the results
                        ds_list.append(ds)
        ds_total = xr.merge(ds_list)
        self.multiple_eval_ds = ds_total
        return self.multiple_eval_ds
        # Keep this for later, as I will work on it at some point

    def run_simulation(self, winddir, windspeed, TI, box, save_figs, scale_obs, debug):
        """
        Run a singel simulation.
        This function might be used for the parallelization of the simulation.
        """
        # Run a singe simulation with the specified conditions.
        # Set the conditions
        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
        # Run the simulation
        ds = self.eval_single(save_figs=save_figs, scale_obs=scale_obs, debug=debug)
        return ds

    def plot_initial(self):
        """
        Plot the initial conditions of the simulation, alongside the turbines with their numbering.
        """

        _, __ = self.env.reset()

        # Define the x, y and z for the plot
        x_mean = self.env.fs.windTurbines.position[0].mean()
        y_mean = self.env.fs.windTurbines.position[1].mean()
        x_range = (
            self.env.fs.windTurbines.position[0].max()
            - self.env.fs.windTurbines.position[0].min()
        )
        y_range = (
            self.env.fs.windTurbines.position[1].max()
            - self.env.fs.windTurbines.position[1].min()
        )
        h = self.env.fs.windTurbines.hub_height()[0]

        ax1, ax2 = plt.subplots(1, 2, figsize=(10, 4))[1]

        # plot in one way
        self.env.fs.show(
            view=XYView(
                x=np.linspace(x_mean - x_range, x_mean + x_range),
                y=np.linspace(y_mean - y_range, y_mean + y_range),
                z=h,
                ax=ax1,
            ),
            flowVisualizer=Flow2DVisualizer(color_bar=False),
            show=False,
        )
        # plot in another way
        self.env.fs.show(
            view=EastNorthView(
                east=np.linspace(x_mean - x_range, x_mean + x_range),
                north=np.linspace(y_mean - y_range, y_mean + y_range),
                z=h,
                ax=ax2,
            ),
            flowVisualizer=Flow2DVisualizer(color_bar=False),
            show=False,
        )
        setup_plot(
            ax=ax1,
            title=f"Rotated view, {self.env.wd} deg",
            xlabel="x [m]",
            ylabel="y [m]",
            grid=False,
        )
        setup_plot(
            ax=ax2,
            title=f"Alligned view, {self.env.wd} deg",
            xlabel="east [m]",
            ylabel="north [m]",
            grid=False,
        )

    def plot_performance(self):
        """
        Plot the performance of the agent, and the baseline farm.
        We could plot the power output, the wind speed, the wind direction, the yaw angles, the turbulence intensity, the wake losses, etc.
        The return is a plot of the performance metrics.
        """
        print("Not implemented yet")

    def save_performance(self):
        """
        Save the performance metrics to a file.
        TODO: Maybe add the options for a specific path to save the file to.
        """
        if self.multiple_eval:
            self.multiple_eval_ds.to_netcdf(self.name + "_eval.nc")
        else:
            print("It doenst look like you have any data to save my guy")

    def load_performance(self, path):
        """
        Load the performance metrics from a file.
        Can be used to see the results from a previous evaluation.
        """
        self.multiple_eval_ds = xr.open_dataset(path)
        self.multiple_eval = True

    def plot_power_farm(
        self, WSS, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the power output for the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        if axs is None:
            fig, axs = plt.subplots(
                len(WSS),
                len(WDS),
                figsize=(4 * int(len(WDS)), 3 * int(len(WSS))),
                sharey=True,
            )
        else:
            fig = axs[0, 0].get_figure()

        for j, WS in enumerate(WSS):
            for i, wd in enumerate(WDS):
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(
                    turbbox=TURBBOX
                ).powerF_a.rolling(time=avg_n, center=True).mean().dropna(
                    "time"
                ).plot.line(x="time", label="Agent", ax=axs[j, i])
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(
                    turbbox=TURBBOX
                ).powerF_b.rolling(time=avg_n, center=True).mean().plot.line(
                    x="time", label="Baseline", ax=axs[j, i]
                )

                if j == 0:  # Only set the top row to have a title
                    axs[j, i].set_title(f"WD ={wd} [deg]")
                else:
                    axs[j, i].set_title("")
                if i == 0:  # Only set the left column to have a y-label
                    axs[j, i].set_ylabel(f"WS ={WS} [m/s]")
                else:
                    axs[j, i].set_ylabel("")

                axs[j, i].set_xlabel("")
                axs[j, i].grid()
                x_start = (
                    data.sel(ws=WS)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .powerF_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=WS)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .powerF_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )
                axs[j, i].set_xlim(x_start, x_end)
        axs[0, 1].legend()
        fig.suptitle(
            f"Power output for agent and baseline, WS = {WSS}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        fig.supylabel("Power [W]", fontsize=15, fontweight="bold")
        fig.supxlabel("Time [s]", fontsize=15, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs

    def plot_farm_inc(
        self, WSS, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the percentage increase in power output for the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        if axs is None:
            fig, axs = plt.subplots(
                len(WSS),
                len(WDS),
                figsize=(4 * int(len(WDS)), 3 * int(len(WSS))),
                sharey=True,
            )
        else:
            fig = axs[0, 0].get_figure()

        for j, WS in enumerate(WSS):
            for i, wd in enumerate(WDS):
                data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(
                    turbbox=TURBBOX
                ).pct_inc.rolling(time=avg_n, center=True).mean().dropna(
                    "time"
                ).plot.line(x="time", ax=axs[j, i])
                if j == 0:  # Only set the top row to have a title
                    axs[j, i].set_title(f"WD ={wd} [deg]")
                else:
                    axs[j, i].set_title("")
                if i == 0:  # Only set the left column to have a y-label
                    axs[j, i].set_ylabel(f"WS ={WS} [m/s]")
                else:
                    axs[j, i].set_ylabel("")

                axs[j, i].set_xlabel("")
                axs[j, i].grid()
                x_start = (
                    data.sel(ws=WS)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .pct_inc.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=WS)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .pct_inc.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )
                axs[j, i].set_xlim(x_start, x_end)

        fig.suptitle(
            f"Power increase, WS = {WSS}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        fig.supylabel("Power increase [%]", fontsize=15, fontweight="bold")
        fig.supxlabel("Time [s]", fontsize=15, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs

    def plot_power_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the power output for each turbine in the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        n_turb = len(data.turb.values)  # The number of turbines in the farm
        n_wds = len(WDS)  # The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(
                n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True
            )
        else:
            fig = axs[0, 0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).powerT_a.rolling(time=avg_n, center=True).mean().dropna(
                    "time"
                ).plot.line(x="time", label="Agent", ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).powerT_b.rolling(time=avg_n, center=True).mean().plot.line(
                    x="time", label="Baseline", ax=axs[i, j]
                )
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")

                x_start = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .powerT_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .powerT_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")

        fig.supylabel("Power [W]", fontsize=15, fontweight="bold")
        fig.supxlabel("Time [s]", fontsize=15, fontweight="bold")
        fig.suptitle(
            f"Power output pr turbine for agent and baseline, ws = {ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        axs[0, 0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_power_farm.png")
        return fig, axs

    def plot_yaw_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the yaw angle for each turbine in the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        n_turb = len(data.turb.values)  # The number of turbines in the farm
        n_wds = len(WDS)  # The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(
                n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True
            )
        else:
            fig = axs[0, 0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).yaw_a.rolling(time=avg_n, center=True).mean().dropna(
                    "time"
                ).plot.line(x="time", label="Agent", ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).yaw_b.rolling(time=avg_n, center=True).mean().plot.line(
                    x="time", label="Baseline", ax=axs[i, j]
                )
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")

                x_start = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .yaw_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .yaw_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")

        fig.supylabel("Yaw offset [deg]", fontsize=15, fontweight="bold")
        fig.supxlabel("Time [s]", fontsize=15, fontweight="bold")
        fig.suptitle(
            f"Yaw angle pr turbine for agent and baseline, ws = {ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        axs[0, 0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_yaw_farm.png")
        return fig, axs

    def plot_speed_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the rotor wind speed for each turbine in the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        n_turb = len(data.turb.values)  # The number of turbines in the farm
        n_wds = len(WDS)  # The number of wind directions we are looking at

        if axs is None:
            fig, axs = plt.subplots(
                n_turb, n_wds, figsize=(18, 9), sharex=True, sharey=True
            )
        else:
            fig = axs[0, 0].get_figure()

        for i in range(n_turb):
            for j, wd in enumerate(WDS):
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).ws_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(
                    x="time", label="Agent", ax=axs[i, j]
                )
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).ws_b.rolling(time=avg_n, center=True).mean().plot.line(
                    x="time", label="Baseline", ax=axs[i, j]
                )
                axs[i, j].set_title(f"WD ={wd}, Turbine {i}")

                x_start = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .ws_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=0)
                    .ws_a.rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )

                axs[i, j].grid()
                axs[i, j].set_xlim(x_start, x_end)
                axs[i, j].set_ylabel(" ")
                axs[i, j].set_xlabel(" ")

        fig.supylabel("Wind speed [m/s]", fontsize=15, fontweight="bold")
        fig.supxlabel("Time [s]", fontsize=15, fontweight="bold")
        fig.suptitle(
            f"Rotor wind speed, ws={ws}, WD = {WDS}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        axs[0, 0].legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.name + "_yaw_farm.png")
        return fig, axs

    def plot_turb(
        self, ws, wd, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):
        """
        Plot the power, yaw and rotor wind speed for each turbine in the farm.
        """
        data = self.multiple_eval_ds  # Just for easier writing
        n_turb = len(data.turb.values)  # The number of turbines in the farm
        # n_wds = len(WDS)  #The number of wind directions we are looking at

        plot_x = ["Power", "Yaw", "Rotor wind speed"]

        if axs is None:
            fig, axs = plt.subplots(n_turb, len(plot_x), figsize=(18, 9), sharex=True)
        else:
            fig = axs[0, 0].get_figure()

        for i in range(n_turb):
            # Bookkeeping for the different variables
            for j, plot_var in enumerate(plot_x):
                if plot_var == "Power":
                    to_plot = "powerT_"
                    plot_title = "Turbine power"
                    y_label = "Power [W]"
                elif plot_var == "Yaw":
                    to_plot = "yaw_"
                    plot_title = "Yaw offset [deg]"
                    y_label = "Yaw offset [deg]"
                elif plot_var == "Rotor wind speed":
                    to_plot = "ws_"
                    plot_title = "Rotor wind speed [m/s]"

                # Set the y axis to be shared between the different plots
                axs[i, j].sharey(axs[0, j])

                # Plot the data
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).data_vars[to_plot + "a"].rolling(
                    time=avg_n, center=True
                ).mean().dropna("time").plot.line(x="time", label="Agent", ax=axs[i, j])
                data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                    turb=i
                ).data_vars[to_plot + "b"].rolling(
                    time=avg_n, center=True
                ).mean().dropna("time").plot.line(
                    x="time", label="Baseline", ax=axs[i, j]
                )

                # Set the title of the plot
                if i == 0:
                    axs[i, j].set_title(plot_title)
                else:
                    axs[i, j].set_title("")

                # Find at set the x-axis limits
                x_start = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=i)
                    .data_vars[to_plot + "a"]
                    .rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.min()
                )
                x_end = (
                    data.sel(ws=ws)
                    .sel(wd=wd)
                    .sel(TI=TI)
                    .sel(turbbox=TURBBOX)
                    .sel(turb=i)
                    .data_vars[to_plot + "a"]
                    .rolling(time=avg_n, center=True)
                    .mean()
                    .dropna("time")
                    .time.values.max()
                )
                axs[i, j].set_xlim(x_start, x_end)

                # Set the y and x labels
                if j == 0:
                    axs[i, j].set_ylabel(f"Turbine {i}")
                else:
                    axs[i, j].set_ylabel(" ")
                if i == n_turb - 1:
                    axs[i, j].set_xlabel("Time [s]")
                else:
                    axs[i, j].set_xlabel(" ")

        fig.suptitle(
            f"Turbine power, yaw and rotor windspeed, ws={ws}, WD = {wd}, TI = {TI}, TurbBox = {TURBBOX}",
            fontsize=15,
            fontweight="bold",
        )
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(self.name + "_turbine_metrics.png")
        return fig, axs
