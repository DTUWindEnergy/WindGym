import warnings

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

import torch

from wetb.gtsdf import (
    gtsdf,
)  # re-exported: patch target `WindGym.agent_eval.gtsdf.load`

from dynamiks.views import XYView
from py_wake.utils.plotting import setup_plot

# Import visualization functions from new modules
from .visualization import (
    plot_power_farm,
    plot_farm_inc,
    plot_power_turb,
    plot_yaw_turb,
    plot_speed_turb,
    plot_turb,
)
from .evaluation.recorder import EpisodeRecorder
from .evaluation.live_figures import (
    LiveFigureWriter,
    assemble_gif,
    resolve_fig_folder,
)
from .evaluation.hawc2_loads import load_hawc2_loads_dataset, close_hawc2_and_cleanup
from .evaluation.selection import select_action

"""
AgentEval is a class that is used to evaluate an agent on the EnvEval environment.
The class is made to evaluate the agent for multiple wind directions, and then save a xarray dataset with the results.

The per-episode pieces (result arrays, live figures, HAWC2 loads, model
action selection) live in ``WindGym.evaluation``.

TODO: We could add in a check that the agent has already been evaluated on a given condition. if yes, then we dont need to simulate it again.
TODO: Add a function to animate the results.
TODO: parallelize the evaluation in eval_multiple()
"""

# Kept under its historical private name for downstream code.
_select_action = select_action


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
    return_loads=False,
    cleanup=True,
    seed=None,
    fig_dir=None,
    *,
    veer=0.0,
    gif_fps=8,
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
    return_loads: If True and the env runs HAWC2 turbines (HTC_path set),
        return the HAWC2 load-channel dataset instead of the power dataset.
        With return_loads=True on a non-HAWC2 env nothing is returned (None).
    seed: Seed passed to env.reset() for reproducible episodes. None keeps
        the previous (unseeded) behaviour.
    fig_dir: Directory for figures when save_figs=True. Defaults to
        "./Temp_Figs_{name}_ws{ws}_wd{wd}/" in the current working directory
        (the historical behaviour).
    veer: Linear veer rate (deg per 100 m, positive = wd increases with
        height, 0 at hub height), pinned for the episode (keyword-only).
        None keeps the config's veer_min/veer_max draw.
    gif_fps: Frames per second of the animated GIF assembled from the saved
        figures (only used when save_figs is True; keyword-only). The GIF
        path is stored in ``ds.attrs["gif_path"]`` (on the loads dataset
        with return_loads).

    """

    device = torch.device("cpu")

    if hasattr(env.unwrapped, "parent_pipes"):
        raise AssertionError(
            "The eval_single_fast function is not compatible with vectorized versions of the environment. Please use unvectorized envs instead."
        )

    env.set_wind_vals(ws=ws, ti=ti, wd=wd, veer=veer)
    baseline_comp = env.Baseline_comp

    scaling = scale_obs if isinstance(scale_obs, list) else [scale_obs]
    if debug:  # If debug, do both.
        scaling = [True, False]
        save_figs = True

    if model is None:
        raise ValueError("You need to specify a model to evaluate the agent.")

    # Calculate the correct number of steps
    step_val = (
        env.sim_steps_per_env_step
    )  # This is the number of steps per environment step
    total_steps = (
        t_sim // env.dt_env + 1
    )  # This is the total number of steps to simulate
    time = total_steps * step_val + 1

    n_turb = env.n_turb  # Number of turbines

    # Steady-state operating point (blade pitch / rotor RPM) is available when
    # the env carries an OperatingPointLookup; the derate control signal is
    # available on any derating env. Both are off for yaw-only envs.
    op_mode = getattr(env, "op_lookup", None) is not None
    log_derate = bool(getattr(env, "derate_action", False))
    tracking = bool(getattr(env, "Track_power", False))

    rec = EpisodeRecorder(
        time,
        n_turb,
        baseline=baseline_comp,
        op_mode=op_mode,
        log_derate=log_derate,
        tracking=tracking,
    )

    # Initialize the environment
    obs, info = env.reset(seed=seed)

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
    rec.record_initial(env)

    # If save_figs is True, initalize the live figure writer here.
    writer = None
    if save_figs:
        writer = LiveFigureWriter(
            env,
            rec,
            folder=resolve_fig_folder(fig_dir, name, env.ws, wd),
            scaling=scaling,
            n_turb=n_turb,
            tracking=tracking,
            op_mode=op_mode,
        )

    # Run the simulation
    for i in range(0, total_steps):
        action = select_action(model, obs, deterministic=deterministic, device=device)

        obs, reward, terminated, truncated, info = env.step(action)

        # The eval loop assumes the env runs as an untruncated "sandbox": it
        # never resets mid-run. A truncation here means the requested t_sim
        # exceeded the env's horizon (time_max) -- and truncation triggers
        # _cleanup_resources(), which frees the flow simulation, so every
        # subsequent step would read freed/garbage state. Fail loudly instead
        # of silently returning corrupt results.
        if truncated:
            raise RuntimeError(
                f"Environment truncated during evaluation at step {i + 1} of "
                f"{total_steps} (env.time_max={env.time_max}s, delay={env.delay}s, "
                f"t_sim={t_sim}s). The eval loop cannot continue past truncation "
                "because the flow simulation is cleaned up on the time limit. "
                "Reduce t_sim or raise the env's time_max/max_time_steps so the "
                "full evaluation fits within one episode."
            )

        # Put the values in the arrays
        rec.record_step(i, step_val, info, reward)

        if writer is not None:
            writer.draw(i)

    # Assemble the saved frames into an animated GIF. A missing encoder must
    # not cost the finished evaluation: warn and keep the frames.
    gif_path = None
    if writer is not None:
        try:
            gif_path = assemble_gif(writer.FOLDER, total_steps, gif_fps)
        except ImportError as e:
            warnings.warn(f"No GIF written ({e}); frames kept in {writer.FOLDER}")

    # Create the dataset
    if not return_loads:
        ds = rec.to_dataset(
            ws=ws,
            wd=wd,
            ti=ti,
            turbbox=turbbox,
            model_step=model_step,
            deterministic=deterministic,
        )
        if gif_path is not None:
            ds.attrs["gif_path"] = gif_path
        # Do this to remove it from memory
        env.timestep = env.time_max
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        return ds
    # Do this if we have the HTC and want the loads as well.
    elif env.HTC_path is not None:
        # If the HTC_path is not None, then ill assume we also want to include the loads
        ds_load = load_hawc2_loads_dataset(
            env, ws=ws, wd=wd, ti=ti, turbbox=turbbox, model_step=model_step
        )
        if gif_path is not None:
            ds_load.attrs["gif_path"] = gif_path
        close_hawc2_and_cleanup(env, cleanup=cleanup, baseline_comp=baseline_comp)
        return ds_load
    # return_loads=True without HAWC2 turbines: nothing to return (historical
    # fall-through, kept explicit).
    return None


class AgentEval:
    def __init__(self, env=None, model=None, name="NoName", t_sim=1000, seed=None):
        # Initialize the evaluater with some default values.
        self.ws = 10.0
        self.ti = 0.05
        self.wd = 270
        self.veer = 0.0  # deg per 100 m, 0 at hub height
        self.yaw = 0.0
        self.turbbox = "Default"

        self.t_sim = t_sim
        # Master seed. When set, eval_single/eval_multiple derive per-episode
        # seeds from it for reproducible evaluations; None = unseeded.
        self.seed = seed

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

    def set_condition(
        self, ws=None, ti=None, wd=None, yaw=None, turbbox=None, veer=None
    ):
        # Set the conditions for the individual evaluation, and then update the env with these values.
        if ws is not None:
            self.ws = ws
        if ti is not None:
            self.ti = ti
        if wd is not None:
            self.wd = wd
        if veer is not None:
            self.veer = veer
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
        self.env.set_wind_vals(ws=self.ws, ti=self.ti, wd=self.wd, veer=self.veer)
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
        self,
        save_figs=False,
        scale_obs=None,
        debug=False,
        deterministic=False,
        return_loads=False,
        seed=None,
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
            veer=self.veer,
            turbbox=self.turbbox,
            save_figs=save_figs,
            scale_obs=scale_obs,
            t_sim=self.t_sim,
            name=self.name,
            debug=debug,
            deterministic=deterministic,
            return_loads=return_loads,
            seed=seed if seed is not None else self.seed,
        )

        self.env.close()  # Close the environment to make sure that we dont have any issues with the turbulence box being in memory.
        return ds

    def eval_multiple(
        self, save_figs=False, scale_obs=None, debug=False, return_loads=False
    ):
        """
        Evaluate the agent on multiple wind directions, wind speeds, turbulence intensities and turbulence boxes.

        """
        i = (
            len(self.winddirs)
            * len(self.windspeeds)
            * len(self.turbintensities)
            * len(self.turbboxes)
        )
        print(
            "Running for a total of ",
            i,
            " simulations.",
        )
        # Flag that we are running multiple evaluations.
        self.multiple_eval = True

        # Derive one seed per episode from the master seed (if set), the same
        # way utils/evaluate_PPO.py does, so multi-condition runs reproduce.
        rng = np.random.default_rng(self.seed) if self.seed is not None else None

        # TODO this should be parallelized.
        ds_list = []
        for winddir in self.winddirs:
            for windspeed in self.windspeeds:
                for TI in self.turbintensities:
                    for box in self.turbboxes:
                        # For all these in the loop... (run_simulation sets
                        # the conditions on the env)
                        episode_seed = (
                            int(rng.integers(2**31)) if rng is not None else None
                        )
                        # Run the simulation
                        ds = self.run_simulation(
                            winddir,
                            windspeed,
                            TI,
                            box,
                            save_figs,
                            scale_obs,
                            debug,
                            return_loads=return_loads,
                            seed=episode_seed,
                        )
                        ds_list.append(ds)
                        i -= 1
                        print("Done with simulation. Missing sims: ", i)
        ds_total = xr.merge(ds_list)
        self.multiple_eval_ds = ds_total
        return self.multiple_eval_ds
        # Keep this for later, as I will work on it at some point

    def run_simulation(
        self,
        winddir,
        windspeed,
        TI,
        box,
        save_figs,
        scale_obs,
        debug,
        *,
        return_loads=False,
        seed=None,
    ):
        """
        Run a singel simulation.
        This function might be used for the parallelization of the simulation.
        """
        # Run a singe simulation with the specified conditions.
        # Set the conditions
        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
        # Run the simulation
        ds = self.eval_single(
            save_figs=save_figs,
            scale_obs=scale_obs,
            debug=debug,
            return_loads=return_loads,
            seed=seed,
        )
        return ds

    def plot_initial(self):
        """
        Plot the initial conditions of the simulation, alongside the turbines with their numbering.
        """

        _, __ = self.env.reset()

        # Define the x, y and z for the plot
        x_mean = self.env.fs.windTurbines.positions_xyz[0].mean()
        y_mean = self.env.fs.windTurbines.positions_xyz[1].mean()
        x_range = (
            self.env.fs.windTurbines.positions_xyz[0].max()
            - self.env.fs.windTurbines.positions_xyz[0].min()
        )
        y_range = (
            self.env.fs.windTurbines.positions_xyz[1].max()
            - self.env.fs.windTurbines.positions_xyz[1].min()
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
            # flowVisualizer=Flow2DVisualizer(color_bar=False),
            # show=False,
        )
        # plot in another way
        # self.env.fs.show(
        #    view=EastNorthView(
        #        east=np.linspace(x_mean - x_range, x_mean + x_range),
        #        north=np.linspace(y_mean - y_range, y_mean + y_range),
        #        z=h,
        #        ax=ax2,
        #    ),
        #    flowVisualizer=Flow2DVisualizer(color_bar=False),
        #    show=False,
        # )
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

    def plot_performance(self):  # pragma: no cover
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
    ):  # pragma: no cover
        """
        Plot the power output for the farm.
        """
        save_path = self.name + "_power_farm.png" if save else None
        return plot_power_farm(
            self.multiple_eval_ds, WSS, WDS, avg_n, TI, TURBBOX, axs, save, save_path
        )

    def plot_farm_inc(
        self, WSS, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):  # pragma: no cover
        """
        Plot the percentage increase in power output for the farm.
        """
        save_path = self.name + "_power_farm_inc.png" if save else None
        return plot_farm_inc(
            self.multiple_eval_ds, WSS, WDS, avg_n, TI, TURBBOX, axs, save, save_path
        )

    def plot_power_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):  # pragma: no cover
        """
        Plot the power output for each turbine in the farm.
        """
        save_path = self.name + "_power_turb.png" if save else None
        return plot_power_turb(
            self.multiple_eval_ds, ws, WDS, avg_n, TI, TURBBOX, axs, save, save_path
        )

    def plot_yaw_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):  # pragma: no cover
        """
        Plot the yaw angle for each turbine in the farm.
        """
        save_path = self.name + "_yaw_turb.png" if save else None
        return plot_yaw_turb(
            self.multiple_eval_ds, ws, WDS, avg_n, TI, TURBBOX, axs, save, save_path
        )

    def plot_speed_turb(
        self, ws, WDS, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):  # pragma: no cover
        """
        Plot the rotor wind speed for each turbine in the farm.
        """
        save_path = self.name + "_speed_turb.png" if save else None
        return plot_speed_turb(
            self.multiple_eval_ds, ws, WDS, avg_n, TI, TURBBOX, axs, save, save_path
        )

    def plot_turb(
        self, ws, wd, avg_n=10, TI=0.07, TURBBOX="Default", axs=None, save=False
    ):  # pragma: no cover
        """
        Plot the power, yaw and rotor wind speed for each turbine in the farm.
        """
        save_path = self.name + "_turbine_metrics.png" if save else None
        return plot_turb(
            self.multiple_eval_ds, ws, wd, avg_n, TI, TURBBOX, axs, save, save_path
        )
