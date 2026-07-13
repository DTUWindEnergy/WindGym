import xarray as xr
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse

from collections import deque
from py_wake.wind_turbines import WindTurbines as WindTurbinesPW

import torch
import torch.nn as nn
import torch.nn.functional as F

from wetb.gtsdf import gtsdf
from wetb.fatigue_tools.fatigue import eq_load

from dynamiks.views import XYView, EastNorthView
from dynamiks.visualizers.flow_visualizers import Flow2DVisualizer
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

"""
AgentEval is a class that is used to evaluate an agent on the EnvEval environment.
The class is made to evaluate the agent for multiple wind directions, and then save a xarray dataset with the results.

TODO: We could add in a check that the agent has already been evaluated on a given condition. if yes, then we dont need to simulate it again.
TODO: Add a function to animate the results.
TODO: parallelize the evaluation in eval_multiple()
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
    return_loads=False,
    cleanup=True,
    seed=None,
    fig_dir=None,
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
    seed: Seed passed to env.reset() for reproducible episodes. None keeps
        the previous (unseeded) behaviour.
    fig_dir: Directory for figures when save_figs=True. Defaults to
        "./Temp_Figs_{name}_ws{ws}_wd{wd}/" in the current working directory
        (the historical behaviour).

    """

    device = torch.device("cpu")

    if hasattr(env.unwrapped, "parent_pipes"):
        raise AssertionError(
            "The eval_single_fast function is not compatible with vectorized versions of the environment. Please use unvectorized envs instead."
        )

    env.set_wind_vals(ws=ws, ti=ti, wd=wd)
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
    n_ws = 1  # Number of wind speeds to simulate
    n_wd = 1  # Number of wind direction simulate
    n_turbbox = 1  # Number of turbulence boxes to simulate
    n_TI = 1  # Number of turbulence intensities to simulate

    # Initialize the arrays to store the results
    # _a is the agent and _b is the baseline
    powerF_a = np.zeros((time), dtype=np.float32)
    powerT_a = np.zeros((time, n_turb), dtype=np.float32)
    yaw_a = np.zeros((time, n_turb), dtype=np.float32)
    ws_a = np.zeros((time, n_turb), dtype=np.float32)
    time_plot = np.zeros((time), dtype=int)
    rew_plot = np.zeros((time), dtype=np.float32)

    # Steady-state operating point (blade pitch / rotor RPM) is available when
    # the env carries an OperatingPointLookup; the derate control signal is
    # available on any derating env. Both are off for yaw-only envs.
    op_mode = getattr(env, "op_lookup", None) is not None
    log_derate = bool(getattr(env, "derate_action", False))
    if op_mode:
        pitch_a = np.zeros((time, n_turb), dtype=np.float32)
        rpm_a = np.zeros((time, n_turb), dtype=np.float32)
    if log_derate:
        derate_a = np.zeros((time, n_turb), dtype=np.float32)

    tracking = bool(getattr(env, "Track_power", False))
    if tracking:
        p_ref = np.zeros((time), dtype=np.float32)
        track_err = np.zeros((time), dtype=np.float32)

    if baseline_comp:
        powerF_b = np.zeros((time), dtype=np.float32)
        powerT_b = np.zeros((time, n_turb), dtype=np.float32)
        yaw_b = np.zeros((time, n_turb), dtype=np.float32)
        ws_b = np.zeros((time, n_turb), dtype=np.float32)
        pct_inc = np.zeros((time), dtype=np.float32)

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
    powerF_a[0] = env.fs.windTurbines.power().sum()
    powerT_a[0] = env.fs.windTurbines.power()
    yaw_a[0] = env.fs.windTurbines.yaw
    ws_a[0] = np.linalg.norm(env.fs.windTurbines.rotor_avg_windspeed, axis=1)
    time_plot[0] = env.fs.time
    # There is no reward at the first time step, so we just set it to zero.
    rew_plot[0] = 0.0

    # reset()'s warm-up already ran _take_measurements, so these exist here.
    if op_mode:
        pitch_a[0] = env.current_pitch
        rpm_a[0] = env.current_rpm
    if log_derate:
        derate_a[0] = env.current_derate

    if tracking:
        p_ref[0] = env.power_setpoint
        track_err[0] = powerF_a[0] - p_ref[0]

    if baseline_comp:
        powerF_b[0] = env.fs_baseline.windTurbines.power().sum()
        powerT_b[0] = env.fs_baseline.windTurbines.power()
        yaw_b[0] = env.fs_baseline.windTurbines.yaw
        ws_b[0] = np.linalg.norm(
            env.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
        )
        # Percentage increase in power output. This should be zero (or close
        # to zero) at the first time step. Baseline power can be 0 (e.g.
        # below cut-in), so guard the division.
        pct_inc[0] = (
            ((powerF_a[0] - powerF_b[0]) / powerF_b[0]) * 100
            if powerF_b[0] != 0
            else 0.0
        )

    # If save_figs is True, initalize some parameters here.
    if save_figs:
        if fig_dir is not None:
            FOLDER = os.path.join(fig_dir, "")
        else:
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

        # Derating / tracking panels: for a derate-only agent the yaw-trainer's
        # right column (yaw + local wind speed) is meaningless (yaw is fixed).
        # Auto-detect the mode from the env and swap in derating + per-turbine
        # power; a yaw run (derate_mode=False) keeps every original panel.
        derate_mode = (
            bool(getattr(env, "derate_action", False))
            and getattr(env, "current_derate", None) is not None
        )
        # A yaw+derate agent steers AND derates: keep the derate layout but add
        # a yaw time-series panel in the spare bottom-right cell. Derate-only
        # envs (yaw_action=False) leave that cell blank ("yaw fixed").
        yaw_active = bool(getattr(env, "yaw_action", True))
        show_yaw_panel = derate_mode and yaw_active
        if derate_mode:
            derate_deq = deque(maxlen=max_deque)
            powerT_deq = deque(maxlen=max_deque)
            derate_deq.append(np.asarray(env.current_derate).copy())
            powerT_deq.append(powerT_a[0].copy())
            powT_max = powerT_a[0].max() * 1.2
        # Two extra right-column panels (blade pitch, rotor RPM) when the env
        # can report its steady-state operating point. Yaw-only runs keep the
        # original 3-row layout untouched.
        op_panels = derate_mode and op_mode
        if op_panels:
            pitch_deq = deque(maxlen=max_deque)
            rpm_deq = deque(maxlen=max_deque)
            pitch_deq.append(pitch_a[0].copy())
            rpm_deq.append(rpm_a[0].copy())
            # Default panel ranges; grown on the fly (like pow_max) whenever
            # the data leaves them. Exception: the pitch axis is HARD-capped
            # at 15 deg — the table's feathered/parked points (pitch ~90) at
            # deep derate + low waked ws would otherwise flatten the panel.
            pitch_lo = min(0.0, float(pitch_a[0].min()) - 0.5)
            pitch_hi = 15.0
            rpm_lo = min(5.0, float(rpm_a[0].min()) - 0.2)
            rpm_hi = max(8.0, float(rpm_a[0].max()) + 0.2)
        if tracking:
            pref_deq = deque(maxlen=max_deque)
            pref_deq.append(p_ref[0])

        # Flow-field extent. x spans the row + margins; y is padded +-2D so a
        # single row sits in a ~3.7:1 rectangle that reads at true (equal) aspect
        # (see below) instead of the old ~6x-stretched square. For multi-row
        # farms this just pads 2D beyond the y-extent, so nothing is clipped.
        D_view = float(np.atleast_1d(env.fs.windTurbines.diameter())[0])
        a = np.linspace(-200 + min(env.x_pos), 300 + max(env.x_pos), 200)
        b = np.linspace(min(env.y_pos) - 2 * D_view, max(env.y_pos) + 2 * D_view, 200)

    # Run the simulation
    for i in range(0, total_steps):
        if hasattr(model, "model_type"):
            if model.model_type == "CleanRL":
                obs = np.expand_dims(obs, 0)
                action, _, _ = model.get_action(
                    torch.Tensor(obs).to(device), deterministic=deterministic
                )
                action = action.detach().cpu().numpy()
                action = action.flatten()
        else:  # This is for the other models (Pywake and such)
            action = model.predict(obs, deterministic=deterministic)[0]

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
        powerF_a[i * step_val + 1 : i * step_val + step_val + 1] = info["powers"].sum(
            axis=1
        )
        powerT_a[i * step_val + 1 : i * step_val + step_val + 1] = info["powers"]
        yaw_a[i * step_val + 1 : i * step_val + step_val + 1] = info["yaws"]
        ws_a[i * step_val + 1 : i * step_val + step_val + 1] = info["windspeeds"]
        time_plot[i * step_val + 1 : i * step_val + step_val + 1] = info["time_array"]
        rew_plot[i * step_val + 1 : i * step_val + step_val + 1] = reward

        if op_mode:
            pitch_a[i * step_val + 1 : i * step_val + step_val + 1] = info["pitches"]
            rpm_a[i * step_val + 1 : i * step_val + step_val + 1] = info["rpms"]
        if log_derate:
            derate_a[i * step_val + 1 : i * step_val + step_val + 1] = info["derates"]

        if tracking:
            # The reference is per env step; the error is at sim resolution.
            p_ref[i * step_val + 1 : i * step_val + step_val + 1] = info[
                "Power reference"
            ]
            track_err[i * step_val + 1 : i * step_val + step_val + 1] = (
                info["powers"].sum(axis=1) - info["Power reference"]
            )

        if baseline_comp:
            powerF_b[i * step_val + 1 : i * step_val + step_val + 1] = info[
                "baseline_powers"
            ].sum(axis=1)
            powerT_b[i * step_val + 1 : i * step_val + step_val + 1] = info[
                "baseline_powers"
            ]
            yaw_b[i * step_val + 1 : i * step_val + step_val + 1] = info[
                "yaws_baseline"
            ]
            ws_b[i * step_val + 1 : i * step_val + step_val + 1] = info[
                "windspeeds_baseline"
            ]

            # Percentage increase in power output. Guard against zero
            # baseline power (e.g. below cut-in) -> report 0 instead of inf.
            agent_farm_power = info["powers"].sum(axis=1)
            base_farm_power = info["baseline_powers"].sum(axis=1)
            pct_inc[i * step_val + 1 : i * step_val + step_val + 1] = (
                np.divide(
                    agent_farm_power - base_farm_power,
                    base_farm_power,
                    out=np.zeros_like(base_farm_power),
                    where=base_farm_power != 0,
                )
                * 100
            )

        if save_figs:
            # The result arrays are at sim resolution while i counts env
            # steps; index the end-of-step sample, not raw i.
            end_idx = i * step_val + step_val
            time_deq.append(time_plot[end_idx])
            pow_deq.append(powerF_a[end_idx])
            yaw_deq.append(yaw_a[end_idx])
            ws_deq.append(ws_a[end_idx])
            if derate_mode:
                derate_deq.append(np.asarray(env.current_derate).copy())
                powerT_deq.append(powerT_a[end_idx].copy())
            if op_panels:
                pitch_deq.append(pitch_a[end_idx].copy())
                rpm_deq.append(rpm_a[end_idx].copy())
            if tracking:
                pref_deq.append(p_ref[end_idx])

            # Wide layout: the right-hand block is a 3x2 grid on a (3, 4)
            # figure grid — left sub-column keeps the original stack, right
            # sub-column adds pitch/RPM, and the spare bottom-right cell hosts
            # the yaw panel for yaw+derate agents (blank for derate-only; the
            # shared legend moved to a figure-level strip below the grid).
            # Otherwise the original (3, 3) layout.
            wide = op_panels or show_yaw_panel
            grid = (3, 4) if wide else (3, 3)
            fig = plt.figure(figsize=(15, 7.5) if wide else (12, 7.5))
            ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=3)

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
            plt.colorbar().set_label("Wind speed [m/s]")

            # This is code taken from PyWake, but slightly modified to fit our needs.
            colors = ["k", "gray", "r", "g"] * 5

            x, y, D = [np.asarray(v) for v in [x_turb, y_turb, wt.diameter()]]
            R = D / 2
            types = np.zeros_like(
                x, dtype=int
            )  # Assuming all turbines are of the same type
            for ii, (x_, y_, r, t, yaw_, tilt_) in enumerate(
                zip(x, y, R, types, yaw, tilt)
            ):
                for wd_ in np.atleast_1d(env.fs.wind_direction):
                    circle = Ellipse(
                        (x_, y_),
                        2 * r * np.sin(np.deg2rad(tilt_)),
                        2 * r,
                        angle=90 - wd_ + yaw_,
                        ec=colors[t],
                        fc="None",
                        lw=2.5,  # thicker rotor bar reads better at true aspect
                    )
                    ax1.add_artist(circle)
                    ax1.plot(x_, y_, ".", color=colors[t])

                for ii, (x_, y_, r) in enumerate(zip(x, y, R)):
                    text = ax1.annotate(
                        ii + 1,
                        (x_ - r, y_ + r),
                        fontsize=10,
                        color="white",
                    )
                    text.set_path_effects(
                        [
                            path_effects.Stroke(linewidth=2, foreground="black"),
                            path_effects.Normal(),
                        ]
                    )

                    # Annotate each turbine with its live derating value.
                    if derate_mode:
                        dtext = ax1.annotate(
                            f"{env.current_derate[ii]:.2f}",
                            (x_ - r, y_ - r),
                            fontsize=10,
                            color="white",
                        )
                        dtext.set_path_effects(
                            [
                                path_effects.Stroke(linewidth=2, foreground="black"),
                                path_effects.Normal(),
                            ]
                        )

            ax1.set_title("Flow field at {} s".format(env.fs.time))
            # True aspect so wakes read as long horizontal streaks and rotors as
            # correctly-proportioned cross-stream bars, instead of the old ~6x
            # vertical smear. Keep the meter ticks the old NullLocator hid. A
            # landscape row letterboxes to a band in the (square-ish) ax1 slot,
            # which is expected for this framing.
            ax1.set_aspect("equal")
            ax1.set_xlabel("x [m]")
            ax1.set_ylabel("y [m]")

            ax2 = plt.subplot2grid(
                grid,
                (0, 2),
            )
            ax3 = plt.subplot2grid(
                grid,
                (1, 2),
            )
            ax4 = plt.subplot2grid(
                grid,
                (2, 2),
            )
            if op_panels:
                ax5 = plt.subplot2grid(grid, (0, 3))
                ax6 = plt.subplot2grid(grid, (1, 3))
                right_axes = [ax2, ax3, ax4, ax5, ax6]
                # Lowest time-series axis of each sub-column gets the time axis
                bottom_axes = [ax4, ax6]
            else:
                right_axes = [ax2, ax3, ax4]
                bottom_axes = [ax4]
            if show_yaw_panel:
                # Yaw panel in the (2, 3) cell; it is now the lowest axis of
                # the right sub-column, so the time label/ticks move to it.
                ax7 = plt.subplot2grid(grid, (2, 3))
                right_axes.append(ax7)
                bottom_axes = [ax4, ax7]

            # Plot the power in ax2 (+ the tracking reference overlay).
            ax2.plot(time_deq, pow_deq, color="orange", label="farm")
            if tracking:
                ax2.plot(time_deq, pref_deq, "k--", label="reference")
                if not op_panels:
                    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax2.set_title("Farm power [W]")

            # Plot per-turbine derating (or yaws) in ax3
            if derate_mode:
                ax3.plot(time_deq, derate_deq, label=np.arange(n_turb))
                ax3.set_title("Turbine derating [-]")
            else:
                ax3.plot(time_deq, yaw_deq, label=np.arange(n_turb))
                ax3.set_title("Turbine yaws [deg]")
            if not op_panels:
                ax3.legend(
                    [f"T{i + 1}" for i in range(n_turb)],
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                )

            # Plot per-turbine power (or rotor windspeeds) in ax4
            if derate_mode:
                ax4.plot(time_deq, powerT_deq, label=np.arange(n_turb))
                ax4.set_title("Turbine power [W]")
            else:
                ax4.plot(time_deq, ws_deq, label=np.arange(n_turb))
                ax4.set_title("Local wind speed [m/s]")

            # Steady-state operating point in ax5/ax6 (surrogate table fidelity)
            if op_panels:
                ax5.plot(time_deq, pitch_deq, label=np.arange(n_turb))
                ax5.set_title("Blade pitch [deg]")
                ax6.plot(time_deq, rpm_deq, label=np.arange(n_turb))
                ax6.set_title("Rotor speed [RPM]")

            # Turbine yaws in the bottom-right cell (yaw+derate agents only)
            if show_yaw_panel:
                ax7.plot(time_deq, yaw_deq, label=np.arange(n_turb))
                ax7.set_title("Turbine yaws [deg]")

            # One shared legend as a horizontal figure-level strip below the
            # right-hand grid (the old in-grid legend cell is now the yaw
            # panel; per-axis outside legends would collide with the extra
            # sub-column).
            fig_legend = None
            if op_panels:
                farm_lines = list(ax2.get_lines())
                turb_lines = list(ax3.get_lines())
                fig_legend = fig.legend(
                    farm_lines + turb_lines,
                    [ln.get_label() for ln in farm_lines]
                    + [f"T{i + 1}" for i in range(n_turb)],
                    loc="upper center",
                    bbox_to_anchor=(0.76, 0.02),
                    ncol=len(farm_lines) + n_turb,
                    frameon=False,
                )

            # Time axis label + ticks live on the bottom panel of each column
            for ax in bottom_axes:
                ax.set_xlabel("Time [s]")

            # Set the x limits for the plots
            for ax in right_axes:
                ax.set_xlim(time_deq[0], time_deq[-1])

            pow_max = max(pow_max, powerF_a[end_idx] * 1.2)
            pow_min = min(pow_min, powerF_a[end_idx] * 0.8)
            if tracking:
                # Keep the reference line inside the frame even when the agent
                # tracks it poorly early on.
                pow_max = max(pow_max, p_ref[end_idx] * 1.2)
                pow_min = min(pow_min, p_ref[end_idx] * 0.8)

            # Set the y limits for the plots. If we go over/under the limits, the plot will adjust the limits.
            ax2.set_ylim(pow_min, pow_max)
            if derate_mode:
                # Fixed derate range [derate_min, derate_max] (+/- epsilon); the
                # per-turbine power axis grows to a running maximum like ax2.
                ax3.set_ylim(env.derate_min - 0.05, env.derate_max + 0.05)
                powT_max = max(powT_max, powerT_a[end_idx].max() * 1.2)
                ax4.set_ylim(0.0, powT_max)
                if op_panels:
                    pitch_lo = min(pitch_lo, float(pitch_a[end_idx].min()) - 0.5)
                    rpm_lo = min(rpm_lo, float(rpm_a[end_idx].min()) - 0.2)
                    rpm_hi = max(rpm_hi, float(rpm_a[end_idx].max()) + 0.2)
                    ax5.set_ylim(pitch_lo, pitch_hi)
                    ax6.set_ylim(rpm_lo, rpm_hi)
                if show_yaw_panel:
                    # Same running-limit rule as the yaw-only branch below.
                    yaw_max = max(yaw_max, max(yaw_a[end_idx]) * 1.2)
                    yaw_min = min(yaw_min, min(yaw_a[end_idx]) * 1.2)
                    ax7.set_ylim(yaw_min, yaw_max)
            else:
                yaw_max = max(yaw_max, max(yaw_a[end_idx]) * 1.2)
                # This value can be negative, so we multiply 1.2, instead of 0.8
                yaw_min = min(yaw_min, min(yaw_a[end_idx]) * 1.2)
                ws_max = max(ws_max, max(ws_a[end_idx]) * 1.2)
                ws_min = min(ws_min, min(ws_a[end_idx]) * 0.8)
                ax3.set_ylim(yaw_min, yaw_max)
                ax4.set_ylim(ws_min, ws_max)
            # ax2.set_xticks([])
            # ax3.set_xticks([])

            # Hide time ticks on everything but the bottom panel of each column
            for ax in right_axes:
                if ax in bottom_axes:
                    # Set the number of ticks on the x-axis to 5
                    ax.locator_params(axis="x", nbins=5)
                else:
                    ax.tick_params(axis="x", colors="white")

            for ax in right_axes:
                ax.grid()

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
                # The figure-level legend hangs below the axes region, so it
                # must be an extra artist or bbox_inches="tight" clips it.
                bbox_extra_artists=tuple(
                    [ax1]
                    + right_axes
                    + ([fig_legend] if fig_legend is not None else [])
                ),
                bbox_inches="tight",
            )
            plt.clf()
            plt.close("all")

    # Reshape the arrays and put them in a xarray dataset
    powerF_a = powerF_a.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    powerT_a = powerT_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    yaw_a = yaw_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    ws_a = ws_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    rew_plot = rew_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)

    # # Then create a xarray dataset with the results
    # Common data variables
    data_vars = {
        "powerF_a": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            powerF_a,
        ),
        "powerT_a": (
            (
                "time",
                "turb",
                "ws",
                "wd",
                "TI",
                "turbbox",
                "model_step",
                "deterministic",
            ),
            powerT_a,
        ),
        "yaw_a": (
            (
                "time",
                "turb",
                "ws",
                "wd",
                "TI",
                "turbbox",
                "model_step",
                "deterministic",
            ),
            yaw_a,
        ),
        "ws_a": (
            (
                "time",
                "turb",
                "ws",
                "wd",
                "TI",
                "turbbox",
                "model_step",
                "deterministic",
            ),
            ws_a,
        ),
        "reward": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            rew_plot,
        ),
    }

    # Add operating-point / derate variables if applicable
    turb_dims = (
        "time",
        "turb",
        "ws",
        "wd",
        "TI",
        "turbbox",
        "model_step",
        "deterministic",
    )
    if op_mode:
        pitch_a = pitch_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        rpm_a = rpm_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        data_vars.update(
            {
                "pitch_a": (turb_dims, pitch_a),
                "rpm_a": (turb_dims, rpm_a),
            }
        )
    if log_derate:
        derate_a = derate_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        data_vars.update({"derate_a": (turb_dims, derate_a)})

    # Add tracking variables if applicable
    if tracking:
        p_ref = p_ref.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        track_err = track_err.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        # Per-condition scalar (no time dim) so it merges across conditions
        # in eval_multiple like any other data variable.
        track_mae = np.full(
            (n_ws, n_wd, n_TI, n_turbbox, 1, 1),
            np.abs(track_err).mean(),
            dtype=np.float32,
        )

        data_vars.update(
            {
                "power_ref": (
                    (
                        "time",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    p_ref,
                ),
                "track_err": (
                    (
                        "time",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    track_err,
                ),
                "track_mae": (
                    ("ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    track_mae,
                ),
            }
        )

    # Add baseline variables if applicable
    if baseline_comp:
        powerF_b = powerF_b.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        powerT_b = powerT_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        yaw_b = yaw_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        ws_b = ws_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        pct_inc = pct_inc.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)

        data_vars.update(
            {
                "powerF_b": (
                    (
                        "time",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    powerF_b,
                ),
                "powerT_b": (
                    (
                        "time",
                        "turb",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    powerT_b,
                ),
                "yaw_b": (
                    (
                        "time",
                        "turb",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    yaw_b,
                ),
                "ws_b": (
                    (
                        "time",
                        "turb",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    ws_b,
                ),
                "pct_inc": (
                    (
                        "time",
                        "ws",
                        "wd",
                        "TI",
                        "turbbox",
                        "model_step",
                        "deterministic",
                    ),
                    pct_inc,
                ),
            }
        )

    # Common coordinates
    coords = {
        "ws": np.array([ws]),
        "wd": np.array([wd]),
        "turb": np.arange(n_turb),
        "time": time_plot,
        "TI": np.array([ti]),
        "turbbox": [turbbox],
        "model_step": np.array([model_step]),
        "deterministic": np.array([deterministic]),
    }

    # Create the dataset
    if not return_loads:
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        # Do this to remove it from memory
        env.timestep = env.time_max
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        return ds
    # Do this if we have the HTC and want the loads as well.
    elif env.HTC_path is not None:
        # If the HTC_path is not None, then ill assume we also want to include the loads
        # First make sure we have written the lates results
        env.wts.h2.write_output()  # I am not sure this is needed tho

        all_data = []
        # For each turbine read the data and put in into an array
        for i in range(n_turb):
            file_name = env.wts.htc_lst[i].output.filename.values[0] + ".hdf5"
            test_string = env.wts.htc_lst[i].modelpath + file_name
            time, data, info = gtsdf.load(test_string)

            # Store each turbine's data in a dictionary
            all_data.append(
                {
                    "Ae rot. torque": data[:, 10],
                    "Ae rot. power": data[:, 11],
                    "Ae rot. thrust": data[:, 12],
                    "WSP gl. coo.,Vx": data[:, 13],
                    "WSP gl. coo.,Vy": data[:, 14],
                    "WSP gl. coo.,Vz": data[:, 15],
                    "Blade_Mx": data[:, 19],
                    "Blade_My": data[:, 20],
                    "Tower_Mx": data[:, 28],
                    "Tower_My": data[:, 29],
                    "yaw_a": data[:, 112],
                    "time": time,
                }
            )

        # Assuming all turbines share the same time vector
        time = all_data[0]["time"]

        # Stack data into arrays with shape (turbine, time)
        blade_mx = np.stack([d["Blade_Mx"] for d in all_data]).T
        blade_my = np.stack([d["Blade_My"] for d in all_data]).T
        tower_mx = np.stack([d["Tower_Mx"] for d in all_data]).T
        tower_my = np.stack([d["Tower_My"] for d in all_data]).T
        Ae_rot_torque = np.stack([d["Ae rot. torque"] for d in all_data]).T
        Ae_rot_power = np.stack([d["Ae rot. power"] for d in all_data]).T
        Ae_rot_thrust = np.stack([d["Ae rot. thrust"] for d in all_data]).T
        WSP_gl_coo_Vx = np.stack([d["WSP gl. coo.,Vx"] for d in all_data]).T
        WSP_gl_coo_Vy = np.stack([d["WSP gl. coo.,Vy"] for d in all_data]).T
        WSP_gl_coo_Vz = np.stack([d["WSP gl. coo.,Vz"] for d in all_data]).T
        yaw_a = np.stack([d["yaw_a"] for d in all_data]).T

        # Reshape the data to match the xarray dataset dimensions
        blade_mx = blade_mx.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        blade_my = blade_my.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        tower_mx = tower_mx.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        tower_my = tower_my.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        Ae_rot_torque = Ae_rot_torque.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        Ae_rot_power = Ae_rot_power.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        Ae_rot_thrust = Ae_rot_thrust.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        WSP_gl_coo_Vx = WSP_gl_coo_Vx.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        WSP_gl_coo_Vy = WSP_gl_coo_Vy.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        WSP_gl_coo_Vz = WSP_gl_coo_Vz.reshape(
            time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1
        )
        yaw_a = yaw_a.reshape(time.shape[0], n_turb, n_ws, n_wd, n_TI, n_turbbox, 1)

        # Create xarray dataset with 'turb' and 'time' dimensions
        ds_load = xr.Dataset(
            data_vars={
                "Blade_Mx": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    blade_mx,
                ),
                "Blade_My": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    blade_my,
                ),
                "Tower_Mx": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    tower_mx,
                ),
                "Tower_My": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    tower_my,
                ),
                "Ae_rot_torque": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    Ae_rot_torque,
                ),
                "Ae_rot_power": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    Ae_rot_power,
                ),
                "Ae_rot_thrust": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    Ae_rot_thrust,
                ),
                "WSP_gl_coo_Vx": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    WSP_gl_coo_Vx,
                ),
                "WSP_gl_coo_Vy": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    WSP_gl_coo_Vy,
                ),
                "WSP_gl_coo_Vz": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    WSP_gl_coo_Vz,
                ),
                "yaw_a": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step"),
                    yaw_a,
                ),
            },
            coords={
                "ws": np.array([ws]),
                "wd": np.array([wd]),
                "turb": np.arange(n_turb),
                "time": time,
                "TI": np.array([ti]),
                "turbbox": [turbbox],
                "model_step": np.array([model_step]),
            },
        )
        # Clean up also.
        # To make sure that the turbulence box is removed from memory, we set the current timestep to be equal to the max, and then do one last step.
        # This clears the turbulence box from memory, and makes sure that we dont have any issues with the turbulence box being in memory.
        env.wts.h2.close()
        if baseline_comp:
            env.wts_baseline.h2.close()

        if cleanup:
            env._deleteHAWCfolder()
            env.fs = None
            env.site = None
            env.farm_measurements = None
            del env.fs
            del env.site
            del env.farm_measurements

            if baseline_comp:
                env.fs_baseline = None
                env.site_base = None
                del env.fs_baseline
                del env.site_base

        return ds_load


class AgentEval:
    def __init__(self, env=None, model=None, name="NoName", t_sim=1000, seed=None):
        # Initialize the evaluater with some default values.
        self.ws = 10.0
        self.ti = 0.05
        self.wd = 270
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
                        # For all these in the loop...
                        # Set the conditions
                        self.set_condition(ws=windspeed, ti=TI, wd=winddir, turbbox=box)
                        episode_seed = (
                            int(rng.integers(2**31)) if rng is not None else None
                        )
                        # Run the simulation
                        ds = self.eval_single(
                            save_figs=save_figs,
                            scale_obs=scale_obs,
                            debug=debug,
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
