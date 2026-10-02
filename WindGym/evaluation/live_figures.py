"""Live per-step figures for ``eval_single_fast(save_figs=True)``.

The drawing code is a verbatim move from the old ``eval_single_fast`` body:
it relies on pyplot's current-figure state (``plt.subplot2grid``,
``plt.pcolormesh``, ``plt.colorbar``, ``plt.close("all")``) and its output
is pixel-identical to the pre-refactor frames.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from dynamiks.views import XYView
from matplotlib.patches import Ellipse


def resolve_fig_folder(fig_dir, name, env_ws, wd) -> str:
    """Folder (with trailing separator) the frames are written to."""
    if fig_dir is not None:
        return os.path.join(fig_dir, "")
    return "./Temp_Figs_{}_ws{}_wd{}/".format(name, env_ws, wd)


def assemble_gif(folder, n_frames, fps):
    """
    Assemble the img_XXXXX.png frames in `folder` into an animated GIF.

    The frames are first cropped to the smallest common size, since
    bbox_inches="tight" lets the figure size drift a few pixels between
    frames and GIFs need a uniform frame shape. Uses ffmpeg when available
    (two-pass palette for better colors), otherwise falls back to imageio.

    Returns the path of the written GIF, or None if no frames were found.
    """
    from PIL import Image  # matplotlib dependency, always available here

    paths = [os.path.join(folder, "img_{:05d}.png".format(i)) for i in range(n_frames)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None

    sizes = []
    for p in paths:
        with Image.open(p) as im:
            sizes.append(im.size)
    w = min(s[0] for s in sizes)
    h = min(s[1] for s in sizes)
    for p, s in zip(paths, sizes):
        if s != (w, h):
            with Image.open(p) as im:
                im.crop((0, 0, w, h)).save(p)

    gif_path = os.path.join(folder, "animation.gif")
    if shutil.which("ffmpeg") is not None:
        # Scrub LD_LIBRARY_PATH for the child: conda envs put an older
        # libstdc++ on it, which breaks the system ffmpeg's own libraries.
        child_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(folder, "img_%05d.png"),
                    "-vf",
                    "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    "-loop",
                    "0",
                    gif_path,
                ],
                check=True,
                env=child_env,
            )
            return gif_path
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"ffmpeg GIF assembly failed ({e}); falling back to imageio.")
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "GIF assembly needs either ffmpeg on the PATH or the imageio "
            "package (pip install imageio)."
        ) from e
    frames = [imageio.imread(p) for p in paths]
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    return gif_path


class LiveFigureWriter:
    """Renders ``img_{i:05d}.png`` after each env step from an ``EpisodeRecorder``.

    Args:
        env: The evaluation env (flow field, measurements, derate state).
        recorder: The ``EpisodeRecorder`` being filled by the eval loop.
        folder: Output folder (created if missing).
        scaling: List of ``None`` / ``True`` / ``False``: which observation
            text blocks to overlay (scaled and/or raw).
        n_turb: Number of turbines.
        tracking: Whether the env tracks a power reference.
        op_mode: Whether the env reports its operating point (pitch / RPM).
    """

    def __init__(self, env, recorder, *, folder, scaling, n_turb, tracking, op_mode):
        self.env = env
        self.rec = recorder
        self.scaling = scaling
        self.n_turb = n_turb
        self.tracking = tracking
        self.op_mode = op_mode

        self.FOLDER = folder
        if not os.path.exists(self.FOLDER):
            os.makedirs(self.FOLDER)
        rec = recorder
        max_deque = 70
        self.time_deq = deque(maxlen=max_deque)
        self.pow_deq = deque(maxlen=max_deque)
        self.yaw_deq = deque(maxlen=max_deque)
        self.ws_deq = deque(maxlen=max_deque)

        self.time_deq.append(rec.time_plot[0])
        self.pow_deq.append(rec.powerF_a[0])
        self.yaw_deq.append(rec.yaw_a[0])
        self.ws_deq.append(rec.ws_a[0])
        # These are used for y limits on the plot.
        self.pow_max = rec.powerF_a[0] * 1.2
        self.pow_min = rec.powerF_a[0] * 0.8
        self.yaw_max = 5
        self.yaw_min = -5
        self.ws_max = env.ws + 2
        self.ws_min = 3

        # Derating / tracking panels: for a derate-only agent the yaw-trainer's
        # right column (yaw + local wind speed) is meaningless (yaw is fixed).
        # Auto-detect the mode from the env and swap in derating + per-turbine
        # power; a yaw run (derate_mode=False) keeps every original panel.
        self.derate_mode = (
            bool(getattr(env, "derate_action", False))
            and getattr(env, "current_derate", None) is not None
        )
        # A yaw+derate agent steers AND derates: keep the derate layout but add
        # a yaw time-series panel in the spare bottom-right cell. Derate-only
        # envs (yaw_action=False) leave that cell blank ("yaw fixed").
        yaw_active = bool(getattr(env, "yaw_action", True))
        self.show_yaw_panel = self.derate_mode and yaw_active
        if self.derate_mode:
            self.derate_deq = deque(maxlen=max_deque)
            self.powerT_deq = deque(maxlen=max_deque)
            self.derate_deq.append(np.asarray(env.current_derate).copy())
            self.powerT_deq.append(rec.powerT_a[0].copy())
            self.powT_max = rec.powerT_a[0].max() * 1.2
        # Two extra right-column panels (blade pitch, rotor RPM) when the env
        # can report its steady-state operating point. Yaw-only runs keep the
        # original 3-row layout untouched.
        self.op_panels = self.derate_mode and op_mode
        if self.op_panels:
            self.pitch_deq = deque(maxlen=max_deque)
            self.rpm_deq = deque(maxlen=max_deque)
            self.pitch_deq.append(rec.pitch_a[0].copy())
            self.rpm_deq.append(rec.rpm_a[0].copy())
            # Default panel ranges; grown on the fly (like pow_max) whenever
            # the data leaves them. Exception: the pitch axis is HARD-capped
            # at 15 deg — the table's feathered/parked points (pitch ~90) at
            # deep derate + low waked ws would otherwise flatten the panel.
            self.pitch_lo = min(0.0, float(rec.pitch_a[0].min()) - 0.5)
            self.pitch_hi = 15.0
            self.rpm_lo = min(5.0, float(rec.rpm_a[0].min()) - 0.2)
            self.rpm_hi = max(8.0, float(rec.rpm_a[0].max()) + 0.2)
        if tracking:
            self.pref_deq = deque(maxlen=max_deque)
            self.pref_deq.append(rec.p_ref[0])

        # Flow-field extent. x spans the row + margins; y is padded +-2D so a
        # single row sits in a ~3.7:1 rectangle that reads at true (equal) aspect
        # (see below) instead of the old ~6x-stretched square. For multi-row
        # farms this just pads 2D beyond the y-extent, so nothing is clipped.
        D_view = float(np.atleast_1d(env.fs.windTurbines.diameter())[0])
        self.a = np.linspace(-200 + min(env.x_pos), 300 + max(env.x_pos), 200)
        self.b = np.linspace(
            min(env.y_pos) - 2 * D_view, max(env.y_pos) + 2 * D_view, 200
        )

    def draw(self, i: int) -> str:
        """Render the frame for env step ``i``; returns the written path."""
        env = self.env
        rec = self.rec
        n_turb = self.n_turb
        tracking = self.tracking
        derate_mode = self.derate_mode
        op_panels = self.op_panels
        show_yaw_panel = self.show_yaw_panel
        step_val = env.sim_steps_per_env_step

        # The result arrays are at sim resolution while i counts env
        # steps; index the end-of-step sample, not raw i.
        end_idx = i * step_val + step_val
        self.time_deq.append(rec.time_plot[end_idx])
        self.pow_deq.append(rec.powerF_a[end_idx])
        self.yaw_deq.append(rec.yaw_a[end_idx])
        self.ws_deq.append(rec.ws_a[end_idx])
        if derate_mode:
            self.derate_deq.append(np.asarray(env.current_derate).copy())
            self.powerT_deq.append(rec.powerT_a[end_idx].copy())
        if op_panels:
            self.pitch_deq.append(rec.pitch_a[end_idx].copy())
            self.rpm_deq.append(rec.rpm_a[end_idx].copy())
        if tracking:
            self.pref_deq.append(rec.p_ref[end_idx])

        time_deq = self.time_deq

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

        view = XYView(z=70, x=self.a, y=self.b, ax=fig.gca(), adaptive=False)

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
        ax2.plot(time_deq, self.pow_deq, color="orange", label="farm")
        if tracking:
            ax2.plot(time_deq, self.pref_deq, "k--", label="reference")
            if not op_panels:
                ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax2.set_title("Farm power [W]")

        # Plot per-turbine derating (or yaws) in ax3
        if derate_mode:
            ax3.plot(time_deq, self.derate_deq, label=np.arange(n_turb))
            ax3.set_title("Turbine derating [-]")
        else:
            ax3.plot(time_deq, self.yaw_deq, label=np.arange(n_turb))
            ax3.set_title("Turbine yaws [deg]")
        if not op_panels:
            ax3.legend(
                [f"T{i + 1}" for i in range(n_turb)],
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )

        # Plot per-turbine power (or rotor windspeeds) in ax4
        if derate_mode:
            ax4.plot(time_deq, self.powerT_deq, label=np.arange(n_turb))
            ax4.set_title("Turbine power [W]")
        else:
            ax4.plot(time_deq, self.ws_deq, label=np.arange(n_turb))
            ax4.set_title("Local wind speed [m/s]")

        # Steady-state operating point in ax5/ax6 (surrogate table fidelity)
        if op_panels:
            ax5.plot(time_deq, self.pitch_deq, label=np.arange(n_turb))
            ax5.set_title("Blade pitch [deg]")
            ax6.plot(time_deq, self.rpm_deq, label=np.arange(n_turb))
            ax6.set_title("Rotor speed [RPM]")

        # Turbine yaws in the bottom-right cell (yaw+derate agents only)
        if show_yaw_panel:
            ax7.plot(time_deq, self.yaw_deq, label=np.arange(n_turb))
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

        self.pow_max = max(self.pow_max, rec.powerF_a[end_idx] * 1.2)
        self.pow_min = min(self.pow_min, rec.powerF_a[end_idx] * 0.8)
        if tracking:
            # Keep the reference line inside the frame even when the agent
            # tracks it poorly early on.
            self.pow_max = max(self.pow_max, rec.p_ref[end_idx] * 1.2)
            self.pow_min = min(self.pow_min, rec.p_ref[end_idx] * 0.8)

        # Set the y limits for the plots. If we go over/under the limits, the plot will adjust the limits.
        ax2.set_ylim(self.pow_min, self.pow_max)
        if derate_mode:
            # Fixed derate range [derate_min, derate_max] (+/- epsilon); the
            # per-turbine power axis grows to a running maximum like ax2.
            ax3.set_ylim(env.derate_min - 0.05, env.derate_max + 0.05)
            self.powT_max = max(self.powT_max, rec.powerT_a[end_idx].max() * 1.2)
            ax4.set_ylim(0.0, self.powT_max)
            if op_panels:
                self.pitch_lo = min(
                    self.pitch_lo, float(rec.pitch_a[end_idx].min()) - 0.5
                )
                self.rpm_lo = min(self.rpm_lo, float(rec.rpm_a[end_idx].min()) - 0.2)
                self.rpm_hi = max(self.rpm_hi, float(rec.rpm_a[end_idx].max()) + 0.2)
                ax5.set_ylim(self.pitch_lo, self.pitch_hi)
                ax6.set_ylim(self.rpm_lo, self.rpm_hi)
            if show_yaw_panel:
                # Same running-limit rule as the yaw-only branch below.
                self.yaw_max = max(self.yaw_max, max(rec.yaw_a[end_idx]) * 1.2)
                self.yaw_min = min(self.yaw_min, min(rec.yaw_a[end_idx]) * 1.2)
                ax7.set_ylim(self.yaw_min, self.yaw_max)
        else:
            self.yaw_max = max(self.yaw_max, max(rec.yaw_a[end_idx]) * 1.2)
            # This value can be negative, so we multiply 1.2, instead of 0.8
            self.yaw_min = min(self.yaw_min, min(rec.yaw_a[end_idx]) * 1.2)
            self.ws_max = max(self.ws_max, max(rec.ws_a[end_idx]) * 1.2)
            self.ws_min = min(self.ws_min, min(rec.ws_a[end_idx]) * 0.8)
            ax3.set_ylim(self.yaw_min, self.yaw_max)
            ax4.set_ylim(self.ws_min, self.ws_max)
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

        img_name = self.FOLDER + "img_{:05d}.png".format(i)

        # Add a text to the plot with the sensor values
        for scale in self.scaling:  # scaling can be a list with True and False. If True, we add the scaled observations to the plot. If False, we only add the unscaled observations.
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
                [ax1] + right_axes + ([fig_legend] if fig_legend is not None else [])
            ),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close("all")
        return img_name
