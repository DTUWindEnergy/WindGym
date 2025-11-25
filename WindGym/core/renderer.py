"""
Rendering module for WindGym wind farm environments.

This module handles all visualization and rendering functionality,
separating it from the core environment logic.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython import display

from dynamiks.views import XYView, EastNorthView
from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from dynamiks.utils.geometry import get_east_north_height
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse


class WindFarmRenderer:
    """
    Handles rendering of wind farm environments.

    Supports multiple render modes:
    - "rgb_array": Return RGB frames for recording/saving
    - "human": Display frames in a window for human viewing
    - None: No rendering

    Also provides utility methods for plotting farm layouts and frames.
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the renderer.

        Args:
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        self.render_mode = render_mode

        # Rendering objects (initialized lazily)
        self.figure = None
        self.ax = None
        self.view = None
        self.a = None  # x-axis linspace
        self.b = None  # y-axis linspace

    def init_render(self, fs, turbine):
        """
        Initialize rendering objects.

        This creates the matplotlib figure, axis, and XYView for rendering.
        Should be called after the flow simulation is created.

        Args:
            fs: Flow simulation object
            turbine: Turbine object (for hub_height)
        """
        plt.ion()
        x_turb, y_turb = fs.windTurbines.positions_xyz[:2]

        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.a = np.linspace(-200 + min(x_turb), 1000 + max(x_turb), 250)
        self.b = np.linspace(-200 + min(y_turb), 200 + max(y_turb), 250)

        self.view = XYView(
            z=turbine.hub_height(), x=self.a, y=self.b, ax=self.ax, adaptive=False
        )

        plt.close()

    def render(self, fs, fs_baseline=None, probes=None, turbine=None):
        """
        Main render method - routes to appropriate rendering function.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            probes: Optional list of wind probes
            turbine: Turbine object for lazy initialization

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            # Return the RGB frame (for recording, saving, etc.)
            return self._render_frame(fs, fs_baseline, probes, turbine=turbine)
        elif self.render_mode == "human":
            # Show the frame in a window (will initialize lazily if needed)
            frame = self._render_frame_for_human(
                fs, fs_baseline, probes, turbine=turbine
            )
            plt.imshow(frame)
            plt.axis("off")
            plt.title("Wind Farm Environment - Render")
            plt.show(block=False)
            plt.pause(0.001)  # Pause to allow window to update
            return None
        else:
            return None

    def _render_frame_for_human(
        self,
        fs,
        fs_baseline=None,
        probes=None,
        baseline: bool = False,
        turbine=None,
    ):
        """
        Render the environment for human viewing with full details.

        Includes wind speed heatmap, turbines, and optional probes with arrows.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            probes: Optional list of wind probes
            baseline: Whether to render baseline instead of agent
            turbine: Turbine object (for view if not initialized)

        Returns:
            np.ndarray: RGB frame
        """
        plt.ioff()  # Non-interactive mode
        fig, ax1 = plt.subplots(figsize=(18, 6))

        fs_use = fs_baseline if baseline else fs

        # Ensure view is initialized
        if self.view is None and turbine is not None:
            self.init_render(fs, turbine)

        uvw = fs_use.get_windspeed(self.view, include_wakes=True, xarray=True)

        wt = fs_use.windTurbines
        x_turb, y_turb = fs_use.windTurbines.positions_xyz[:2]
        yaw, tilt = wt.yaw_tilt()

        mesh = ax1.pcolormesh(
            uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest", cmap="viridis"
        )
        plt.colorbar(mesh, ax=ax1, label="Wind Speed (m/s)")

        WindTurbinesPW.plot_xy(
            fs_use.windTurbines,
            x_turb,
            y_turb,
            types=fs_use.windTurbines.types,
            wd=fs_use.wind_direction,
            ax=ax1,
            yaw=yaw,
            tilt=tilt,
        )
        ax1.set_aspect("equal", adjustable="datalim")

        # Plot probes with color depending on probe type
        if probes is not None and len(probes) > 0:
            for probe in probes:
                x, y, _ = probe.position
                probe_type = probe.probe_type.upper()

                # Determine color and label
                if probe_type == "WS":
                    color = "red"
                    label = "WS Probe"
                    value = float(probe.read())
                    text = f"{value:.2f} m/s"
                elif probe_type == "TI":
                    color = "blue"
                    label = "TI Probe"
                    value = float(probe.read())
                    text = f"{value:.2f} TI"
                else:
                    color = "gray"
                    label = "Unknown"
                    text = "N/A"

                ax1.scatter(x, y, color=color, s=25, marker="o", label=label)
                ax1.text(
                    x + 5,
                    y + 5,
                    text,
                    color="black",
                    fontsize=8,
                    bbox=dict(facecolor="none", alpha=0.6, edgecolor="none"),
                )

                speed = float(probe.read())
                arrow_length = speed * 5
                # Draw inflow direction arrow
                inflow_angle = probe.get_inflow_angle_to_turbine()
                dx = arrow_length * np.cos(inflow_angle)
                dy = arrow_length * np.sin(inflow_angle)

                ax1.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    width=1.5,
                    head_width=5.0,
                    head_length=7.0,
                    fc=color,
                    ec=color,
                    alpha=0.8,
                    length_includes_head=True,
                )

            ax1.set_title(f"Flow field at {fs_use.time} s")

            # Avoid duplicate legend entries
            handles, labels_list = ax1.get_legend_handles_labels()
            if labels_list.count("Probe") > 1:
                unique = dict(zip(labels_list, handles))
                ax1.legend(unique.values(), unique.keys())

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # Get RGB only

        plt.close(fig)  # Avoid memory leaks
        return frame

    def _render_frame(
        self,
        fs,
        fs_baseline=None,
        probes=None,
        baseline: bool = False,
        turbine=None,
        ws=None,
    ):
        """
        Render the current environment state and return RGB array.

        Simpler rendering compared to _render_frame_for_human,
        reuses the figure and axis from init_render.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            probes: Optional list of wind probes (not used in this method)
            baseline: Whether to render baseline instead of agent
            turbine: Turbine object (for view if not initialized)
            ws: Wind speed (for color scale)

        Returns:
            np.ndarray: RGB frame
        """
        # Ensure render objects are initialized
        if self.view is None:
            if turbine is None:
                raise RuntimeError(
                    "Renderer not initialized and turbine not provided. "
                    "Call init_render() first."
                )
            self.init_render(fs, turbine)

        # Use the figure and axis created during initialization
        fig = self.figure
        ax = self.ax
        ax.cla()  # Clear the axis for the new frame

        fs_use = fs_baseline if baseline else fs

        # Define a temporary view for this frame's plot
        temp_view = XYView(
            z=turbine.hub_height() if turbine else self.view.z,
            x=self.a,
            y=self.b,
            ax=ax,
            adaptive=False,
        )
        uvw = fs_use.get_windspeed(temp_view, include_wakes=True, xarray=True)

        # Plot the wind speed heatmap
        vmax = (ws + 2) if ws is not None else None
        ax.pcolormesh(
            uvw.x.values,
            uvw.y.values,
            uvw[0].T,
            shading="auto",
            cmap="viridis",
            vmin=3,
            vmax=vmax,
        )

        # Get turbine coordinates
        x_turb, y_turb, _ = fs_use.windTurbines.positions_xyz

        # Plot the turbines
        WindTurbinesPW.plot_xy(
            fs_use.windTurbines,
            x_turb,
            y_turb,
            wd=fs_use.wind_direction,
            yaw=fs_use.windTurbines.yaw,
            ax=ax,
        )

        ax.set_title(f"Flow Field at Time: {fs_use.time:.1f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        # Capture canvas to NumPy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # RGB only

        return frame

    def plot_farm(
        self,
        fs,
        fs_baseline=None,
        turbine=None,
        baseline: bool = False,
        fix_turbines: bool = True,
    ):
        """
        Plot the entire farm layout (legacy method for IPython notebooks).

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            turbine: Turbine object
            baseline: Whether to plot baseline instead of agent
        """
        if turbine is not None:
            self.init_render(fs, turbine)
        self._render_farm(fs, fs_baseline, baseline, fix_turbines)

    def _render_farm(
        self, fs, fs_baseline=None, baseline: bool = False, fix_turbines: bool = True
    ):
        """
        Internal farm rendering for IPython notebooks.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            baseline: Whether to render baseline instead of agent
            fix_turbines: Whether to fix turbine positions
        """
        plt.ion()
        ax1 = plt.gca()

        fs_use = fs_baseline if baseline else fs
        wt = fs_use.windTurbines

        if fix_turbines:
            view = EastNorthView(z=70, x=self.a, y=self.b, adaptive=False)
            X, Y = view.XY(
                wind_direction=fs_use.wind_direction, center_offset=fs_use.center_offset
            )
            x_turb, y_turb = get_east_north_height(
                xyz=wt.positions_xyz,
                wind_direction=fs_use.wind_direction,
                center_offset=fs_use.center_offset,
            )[:2]
            yaw, tilt = wt.yaw_tilt()
            yaw_plot = yaw - fs_use.wind_direction + 90

        else:
            view = XYView(z=70, x=self.a, y=self.b, adaptive=False)
            x_turb, y_turb = wt.positions_xyz[:2]
            X, Y = view.XY()
            yaw, tilt = wt.yaw_tilt()
            yaw_plot = yaw

        uvw = fs_use.get_windspeed(view, include_wakes=True, xarray=True)

        ax1.pcolormesh(X, Y, uvw[0].T, shading="nearest")

        # Plot turbines
        D = wt.diameter()
        R = D / 2

        for ii, (x_, y_, r, yaw_, tilt_) in enumerate(
            zip(x_turb, y_turb, R, yaw_plot, tilt)
        ):
            # Draw turbine as ellipse
            circle = Ellipse(
                (x_, y_),
                2 * r * np.sin(np.deg2rad(tilt_)),
                2 * r,
                angle=yaw_,
                ec="k",
                fc="None",
            )
            ax1.add_artist(circle)
            ax1.plot(x_, y_, ".", color="k")

            # Add turbine label
            text = ax1.annotate(
                f"T {ii + 1}",
                (x_ - r, y_ + r * 1.75),
                fontsize=15,
                color="white",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ]
            )

        ax1.set_xlim(min(self.a), max(self.a))
        ax1.set_ylim(min(self.b), max(self.b))

        # plt.pcolormesh(uvw.x.values, uvw.y.values, uvw[0].T, shading="nearest")
        # WindTurbinesPW.plot_xy(
        #     fs_use.windTurbines,
        #     x_turb,
        #     y_turb,
        #     wd=fs_use.wind_direction,
        #     ax=ax1,
        #     yaw=yaw,
        #     tilt=tilt,
        # )
        ax1.set_title("Flow field at {} s".format(fs_use.time))
        ax1.set_aspect("equal", adjustable="box")
        # display.display(plt.gcf())
        # display.clear_output(wait=True)

    def plot_frame(self, fs, fs_baseline=None, turbine=None, baseline: bool = False):
        """
        Plot a single frame of the flow field and turbines.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            turbine: Turbine object
            baseline: Whether to plot baseline instead of agent
        """
        if turbine is not None:
            self.init_render(fs, turbine)
        self._render_frame(fs, fs_baseline, baseline=baseline, turbine=turbine)

    def close(self):
        """Close any open matplotlib figures."""
        plt.close()
        self.figure = None
        self.ax = None
        self.view = None
