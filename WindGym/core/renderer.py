"""
Rendering module for WindGym wind farm environments.

This module handles all visualization and rendering functionality,
separating it from the core environment logic.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from dynamiks.views import XYView, EastNorthView
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

    def __init__(
        self,
        render_mode: Optional[str] = None,
        fix_turbines: bool = False,
        show_indices: bool = True,
        fontsize: int = 15,
        axes_lw: float = 1.0,
        colorbar_vmax_step: float = 2.0,
    ):
        self.render_mode = render_mode
        self.fix_turbines = fix_turbines
        self.show_indices = show_indices
        self.fontsize = fontsize
        self.axes_lw = axes_lw
        self.colorbar_vmax_step = colorbar_vmax_step

        # Rendering objects (initialized lazily)
        self.view = None
        self.a = None  # x-axis linspace
        self.b = None  # y-axis linspace

    def init_render(self, fs, turbine):
        """
        Initialize the grid extents and XYView used by all render methods.

        Args:
            fs: Flow simulation object
            turbine: Turbine object (for hub_height)
        """
        x_turb, y_turb = fs.windTurbines.positions_xyz[:2]
        # y margin scales with rotor size so large turbines (IEA22, D=280 m)
        # keep their deflected wakes inside the view.
        D = float(np.max(turbine.diameter()))

        y_pad = max(3 * D, 0.2 * (max(y_turb) - min(y_turb) + 1))
        self.a = np.linspace(-3 * D + min(x_turb), 1500 + max(x_turb), 250)
        self.b = np.linspace(-y_pad + min(y_turb), y_pad + max(y_turb), 250)

        self.view = XYView(z=turbine.hub_height(), x=self.a, y=self.b, adaptive=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self, fs, turbine):
        """Initialize view lazily, raising if turbine is not available."""
        if self.view is None:
            if turbine is None:
                raise RuntimeError(
                    "Renderer not initialized and turbine not provided. "
                    "Call init_render() first or pass turbine."
                )
            self.init_render(fs, turbine)

    @staticmethod
    def _resolve_fs(fs, fs_baseline, baseline):
        """Return the appropriate flow simulation based on the baseline flag."""
        return fs_baseline if baseline else fs

    @staticmethod
    def _draw_turbines(
        ax, x_turb, y_turb, R, yaw_plot, tilt, fontsize=15, show_indices=True
    ):
        """Draw turbines as ellipses on ax, with optional index labels."""
        for ii, (x_, y_, r, yaw_, tilt_) in enumerate(
            zip(x_turb, y_turb, R, yaw_plot, tilt)
        ):
            ax.add_artist(
                Ellipse(
                    (x_, y_),
                    2 * r * np.sin(np.deg2rad(tilt_)),
                    2 * r,
                    angle=yaw_,
                    ec="k",
                    fc="None",
                )
            )
            ax.plot(x_, y_, ".", color="k")
            if show_indices:
                text = ax.annotate(
                    f"T {ii + 1}",
                    (x_ - r, y_ + r * 1.75),
                    fontsize=fontsize,
                    color="white",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2, foreground="black"),
                        path_effects.Normal(),
                    ]
                )

    @staticmethod
    def _probe_xy(probe, fs, fix_turbines):
        """Return the (x, y) position of a probe in the current plot frame."""
        px, py, pz = probe.position
        if fix_turbines:
            ep, np_ = get_east_north_height(
                xyz=np.array([[px], [py], [pz]]),
                wind_direction=fs.wind_direction,
                center_offset=fs.center_offset,
            )[:2]
            return float(ep[0]), float(np_[0])
        return px, py

    @staticmethod
    def _fig_to_rgb(fig):
        """Render a matplotlib figure to an RGB numpy array and close it."""
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return frame

    def _build_view(self, fs, fix_turbines):
        """
        Build a view and return coordinate arrays for the given flow simulation.

        Args:
            fs: Flow simulation object
            fix_turbines: If True use EastNorthView; otherwise use XYView.

        Returns:
            view, X, Y, x_turb, y_turb, yaw, tilt, yaw_plot
        """
        wt = fs.windTurbines
        yaw, tilt = wt.yaw_tilt()

        if fix_turbines:
            view = EastNorthView(z=self.view.z, x=self.a, y=self.b, adaptive=False)
            X, Y = view.XY(
                wind_direction=fs.wind_direction, center_offset=fs.center_offset
            )
            x_turb, y_turb = get_east_north_height(
                xyz=wt.positions_xyz,
                wind_direction=fs.wind_direction,
                center_offset=fs.center_offset,
            )[:2]
            yaw_plot = yaw - fs.wind_direction + 90
        else:
            view = XYView(z=self.view.z, x=self.a, y=self.b, adaptive=False)
            X, Y = view.XY()
            x_turb, y_turb, _ = wt.positions_xyz
            yaw_plot = yaw

        return view, X, Y, x_turb, y_turb, yaw, tilt, yaw_plot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self, fs, fs_baseline=None, probes=None, turbine=None, fix_turbines=False
    ):
        """
        Main render method.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            probes: Optional list of wind probes
            turbine: Turbine object for lazy initialization
            fix_turbines: If True use EastNorthView (farm fixed, wind rotates)

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None

        frame = self._render_frame(
            fs, fs_baseline, probes=probes, turbine=turbine, fix_turbines=fix_turbines
        )

        if self.render_mode == "human":
            plt.imshow(frame)
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.001)
            return None

        return frame  # "rgb_array"

    def _render_frame(
        self,
        fs,
        fs_baseline=None,
        probes=None,
        baseline: bool = False,
        turbine=None,
        fix_turbines: bool = False,
    ):
        """
        Render the current environment state and return an RGB array.

        Args:
            fs: Flow simulation object
            fs_baseline: Optional baseline flow simulation
            probes: Optional list of wind probes
            baseline: Whether to render baseline instead of agent
            turbine: Turbine object (for view if not initialized)
            ws: Approximate free-stream wind speed used to set colorbar vmax
            fix_turbines: If True use EastNorthView (farm fixed, wind rotates)

        Returns:
            np.ndarray: RGB frame (H x W x 3)
        """
        self._ensure_initialized(fs, turbine)

        plt.ioff()
        fig_h = 6
        data_aspect = (max(self.a) - min(self.a)) / (max(self.b) - min(self.b))
        fig, ax = plt.subplots(
            figsize=(fig_h * data_aspect + 1.5, fig_h), layout="constrained"
        )

        fs_use = self._resolve_fs(fs, fs_baseline, baseline)
        wt = fs_use.windTurbines

        view, X, Y, x_turb, y_turb, _, tilt, yaw_plot = self._build_view(
            fs_use, fix_turbines
        )
        uvw = fs_use.get_windspeed(view, include_wakes=True, xarray=True)

        raw = float(uvw[0].max())
        vmax = np.ceil(raw / self.colorbar_vmax_step) * self.colorbar_vmax_step
        mesh = ax.pcolormesh(
            X,
            Y,
            uvw[0].T,
            shading="nearest",
            cmap="viridis",
            vmin=3.0,
            vmax=vmax,
        )
        cbar = plt.colorbar(mesh, ax=ax, label="Wind Speed (m/s)")
        cbar.ax.tick_params(labelsize=self.fontsize)
        cbar.set_label("Wind Speed (m/s)", fontsize=self.fontsize)

        self._draw_turbines(
            ax,
            x_turb,
            y_turb,
            wt.diameter() / 2,
            yaw_plot,
            tilt,
            fontsize=self.fontsize,
            show_indices=self.show_indices,
        )

        ax.set_xlim(min(self.a), max(self.a))
        ax.set_ylim(min(self.b), max(self.b))
        ax.set_title(f"Flow field at {fs_use.time:.1f} s", fontsize=self.fontsize)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.fontsize,
            width=self.axes_lw,
            length=6,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(self.axes_lw)
        ax.set_xlabel("x (m)", fontsize=self.fontsize)
        ax.set_ylabel("y (m)", fontsize=self.fontsize)

        if probes:
            for probe in probes:
                x, y = self._probe_xy(probe, fs_use, fix_turbines)

                probe_type = probe.probe_type.upper()
                if probe_type == "WS":
                    color, label = "red", "WS Probe"
                    text = f"{float(probe.read()):.2f} m/s"
                elif probe_type == "TI":
                    color, label = "blue", "TI Probe"
                    text = f"{float(probe.read()):.2f} TI"
                else:
                    color, label, text = "gray", "Unknown", "N/A"

                ax.scatter(x, y, color=color, s=25, marker="o", label=label)
                ax.text(
                    x + 5,
                    y + 5,
                    text,
                    color="black",
                    fontsize=8,
                    bbox=dict(facecolor="none", alpha=0.6, edgecolor="none"),
                )

                speed = float(probe.read())
                inflow_angle = probe.get_inflow_angle_to_turbine()
                if fix_turbines:
                    inflow_angle += np.pi + np.deg2rad(90 - fs_use.wind_direction)
                arrow_length = speed * 5
                ax.arrow(
                    x,
                    y,
                    arrow_length * np.cos(inflow_angle),
                    arrow_length * np.sin(inflow_angle),
                    width=1.5,
                    head_width=5.0,
                    head_length=7.0,
                    fc=color,
                    ec=color,
                    alpha=0.8,
                    length_includes_head=True,
                )

            handles, labels_list = ax.get_legend_handles_labels()
            if labels_list.count("Probe") > 1:
                unique = dict(zip(labels_list, handles))
                ax.legend(unique.values(), unique.keys())

        return self._fig_to_rgb(fig)

    def get_flow_field(self, fs, probes=None, turbine=None, fix_turbines=False):
        """
        Return raw flow field data for custom plotting.

        Instead of rendering to pixels, this gives you the underlying arrays
        so you can build your own matplotlib figures with full control over
        colorbars, probe overlays, annotations, etc.

        Args:
            fs: Flow simulation object
            probes: Optional list of wind probes
            turbine: Turbine object (only needed if view not yet initialized)
            fix_turbines: If True, use EastNorthView so the farm layout stays
                fixed on screen and the wind/wake pattern rotates with changing
                wind direction. If False (default), use XYView (raw simulation
                coordinates) where the farm rotates and wind always comes from
                the left.

        Returns:
            dict with keys:
                uvw       - xarray DataArray (shape [3, nx, ny]), wind speed components
                x         - 2D np.ndarray, x-axis grid coordinates [m]
                y         - 2D np.ndarray, y-axis grid coordinates [m]
                x_turb    - 1D np.ndarray, turbine x positions [m]
                y_turb    - 1D np.ndarray, turbine y positions [m]
                wd        - float, wind direction [deg]
                yaw       - 1D np.ndarray, turbine yaw angles [deg] (plot-frame)
                time      - float, simulation time [s]
                probes              - list of probe objects (same as input), or []
                probe_inflow_angles - list of floats, inflow angle [rad] per probe (plot-frame corrected)
                wind_turbines       - WindTurbines instance
        """
        self._ensure_initialized(fs, turbine)

        view, X, Y, x_turb, y_turb, yaw, tilt, yaw_plot = self._build_view(
            fs, fix_turbines
        )
        uvw = fs.get_windspeed(view, include_wakes=True, xarray=True)

        probe_list = probes or []
        probe_positions = [self._probe_xy(p, fs, fix_turbines) for p in probe_list]
        probe_inflow_angles = []
        for p in probe_list:
            angle = p.get_inflow_angle_to_turbine()
            if fix_turbines:
                angle += np.pi + np.deg2rad(90 - fs.wind_direction)
            probe_inflow_angles.append(angle)

        return {
            "uvw": uvw,
            "x": X,
            "y": Y,
            "x_turb": x_turb,
            "y_turb": y_turb,
            "wd": fs.wind_direction,
            "yaw": yaw,
            "yaw_plot": yaw_plot,
            "tilt": tilt,
            "diameter": fs.windTurbines.diameter(),
            "time": fs.time,
            "probes": probe_list,
            "probe_positions": probe_positions,
            "probe_inflow_angles": probe_inflow_angles,
            "wind_turbines": fs.windTurbines,
            "fix_turbines": fix_turbines,
        }

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

        fs_use = self._resolve_fs(fs, fs_baseline, baseline)

        view, X, Y, x_turb, y_turb, _, tilt, yaw_plot = self._build_view(
            fs_use, fix_turbines
        )
        uvw = fs_use.get_windspeed(view, include_wakes=True, xarray=True)

        ax1.pcolormesh(X, Y, uvw[0].T, shading="nearest")

        wt = fs_use.windTurbines
        self._draw_turbines(
            ax1,
            x_turb,
            y_turb,
            wt.diameter() / 2,
            yaw_plot,
            tilt,
            show_indices=self.show_indices,
        )

        ax1.set_xlim(min(self.a), max(self.a))
        ax1.set_ylim(min(self.b), max(self.b))
        ax1.set_title("Flow field at {} s".format(fs_use.time))
        ax1.set_aspect("equal", adjustable="box")

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
        # close("all") so human-mode figures are released too, not just the
        # current figure.
        plt.close("all")
        self.view = None
