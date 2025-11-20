"""
Turbine-level plotting functions for wind farm evaluation results.
"""

import matplotlib.pyplot as plt
from .plot_utils import calculate_time_limits


def plot_power_turb(
    data,
    ws,
    WDS,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the power output for each turbine in the farm.

    Args:
        data: xarray Dataset with evaluation results
        ws: Wind speed to plot
        WDS: List of wind directions to plot
        avg_n: Rolling average window size
        TI: Turbulence intensity
        TURBBOX: Turbulence box identifier
        axs: Matplotlib axes array (optional)
        save: Whether to save the figure
        save_path: Path to save figure (uses default if None)

    Returns:
        Tuple of (fig, axs)
    """
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
            ).powerT_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(
                x="time", label="Agent", ax=axs[i, j]
            )
            data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                turb=i
            ).powerT_b.rolling(time=avg_n, center=True).mean().plot.line(
                x="time", label="Baseline", ax=axs[i, j]
            )
            axs[i, j].set_title(f"WD ={wd}, Turbine {i}")

            x_start, x_end = calculate_time_limits(
                data, ws, wd, TI, TURBBOX, "powerT_a", avg_n, turb_idx=0
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
        save_path = save_path or "power_turbines.png"
        plt.savefig(save_path)

    return fig, axs


def plot_yaw_turb(
    data,
    ws,
    WDS,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the yaw angle for each turbine in the farm.

    Args:
        data: xarray Dataset with evaluation results
        ws: Wind speed to plot
        WDS: List of wind directions to plot
        avg_n: Rolling average window size
        TI: Turbulence intensity
        TURBBOX: Turbulence box identifier
        axs: Matplotlib axes array (optional)
        save: Whether to save the figure
        save_path: Path to save figure (uses default if None)

    Returns:
        Tuple of (fig, axs)
    """
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
            ).yaw_a.rolling(time=avg_n, center=True).mean().dropna("time").plot.line(
                x="time", label="Agent", ax=axs[i, j]
            )
            data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                turb=i
            ).yaw_b.rolling(time=avg_n, center=True).mean().plot.line(
                x="time", label="Baseline", ax=axs[i, j]
            )
            axs[i, j].set_title(f"WD ={wd}, Turbine {i}")

            x_start, x_end = calculate_time_limits(
                data, ws, wd, TI, TURBBOX, "yaw_a", avg_n, turb_idx=0
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
        save_path = save_path or "yaw_turbines.png"
        plt.savefig(save_path)

    return fig, axs


def plot_speed_turb(
    data,
    ws,
    WDS,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the rotor wind speed for each turbine in the farm.

    Args:
        data: xarray Dataset with evaluation results
        ws: Wind speed to plot
        WDS: List of wind directions to plot
        avg_n: Rolling average window size
        TI: Turbulence intensity
        TURBBOX: Turbulence box identifier
        axs: Matplotlib axes array (optional)
        save: Whether to save the figure
        save_path: Path to save figure (uses default if None)

    Returns:
        Tuple of (fig, axs)
    """
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

            x_start, x_end = calculate_time_limits(
                data, ws, wd, TI, TURBBOX, "ws_a", avg_n, turb_idx=0
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
        save_path = save_path or "windspeed_turbines.png"
        plt.savefig(save_path)

    return fig, axs


def plot_turb(
    data,
    ws,
    wd,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the power, yaw and rotor wind speed for each turbine in the farm.

    Args:
        data: xarray Dataset with evaluation results
        ws: Wind speed to plot
        wd: Wind direction to plot
        avg_n: Rolling average window size
        TI: Turbulence intensity
        TURBBOX: Turbulence box identifier
        axs: Matplotlib axes array (optional)
        save: Whether to save the figure
        save_path: Path to save figure (uses default if None)

    Returns:
        Tuple of (fig, axs)
    """
    n_turb = len(data.turb.values)  # The number of turbines in the farm

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
            elif plot_var == "Yaw":
                to_plot = "yaw_"
                plot_title = "Yaw offset [deg]"
            elif plot_var == "Rotor wind speed":
                to_plot = "ws_"
                plot_title = "Rotor wind speed [m/s]"

            # Set the y axis to be shared between the different plots
            axs[i, j].sharey(axs[0, j])

            # Plot the data
            data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                turb=i
            ).data_vars[to_plot + "a"].rolling(time=avg_n, center=True).mean().dropna(
                "time"
            ).plot.line(x="time", label="Agent", ax=axs[i, j])
            data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).sel(
                turb=i
            ).data_vars[to_plot + "b"].rolling(time=avg_n, center=True).mean().dropna(
                "time"
            ).plot.line(x="time", label="Baseline", ax=axs[i, j])

            # Set the title of the plot
            if i == 0:
                axs[i, j].set_title(plot_title)
            else:
                axs[i, j].set_title("")

            # Find at set the x-axis limits
            x_start, x_end = calculate_time_limits(
                data, ws, wd, TI, TURBBOX, to_plot + "a", avg_n, turb_idx=i
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
        save_path = save_path or "turbine_metrics.png"
        plt.savefig(save_path)

    return fig, axs
