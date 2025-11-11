"""
Farm-level plotting functions for wind farm evaluation results.
"""

import matplotlib.pyplot as plt
from .plot_utils import setup_wind_grid_axes, calculate_time_limits


def plot_power_farm(
    data,
    WSS,
    WDS,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the power output for the farm.

    Args:
        data: xarray Dataset with evaluation results
        WSS: List of wind speeds to plot
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
            data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_a.rolling(
                time=avg_n, center=True
            ).mean().dropna("time").plot.line(x="time", label="Agent", ax=axs[j, i])
            data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).powerF_b.rolling(
                time=avg_n, center=True
            ).mean().plot.line(x="time", label="Baseline", ax=axs[j, i])

            setup_wind_grid_axes(axs, j, i, WSS, WDS, WS, wd)

            x_start, x_end = calculate_time_limits(
                data, WS, wd, TI, TURBBOX, "powerF_a", avg_n
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
        save_path = save_path or "power_farm.png"
        plt.savefig(save_path)

    return fig, axs


def plot_farm_inc(
    data,
    WSS,
    WDS,
    avg_n=10,
    TI=0.07,
    TURBBOX="Default",
    axs=None,
    save=False,
    save_path=None,
):  # pragma: no cover
    """
    Plot the percentage increase in power output for the farm.

    Args:
        data: xarray Dataset with evaluation results
        WSS: List of wind speeds to plot
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
            data.sel(ws=WS).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX).pct_inc.rolling(
                time=avg_n, center=True
            ).mean().dropna("time").plot.line(x="time", ax=axs[j, i])

            setup_wind_grid_axes(axs, j, i, WSS, WDS, WS, wd)

            x_start, x_end = calculate_time_limits(
                data, WS, wd, TI, TURBBOX, "pct_inc", avg_n
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
        save_path = save_path or "power_farm_increase.png"
        plt.savefig(save_path)

    return fig, axs
