"""
Shared plotting utilities for wind farm evaluation visualizations.
"""

import matplotlib.pyplot as plt
from typing import Tuple, Any


def setup_wind_grid_axes(
    axs: Any,
    j: int,
    i: int,
    WSS: list,
    WDS: list,
    WS: float,
    wd: float,
    add_grid: bool = True,
) -> None:
    """
    Configure axes for wind condition grid plots.

    Args:
        axs: Matplotlib axes array
        j: Row index
        i: Column index
        WSS: List of wind speeds
        WDS: List of wind directions
        WS: Current wind speed
        wd: Current wind direction
        add_grid: Whether to add grid lines
    """
    if j == 0:  # Only set the top row to have a title
        axs[j, i].set_title(f"WD ={wd} [deg]")
    else:
        axs[j, i].set_title("")

    if i == 0:  # Only set the left column to have a y-label
        axs[j, i].set_ylabel(f"WS ={WS} [m/s]")
    else:
        axs[j, i].set_ylabel("")

    axs[j, i].set_xlabel("")

    if add_grid:
        axs[j, i].grid()


def calculate_time_limits(
    data: Any,
    ws: float,
    wd: float,
    TI: float,
    TURBBOX: str,
    variable: str,
    avg_n: int = 10,
    turb_idx: int = None,
) -> Tuple[float, float]:
    """
    Calculate x-axis time limits for a given data selection.

    Args:
        data: xarray Dataset
        ws: Wind speed
        wd: Wind direction
        TI: Turbulence intensity
        TURBBOX: Turbulence box identifier
        variable: Variable name to calculate limits for
        avg_n: Rolling average window size
        turb_idx: Turbine index (None for farm-level data)

    Returns:
        Tuple of (x_start, x_end) time limits
    """
    selection = data.sel(ws=ws).sel(wd=wd).sel(TI=TI).sel(turbbox=TURBBOX)

    if turb_idx is not None:
        selection = selection.sel(turb=turb_idx)

    rolled = selection[variable].rolling(time=avg_n, center=True).mean().dropna("time")

    x_start = rolled.time.values.min()
    x_end = rolled.time.values.max()

    return x_start, x_end
