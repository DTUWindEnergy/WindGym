import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

"""
This module provides functions to generate different layouts for wind farms.

Generate_circular_farm functions create circular arrangements of turbines,
- Originally coded by #JensMVPeter 
"""


def generate_square_grid(turbine, nx, ny, xDist, yDist):
    """
    Create a square grid of turbines.

    Parameters
    ----------
    turbine : WindTurbine
        The wind turbine object.
    nx : int
        Number of turbines in the x-direction.
    ny : int
        Number of turbines in the y-direction.
    xDist : float
        Diameter distance between turbines in the x-direction.
    yDist : float
        Diameter distance between turbines in the y-direction.

    Returns
    -------
    np.ndarray
        Array of turbine positions.
    """
    D = turbine.diameter()
    x = np.linspace(0, D * xDist * (nx - 1), nx)
    y = np.linspace(0, D * yDist * (ny - 1), ny)
    xv, yv = np.meshgrid(x, y, indexing="xy")

    x_pos = xv.flatten()
    y_pos = yv.flatten()

    return x_pos, y_pos


def generate_right_triangle_grid(
    turbine,
    nx: int,
    ny: int,
    xDist: float,
    yDist: float,
    orientation: str = "lower_left",
):
    """
    Create a right-triangle (right-angle) grid of turbines.

    The triangle is created by stacking columns. Column i (0-indexed) contains
    min(i+1, ny) turbines (so the first column has 1, the second 2, ... up to ny).
    Legs of the right triangle lie along the x- and y-axes for the "lower_left"
    orientation. Other orientations are produced by reflections.

    Parameters
    ----------
    turbine : WindTurbine
        The wind turbine object.
    nx : int
        Number of columns along the x-direction (length of one leg).
    ny : int
        Maximum number of turbines along the y-direction (length of the other leg).
    xDist : float
        Distance between turbines in the x-direction, in rotor diameters.
    yDist : float
        Distance between turbines in the y-direction, in rotor diameters.
    orientation : str
        One of "lower_left" (default), "lower_right", "upper_left", "upper_right".
        Controls where the right angle is placed.

    Returns
    -------
    np.ndarray, np.ndarray
        Arrays of x and y positions.
    """
    if nx <= 0 or ny <= 0:
        return np.array([]), np.array([])

    D = turbine.diameter()
    x_spacing = xDist * D
    y_spacing = yDist * D

    x_list = []
    y_list = []

    # Build a triangle with the right angle at the lower-left corner:
    for i in range(nx):
        count = min(i + 1, ny)  # number of turbines in this column
        for j in range(count):
            x_list.append(i * x_spacing)
            y_list.append(j * y_spacing)

    x_arr = np.array(x_list, dtype=float)
    y_arr = np.array(y_list, dtype=float)

    # Reflect according to orientation
    max_x = (nx - 1) * x_spacing
    max_y = (ny - 1) * y_spacing

    if orientation == "lower_left":
        return x_arr, y_arr
    elif orientation == "lower_right":
        return max_x - x_arr, y_arr
    elif orientation == "upper_left":
        return x_arr, max_y - y_arr
    elif orientation == "upper_right":
        return max_x - x_arr, max_y - y_arr
    else:
        raise ValueError(
            "orientation must be one of 'lower_left','lower_right','upper_left','upper_right'"
        )


def generate_circle(n, r, angle_offset=0):
    """
    Generate a circular grid of n points with radius r.
    """
    if n > 1:
        t = np.linspace(0, 2 * np.pi, n + 1)[:n]
    else:
        t = [0]
    t += np.deg2rad(angle_offset)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def generate_cirular_farm(
    n_list: ArrayLike, turbine, r_dist: float = 5, angle_offset_list: ArrayLike = None
):
    """
    Generate a circular farm of n circular grids with radius r and m points.
    """
    D = turbine.diameter()
    r_list = np.arange(len(n_list), dtype=float)
    if n_list[0] != 1:
        r_list = r_list + 0.5
    r_list *= r_dist * D  # Diameter factors

    x = []
    y = []
    if angle_offset_list is None:
        angle_offset_list = np.zeros(len(n_list))
    for n, r, angle_offset in zip(n_list, r_list, angle_offset_list):
        x_, y_ = generate_circle(n, r, angle_offset)
        x.append(x_)
        y.append(y_)
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y


def generate_staggered_grid(
    turbine, nx, ny, xDist, yDist, x_stagger_offset=None, y_stagger_offset=None
):
    """
    Create a staggered grid of turbines with column- or row-based offsets.

    Parameters
    ----------
    turbine : WindTurbine
        The wind turbine object.
    nx : int
        Number of turbines in the x-direction.
    ny : int
        Number of turbines in the y-direction.
    xDist : float
        Distance between turbines in the x-direction, in rotor diameters.
    yDist : float
        Distance between turbines in the y-direction, in rotor diameters.
    x_stagger_offset : list[float] or None
        List of horizontal offsets (in rotor diameters) per column.
    y_stagger_offset : list[float] or None
        List of vertical offsets (in rotor diameters) per column.

    Returns
    -------
    np.ndarray
        Array of turbine positions.
    """
    D = turbine.diameter()
    x_list = []
    y_list = []

    # Ensure offset lists are the correct length or default to zeros
    if x_stagger_offset is None:
        x_stagger_offset = [0.0] * nx
    if y_stagger_offset is None:
        y_stagger_offset = [0.0] * nx

    for i in range(nx):
        for j in range(ny):
            x = i * xDist * D + x_stagger_offset[i] * D
            y = j * yDist * D + y_stagger_offset[i] * D
            x_list.append(x)
            y_list.append(y)

    return np.array(x_list), np.array(y_list)


def generate_line_dots_multiple_thetas(X, spacing, thetas, turbine):
    """
    Generate X evenly spaced dots on straight lines for multiple angles.
    Uses a set to ensure points are unique.

    Returns:
        xs : np.ndarray of x positions
        ys : np.ndarray of y positions
    """
    unique_points = set()

    D = turbine.diameter()
    spacing *= D  # Convert spacing to meters
    for theta in thetas:
        theta_rad = np.deg2rad(theta)

        # Step along the line direction
        dx = spacing * np.cos(theta_rad)
        dy = spacing * np.sin(theta_rad)

        # Center the line
        x0 = -dx * (X - 1) / 2
        y0 = -dy * (X - 1) / 2

        xs = x0 + dx * np.arange(X)
        ys = y0 + dy * np.arange(X)

        # Add rounded points to ensure uniqueness
        for x, y in zip(xs, ys):
            unique_points.add((round(x, 1), round(y, 1)))

    # Convert the set to arrays
    unique_points = np.array(list(unique_points))
    xs = unique_points[:, 0]
    ys = unique_points[:, 1]

    return xs, ys


def generate_diamond_grid(turbine, n, xDist, yDist):
    """
    Create a diamond-shaped grid of turbines.

    The diamond has a single turbine at the top and bottom, widening to
    a maximum width of *n* turbines at the middle row. The total number
    of rows is ``2*n - 1``.

    Parameters
    ----------
    turbine : WindTurbine
        The wind turbine object.
    n : int
        Half-width of the diamond (number of turbines in the widest row).
    xDist : float
        Distance between turbines in the x-direction, in rotor diameters.
    yDist : float
        Distance between turbines in the y-direction, in rotor diameters.

    Returns
    -------
    np.ndarray, np.ndarray
        Arrays of x and y positions.
    """
    if n <= 0:
        return np.array([]), np.array([])

    D = turbine.diameter()
    x_spacing = xDist * D
    y_spacing = yDist * D

    x_list = []
    y_list = []

    # 2*n - 1 rows: row 0 has 1 turbine, row n-1 has n, row 2*n-2 has 1
    num_rows = 2 * n - 1
    for row in range(num_rows):
        # Number of turbines in this row
        count = n - abs(row - (n - 1))
        # Center the row horizontally
        x_offset = -(count - 1) / 2.0 * x_spacing
        for col in range(count):
            x_list.append(x_offset + col * x_spacing)
            y_list.append(row * y_spacing)

    return np.array(x_list), np.array(y_list)


def plot_farm(x, y, turbine=None, D=None):
    """
    Plots the turbines in the farm layout, and their minimum distance to the closest turbine
    """
    if D is None and turbine is None:
        D = 1.0  # Default diameter if not provided
    elif turbine is not None:
        D = turbine.diameter()

    min_distances = []
    for i in range(len(x)):
        distances = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
        distances[i] = np.inf  # Ignore self-distance
        min_distance = np.min(distances)
        min_distances.append(min_distance)
    min_distances = np.array(min_distances) / D  # Convert to diameter units
    plt.scatter(x, y, c=min_distances, cmap="viridis", label="Min Distance (m)")
    plt.colorbar(label="Min Distance (Diameter Units)")
    # plt.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', label='Turbines'),
    #                     Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='Min Distance')])
    plt.show()
