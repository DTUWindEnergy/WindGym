import numpy as np
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.hornsrev1 import V80

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

import matplotlib.pyplot as plt
from .BaseAgent import BaseAgent
import warnings

"""
The PyWakeAgent is a class that is used to optimize the yaw angles of a wind farm using the PyWake library.
It interfaces with the AgentEval class in the dtu_wind_gym library.
Based on the global wind conditons it can optimize the yaw angles and then use them during the simulation.
"""


class PyWakeAgent(BaseAgent):
    def __init__(
        self,
        x_pos,
        y_pos,
        wind_speed=8,
        wind_dir=270,
        TI=0.07,
        yaw_max=45,
        yaw_min=-45,
        refine_pass_n=5,
        yaw_n=5,
        turbine=V80(),
    ):
        # This is used in a hasattr in the AgentEval class.
        self.pywakeagent = True
        self.optimized = False  # Is false before we have optimized the farm.
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        # choosing the flow cases for the optimization
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_dir])
        self.TI = TI

        self.refine_pass_n = refine_pass_n
        self.yaw_n = yaw_n

        # Define the farm.
        site = LillgrundSite()
        self.turbine = turbine

        # Check if x_pos or y_pos are lists if so then convert them to numpy arrays
        if isinstance(x_pos, list):  # pragma: no cover
            x_pos = np.array(x_pos)
        if isinstance(y_pos, list):  # pragma: no cover
            y_pos = np.array(y_pos)
        if len(x_pos) != len(y_pos):  # pragma: no cover
            raise ValueError("x_pos and y_pos must have the same length.")

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_wt = len(x_pos)

        site.initial_position = np.array([x_pos, y_pos]).T

        self.wf_model = Blondel_Cathelain_2020(
            site,
            turbine,
            turbulenceModel=CrespoHernandez(),
            deflectionModel=JimenezWakeDeflection(),
        )

        # initial condition of yaw angles
        self.yaw_zero = np.zeros((self.n_wt, 1, 1))
        self.reset()

    def update_wind(self, wind_speed, wind_direction, TI):
        """
        Update the wind conditions for the agent.
        """
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_direction])
        self.TI = TI
        self.reset()

    def reset(self):
        """
        Reset the wind things for the objective.
        """
        self.optimized = False

    def optimize(self):
        """
        Optimizes the yaw angles of the wind farm.
        """
        yaws = yaw_optimizer_srf_vect(
            x=self.x_pos,
            y=self.y_pos,
            wffm=self.wf_model,
            wd=self.wdir,
            ws=self.wsp,
            ti=self.TI,
            refine_pass_n=self.refine_pass_n,
            yaw_n=self.yaw_n,
            nn_cpu=1,
            sort_reverse=False,
        )

        self.optimized_yaws = yaws.squeeze()

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        if self.optimized is False:
            self.optimize()
            self.action = self.scale_yaw(self.optimized_yaws)

        return self.action, None

    def plot_flow(self):
        """
        Plot the flowfield of the wind farm.
        """
        if self.optimized is False:
            self.optimize()
        simulationResult = self.wf_model(
            self.x_pos,
            self.y_pos,
            wd=self.wdir,
            ws=self.wsp,
            yaw=self.optimized_yaws,
            tilt=0,
        )
        plt.figure(figsize=(12, 4))
        simulationResult.flow_map().plot_wake_map()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()


def yaw_optimizer_srf_vect(
    x, y, wffm, wd, ws, ti=0.04, refine_pass_n=4, yaw_n=5, nn_cpu=1, sort_reverse=False
):
    """
    This is the Serial-Refine Method for yaw optimization, implemented in PyWake.
    This was done by Deniz. I just copied the function into this file.

    Optimizes turbine yaw angles over arrays of wind directions and wind speeds
    using a Serial-Refine Method for PyWake.

    This version vectorizes the wind direction (wd) dimension by evaluating the candidate
    yaw configurations for all wind directions at once. The candidate offset dimension is looped
    over (typically small), while wd is fed in vectorized.

    Parameters
    ----------
    x : array_like, shape (n_wt,)
        x coordinates of the turbines.
    y : array_like, shape (n_wt,)
        y coordinates of the turbines.
    wffm : EngineeringWindFarmModel Object
        PyWake wind farm flow model.
    wd : array_like or float
        Wind direction(s) (in meteorological convention, degrees).
    ws : array_like or float
        Wind speed(s) (m/s).
    ti : array_like or float
        Turbulence intensity.
    refine_pass_n : int, optional
        Number of refine passes.
    yaw_n : int, optional
        Number of candidate yaw offsets to test at each update step.
    nn_cpu : int, optional
        Number of CPUs to use.
    sort_reverse : bool, optional
        Whether to reverse turbine sorting (downstream-to-upstream).

    Returns
    -------
    yaw_opt : ndarray, shape (n_wt, n_wd, n_ws)
        The optimized yaw angles for each turbine, wind direction, and wind speed.
    """
    yaw_max = 30  # Maximum yaw angle (degrees)
    # add delta to wd to fix perfect alignment cases with 2 maxima
    wd = np.atleast_1d(wd) + 1e-3  # shape: (n_wd,)
    ws = np.atleast_1d(ws)  # shape: (n_ws,)
    ti = np.atleast_1d(ti)
    n_wt = len(x)
    n_wd = len(wd)
    n_ws = len(ws)

    # Initialize yaw angles: shape (n_wt, n_wd, n_ws)
    yaw_opt = np.zeros((n_wt, n_wd, n_ws))

    # Compute baseline power for all conditions.
    po = wffm(x, y, wd=wd, ws=ws, TI=ti, yaw=yaw_opt, tilt=0, n_cpu=nn_cpu).Power.values
    # Sum power over turbines -> shape (n_wd, n_ws)
    power_current = np.sum(po, axis=0)

    # Compute turbine ordering for each wind direction.
    # Convert meteorological wd to mathematical angle (so that x aligns with wind)
    theta = np.radians((270 - wd) % 360)  # shape: (n_wd,)
    # Compute rotated x-coordinate: shape (n_wd, n_wt)
    x_rotated = (
        x[None, :] * np.cos(theta)[:, None] + y[None, :] * np.sin(theta)[:, None]
    )
    # For each wind direction, sort turbines upstream-to-downstream.
    turbines_ordered = np.argsort(x_rotated, axis=1)  # shape: (n_wd, n_wt)
    if sort_reverse:
        turbines_ordered = np.argsort(-x_rotated, axis=1)
        # print('serial-refine sorting upstream to downstream')

    current_offset_range = yaw_max

    # Begin refine passes.
    for s in range(refine_pass_n):
        # print(f"Serial refine pass {s + 1}/{refine_pass_n}")
        # Create a symmetric grid of candidate offsets.
        candidate_offsets = np.linspace(
            -current_offset_range, current_offset_range, yaw_n
        )
        current_offset_range /= 2.0

        # Loop over turbine ordering positions.
        for pos in range(n_wt):
            # For each wd, select the turbine at ordering position "pos"
            turb_idx = turbines_ordered[:, pos]  # shape: (n_wd,)
            # Get current yaw for these turbines (for each wd): shape (n_wd, n_ws)
            current_yaws = yaw_opt[turb_idx, np.arange(n_wd), :]
            # Compute candidate yaw values: for each wd, candidate_yaws has shape (yaw_n, n_ws)
            candidate_yaws = current_yaws[:, None, :] + candidate_offsets[None, :, None]

            # Prepare to store candidate total power (summed over turbines) for each wd and candidate.
            candidate_power = np.empty((n_wd, yaw_n, n_ws))

            # Loop over candidate offsets (yaw_n is typically small)
            for j in range(yaw_n):
                # Create a candidate yaw configuration for all turbines, for all wd and ws.
                candidate_yaw_config = np.copy(yaw_opt)  # shape: (n_wt, n_wd, n_ws)
                # For each wd, update only the turbine being updated with its candidate yaw.
                candidate_yaw_config[turb_idx, np.arange(n_wd), :] = candidate_yaws[
                    :, j, :
                ]

                # Evaluate the candidate configuration for all wd at once.
                p = wffm(
                    x,
                    y,
                    wd=wd,
                    ws=ws,
                    TI=ti,
                    yaw=candidate_yaw_config,
                    tilt=0,
                    n_cpu=nn_cpu,
                ).Power.values
                # Sum power over turbines: shape (n_wd, n_ws)
                candidate_power[:, j, :] = np.sum(p, axis=0)

            # For each wd and wind speed, select the candidate offset that yields the highest power.
            best_candidate_idx = np.argmax(
                candidate_power, axis=1
            )  # shape: (n_wd, n_ws)
            best_candidate_power = np.max(
                candidate_power, axis=1
            )  # shape: (n_wd, n_ws)

            # Update yaw if improvement is found.
            ws_idx = np.arange(n_ws)
            for i in range(n_wd):
                improvement_mask = best_candidate_power[i] > power_current[i]
                if np.any(improvement_mask):
                    # For element-wise selection, index with ws_idx.
                    best_yaws = candidate_yaws[
                        i, best_candidate_idx[i], ws_idx
                    ]  # shape: (n_ws,)
                    yaw_opt[turb_idx[i], i, improvement_mask] = best_yaws[
                        improvement_mask
                    ]
                    power_current[i, improvement_mask] = best_candidate_power[
                        i, improvement_mask
                    ]
    if np.any(yaw_opt < -yaw_max) or np.any(yaw_opt > yaw_max):
        warnings.warn("Optimal setpoints outside [-yaw_max, yaw_max] range.")
        yaw_opt = np.clip(yaw_opt, a_min=-yaw_max, a_max=yaw_max)
    return yaw_opt
