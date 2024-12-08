import numpy as np
from py_wake.examples.data.hornsrev1 import V80

import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz
from floris import FlorisModel, TimeSeries
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.turbine_library import build_cosine_loss_turbine_dict

import matplotlib.pyplot as plt
from .BaseAgent import BaseAgent
import os
"""
The FlorisAgent is a class that is used to optimize the yaw angles of a wind farm using the PyWake library.
It interfaces with the AgentEval class in the dtu_wind_gym library.
Based on the global wind conditons it can optimize the yaw angles and then use them during the simulation.
"""


class FlorisAgent(BaseAgent):
    def __init__(self, x_pos, y_pos,
                 wind_speed=8, wind_dir=270, TI=0.07, yaw_max=45,
                 yaw_min=-45,
                 turbine=V80()):
        # This is used in a hasattr in the AgentEval class.
        self.florisagent = True
        self.optimized = False  # Is false before we have optimized the farm.

        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        # choosing the flow cases for the optimization
        self.wsp = wind_speed
        self.wdir = wind_dir
        self.TI = TI

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_wt = len(x_pos)

        # Define the floris farm:
        current_path = os.path.dirname(os.path.realpath(__file__))
        # print("The current path is: ", current_path)
        self.fmodel = FlorisModel(f"{current_path}/gch.yaml")

        # Turn the pywake turbine into a floris turbine
        wind_speeds = np.linspace(1, 30, 100)

        # Generate an example turbine power and thrust curve for use in the FLORIS model
        pywake_power = turbine.power(wind_speeds)
        power_coeffs = pywake_power[1:] / (0.5 * turbine.diameter()
                                           ** 2 * np.pi / 4 * 1.225 * wind_speeds[1:] ** 3)
        turbine_data_dict = {
            "wind_speed": list(wind_speeds),
            "power_coefficient": [0] + list(power_coeffs),
            "thrust_coefficient": list(turbine.ct(wind_speeds)),
        }

        turbine_dict = build_cosine_loss_turbine_dict(
            turbine_data_dict,
            turbine.name(),
            file_name=None,
            generator_efficiency=1,
            hub_height=turbine.hub_height(),
            cosine_loss_exponent_yaw=1.88,
            cosine_loss_exponent_tilt=1.88,
            rotor_diameter=turbine.diameter(),
            TSR=8,
            ref_air_density=1.225,
            ref_tilt=5,
        )

        # Replace the turbine(s) in the FLORIS model with the created one
        self.fmodel.set(
            layout_x=self.x_pos,
            layout_y=self.y_pos,
            wind_directions=np.array([self.wdir]),
            wind_speeds=np.array([self.wsp]),
            turbulence_intensities=np.array([self.TI]),
            turbine_type=[turbine_dict],
            reference_wind_height=self.fmodel.reference_wind_height
        )

        # initial condition of yaw angles
        self.yaw_zero = np.zeros((self.n_wt, 1, 1))
        self.reset()

    def update_wind(self, wind_speed, wind_direction, TI):
        """
        Update the wind conditions for the agent.
        """
        self.wsp = wind_speed
        self.wdir = wind_direction
        self.TI = TI
        self.reset()

    def reset(self):
        """
        Reset the wind things for the objective. 
        """
        self.optimized = False

        self.fmodel.set(
            layout_x=self.x_pos,
            layout_y=self.y_pos,
            wind_directions=np.array([self.wdir]),
            wind_speeds=np.array([self.wsp]),
            turbulence_intensities=np.array([self.TI]),
        )

    def optimize(self):
        """
        Optimizes the yaw angles of the wind farm.
        """
        # Initialize optimizer object and run optimization using the Serial-Refine method
        yaw_opt = YawOptimizationSR(
            self.fmodel, maximum_yaw_angle=self.yaw_max, minimum_yaw_angle=self.yaw_min)

        df_opt = yaw_opt.optimize(print_progress=False)

        self.optimized = True  # Now the farm has been optimized
        self.optimized_yaws = df_opt["yaw_angles_opt"][0]

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        if self.optimized == False:
            self.optimize()
            self.action = self.scale_yaw(self.optimized_yaws)

        return self.action, None

    def plot_flow(self):
        """
        Plot the flowfield of the wind farm.
        """
        if self.optimized == False:
            self.optimize()

        fig, ax = plt.subplots()
        horizontal_plane = self.fmodel.calculate_horizontal_plane(
            height=self.fmodel.core.farm.hub_heights[0])
        flowviz.visualize_cut_plane(horizontal_plane, ax=ax)
        layoutviz.plot_turbine_rotors(
            self.fmodel, ax=ax, yaw_angles=self.optimized_yaws)
        ax.set_title("Optimized yaw angles")
        plt.show()
