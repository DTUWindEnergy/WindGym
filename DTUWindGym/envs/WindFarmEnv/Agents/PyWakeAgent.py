import numpy as np
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from py_wake.examples.data.hornsrev1 import V80

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

import matplotlib.pyplot as plt
from .BaseAgent import BaseAgent

"""
The PyWakeAgent is a class that is used to optimize the yaw angles of a wind farm using the PyWake library.
It interfaces with the AgentEval class in the dtu_wind_gym library.
Based on the global wind conditons it can optimize the yaw angles and then use them during the simulation.
"""


class PyWakeAgent(BaseAgent):
    def __init__(self, x_pos, y_pos,
                 wind_speed=8, wind_dir=270, TI=0.07, yaw_max=45,
                 yaw_min=-45, maxiter=100, tol=1e-4, ec=1e-4,
                 turbine=V80()):
        # This is used in a hasattr in the AgentEval class.
        self.pywakeagent = True
        self.optimized = False  # Is false before we have optimized the farm.
        self.maxiter = maxiter
        self.tol = tol
        self.ec = ec
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        # choosing the flow cases for the optimization
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_dir])
        self.TI = TI

        # Define the farm.
        site = LillgrundSite()
        self.turbine = turbine

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.n_wt = len(x_pos)

        site.initial_position = np.array([x_pos, y_pos]).T

        self.wf_model = Blondel_Cathelain_2020(
            site, turbine, turbulenceModel=CrespoHernandez(), deflectionModel=JimenezWakeDeflection())

        # initial condition of yaw angles
        self.yaw_zero = np.zeros((self.n_wt, 1, 1))
        self.reset()

    def power_func(self, yaw_ilk):
        """
        Function to calculate the power output of the wind farm.
        """
        simres = self.wf_model(self.x_pos, self.y_pos, wd=self.wdir,
                               ws=self.wsp, yaw=yaw_ilk, tilt=0)
        aep = simres.aep().sum()
        return aep

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

        self.pywakefarm = self.wf_model(self.x_pos, self.y_pos,
                                        wd=self.wdir, ws=self.wsp,
                                        yaw=self.yaw_zero, tilt=0, TI=self.TI)

        self.cost_comp = CostModelComponent(input_keys=[('yaw_ilk', np.zeros((self.n_wt, 1, 1)))],
                                            n_wt=self.n_wt,
                                            cost_function=self.power_func,
                                            objective=True,
                                            maximize=True,
                                            output_keys=[('AEP', 0)]
                                            )

        self.problem = TopFarmProblem(design_vars={'yaw_ilk': (self.yaw_zero, self.yaw_min, self.yaw_max)},  # setting up initial values and lower and upper bounds for yaw angles
                                      n_wt=self.n_wt,
                                      cost_comp=self.cost_comp,
                                      driver=EasyScipyOptimizeDriver(
                                          optimizer='COBYLA', maxiter=self.maxiter, tol=self.tol),
                                      plot_comp=NoPlot(),
                                      )

    def optimize(self):
        """
        Optimizes the yaw angles of the wind farm.
        """
        _, state, self.info = self.problem.optimize()
        self.optimized = True  # Now the farm has been optimized
        self.optimized_yaws = state['yaw_ilk'][:, 0, 0]

    def predict(self, *args, **kwargs):
        """
        This class pretends to be an agent, so we need to have a predict function.
        If we havent called the optimize function, we do that now, and return the action
        Note that we dont use the obs or the deterministic arguments.
        """

        if self.optimized == False:
            self.optimize()
            self.action = self.scale_yaw()

        return self.action, None

    def plot_flow(self):
        """
        Plot the flowfield of the wind farm.
        """
        if self.optimized == False:
            self.optimize()
        simulationResult = self.wf_model(
            self.x_pos, self.y_pos, wd=self.wdir, ws=self.wsp, yaw=self.optimized_yaws, tilt=0)
        plt.figure(figsize=(12, 4))
        simulationResult.flow_map().plot_wake_map()
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()
