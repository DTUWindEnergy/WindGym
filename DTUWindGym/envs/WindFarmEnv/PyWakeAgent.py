import numpy as np
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
# from py_wake.deficit_models.gaussian import BastankhahGaussian, BlondelSuperGaussianDeficit2023

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from py_wake.examples.data.hornsrev1 import V80

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

import matplotlib.pyplot as plt

class PyWakeAgent():
    def __init__(self, x_pos, y_pos,
                 wind_speed = 8, wind_dir = 270, TI=0.07,yaw_max=45, 
                 yaw_min=-45, maxiter=100, tol=1e-4, ec=1e-4, 
                 turbine = V80()):
        self.pywakeagent = True #This is used in a hasattr. Dont ask me why. Im bad at python. 
        self.optimized = False #Is false before we have optimized the farm.
        self.maxiter=maxiter
        self.tol=tol
        self.ec=ec
        self.yaw_max=yaw_max
        self.yaw_min=yaw_min
        #choosing the flow cases for the optimization
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_dir])
        self.TI = TI

        #Define the farm. 
        site = LillgrundSite()
        self.turbine = turbine

        self.x_pos = x_pos
        self.y_pos = y_pos #Note we move the farm 200 units up. This is done because I think I saw some weird behaviour with y = 0 :/ 
        self.n_wt = len(x_pos)

        site.initial_position = np.array([x_pos, y_pos]).T

        self.wf_model = Blondel_Cathelain_2020(site, turbine, turbulenceModel=CrespoHernandez(), deflectionModel=JimenezWakeDeflection())

        self.yaw_zero = np.zeros((self.n_wt,1,1))  #initial condition of yaw angles
        #Define the farm, the wind farm model and the initial yaw angles
        self.reset()

    def power_func(self, yaw_ilk):
        simres = self.wf_model(self.x_pos, self.y_pos, wd=self.wdir, 
                                 ws=self.wsp, yaw=yaw_ilk, tilt=0)
        # power = simres.Power.sum()
        aep = simres.aep().sum()
        return aep

    def update_wind(self, wind_speed, wind_direction, TI):
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_direction])
        self.TI = TI
        self.reset()

    def reset(self):
        #Reset the wind things for the objective. 
        self.optimized = False

        self.pywakefarm = self.wf_model(self.x_pos, self.y_pos, 
                                wd=self.wdir, ws=self.wsp, 
                                yaw=self.yaw_zero, tilt=0, TI=self.TI)
        
        self.cost_comp = CostModelComponent(input_keys=[('yaw_ilk', np.zeros((self.n_wt, 1, 1)))],
                                                n_wt = self.n_wt,
                                                cost_function = self.power_func,
                                                objective=True,
                                                maximize=True,
                                                output_keys=[('AEP', 0)]
                                                )

        self.problem = TopFarmProblem(design_vars={'yaw_ilk': (self.yaw_zero, self.yaw_min, self.yaw_max)},  #setting up initial values and lower and upper bounds for yaw angles
                                n_wt=self.n_wt,
                                cost_comp=self.cost_comp,
                                driver=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=self.maxiter, tol=self.tol),
                                plot_comp=NoPlot(),
                                )

    def optimize(self):
        _, state, self.info = self.problem.optimize()
        self.optimized = True #Now the farm has been optimized
        self.optimized_yaws = state['yaw_ilk'][:,0,0]

    def update_wind(self, wind_speed, wind_direction, TI):
        #Update the windconditions for the agent. Then reset the agent, so it is ready for optimization. 
        self.wsp = np.asarray([wind_speed])
        self.wdir = np.asarray([wind_direction])
        self.TI = TI
        self.reset()

    def scale_yaw(self):
        #Make the scaling of the yaw angles.
        #Scaled from [max, min] to [-1, 1]
        self.action = (self.optimized_yaws-self.yaw_min)/(self.yaw_max-self.yaw_min)*2-1

    def predict(self, *args, **kwargs): #This is to make the agent compatible with the agent framework.
        #Note that we dont use the obs or the deterministic arguments.
        if self.optimized == False:
            self.optimize()
            self.scale_yaw()
        return self.action, None #Return the optimized (scaled) yaws and None as to make it fit with the agent framework.
    
    def plot_flow(self):
        if self.optimized == False:
            self.optimize()
        simulationResult = self.wf_model(self.x_pos,self.y_pos,wd=self.wdir, ws=self.wsp, yaw=self.optimized_yaws, tilt=0)
        plt.figure(figsize=(12,4))
        simulationResult.flow_map().plot_wake_map()
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()