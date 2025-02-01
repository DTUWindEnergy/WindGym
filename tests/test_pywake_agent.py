from DTUWindGym.envs.WindFarmEnv.Agents import PyWakeAgent
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


def test_power_optimization():
    # Initialize parameters
    x_pos = [0, 500]
    y_pos = [0, 0]
    wind_speed = 6
    wind_dir = 270
    TI = 0.02

    # Setup wind farm models
    site = LillgrundSite()
    turbine = V80()
    wf_model = Blondel_Cathelain_2020(
        site,
        turbine,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
    )

    def compute_power(yaws):
        return wf_model(
            x=x_pos, y=y_pos, ws=wind_speed, wd=wind_dir, TI=TI, tilt=0, yaw=yaws
        ).Power.sum()

    # Compute nominal power
    nominal_power = compute_power([30, 0])

    # Run optimization
    agent = PyWakeAgent(
        x_pos=x_pos, y_pos=y_pos, wind_speed=wind_speed, wind_dir=wind_dir, TI=TI
    )
    agent.optimize()
    agent_power = compute_power(agent.optimized_yaws)

    # Assert optimization improved power
    assert agent_power >= nominal_power
