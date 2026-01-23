from pathlib import Path

import numpy as np
import pytest
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.hornsrev1 import V80
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

from WindGym import FarmEval
from WindGym.Agents import (
    BaseAgent,
    ConstantAgent,
    GreedyAgent,
    PyWakeAgent,
    RandomAgent,
)
from WindGym.utils.generate_layouts import generate_square_grid


@pytest.fixture
def base_example_data_path():
    """Provides path to the example configuration directory"""
    return Path("examples/EnvConfigs")


@pytest.fixture(params=["2turb.yaml", "Env1.yaml"])
def example_config_path(base_example_data_path, request):
    """Provides paths to multiple example configurations, testing environment with different setups"""
    return base_example_data_path / request.param


@pytest.mark.slow
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
        x_pos=x_pos,
        y_pos=y_pos,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        TI=TI,
        look_up=True,
    )
    agent.plot_flow(show=False)  # runs optimize
    agent_power = compute_power(agent.optimized_yaws)

    # Assert optimization improved power
    assert agent_power >= nominal_power


def test_greedy_agent_local_controller(base_example_data_path):
    """
    Tests the GreedyAgent with the 'local' controller type.
    Verifies that the predict method returns a correctly shaped and scaled action.
    """
    yaml_path = base_example_data_path / Path("Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=2, yDist=2)
    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        yaw_init="Zeros",  # Start with zero yaw offset
        seed=0,
        Baseline_comp=False,
        turbtype="None",  # Key optimization for speed
        n_passthrough=0.1,  # Very short episodes for fast tests
        burn_in_passthroughs=0.0001,
    )
    env.reset()  # Initialize flow simulation `fs`

    agent = GreedyAgent(
        env=env,
        type="local",
        yaw_max=env.yaw_max,
        yaw_min=env.yaw_min,
        yaw_step=env.yaw_step_sim,
    )

    action, state = agent.predict()

    assert (
        state is None
    ), "GreedyAgent's predict method should return None for the state."
    assert isinstance(action, np.ndarray), "Action should be a numpy array."
    assert (
        action.shape == env.action_space.shape
    ), f"Action shape mismatch. Expected {env.action_space.shape}, got {action.shape}."
    assert np.all(action >= -1) and np.all(
        action <= 1
    ), "Action values should be scaled between -1 and 1."
    env.close()


def test_greedy_agent_global_controller(base_example_data_path):
    """
    Tests the GreedyAgent with the 'global' controller type.
    Verifies that the predict method returns a correctly shaped and scaled action.
    """
    yaml_path = base_example_data_path / Path("Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=4, yDist=4)

    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        yaw_init="Random",  # Test with different init
        seed=1,  # Use a different seed
        Baseline_comp=False,
        turbtype="None",
        n_passthrough=0.1,
        burn_in_passthroughs=0.01,
    )
    env.reset()  # Initialize flow simulation `fs`

    agent = GreedyAgent(
        env=env,
        type="global",
        yaw_max=env.yaw_max,
        yaw_min=env.yaw_min,
        yaw_step=env.yaw_step_sim,
    )

    action, state = agent.predict()

    assert (
        state is None
    ), "GreedyAgent's predict method should return None for the state."
    assert isinstance(action, np.ndarray), "Action should be a numpy array."
    assert (
        action.shape == env.action_space.shape
    ), f"Action shape mismatch. Expected {env.action_space.shape}, got {action.shape}."
    assert np.all(action >= -1) and np.all(
        action <= 1
    ), "Action values should be scaled between -1 and 1."
    env.close()


def test_base_agent(base_example_data_path):
    base_agent = BaseAgent()
    assert base_agent.predict() is None


def test_random_agent(base_example_data_path):
    yaml_path = base_example_data_path / Path("Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=4, yDist=4)

    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        yaw_init="Zeros",  # always start at zero yaw offset ,
        seed=1,
        turbtype="None",  # Key optimization for speed
        n_passthrough=0.1,  # Very short episodes for fast tests
        burn_in_passthroughs=0.0001,
    )
    random_agent = RandomAgent(env=env)
    action, state = random_agent.predict()

    assert (
        state is None
    ), "RandomAgent's predict method should return None for the state."
    assert isinstance(action, np.ndarray), "Action should be a numpy array."
    assert (
        action.shape == env.action_space.shape
    ), f"Action shape mismatch. Expected {env.action_space.shape}, got {action.shape}."
    assert np.all(action >= -1) and np.all(
        action <= 1
    ), "Action values should be scaled between -1 and 1."
    env.close()
