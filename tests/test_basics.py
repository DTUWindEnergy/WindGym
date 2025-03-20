import pytest
import numpy as np
from pathlib import Path
from py_wake.examples.data.hornsrev1 import V80
from stable_baselines3 import PPO
from WindGym import WindFarmEnv
from WindGym import FarmEval, AgentEval, PyWakeAgent
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from gymnasium.utils.env_checker import check_env


@pytest.fixture
def turbine():
    """Provides a standard wind turbine configuration for testing"""
    return V80()


@pytest.fixture
def base_example_data_path():
    """Provides path to the example configuration directory"""
    return Path("examples/EnvConfigs")


@pytest.fixture(params=["2turb.yaml", "Env1.yaml"])
def example_config_path(base_example_data_path, request):
    """Provides paths to multiple example configurations, testing environment with different setups"""
    return base_example_data_path / request.param


@pytest.fixture(scope="session")
def mann_turbulence_field():
    """
    Generate a single Mann turbulence field to be reused across all tests.
    Using session scope ensures this is only generated once per test run.
    """
    tf_agent = MannTurbulenceField.generate(
        alphaepsilon=0.1,
        L=33.6,
        Gamma=3.9,
        Nxyz=(1024, 128, 32),  # Reduced size for testing
        dxyz=(3.0, 3.0, 3.0),
        seed=1234,  # Fixed seed for reproducibility
    )
    return tf_agent


@pytest.fixture
def wind_farm_env(turbine, mann_turbulence_field, monkeypatch):
    """
    Creates a wind farm environment instance for testing, ensuring proper cleanup.
    Uses the pre-generated turbulence field instead of generating a new one.
    """

    def mock_generate(*args, **kwargs):
        """Returns our pre-generated turbulence field"""
        return mann_turbulence_field

    # Replace turbulence field generation with our mock
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate", mock_generate
    )

    env = WindFarmEnv(
        turbine=turbine,
        n_passthrough=2,
        yaml_path=Path("examples/EnvConfigs/2turb.yaml"),
        turbtype="MannFixed",  # Using fixed turbulence type
    )

    yield env
    env.close()


@pytest.fixture
def evaluation_params():
    """Defines minimal evaluation parameters focused on testing core functionality"""
    return {
        "winddirs": [260],
        "windspeeds": [10],
        "turbintensities": [0.07],
        "turbboxes": ["Default"],
        "t_sim": 2,  # Keep simulation time short for testing
    }


@pytest.fixture
def trained_agent():
    """
    Loads a pre-trained agent that was successful in wind farm control.
    This agent should be stored in a consistent location in the repo.
    """
    # Load from a saved checkpoint in the repo
    model_path = Path("WindGym/Examples/PPO_2975000.zip")
    model = PPO.load(model_path)
    return model


def test_environment_initialization(wind_farm_env):
    """
    Validates that the environment initializes correctly with proper spaces and initial state.
    This ensures basic functionality before testing more complex operations.
    """
    # Verify observation and action spaces exist and are properly configured
    assert wind_farm_env.observation_space is not None
    assert wind_farm_env.action_space is not None
    assert isinstance(wind_farm_env.observation_space.sample(), np.ndarray)
    assert isinstance(wind_farm_env.action_space.sample(), np.ndarray)

    # Check that the turbine configuration is correct
    assert wind_farm_env.n_turb > 0
    assert hasattr(wind_farm_env, "turbine")


def test_environment_reset(wind_farm_env):
    """
    Tests the environment reset functionality, ensuring it returns valid observations
    and information. This is crucial as reset is called at the start of each episode.
    """
    obs, info = wind_farm_env.reset()

    # Validate observation structure
    assert isinstance(obs, np.ndarray)
    assert obs.shape == wind_farm_env.observation_space.shape
    assert not np.any(np.isnan(obs))

    # Validate info dictionary contains expected keys
    assert isinstance(info, dict)
    expected_keys = [
        "yaw angles agent",
        "Wind speed Global",
        "Wind direction Global",
        "Power agent",
    ]
    for key in expected_keys:
        assert key in info


def test_environment_step(wind_farm_env):
    """
    Tests the environment step function, validating state transitions, rewards,
    and information flow. This ensures the core simulation mechanics work correctly.
    """
    obs, info = wind_farm_env.reset()

    # Test multiple steps to ensure consistent behavior
    for _ in range(3):
        # Use zero action first to test baseline behavior
        action = np.zeros(wind_farm_env.action_space.shape)
        obs, reward, terminated, truncated, info = wind_farm_env.step(action)

        # Validate step outputs
        assert isinstance(obs, np.ndarray)
        assert obs.shape == wind_farm_env.observation_space.shape
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "Power agent" in info
        assert info["Power agent"] >= 0  # Power should never be negative


def test_random_actions(wind_farm_env):
    """
    Tests the environment's response to random actions, ensuring stability
    and proper bounds handling. This helps verify the environment can handle
    unexpected inputs gracefully.
    """
    wind_farm_env.reset()

    for _ in range(3):
        random_action = wind_farm_env.action_space.sample()
        obs, reward, terminated, truncated, info = wind_farm_env.step(random_action)

        # Check that random actions don't break the environment
        assert not np.any(np.isnan(obs))
        assert not np.isnan(reward)
        assert isinstance(info["Power agent"], (int, float))
        assert info["Power agent"] >= 0


def test_yaw_angle_limits(wind_farm_env):
    """
    Verifies that yaw angles remain within specified limits regardless of actions.
    This is crucial for realistic wind farm behavior and safety constraints.
    """
    wind_farm_env.reset()

    # Test extreme actions to ensure yaw limits are respected
    extreme_action = np.ones(wind_farm_env.action_space.shape)
    _, _, _, _, info = wind_farm_env.step(extreme_action)

    yaw_angles = info["yaw angles agent"]
    assert np.all(yaw_angles >= wind_farm_env.yaw_min)
    assert np.all(yaw_angles <= wind_farm_env.yaw_max)


def test_power_calculation(wind_farm_env):
    """
    Tests that power calculations are reasonable and consistent.
    This ensures the core purpose of the wind farm simulation is working correctly.
    """
    wind_farm_env.reset()

    # Collect power values over several steps
    power_values = []
    for _ in range(3):
        action = wind_farm_env.action_space.sample()
        _, _, _, _, info = wind_farm_env.step(action)
        power_values.append(info["Power agent"])

    # Verify power values are reasonable
    assert all(p >= 0 for p in power_values)  # Power should never be negative
    assert all(isinstance(p, (int, float)) for p in power_values)
    assert not any(np.isnan(p) for p in power_values)


def test_wind_conditions(wind_farm_env):
    """
    Verifies that wind conditions are properly initialized and maintained.
    This ensures the environmental conditions affecting the wind farm are realistic.
    """
    _, info = wind_farm_env.reset()

    # Check wind speed and direction are within expected ranges
    assert wind_farm_env.ws_min <= info["Wind speed Global"] <= wind_farm_env.ws_max
    assert wind_farm_env.wd_min <= info["Wind direction Global"] <= wind_farm_env.wd_max

    # Verify turbine-specific wind measurements exist
    assert "Wind speed at turbines" in info
    assert "Wind direction at turbines" in info
    assert len(info["Wind speed at turbines"]) == wind_farm_env.n_turb


# def test_extreme_wind_conditions(wind_farm_env):
#    """Test environment behavior at minimum and maximum wind speeds"""
#    wind_farm_env.set_wind_vals(ws=wind_farm_env.ws_min, wd=270, ti=0.07)
#    obs_min, _ = wind_farm_env.reset()
#    assert not np.any(np.isnan(obs_min))
#
#    wind_farm_env.set_wind_vals(ws=wind_farm_env.ws_max, wd=270, ti=0.07)
#    obs_max, _ = wind_farm_env.reset()
#    assert not np.any(np.isnan(obs_max))


def test_reward_functions(wind_farm_env):
    """Test different reward formulations behave as expected"""
    wind_farm_env.reset()

    # Test baseline reward
    if wind_farm_env.power_reward == "Baseline":
        _, reward, _, _, _ = wind_farm_env.step(
            np.zeros(wind_farm_env.action_space.shape)
        )
        assert isinstance(reward, float)

    # Test action penalty
    large_action = np.ones(wind_farm_env.action_space.shape)
    _, reward_large, _, _, _ = wind_farm_env.step(large_action)
    small_action = np.zeros(wind_farm_env.action_space.shape)
    _, reward_small, _, _, _ = wind_farm_env.step(small_action)
    # assert reward_small > reward_large  # Smaller actions should get less penalty


# def test_measurement_system(wind_farm_env):
#    """Test that measurements include appropriate noise and history"""
#    wind_farm_env.reset()
#
#    # Get measurements over several steps
#    measurements = []
#    for _ in range(10):
#        obs, _, _, _, _ = wind_farm_env.step(np.zeros(wind_farm_env.action_space.shape))
#        measurements.append(obs)
#
#    measurements = np.array(measurements)
#
#    # If noise is enabled, measurements should vary
#    if wind_farm_env.noise:
#        assert np.std(measurements, axis=0).any() > 0
#
#    # Check history length matches configuration
#    assert len(wind_farm_env.farm_measurements.get_ws_history()) == \
# @           min(10, wind_farm_env.farm_measurements.max_hist())

# def test_wake_effects(wind_farm_env):
#    """Test that downstream turbines experience wake effects"""
#    wind_farm_env.reset()
#
#    # Get wind speeds at each turbine
#    _, info = wind_farm_env.reset()
#    turbine_speeds = info["Wind speed at turbines"]
#
#    # In aligned conditions, downstream turbines should see lower wind speeds
#    downstream_mask = wind_farm_env.x_pos > wind_farm_env.x_pos.min()
#    if any(downstream_mask):
#        assert np.mean(turbine_speeds[downstream_mask]) < \
#               np.mean(turbine_speeds[~downstream_mask])


def test_ppo_compatibility(wind_farm_env):
    """Test that environment works with PPO algorithm"""
    model = PPO("MlpPolicy", wind_farm_env, verbose=0)
    model.learn(total_timesteps=10)  # Just verify it runs without errors

    obs, _ = wind_farm_env.reset()
    action, _ = model.predict(obs)
    assert wind_farm_env.action_space.contains(action)


def eval_pretrained_agent(base_example_data_path):
    model_name = base_example_data_path / Path(
        "PreTrainedPPO"
    )  # Name of the model. Used for the naming
    model_step = base_example_data_path / Path(
        "PPO_2975000"
    )  # Name of the agent .zip file
    SEED = 1  # What seed to use for the evaluation
    t_sim = 20  # How many seconds to simulate
    yaml_path = base_example_data_path / Path("Env1.yaml")  # Path to the yaml file

    env = FarmEval(
        turbine=V80(),
        yaml_path=yaml_path,
        yaw_init="Zeros",  # always start at zero yaw offset ,
        seed=SEED,
    )

    model = PPO.load(model_step)
    tester = AgentEval(
        env=env, model=model, name=model_name + "_" + model_step, t_sim=t_sim
    )

    tester.set_conditions(
        winddirs=[260], windspeeds=[10], turbintensities=[0.07], turbboxes=["Default"]
    )

    multi_ds = tester.eval_multiple(
        save_figs=False, debug=False
    )  # Dont make the figures, just return the data

    tester.save_performance()

    model = PyWakeAgent(x_pos=env.x_pos, y_pos=env.y_pos)
    tester = AgentEval(env=env, model=model, name="PyWakeAgent", t_sim=t_sim)

    tester.set_conditions(
        winddirs=[260], windspeeds=[10], turbintensities=[0.07], turbboxes=["Default"]
    )
    multi_ds = tester.eval_multiple(
        save_figs=False, debug=False
    )  # Dont make the figures, just return the data
    tester.save_performance()


def test_set_windconditions_with_site(wind_farm_env):
    """
    Test that wind conditions are properly sampled when using a PyWake site.
    This test verifies that:
    1. Wind speeds and directions are sampled from the site's distributions
    2. Values remain within the environment's configured limits
    3. Multiple calls produce different but valid values
    4. Turbulence intensity is still sampled uniformly
    """
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site

    # Setup the site
    site = Hornsrev1Site()
    wind_farm_env.sample_site = site

    # Sample multiple times to check distribution
    samples = []
    for _ in range(10):
        wind_farm_env._set_windconditions()
        samples.append(
            {"ws": wind_farm_env.ws, "wd": wind_farm_env.wd, "ti": wind_farm_env.ti}
        )

    # Check all samples are within configured limits
    for sample in samples:
        assert wind_farm_env.ws_min <= sample["ws"] <= wind_farm_env.ws_max
        assert wind_farm_env.wd_min <= sample["wd"] <= wind_farm_env.wd_max
        assert wind_farm_env.TI_min <= sample["ti"] <= wind_farm_env.TI_max

    # Verify we get different values (sampling is working)
    wind_speeds = [s["ws"] for s in samples]
    wind_directions = [s["wd"] for s in samples]
    assert len(set(wind_speeds)) > 1, "Wind speeds are not varying"
    assert len(set(wind_directions)) > 1, "Wind directions are not varying"

    # Check that TI is still uniformly distributed between min and max
    ti_values = [s["ti"] for s in samples]
    assert len(set(ti_values)) > 1, "TI values are not varying"
    assert all(wind_farm_env.TI_min <= ti <= wind_farm_env.TI_max for ti in ti_values)


def test_check_env(wind_farm_env):
    """Test that the environment passes the gymnasium check"""
    check_env(wind_farm_env)
