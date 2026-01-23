from pathlib import Path

import numpy as np
import pytest
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from gymnasium.utils.env_checker import check_env
from py_wake.examples.data.hornsrev1 import V80
from stable_baselines3 import PPO

from WindGym import AgentEval, AgentEvalFast, FarmEval, PyWakeAgent, WindFarmEnv
from WindGym.Agents import BaseAgent, ConstantAgent, RandomAgent
from WindGym.utils.generate_layouts import generate_square_grid


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

    x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=4, yDist=4)

    env = WindFarmEnv(
        turbine=turbine,
        x_pos=x_pos,
        y_pos=y_pos,
        n_passthrough=0.1,
        config=Path("examples/EnvConfigs/2turb.yaml"),
        turbtype="None",
        burn_in_passthroughs=0.0001,  # Minimal burn-in for faster tests
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
        assert isinstance(reward, (np.float32, float))
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

        # Action penalties are moved into the reward calculator now.
        # # Check that the action penalty is applied correctly
        # wind_farm_env.action_penalty = 1.0
        # wind_farm_env.action_penalty_type = "Change"
        # pen_1 = wind_farm_env._action_penalty()

        # wind_farm_env.action_penalty_type = "Total"
        # pen_2 = wind_farm_env._action_penalty()

        # # Assert that both should be above 0
        # assert pen_1 >= 0, "The penalty should be above 0"
        # assert pen_2 >= 0, "The penalty should be above 0"


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
    assert (
        wind_farm_env.farm_measurements.farm_mes.ws_min
        <= info["Wind speed Global"]
        <= wind_farm_env.farm_measurements.farm_mes.ws_max
    )
    assert (
        wind_farm_env.farm_measurements.farm_mes.wd_min
        <= info["Wind direction Global"]
        <= wind_farm_env.farm_measurements.farm_mes.wd_max
    )

    # Verify turbine-specific wind measurements exist
    assert "Wind speed at turbines" in info
    assert "Wind direction at turbines" in info
    assert len(info["Wind speed at turbines"]) == wind_farm_env.n_turb


def test_reward_functions(wind_farm_env):
    """Test different reward formulations behave as expected"""
    wind_farm_env.reset(seed=42)  # Use fixed seed for reproducibility

    # Test baseline reward returns valid type
    if wind_farm_env.power_reward == "Baseline":
        _, reward, _, _, _ = wind_farm_env.step(
            np.zeros(wind_farm_env.action_space.shape)
        )
        assert isinstance(reward, (np.float32, float))
        assert np.isfinite(reward), "Reward should be finite"

    # Test that rewards are computed correctly regardless of action
    # Note: The config uses action_penalty=0.0, so we can't test penalty effects.
    # Instead, verify rewards are valid numbers for different actions.
    large_action = np.ones(wind_farm_env.action_space.shape)
    _, reward_large, _, _, _ = wind_farm_env.step(large_action)
    assert isinstance(reward_large, (np.float32, float))
    assert np.isfinite(reward_large), "Reward for large action should be finite"

    small_action = np.zeros(wind_farm_env.action_space.shape)
    _, reward_small, _, _, _ = wind_farm_env.step(small_action)
    assert isinstance(reward_small, (np.float32, float))
    assert np.isfinite(reward_small), "Reward for small action should be finite"

    # Only test penalty ordering if action_penalty is configured
    if wind_farm_env.action_penalty > 0.001:
        assert reward_small >= reward_large, "Smaller actions should get less penalty"


def test_ppo_compatibility(wind_farm_env):
    """
    Test that a PPO model can be initialized and interact with the environment
    for a few steps without errors, and that the outputs are valid.
    """
    # 1. Initialize PPO model
    # Use the 'wind_farm_env' fixture directly as the environment for PPO
    model = PPO("MlpPolicy", wind_farm_env, verbose=0)

    # Assert that the model object is created successfully
    assert model is not None, "PPO model should be initialized."
    assert hasattr(model, "policy"), "PPO model should have a policy object."
    # Corrected assertion for the value network: it's typically 'critic' or 'value_net' on the policy object
    assert hasattr(model.policy, "critic") or hasattr(
        model.policy, "value_net"
    ), "PPO model's policy should have a value network (critic or value_net)."

    # 2. Explicitly call reset and capture initial state
    obs, info = wind_farm_env.reset()

    # Assert initial observation and info are valid
    assert isinstance(obs, np.ndarray), "Initial observation should be a numpy array."
    assert (
        obs.shape == wind_farm_env.observation_space.shape
    ), f"Initial observation shape mismatch. Expected {wind_farm_env.observation_space.shape}, got {obs.shape}."
    assert not np.any(np.isnan(obs)), "Initial observation should not contain NaNs."
    assert isinstance(info, dict), "Initial info should be a dictionary."
    assert "Power agent" in info, "Initial info should contain 'Power agent'."
    assert info["Power agent"] >= 0, "Initial 'Power agent' should be non-negative."

    # 3. Take a few steps manually
    num_steps_to_take = 2
    for i in range(num_steps_to_take):
        # Predict action from the model
        action, _ = model.predict(
            obs, deterministic=True
        )  # PPO needs an observation to predict

        # Assert action is valid before stepping
        assert isinstance(
            action, np.ndarray
        ), f"Action at step {i} should be a numpy array."
        assert (
            action.shape == wind_farm_env.action_space.shape
        ), f"Action shape mismatch at step {i}. Expected {wind_farm_env.action_space.shape}, got {action.shape}."
        assert wind_farm_env.action_space.contains(
            action
        ), f"Action {action} at step {i} is outside action space bounds {wind_farm_env.action_space}."

        # Step the environment
        obs, reward, terminated, truncated, info = wind_farm_env.step(action)

        # Assert outputs of the step
        assert isinstance(
            obs, np.ndarray
        ), f"Observation at step {i} should be a numpy array."
        assert (
            obs.shape == wind_farm_env.observation_space.shape
        ), f"Observation shape mismatch at step {i}. Expected {wind_farm_env.observation_space.shape}, got {obs.shape}."
        assert not np.any(
            np.isnan(obs)
        ), f"Observation at step {i} should not contain NaNs."

        assert isinstance(
            reward, (np.float32, float)
        ), f"Reward at step {i} should be a float."
        assert not np.isnan(reward), f"Reward at step {i} should not be NaN."

        assert isinstance(
            terminated, bool
        ), f"Terminated flag at step {i} should be a boolean."
        assert isinstance(
            truncated, bool
        ), f"Truncated flag at step {i} should be a boolean."

        assert isinstance(info, dict), f"Info at step {i} should be a dictionary."
        assert "Power agent" in info, f"Info at step {i} should contain 'Power agent'."
        assert (
            info["Power agent"] >= 0
        ), f"'Power agent' at step {i} should be non-negative."

        # Handle episode termination/truncation (and reset for next step if needed)
        if terminated or truncated:
            # Reset if there are more steps planned in this loop
            if i < num_steps_to_take - 1:
                obs, _ = wind_farm_env.reset()


def eval_pretrained_agent(base_example_data_path):
    model_name = base_example_data_path / Path(
        "PreTrainedPPO"
    )  # Name of the model. Used for the naming
    model_step = base_example_data_path / Path(
        "PPO_2975000"
    )  # Name of the agent .zip file
    SEED = 1  # What seed to use for the evaluation
    t_sim = 20  # How many seconds to simulate
    yaml_path = base_example_data_path / Path("Env1.yaml")  # Path to the yaml file2

    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=4, yDist=4)

    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        yaw_init="Zeros",  # always start at zero yaw offset ,
        seed=SEED,
        n_passthrough=0.1,
        burn_in_passthroughs=0.01,
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

    model = PyWakeAgent(x_pos=env.x_pos, y_pos=env.y_pos, env=env)
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
    for _ in range(40):
        wind_farm_env._set_windconditions()
        samples.append(
            {"ws": wind_farm_env.ws, "wd": wind_farm_env.wd, "ti": wind_farm_env.ti}
        )

    # Check all samples are within configured limits
    for sample in samples:
        assert (
            wind_farm_env.ws_inflow_min <= sample["ws"] <= wind_farm_env.ws_inflow_max
        )
        assert (
            wind_farm_env.wd_inflow_min <= sample["wd"] <= wind_farm_env.wd_inflow_max
        )
        assert (
            wind_farm_env.TI_inflow_min <= sample["ti"] <= wind_farm_env.TI_inflow_max
        )

    # Verify we get different values (sampling is working)
    wind_speeds = [s["ws"] for s in samples]
    wind_directions = [s["wd"] for s in samples]
    assert len(set(wind_speeds)) > 1, "Wind speeds are not varying"
    assert len(set(wind_directions)) > 1, "Wind directions are not varying"

    # Check that TI is still uniformly distributed between min and max
    ti_values = [s["ti"] for s in samples]
    assert len(set(ti_values)) > 1, "TI values are not varying"
    assert all(
        wind_farm_env.TI_inflow_min <= ti <= wind_farm_env.TI_inflow_max
        for ti in ti_values
    )


@pytest.mark.slow
def test_check_env(wind_farm_env):
    """
    Test that the environment passes the gymnasium check_env validation.

    check_env validates:
    - Observation space bounds and types
    - Action space bounds and types
    - Reset returns valid observations
    - Step returns valid observations, rewards, and info
    - Episode termination/truncation handling
    """
    # Run the gymnasium check - this raises on failure
    check_env(wind_farm_env, skip_render_check=True)

    # Additional validation that check_env doesn't cover
    obs, info = wind_farm_env.reset()

    # Verify observation is within declared bounds
    assert wind_farm_env.observation_space.contains(
        obs
    ), f"Observation {obs} is outside declared observation space bounds"

    # Verify action space is properly bounded
    action = wind_farm_env.action_space.sample()
    assert wind_farm_env.action_space.contains(
        action
    ), "Sampled action outside action space"

    # Step and verify outputs
    obs, reward, terminated, truncated, info = wind_farm_env.step(action)
    assert wind_farm_env.observation_space.contains(
        obs
    ), "Post-step observation outside bounds"
    assert np.isfinite(reward), f"Reward {reward} is not finite"


@pytest.mark.slow
def test_fast_eval():
    """
    Test the fast evaluation of the environment using a pre-trained agent.
    This ensures that the environment can be evaluated quickly and efficiently.
    """
    SEED = 1
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=4, yDist=4)

    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config="examples/EnvConfigs/Env1.yaml",
        turbtype="None",
        yaw_init="Zeros",  # always start at zero yaw offset ,
        seed=SEED,
        n_passthrough=0.1,
        burn_in_passthroughs=0.0001,
    )
    n_turb = env.n_turb  # The Env1.yams file has 4 turbines
    WS_SIM = 10  # Wind speed in m/s
    WD_SIM = 270  # Wind direction in degrees
    TI_SIM = 0.07  # Turbulence intensity

    yaw_goal = np.zeros(n_turb)  # yaw angles in radians
    yaw_goal[0] = -10
    yaw_goal[1] = 20
    model = ConstantAgent(yaw_angles=list(yaw_goal))  # yaw angles in degrees

    ds = AgentEvalFast(
        env,
        model,
        1,
        ws=WS_SIM,
        ti=TI_SIM,
        wd=WD_SIM,
        turbbox="Default",  # BARE EN STRING
        t_sim=50,
        deterministic=True,
        # debug=True,
    )  # Default values

    assert ds is not None, "Fast evaluation failed to return a dataset"
    assert len(ds) > 0, "Fast evaluation dataset is empty"
    assert np.allclose(
        ds.yaw_a[0].values.flatten(), np.zeros(n_turb)
    ), "Yaw angles did not initialize as expected"
    assert np.allclose(
        ds.yaw_a[-1].values.flatten(), yaw_goal
    ), "Yaw angles did not reach their goal"
    assert np.allclose(
        ds.ws[0].values.flatten(), WS_SIM
    ), "Wind speed did not initialize as expected"
    assert np.allclose(
        ds.TI[0].values.flatten(), TI_SIM
    ), "Turbulence intensity did not initialize as expected"
    assert np.allclose(
        ds.wd[0].values.flatten(), WD_SIM
    ), "Wind direction did not initialize as expected"
    assert np.allclose(
        ds.turb.values.size, n_turb
    ), "The number of turbines is not as expected"


@pytest.mark.slow
def test_fast_eval_debug():
    """
    Test the fast evaluation of the environment using a pre-trained agent.
    This ensures that the environment can be evaluated quickly and efficiently.
    Verifies that the PyWakeAgent produces valid outputs with debug mode enabled.
    """
    SEED = 1
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=4, yDist=4)

    env = FarmEval(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config="examples/EnvConfigs/Env1.yaml",
        turbtype="None",
        yaw_init="Zeros",  # always start at zero yaw offset ,
        seed=SEED,
        n_passthrough=0.1,  # Very short episodes for fast tests
        burn_in_passthroughs=0.0001,
    )
    n_turb = env.n_turb

    WS_SIM = 10  # Wind speed in m/s
    WD_SIM = 270  # Wind direction in degrees
    TI_SIM = 0.07  # Turbulence intensity

    model = PyWakeAgent(x_pos=env.x_pos, y_pos=env.y_pos, env=env)

    ds = AgentEvalFast(
        env,
        model,
        1,
        ws=WS_SIM,
        ti=TI_SIM,
        wd=WD_SIM,
        turbbox="None",
        t_sim=5,
        deterministic=True,
        debug=True,
        save_figs=True,
    )

    # Verify the dataset is valid and contains expected data
    assert ds is not None, "Debug evaluation failed to return a dataset"
    assert len(ds) > 0, "Debug evaluation dataset is empty"

    # Verify wind conditions are correctly recorded
    assert np.allclose(
        ds.ws[0].values.flatten(), WS_SIM
    ), "Wind speed did not initialize as expected"
    assert np.allclose(
        ds.TI[0].values.flatten(), TI_SIM
    ), "Turbulence intensity did not initialize as expected"
    assert np.allclose(
        ds.wd[0].values.flatten(), WD_SIM
    ), "Wind direction did not initialize as expected"

    # Verify turbine count matches
    assert (
        ds.turb.values.size == n_turb
    ), f"Expected {n_turb} turbines, got {ds.turb.values.size}"

    # Verify power values are positive and realistic (not zero, not unreasonably high)
    # V80 rated power is ~2MW, so 4 turbines should produce < 10MW total
    final_power = ds.powerF_a[-1].values.flatten()[0]
    assert final_power > 0, "Final power should be positive"
    assert (
        final_power < 10e6
    ), f"Final power {final_power} exceeds realistic bounds for 4 V80 turbines"

    # Verify yaw angles are within physical limits
    final_yaws = ds.yaw_a[-1].values.flatten()
    assert np.all(final_yaws >= env.yaw_min), "Yaw angles below minimum"
    assert np.all(final_yaws <= env.yaw_max), "Yaw angles above maximum"

    env.close()


def test_env_features(wind_farm_env):
    """Test various environment features and utility methods."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    # Test plot_frame can be called without crashing (returns None, just displays)
    # Note: plot_frame displays plots but doesn't return the figure object
    wind_farm_env.plot_frame(baseline=False)
    wind_farm_env.plot_frame(baseline=True)
    plt.close("all")  # Clean up any created figures

    # Verify feature counts are consistent
    num_raw = wind_farm_env._get_num_raw_features()
    num_observed = wind_farm_env.farm_measurements.observed_variables()
    assert num_raw > 0, "Number of raw features should be positive"
    assert num_observed > 0, "Number of observed variables should be positive"
    assert (
        num_raw <= num_observed
    ), "The number of raw features should be less than or equal to the number of observed variables"

    # Verify environment core attributes are properly set
    assert wind_farm_env.n_turb > 0, "Environment should have at least one turbine"
    assert (
        wind_farm_env.yaw_min < wind_farm_env.yaw_max
    ), "Yaw limits should be properly ordered"
    assert wind_farm_env.dt_sim > 0, "Simulation timestep should be positive"
    assert wind_farm_env.dt_env > 0, "Environment timestep should be positive"
