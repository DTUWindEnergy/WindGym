import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock, patch

# Import the class to be tested
from WindGym.wrappers.power_wrapper import PowerWrapper


@pytest.fixture
def mock_env():
    """Create a mock environment that simulates WindFarmEnv."""
    env = MagicMock(spec=gym.Env)
    env.x_pos = np.array([0, 100])
    env.y_pos = np.array([0, 0])
    env.turbine = "mock_turbine"
    env.n_turb = 2
    env.rated_power = 5000.0  # 5 MW
    env.current_yaw = np.array([0.0, 0.0])

    # Mock the reset and step methods
    env.reset.return_value = (np.array([0.1, 0.2]), {"initial_info": True})
    env.step.return_value = (
        np.array([0.3, 0.4]),  # obs
        0.8,  # env_reward
        False,  # terminated
        False,  # truncated
        {"env_info": True},  # info
    )
    return env


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_initialization(MockPyWakeAgent, mock_env):
    """Test the initialization of the PowerWrapper."""
    n_envs = 3
    print(help(PowerWrapper))
    wrapper = PowerWrapper(env=mock_env, n_envs=n_envs)

    # Assert PyWakeAgent was initialized correctly
    MockPyWakeAgent.assert_called_once_with(
        x_pos=mock_env.x_pos,
        y_pos=mock_env.y_pos,
        turbine=mock_env.turbine,
        env=mock_env,
    )

    # Assert wrapper attributes are set
    assert wrapper.n_envs == n_envs
    assert wrapper.current_step == 0
    assert callable(wrapper.weight_function)


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_reset(MockPyWakeAgent, mock_env):
    """Test the reset method of the PowerWrapper."""
    # Setup mock agent instance
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.calc_power.return_value = 5000.0

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=1)
    obs, info = wrapper.reset()

    # Check that the environment's reset was called
    mock_env.reset.assert_called_once()

    assert wrapper.pywake_baseline_power == 5000.0

    # Check the returned observation and info
    assert np.array_equal(obs, np.array([0.1, 0.2]))
    assert info["initial_info"] is True
    assert "curriculum_weight" in info
    assert info["curriculum_weight"] == 1.0  # Default weight_function at step 0


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_step_baseline_reward(MockPyWakeAgent, mock_env):
    """Test the step method with 'Baseline' power reward."""
    # Setup mock agent instance and environment
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.calc_power.side_effect = [
        6000.0,  # Baseline power during reset
        6600.0,  # Agent power during step
    ]
    mock_env.power_reward = "Baseline"
    action = np.array([0.1, 0.1])

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=1)
    wrapper.reset()  # This will set the baseline power to 6000.0

    obs, new_reward, terminated, truncated, info = wrapper.step(action)

    # Check env.step was called
    mock_env.step.assert_called_with(action)

    # Check reward calculation
    # wrapper_reward = 6600 / 6000 - 1 = 0.1
    # env_reward = 0.8
    # new_reward = 1.0 * 0.8 + (1 - 1.0) * 0.1 = 0.8
    assert new_reward == pytest.approx(0.8)
    assert info["wrapper_reward"] == pytest.approx(0.1)

    # Check info dict and return values
    assert np.array_equal(obs, np.array([0.3, 0.4]))
    assert info["env_info"] is True
    assert "curriculum_weight" in info
    assert "current_step" in info
    assert not terminated
    assert not truncated


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_step_power_avg_reward(MockPyWakeAgent, mock_env):
    """Test the step method with 'Power_avg' power reward."""
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.calc_power.return_value = 7000.0
    mock_env.power_reward = "Power_avg"
    action = np.array([0.1, 0.1])

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=1)
    wrapper.reset()
    obs, new_reward, terminated, truncated, info = wrapper.step(action)

    # Check reward calculation
    # wrapper_reward = 7000.0 / 2 / 5000.0 = 0.7
    # env_reward = 0.8
    # new_reward = 1.0 * 0.8 + (1 - 1.0) * 0.7 = 0.8
    assert new_reward == pytest.approx(0.8)
    assert info["wrapper_reward"] == pytest.approx(0.7)


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_step_unknown_reward_type(MockPyWakeAgent, mock_env):
    """Test that an unknown power_reward type raises a ValueError."""
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.power.return_value = 1.0
    mock_env.power_reward = "UnknownReward"
    action = np.array([0.1, 0.1])

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=1)
    wrapper.reset()

    with pytest.raises(ValueError, match="Unknown power_reward type: UnknownReward"):
        wrapper.step(action)


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_step_reward_weighting(MockPyWakeAgent, mock_env):
    """Test the weighting between environment and wrapper rewards."""
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.calc_power.return_value = 7000.0
    mock_env.power_reward = "Power_avg"
    action = np.array([0.1, 0.1])

    # Create a weight function that returns 0.25
    def weight_function(step):
        return 0.25

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=1, weight_function=weight_function)
    wrapper.reset()
    obs, new_reward, terminated, truncated, info = wrapper.step(action)

    # Check reward calculation with weighting
    wrapper_reward = 7000.0 / 2 / 5000.0  # 0.7
    env_reward = 0.8
    env_w = 0.25
    expected_reward = env_w * env_reward + (1 - env_w) * wrapper_reward
    # expected_reward = 0.25 * 0.8 + 0.75 * 0.7 = 0.2 + 0.525 = 0.725
    assert new_reward == pytest.approx(expected_reward)
    assert info["curriculum_weight"] == pytest.approx(env_w)


@patch("WindGym.wrappers.power_wrapper.PyWakeAgent")
def test_step_counter_and_info(MockPyWakeAgent, mock_env):
    """Test that the step counter and info dictionary are updated correctly."""
    mock_agent_instance = MockPyWakeAgent.return_value
    mock_agent_instance.calc_power.return_value = 1.0
    mock_env.power_reward = "Baseline"
    action = np.array([0.1, 0.1])
    n_envs = 4

    # Mock environment state
    mock_env.ws = 8.0
    mock_env.wd = 270.0
    mock_env.ti = 0.1

    wrapper = PowerWrapper(env=mock_env, n_envs=n_envs)
    wrapper.reset()
    assert wrapper.current_step == 0

    _, _, _, _, info = wrapper.step(action)
    assert wrapper.current_step == 4
    assert (
        info["current_step"] == 0
    )  # The info dict shows the step *before* the increment

    _, _, _, _, info = wrapper.step(action)
    assert wrapper.current_step == 8
    assert info["current_step"] == 4
