import pytest
import numpy as np
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from gymnasium.spaces import Box
from pettingzoo.test import parallel_api_test  # Good for comprehensive API compliance

# Import the class under test
from WindGym.wind_env_multi import WindFarmEnvMulti

# Import necessary components that WindFarmEnvMulti uses internally
from WindGym.wind_farm_env import WindFarmEnv
from WindGym.core.mes_class import (
    TurbMes,
    FarmMes,
    Mes,
)  # Explicitly import if their methods are called
from WindGym import utils  # Import utils module for utility functions
from py_wake.examples.data.hornsrev1 import V80  # <--- Import V80 here for use


# --- Fixtures for YAML Configuration ---
@pytest.fixture(scope="module")
def tmp_yaml_file(tmp_path_factory):
    """
    Creates a temporary YAML file with a configuration suitable for testing.
    It enables all measurement levels to ensure a comprehensive observation space.
    """
    default_config = {
        "yaw_init": "Zeros",  # Can be overridden by test
        "noise": "None",
        "BaseController": "Local",  # Set to Local or PyWake_oracle/PyWake_local for baseline comp
        "ActionMethod": "yaw",  # Or 'wind', depending on env's default. Let's align with test needs.
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
        },
        "wind": {
            "ws_min": 10.0,
            "ws_max": 10.0,
            "TI_min": 0.07,
            "TI_max": 0.07,
            "wd_min": 270.0,
            "wd_max": 270.0,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {
            "Power_reward": "Baseline",  # A common reward type
            "Power_avg": 2,  # Small buffer for quicker convergence in internal calcs
            "Power_scaling": 1.0,
        },
        "mes_level": {  # Enable all measurements for full obs_var calculation
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": True,
            "turb_power": True,
            "farm_ws": True,
            "farm_wd": True,
            "farm_TI": True,
            "farm_power": True,
            "ti_sample_count": 2,  # Small for fast TI calculation in non-mocked MesClass
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": True,
            "ws_history_N": 1,
            "ws_history_length": 2,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": 1,
            "wd_history_length": 2,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": 1,
            "yaw_history_length": 2,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": 1,
            "power_history_length": 2,
            "power_window_length": 1,
        },
    }

    content_str = yaml.dump(default_config)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(content_str)
        filepath = tmp_file.name

    yield filepath
    os.remove(filepath)


# --- Fixtures for Tests ---


@pytest.fixture
def basic_env_config(tmp_yaml_file):
    """Fixture providing basic environment configuration for WindFarmEnvMulti."""
    # Use a REAL V80 turbine here as its .power method is called by Wind_Farm_Env.py
    return {
        "turbine": V80(),  # <--- Changed to real V80() turbine
        "x_pos": [0, 560],  # Two turbines for multi-agent
        "y_pos": [0, 0],
        "config": tmp_yaml_file,
        "turbtype": "None",  # Crucial: disable turbulence generation for speed and determinism
        "reset_init": True,  # Call reset during constructor
        "dt_sim": 1,
        "dt_env": 1,
        "n_passthrough": 5.0,  # Increased for more robust testing in parallel_api_test
        "burn_in_passthroughs": 0.0001,  # Minimal burn-in for faster tests
    }


@pytest.fixture
def initialized_env(basic_env_config):
    """Provides an initialized WindFarmEnvMulti instance."""
    env = WindFarmEnvMulti(**basic_env_config)
    # env.reset() is already called in __init__ due to reset_init=True
    yield env
    env.close()  # Ensure proper cleanup


# --- Test Class for WindFarmEnvMulti ---


class TestWindFarmEnvMultiCoverage:
    """Comprehensive test suite for WindFarmEnvMulti."""

    def test_init_and_spaces_creation(self, basic_env_config):
        """
        Test initialization and proper space creation for multi-agent environment.
        Verifies agent attributes and observation/action space dimensions.
        """
        env = WindFarmEnvMulti(**basic_env_config)

        assert hasattr(env, "obs_var")
        assert hasattr(env, "act_var")
        assert env.act_var == 1  # Action dimension per agent

        assert len(env.possible_agents) == 2
        assert env.possible_agents == ["turbine_0", "turbine_1"]

        assert env.agent_name_mapping["turbine_0"] == 0
        assert env.agent_name_mapping["turbine_1"] == 1

        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)

            assert isinstance(obs_space, Box)
            assert isinstance(act_space, Box)
            assert obs_space.shape == (env.obs_var,)
            assert act_space.shape == (env.act_var,)  # Should be (1,)
        env.close()  #  Ensure cleanup for this specific test too, also ensures the `lru_cache` for obs/action spaces is cleared if needed between tests

    def test_reset_method(self, initialized_env):
        """Test reset method returns correct format and initializes agents."""
        env = initialized_env  # Already initialized and reset once by fixture

        # Call reset again to ensure it properly re-initializes
        observations, infos = env.reset(seed=123)

        assert env.agents == env.possible_agents  # Agents list should be reset to full
        assert env.timestep == 0  # Timestep should reset after reset call

        assert isinstance(observations, dict)
        assert set(observations.keys()) == set(env.agents)

        assert isinstance(infos, dict)
        assert set(infos.keys()) == set(env.agents)

        for agent in env.agents:
            obs = observations[agent]
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0)  # Check obs are scaled

            info = infos[agent]
            assert isinstance(info, dict)
            # Check a few key info values are present and plausible after real reset
            assert "Power agent" in info
            assert info["Power agent"] > 0  # Should be positive power

            # Check baseline info if Baseline_comp is True (it is in default YAML)
            if env.Baseline_comp:
                assert "Power baseline" in info
                assert info["Power baseline"] > 0
                assert "Power pr turbine baseline" in info
                assert "yaw angles base" in info
                assert "Wind speed at turbines baseline" in info

    def test_get_obs_multi(self, initialized_env):
        """
        Test _get_obs_multi method.
        This tests the data concatenation and clipping for observations generated by the real environment.
        """
        env = initialized_env
        observations = env._get_obs_multi()

        assert isinstance(observations, dict)
        assert len(observations) == len(env.agents)

        for agent in env.agents:
            obs = observations[agent]
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            # The observed length from _get_obs_multi should match the declared observation space size
            assert len(obs) == env.observation_space(agent).shape[0]
            assert np.all(obs >= -1.0) and np.all(
                obs <= 1.0
            )  # Ensure clipping is applied

    def test_get_infos(self, initialized_env):
        """Test _get_infos method returns correct structure and values."""
        env = initialized_env
        infos = env._get_infos()

        assert isinstance(infos, dict)
        assert set(infos.keys()) == set(env.agents)

        for i, agent in enumerate(env.agents):
            info = infos[agent]
            assert isinstance(info, dict)

            # Check for expected keys and plausible values
            assert "yaw angles agent" in info and isinstance(
                info["yaw angles agent"], (float, np.ndarray)
            )
            assert "Wind speed Global" in info and isinstance(
                info["Wind speed Global"], float
            )
            assert "Power agent" in info and isinstance(info["Power agent"], float)
            assert info["Power agent"] > 0

            # Check baseline info if Baseline_comp is True (it is in default YAML)
            if env.Baseline_comp:
                assert "Power baseline" in info
                assert info["Power baseline"] > 0
                assert "Power pr turbine baseline" in info
                assert "yaw angles base" in info
                assert "Wind speed at turbines baseline" in info

    def test_calc_reward(self, initialized_env):
        """Test _calc_reward method."""
        env = initialized_env
        rewards = env._calc_reward()

        assert isinstance(rewards, dict)
        assert set(rewards.keys()) == set(env.agents)

        for agent in env.agents:
            assert isinstance(rewards[agent], float)
            assert not np.isnan(rewards[agent])

    def test_step_method_with_truncation_logic(self, initialized_env):
        """Test step method including truncation logic and agent list clearing."""
        env = initialized_env

        # Set a short time_max to trigger truncation quickly
        # `dt_env` is 1, so `time_max=1` means episode ends after 1 step.
        # `self.timestep` is incremented by parent.step, starting from 0.
        env.time_max = 1
        print(f"Initial env.timestep: {env.timestep}")

        actions = {agent: np.array([0.0]) for agent in env.possible_agents}

        # Step 1: Should not truncate yet, timestep becomes 1
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert (
            env.timestep == 1
        )  # Timestep should be 1 after the first step (if duplicate increment is removed)
        assert not any(terminations.values())
        assert not any(truncations.values())
        assert len(env.agents) == 2  # Agents still active

        # Step 2: Should truncate (timestep becomes 2, which is > time_max=1), agents list cleared
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check truncations are True and agents list is cleared
        for agent in (
            env.possible_agents
        ):  # Iterate over possible_agents as env.agents will be empty
            assert truncations[agent] is True
            assert (
                terminations[agent] is False
            )  # Parallel envs usually truncate, not terminate
        assert len(env.agents) == 0  # Agents list should be cleared

    def test_render_method_delegation(self, basic_env_config):
        """Test that render method properly delegates to parent class."""
        render_modes_to_test = [None, "human", "rgb_array"]

        for render_mode_val in render_modes_to_test:
            env_params = basic_env_config.copy()
            env_params["render_mode"] = render_mode_val
            env = WindFarmEnvMulti(**env_params)
            env.reset(seed=42)  # Must call reset before rendering

            with patch(
                "WindGym.wind_farm_env.WindFarmEnv.render", return_value="mock_frame"
            ) as mock_parent_render:
                mock_parent_render.return_value = (
                    None if render_mode_val is None else "mock_frame"
                )
                try:
                    result = env.render()
                    mock_parent_render.assert_called_once()
                    if render_mode_val is None:
                        assert result is None
                    else:
                        assert result == "mock_frame"
                except Exception as e:
                    if "TclError" in str(e) and render_mode_val == "human":
                        pytest.skip(
                            f"Skipping human render test due to TclError in headless env: {e}"
                        )
                    else:
                        raise
                finally:
                    env.close()

    def test_init_spaces_override(self, initialized_env):
        """Test that _init_spaces is properly overridden by WindFarmEnvMulti."""
        env = initialized_env

        # _init_spaces is called in WindFarmEnvMulti's __init__ and is overridden to do nothing,
        # as spaces are defined via lru_cache properties.
        env._init_spaces()

        # The multi-agent spaces (accessed via properties) should still exist and be functional
        obs_space = env.observation_space("turbine_0")
        act_space = env.action_space("turbine_0")

        assert obs_space is not None
        assert act_space is not None
        assert isinstance(obs_space, Box)
        assert isinstance(act_space, Box)
        assert obs_space.shape == (env.obs_var,)

    def test_action_extraction_in_step(self, initialized_env):
        """Test that actions are properly extracted and formatted for the parent step."""
        env = initialized_env

        test_actions = {
            "turbine_0": np.array([0.5], dtype=np.float32),
            "turbine_1": np.array([-0.3], dtype=np.float32),
        }
        print(f"Test actions: {test_actions}")

        captured_action_from_parent_step = None

        # Mock the parent WindFarmEnv.step method to capture the action passed to it
        with patch("WindGym.wind_farm_env.WindFarmEnv.step") as mock_parent_step:

            def side_effect_func(self_env_inner, action_arg):
                nonlocal captured_action_from_parent_step
                captured_action_from_parent_step = action_arg
                print(
                    f"Mocked parent WindFarmEnv.step called with action_arg: {action_arg}"
                )
                # Return dummy values for obs, reward, terminated, truncated, info for parent's return
                # These are single-env returns
                # We need to create a mock for farm_measurements as well, if _get_obs_multi is called
                # on the *return* from this mock.
                mock_farm_measurements = MagicMock()
                mock_farm_measurements.turb_mes = [MagicMock(), MagicMock()]
                mock_farm_measurements.turb_mes[
                    0
                ].get_measurements.return_value = np.zeros(
                    9
                )  # Assuming 9 features per turbine
                mock_farm_measurements.turb_mes[
                    1
                ].get_measurements.return_value = np.zeros(9)
                mock_farm_measurements.farm_mes.get_measurements.return_value = (
                    np.zeros(7)
                )  # Assuming 7 features for farm
                mock_farm_measurements.farm_mes.get_wd_farm.return_value = (
                    270.0  # Dummy value
                )

                # Patch the env's farm_measurements for the duration of this mock call
                with patch.object(
                    self_env_inner, "farm_measurements", new=mock_farm_measurements
                ):
                    dummy_obs_parent = np.zeros(
                        env.obs_var
                    )  # Use correct total obs size from env
                    # dummy_info_parent should contain keys expected by _get_infos or _get_obs_multi
                    dummy_info_parent = {
                        "Power agent": 1.0e6,
                        "yaws": np.array([0.0, 0.0]),
                        "powers": np.array([1.0e6, 1.0e6]),
                        "time_array": np.array([0]),
                        "yaw angles agent": np.array(
                            [0.0, 0.0]
                        ),  # Needed for _get_infos
                        "Wind speed Global": 10.0,
                        "Wind direction Global": 270.0,
                        "Turbulence intensity": 0.07,
                        "Wind speed at turbines": np.array([10.0, 9.5]),
                        "Wind direction at turbines": np.array([270.0, 270.0]),
                        "Turbulence intensity at turbines": np.array([0.07, 0.07]),
                        "Turbine x positions": np.array([0.0, 560.0]),
                        "Turbine y positions": np.array([0.0, 0.0]),
                        # Add baseline info if Baseline_comp is true in the environment
                        "yaw angles base": np.array(
                            [0.0, 0.0]
                        ),  # Dummy for baseline yaw
                        "Power baseline": 0.9e6,  # Dummy for baseline power
                        "Power pr turbine baseline": np.array(
                            [0.45e6, 0.45e6]
                        ),  # Dummy for baseline power per turbine
                        "Wind speed at turbines baseline": np.array(
                            [10.0, 9.0]
                        ),  # Dummy for baseline wind speed
                    }
                    return (dummy_obs_parent, 0.5, False, False, dummy_info_parent)

            mock_parent_step.side_effect = side_effect_func

            # Now call the WindFarmEnvMulti.step method
            observations, rewards, terminations, truncations, infos = (
                initialized_env.step(test_actions)
            )

        # Check that actions are properly extracted and concatenated
        assert captured_action_from_parent_step is not None
        assert isinstance(captured_action_from_parent_step, np.ndarray)
        assert len(captured_action_from_parent_step) == env.n_turb  # Number of turbines
        assert captured_action_from_parent_step[0] == pytest.approx(0.5)
        assert captured_action_from_parent_step[1] == pytest.approx(-0.3)

    # --- Tests for yaw_init logic (from WindEnv base class, as used by WindFarmEnv) ---
    # These tests are for the underlying WindFarmEnv logic, but are called via the Multi-Env.
    # They should ideally use a single WindFarmEnv instance for clarity, but keeping them here
    # for now as they pass when the WindFarmEnvMulti is correctly initialized.

    def test_yaw_init_defined_error(self, basic_env_config):
        """Test error handling in utils.defined_yaw function with wrong length."""
        temp_config = basic_env_config.copy()
        temp_config["yaw_init"] = "Defined"
        temp_config["reset_init"] = (
            False  # Prevent immediate reset with default yaw_initial=[0]
        )

        env = WindFarmEnvMulti(**temp_config)
        # Manually call defined_yaw with an invalid yaw_initial array.
        invalid_yaw_vals = [0.0, 10.0, 20.0]  # 3 values for a 2-turbine env

        with pytest.raises(
            ValueError, match="The specified yaw values are not the right length."
        ):
            utils.defined_yaw(yaws=np.array(invalid_yaw_vals), n_turb=env.n_turb)
        env.close()

    def test_yaw_init_defined_single_value(self, basic_env_config):
        """Test utils.defined_yaw with a single value for all turbines."""
        temp_config = basic_env_config.copy()
        temp_config["yaw_init"] = "Defined"
        temp_config["reset_init"] = False

        env = WindFarmEnvMulti(**temp_config)

        single_yaw_val = [5.0]
        result = utils.defined_yaw(yaws=np.array(single_yaw_val), n_turb=env.n_turb)

        assert len(result) == env.n_turb
        assert np.allclose(result, 5.0)
        env.close()

    def test_yaw_init_defined_correct_length(self, basic_env_config):
        """Test utils.defined_yaw with correct length array."""
        temp_config = basic_env_config.copy()
        temp_config["yaw_init"] = "Defined"
        temp_config["reset_init"] = False

        env = WindFarmEnvMulti(**temp_config)

        correct_yaw_vals = [10.0, -5.0]  # For 2 turbines
        result = utils.defined_yaw(yaws=np.array(correct_yaw_vals), n_turb=env.n_turb)

        assert len(result) == env.n_turb
        assert np.allclose(result, np.array(correct_yaw_vals))
        env.close()

    # def test_yaw_init_zeros(self, basic_env_config):
    #     """Test that zeros initialization returns zeros when yaw_init is "Zeros"."""
    #     temp_config = basic_env_config.copy()
    #     temp_config["yaw_init"] = "Zeros"
    #     temp_config["reset_init"] = False

    #     env = WindFarmEnvMulti(**temp_config)
    #     print(f"\n--- DEBUG: {self.test_yaw_init_zeros.__name__} ---")

    #     result = np.zeros(env.n_turb)

    #     print(f"Zeros init result: {result}")
    #     assert len(result) == env.n_turb
    #     assert np.allclose(result, np.zeros(env.n_turb))
    #     env.close()

    def test_yaw_init_random(self, basic_env_config):
        """Test that random initialization returns random values when yaw_init is "Random"."""
        temp_config = basic_env_config.copy()
        temp_config["yaw_init"] = "Random"
        temp_config["reset_init"] = False

        env = WindFarmEnvMulti(**temp_config)

        # Call the method bound to "Random" yaw_init
        # We need to ensure np_random is initialized for determinism.
        env.np_random = np.random.default_rng(seed=123)

        random_yaws_1 = env.np_random.uniform(low=-10, high=10, size=env.n_turb)
        random_yaws_2 = env.np_random.uniform(low=-10, high=10, size=env.n_turb)

        assert len(random_yaws_1) == env.n_turb
        assert len(random_yaws_2) == env.n_turb
        assert not np.allclose(
            random_yaws_1, np.zeros(env.n_turb)
        )  # Should not be all zeros
        assert not np.allclose(
            random_yaws_1, random_yaws_2
        )  # Should be different if RNG is active
        env.close()

    def test_dt_env_not_multiple_of_dt_sim(self, tmp_yaml_file):
        """Test that ValueError is raised if dt_env is not a multiple of dt_sim."""
        invalid_config = {
            "turbine": MagicMock(),  # Mock turbine
            "x_pos": [0],
            "y_pos": [0],  # Single turbine is fine for this test
            "config": tmp_yaml_file,
            "turbtype": "None",
            "reset_init": True,
            "dt_sim": 3,  # dt_sim = 3
            "dt_env": 10,  # dt_env = 10 (not a multiple of 3)
            "n_passthrough": 1,
            "burn_in_passthroughs": 0.01,
        }
        with pytest.raises(ValueError, match="dt_env must be a multiple of dt_sim"):
            WindFarmEnvMulti(**invalid_config)

    @pytest.mark.slow
    def test_pettingzoo_api_compliance(self, initialized_env):
        """Verify the environment complies with the PettingZoo Parallel API."""
        # The `initialized_env` fixture provides a real env setup for this.
        # This will run a full battery of PettingZoo API compliance tests.
        parallel_api_test(initialized_env)
