import pytest
import numpy as np
from WindGym.Agents.PyWakeAgent import PyWakeAgent, NoisyPyWakeAgent

# Import real components needed for the tests
from WindGym.core.measurement_manager import (
    MeasurementManager,
    MeasurementType,
    MeasurementSpec,  # FIX: Add missing import
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    HybridNoiseModel,
    NoisyWindFarmEnv,
)

# --- Noise Model Unit Tests (Unchanged) ---


class TestWhiteNoiseModel:
    def test_initialization(self):
        std_devs = {MeasurementType.WIND_SPEED: 0.5}
        model = WhiteNoiseModel(std_devs)
        assert model.noise_std_devs == std_devs

    def test_apply_noise(self, basic_specs):
        std_devs = {MeasurementType.WIND_SPEED: 1.0}
        model = WhiteNoiseModel(std_devs)
        clean_obs = np.array([0.5, 0.5], dtype=np.float32)
        noisy_obs = model.apply_noise(clean_obs, basic_specs, np.random.default_rng(42))
        assert not np.allclose(noisy_obs, clean_obs)
        assert noisy_obs[1] == clean_obs[1]


class TestEpisodicBiasNoiseModel:
    def test_bias_is_constant_within_episode(self, basic_specs):
        bias_ranges = {MeasurementType.WIND_SPEED: (1.0, 1.0)}
        model = EpisodicBiasNoiseModel(bias_ranges)
        clean_obs = np.array([0.0, 0.0], dtype=np.float32)
        model.reset_noise(basic_specs, np.random.default_rng(42))
        noisy_obs_1 = model.apply_noise(
            clean_obs, basic_specs, np.random.default_rng(42)
        )
        noisy_obs_2 = model.apply_noise(
            clean_obs, basic_specs, np.random.default_rng(42)
        )
        assert np.allclose(noisy_obs_1, noisy_obs_2)

    def test_bias_resamples_across_episodes(self, basic_specs):
        bias_ranges = {MeasurementType.WIND_SPEED: (-2.0, 2.0)}
        model = EpisodicBiasNoiseModel(bias_ranges)
        clean_obs = np.array([0.0, 0.0], dtype=np.float32)
        rng1, rng2 = np.random.default_rng(42), np.random.default_rng(43)
        model.reset_noise(basic_specs, rng1)
        noisy_obs_1 = model.apply_noise(clean_obs, basic_specs, rng1)
        model.reset_noise(basic_specs, rng2)
        noisy_obs_2 = model.apply_noise(clean_obs, basic_specs, rng2)
        assert not np.allclose(noisy_obs_1, noisy_obs_2)


class TestHybridNoiseModel:
    def test_hybrid_noise_combines_models(self, basic_specs):
        clean_obs = np.zeros(2, dtype=np.float32)

        # Path 1: "Expected" - Replicate the hybrid model's internal calls
        rng_expected = np.random.default_rng(42)
        bias_model_expected = EpisodicBiasNoiseModel(
            {MeasurementType.WIND_SPEED: (1.0, 1.0)}
        )
        bias_model_expected.reset_noise(basic_specs, rng_expected)
        white_model_expected = WhiteNoiseModel({MeasurementType.WIND_SPEED: 1.0})
        # Note: The order of application matters in HybridNoiseModel.
        # It applies models sequentially to the *current* observations.
        # So, we should apply white noise first, then bias to the result of white noise.
        intermediate_obs_w = white_model_expected.apply_noise(
            clean_obs, basic_specs, rng_expected
        )
        final_obs_b_after_w = bias_model_expected.apply_noise(
            intermediate_obs_w, basic_specs, rng_expected
        )
        expected_total_noise = final_obs_b_after_w - clean_obs

        # Path 2: "Actual" - Use the hybrid model directly
        rng_hybrid = np.random.default_rng(42)
        hybrid_model = HybridNoiseModel(
            [
                WhiteNoiseModel({MeasurementType.WIND_SPEED: 1.0}),
                EpisodicBiasNoiseModel({MeasurementType.WIND_SPEED: (1.0, 1.0)}),
            ]
        )

        hybrid_model.reset_noise(basic_specs, rng_hybrid)
        actual_noisy_obs = hybrid_model.apply_noise(clean_obs, basic_specs, rng_hybrid)
        actual_total_noise = actual_noisy_obs - clean_obs

        assert np.allclose(actual_total_noise, expected_total_noise)


# --- Integration / Smoke Tests using a Real Environment ---


class TestManagerWithRealEnv:
    def test_manager_builds_specs_from_real_env(self, smoke_test_env):
        manager = MeasurementManager(smoke_test_env, seed=42)
        specs = manager.specs

        # From smoke_test_env YAML: 2 turbines * (ws+wd+yaw) + 1 farm * (ws+TI) = 2*3 + 2 = 8
        assert len(specs) == 8
        assert all(isinstance(s, MeasurementSpec) for s in specs)
        assert specs[0].name == "turb_0/ws_current"
        assert (
            specs[7].name == "farm/TI"
        )  # This is correct based on original YAML and code

    def test_manager_applies_noise_and_clips(self, smoke_test_env):
        manager = MeasurementManager(smoke_test_env, seed=42)
        # Apply a large bias to ensure clipping occurs
        # The range for WS is 2.0 to 25.0 in smoke_test_env.
        # Scaled to [-1, 1], so a clean_obs of 0 (midpoint) is 13.5 m/s.
        # A bias of 200 m/s will definitely push it past max_val (25.0) and thus clip to 1.0.
        bias_model = EpisodicBiasNoiseModel(
            {MeasurementType.WIND_SPEED: (200.0, 200.0)}
        )
        manager.set_noise_model(bias_model)
        manager.reset_noise()

        clean_obs = np.zeros_like(smoke_test_env.observation_space.sample())
        noisy_obs, info = manager.apply_noise(clean_obs)

        # Assuming turb_0/ws_current is the first element and has WS type
        # Check that the scaled value is clipped to 1.0
        assert noisy_obs[0] == pytest.approx(1.0)
        assert np.all(noisy_obs <= 1.0 + 1e-6) and np.all(
            noisy_obs >= -1.0 - 1e-6
        )  # Add tolerance


class TestNoisyWindFarmEnvWrapperSmoke:
    def test_wrapper_instantiation_and_spaces(self, smoke_test_env):
        manager = MeasurementManager(smoke_test_env, seed=42)
        wrapped_env = NoisyWindFarmEnv(
            base_env_class=lambda **kwargs: smoke_test_env, measurement_manager=manager
        )
        assert wrapped_env.observation_space == smoke_test_env.observation_space
        assert wrapped_env.action_space == smoke_test_env.action_space

    def test_wrapper_adds_noise_on_reset_and_step(self, smoke_test_env):
        manager = MeasurementManager(smoke_test_env, seed=123)
        # Using a small non-zero noise to ensure it's not the exact same as clean_obs
        noise_model = WhiteNoiseModel(noise_std_devs={MeasurementType.WIND_SPEED: 0.1})
        manager.set_noise_model(noise_model)

        wrapped_env = NoisyWindFarmEnv(
            base_env_class=lambda **kwargs: smoke_test_env, measurement_manager=manager
        )

        # Test reset
        noisy_obs_reset, info_reset = wrapped_env.reset()

        assert "clean_obs" in info_reset
        assert not np.allclose(noisy_obs_reset, info_reset["clean_obs"])
        assert "noise_info" in info_reset
        assert info_reset["noise_info"]["noise_type"] == "white"

        # Test step
        action = wrapped_env.action_space.sample()
        noisy_obs_step, _, _, _, info_step = wrapped_env.step(action)

        assert "clean_obs" in info_step
        assert not np.allclose(noisy_obs_step, info_step["clean_obs"])
        assert "noise_info" in info_step
        assert info_step["noise_info"]["noise_type"] == "white"


class TestNoisyPyWakeAgent:
    """Tests for the NoisyPyWakeAgent."""

    def test_initialization(self, smoke_test_env):
        """Test that the agent initializes correctly."""
        manager = MeasurementManager(smoke_test_env, seed=42)
        agent = NoisyPyWakeAgent(
            measurement_manager=manager,
            x_pos=smoke_test_env.x_pos,
            y_pos=smoke_test_env.y_pos,
            turbine=smoke_test_env.turbine,
        )
        assert agent.mm is manager
        assert hasattr(agent, "predict")
        assert hasattr(agent, "optimize")

    def test_estimate_wind_from_obs(self, smoke_test_env):
        """
        Tests the core logic of _estimate_wind_from_obs by providing a
        known observation vector and checking the un-scaled, averaged output.
        """
        # 1. Setup
        manager = MeasurementManager(smoke_test_env, seed=42)
        agent = NoisyPyWakeAgent(
            measurement_manager=manager,
            x_pos=smoke_test_env.x_pos,
            y_pos=smoke_test_env.y_pos,
            turbine=smoke_test_env.turbine,
        )

        # 2. Create a known observation vector based on env specs
        # Updated WS physical range for unscaling: [2.0, 25.0] -> smoke_test_env now passes these explicitly
        # Updated WD physical range for unscaling: [265.0, 275.0] -> smoke_test_env now passes these explicitly

        # Get the actual specs from the manager for dynamic indexing
        ws_spec_0 = next(s for s in manager.specs if s.name == "turb_0/ws_current")
        wd_spec_0 = next(s for s in manager.specs if s.name == "turb_0/wd_current")
        ws_spec_1 = next(s for s in manager.specs if s.name == "turb_1/ws_current")
        wd_spec_1 = next(s for s in manager.specs if s.name == "turb_1/wd_current")
        farm_ws_spec = next(s for s in manager.specs if s.name == "farm/ws_current")

        obs = np.zeros(manager.env.observation_space.shape[0], dtype=np.float32)

        # Define target physical values within the configured ranges
        target_ws_0 = 13.5
        target_wd_0 = 267.5
        target_ws_1 = 19.25
        target_wd_1 = 265.0
        target_farm_ws = 13.5

        # Scale physical values to [-1, 1] range for the observation vector
        # Use MeasurementManager's scaling methods for consistency
        scaled_ws_0 = manager._scale_value(
            np.array([target_ws_0]), ws_spec_0.min_val, ws_spec_0.max_val
        ).item()
        scaled_wd_0 = manager._scale_value(
            np.array([target_wd_0]), wd_spec_0.min_val, wd_spec_0.max_val
        ).item()
        scaled_ws_1 = manager._scale_value(
            np.array([target_ws_1]), ws_spec_1.min_val, ws_spec_1.max_val
        ).item()
        scaled_wd_1 = manager._scale_value(
            np.array([target_wd_1]), wd_spec_1.min_val, wd_spec_1.max_val
        ).item()
        scaled_farm_ws = manager._scale_value(
            np.array([target_farm_ws]), farm_ws_spec.min_val, farm_ws_spec.max_val
        ).item()

        obs[ws_spec_0.index_range[0]] = scaled_ws_0
        obs[wd_spec_0.index_range[0]] = scaled_wd_0
        obs[ws_spec_1.index_range[0]] = scaled_ws_1
        obs[wd_spec_1.index_range[0]] = scaled_wd_1
        obs[farm_ws_spec.index_range[0]] = scaled_farm_ws

        # 3. Manually calculate expected averaged physical values
        expected_ws_avg = np.mean([target_ws_0, target_ws_1, target_farm_ws])
        expected_wd_avg = np.mean(
            [target_wd_0, target_wd_1]
        )  # Only turbine WDs are active in smoke_test_env by default

        # 4. Run the method and assert
        est_ws, est_wd = agent._estimate_wind_from_obs(obs)

        assert np.isclose(est_ws, expected_ws_avg)
        assert np.isclose(est_wd, expected_wd_avg)

    def test_predict_reoptimizes_based_on_noisy_obs(self, smoke_test_env):
        """
        Tests that predict() uses the noisy observation to re-optimize,
        resulting in a different action than an agent with perfect information.
        """
        # 1. Setup Manager and Agents
        manager = MeasurementManager(smoke_test_env, seed=42)

        # Agent with perfect information
        perfect_agent = PyWakeAgent(
            x_pos=smoke_test_env.x_pos,
            y_pos=smoke_test_env.y_pos,
            turbine=smoke_test_env.turbine,
        )

        # Agent that receives noisy observations
        noisy_agent = NoisyPyWakeAgent(
            measurement_manager=manager,
            x_pos=smoke_test_env.x_pos,
            y_pos=smoke_test_env.y_pos,
            turbine=smoke_test_env.turbine,
        )

        # 2. Define wind conditions
        true_ws, true_wd, true_ti = 10.0, 270.0, 0.07  # From smoke_test_env config

        # 3. Get action from the agent with perfect info
        perfect_agent.update_wind(
            wind_speed=true_ws, wind_direction=true_wd, TI=true_ti
        )
        # Set agent's env to the unwrapped base env for direct access, as if it has perfect info
        perfect_agent.env = (
            smoke_test_env.unwrapped
        )  # Use unwrapped for direct access to attributes
        action_perfect, _ = perfect_agent.predict(
            obs=None
        )  # obs=None because it has perfect info

        noisy_agent.env = smoke_test_env  # Noisy agent operates on the wrapped env

        # 4. Create a noisy observation vector implying different wind conditions
        # Get the actual specs from the manager for dynamic indexing
        ws_spec_0 = next(s for s in manager.specs if s.name == "turb_0/ws_current")
        wd_spec_0 = next(s for s in manager.specs if s.name == "turb_0/wd_current")
        ws_spec_1 = next(s for s in manager.specs if s.name == "turb_1/ws_current")
        wd_spec_1 = next(s for s in manager.specs if s.name == "turb_1/wd_current")
        farm_ws_spec = next(s for s in manager.specs if s.name == "farm/ws_current")

        # Target implied noisy wind conditions
        implied_ws = 20.0
        implied_wd = 274.0

        scaled_ws_implied = manager._scale_value(
            np.array([implied_ws]), ws_spec_0.min_val, ws_spec_0.max_val
        ).item()
        scaled_wd_implied = manager._scale_value(
            np.array([implied_wd]), wd_spec_0.min_val, wd_spec_0.max_val
        ).item()

        noisy_obs = np.zeros(manager.env.observation_space.shape[0], dtype=np.float32)

        # Populate all relevant observation slots with the implied noisy values
        for spec in manager.specs:
            if spec.measurement_type == MeasurementType.WIND_SPEED:
                noisy_obs[spec.index_range[0] : spec.index_range[1]] = scaled_ws_implied
            elif spec.measurement_type == MeasurementType.WIND_DIRECTION:
                noisy_obs[spec.index_range[0] : spec.index_range[1]] = scaled_wd_implied

        # 5. Get action from the noisy agent
        action_noisy, _ = noisy_agent.predict(obs=noisy_obs)

        # 6. Assertions
        # The noisy agent should have updated its internal state to the estimated values
        assert np.isclose(noisy_agent.wsp[0], implied_ws)
        assert np.isclose(noisy_agent.wdir[0], implied_wd)

        # Because the perceived wind conditions are different, the optimal yaw angles
        # (and thus the scaled actions) must also be different.
        assert not np.allclose(
            action_perfect, action_noisy, atol=1e-5
        )  # Add a small tolerance
