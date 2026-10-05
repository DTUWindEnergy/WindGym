import pytest
import yaml
import numpy as np
from pathlib import Path
import gymnasium as gym

from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80
from gymnasium.utils.env_checker import check_env, data_equivalence
from WindGym.utils.generate_layouts import generate_square_grid

from test_utils import (
    calculate_expected_obs_dim,
    get_base_yaml_dict,
    make_fast_pywake_env,
)


class TestSpecificFeatures:
    @pytest.mark.parametrize(
        "mes_level_config",
        [
            {
                "turb_ws": True,
                "turb_wd": True,
                "turb_TI": True,
                "turb_power": True,
                "farm_ws": True,
                "farm_wd": True,
                "farm_TI": True,
                "farm_power": True,
            },
            {
                "turb_ws": False,
                "turb_wd": False,
                "turb_TI": False,
                "turb_power": False,
                "farm_ws": False,
                "farm_wd": False,
                "farm_TI": False,
                "farm_power": False,
            },
            {
                "turb_ws": True,
                "turb_wd": False,
                "turb_TI": True,
                "turb_power": False,
                "farm_ws": True,
                "farm_wd": False,
                "farm_TI": False,
                "farm_power": True,
            },
        ],
        ids=["All_Features_On", "All_Features_Off", "Mixed_Features"],
    )
    def test_get_num_raw_features_coverage(
        self, temp_yaml_filepath_factory, mes_level_config, mock_mann_methods
    ):
        """Tests `_get_num_raw_features` by varying the `mes_level` config."""
        n_turb_nx, n_turb_ny = 2, 1
        config_dict = get_base_yaml_dict(nx=n_turb_nx, ny=n_turb_ny)
        config_dict["mes_level"] = mes_level_config
        yaml_filepath = temp_yaml_filepath_factory(config_dict, "num_raw_features_test")

        x_pos, y_pos = generate_square_grid(
            turbine=V80(), nx=n_turb_nx, ny=n_turb_ny, xDist=5, yDist=3
        )

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
        )

        expected_count = 0
        n_turb = n_turb_nx * n_turb_ny
        if mes_level_config.get("turb_ws", False):
            expected_count += n_turb
        if mes_level_config.get("turb_wd", False):
            expected_count += n_turb
        if mes_level_config.get("turb_TI", False):
            expected_count += n_turb
        if mes_level_config.get("turb_power", False):
            expected_count += n_turb
        if mes_level_config.get("farm_ws", False):
            expected_count += 1
        if mes_level_config.get("farm_wd", False):
            expected_count += 1
        if mes_level_config.get("farm_TI", False):
            expected_count += 1
        if mes_level_config.get("farm_power", False):
            expected_count += 1

        actual_count = env._get_num_raw_features()

        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} raw features, but got {actual_count}."

        env.close()

    @pytest.mark.parametrize(
        "fill_window_config_val, expected_steps_on_reset_factor_str",
        [(True, "hist_max"), (3, "val_3"), (False, "val_1"), (10, "hist_max_capped")],
    )
    def test_fill_window_logic(
        self,
        temp_yaml_filepath_factory,
        fill_window_config_val,
        expected_steps_on_reset_factor_str,
        mock_mann_methods,
    ):
        base_hist_len = 5
        config_dict = get_base_yaml_dict(history_length=base_hist_len, history_n=1)
        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict, f"fill_window_{str(fill_window_config_val).lower()}"
        )

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        hist_max_calculated = base_hist_len

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            fill_window=fill_window_config_val,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_steps = 0
        if expected_steps_on_reset_factor_str == "hist_max":
            expected_steps = hist_max_calculated
        elif expected_steps_on_reset_factor_str == "val_3":
            expected_steps = 3
        elif expected_steps_on_reset_factor_str == "val_1":
            expected_steps = 1
        elif expected_steps_on_reset_factor_str == "hist_max_capped":
            expected_steps = min(fill_window_config_val, hist_max_calculated)

        assert (
            env.steps_on_reset == expected_steps
        ), f"Calculated hist_max: {hist_max_calculated}"
        check_env(env, skip_render_check=True)
        env.close()

    @pytest.mark.parametrize(
        "turb_wd_enabled, farm_wd_enabled, wd_current, wd_rolling, wd_hist_n",
        [
            (True, True, True, True, 2),
            (False, False, True, False, 0),
            (True, False, True, False, 0),
            (False, True, False, True, 1),
            (True, True, False, False, 0),
        ],
    )
    def test_wind_direction_features_in_observations(
        self,
        temp_yaml_filepath_factory,
        mock_mann_methods,
        turb_wd_enabled,
        farm_wd_enabled,
        wd_current,
        wd_rolling,
        wd_hist_n,
    ):
        # Use nx=2 to avoid time_max=0 issue
        config_dict = get_base_yaml_dict(
            nx=2, ny=1, history_length=5, window_length=2, history_n=max(1, wd_hist_n)
        )

        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        config_dict["mes_level"]["turb_wd"] = turb_wd_enabled
        config_dict["mes_level"]["farm_wd"] = farm_wd_enabled

        config_dict["wd_mes"] = {
            "wd_current": wd_current,
            "wd_rolling_mean": wd_rolling,
            "wd_history_N": wd_hist_n,
            "wd_history_length": 5,
            "wd_window_length": 2,
        }
        config_dict["mes_level"]["turb_ws"] = False
        config_dict["mes_level"]["turb_TI"] = False
        config_dict["mes_level"]["turb_power"] = False
        config_dict["mes_level"]["farm_ws"] = False
        config_dict["mes_level"]["farm_TI"] = False
        config_dict["mes_level"]["farm_power"] = False

        config_dict["ws_mes"]["ws_current"] = False
        config_dict["ws_mes"]["ws_rolling_mean"] = False
        config_dict["ws_mes"]["ws_history_N"] = 0
        config_dict["power_mes"]["power_current"] = False
        config_dict["power_mes"]["power_rolling_mean"] = False
        config_dict["power_mes"]["power_history_N"] = 0
        config_dict["yaw_mes"] = {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 0,
            "yaw_history_length": 5,
            "yaw_window_length": 2,
        }

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict,
            f"obs_wd_{turb_wd_enabled}_{farm_wd_enabled}_{wd_current}_{wd_rolling}_{wd_hist_n}",
        )

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_dim = calculate_expected_obs_dim(config_dict, env.n_turb)
        actual_dim = env.observation_space.shape[0]

        assert (
            actual_dim == expected_dim
        ), f"Obs dim mismatch (YAML: {yaml.dump(config_dict)}). Expected {expected_dim}, got {actual_dim}."

        check_env(env, skip_render_check=True)
        env.close()

    @pytest.mark.parametrize(
        "mes_type_key, current_flag, rolling_flag, history_n_val",
        [
            ("ws_mes", True, False, 0),
            ("ws_mes", False, True, 3),
            ("ws_mes", True, True, 1),
            ("power_mes", True, False, 0),
            ("power_mes", False, True, 2),
        ],
    )
    def test_specific_measurement_configurations(
        self,
        temp_yaml_filepath_factory,
        mock_mann_methods,
        mes_type_key,
        current_flag,
        rolling_flag,
        history_n_val,
    ):
        n_turb_nx = 2  # Use nx=2 to avoid time_max=0
        config_dict = get_base_yaml_dict(
            nx=n_turb_nx,
            ny=1,
            history_length=10,
            window_length=3,
            history_n=max(1, history_n_val),
        )

        config_dict["power_def"]["Power_reward"] = "None"
        config_dict["act_pen"]["action_penalty"] = 0.0

        config_dict["mes_level"] = {
            "turb_ws": False,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        }

        prefix_under_test = mes_type_key.split("_")[0]
        config_dict["mes_level"][f"turb_{prefix_under_test}"] = True
        config_dict["mes_level"][f"farm_{prefix_under_test}"] = True

        config_dict[mes_type_key] = {
            f"{prefix_under_test}_current": current_flag,
            f"{prefix_under_test}_rolling_mean": rolling_flag,
            f"{prefix_under_test}_history_N": history_n_val,
            f"{prefix_under_test}_history_length": 10,
            f"{prefix_under_test}_window_length": 3,
        }

        for other_mes_key_loop in ["ws_mes", "wd_mes", "power_mes"]:
            if other_mes_key_loop != mes_type_key:
                prefix = other_mes_key_loop.split("_")[0]
                config_dict[other_mes_key_loop][f"{prefix}_current"] = False
                config_dict[other_mes_key_loop][f"{prefix}_rolling_mean"] = False
                config_dict[other_mes_key_loop][f"{prefix}_history_N"] = 0

        config_dict["yaw_mes"] = {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 0,
            "yaw_history_length": 10,
            "yaw_window_length": 3,
        }

        yaml_filepath = temp_yaml_filepath_factory(
            config_dict,
            f"obs_cfg_{mes_type_key}_{current_flag}_{rolling_flag}_{history_n_val}",
        )
        x_pos, y_pos = generate_square_grid(
            turbine=V80(), nx=n_turb_nx, ny=1, xDist=5, yDist=3
        )
        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
            Baseline_comp=False,
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )

        expected_dim = calculate_expected_obs_dim(
            config_dict, n_turb_nx
        )  # Pass correct n_turb
        actual_dim = env.observation_space.shape[0]

        assert (
            actual_dim == expected_dim
        ), f"Obs dim mismatch for {mes_type_key} (YAML: {yaml.dump(config_dict)}). Expected {expected_dim}, got {actual_dim}."

        check_env(env, skip_render_check=True)
        env.close()

    def test_track_power_requires_no_power_reward(
        self, temp_yaml_filepath_factory, mock_mann_methods
    ):
        """Track_power is mutually exclusive with a power maximization reward."""
        config_dict = get_base_yaml_dict()
        config_dict["Track_power"] = True  # Power_reward is "Baseline" here
        yaml_filepath = temp_yaml_filepath_factory(config_dict, "track_power_true")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        with pytest.raises(ValueError, match="mutually exclusive"):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                reset_init=True,
                seed=42,
                turbtype="None",
                n_passthrough=0.1,
                burn_in_passthroughs=0.01,
            )

    def test_track_power_env_construction(
        self, temp_yaml_filepath_factory, mock_mann_methods
    ):
        """With Power_reward "None", a tracking env constructs and passes check_env."""
        config_dict = get_base_yaml_dict()
        config_dict["Track_power"] = True
        config_dict["power_def"]["Power_reward"] = "None"
        yaml_filepath = temp_yaml_filepath_factory(config_dict, "track_power_valid")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            reset_init=True,
            seed=42,
            turbtype="None",
            n_passthrough=0.1,
            burn_in_passthroughs=0.01,
        )
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Episode accounting: truncation after ceil(time_max / step-seconds) env steps
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("dt", [1, 2])
def test_truncation_episode_length(dt):
    """Episode ends after ceil(time_max / dt_env) env steps (time_max is in
    seconds; comparing step counts against seconds gave up-to-2x episodes)."""
    env = make_fast_pywake_env(dt_sim=dt, dt_env=dt)
    env.reset(seed=0)
    expected_steps = int(np.ceil(env.time_max / env.dt_env))

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    truncated = False
    for _ in range(expected_steps + 10):  # safety bound
        _, _, _, truncated, _ = env.step(action)
        steps += 1
        if truncated:
            break

    assert truncated
    assert steps == expected_steps
    env.close()


# ---------------------------------------------------------------------------
# delay / ignore_steps: the agent-cadence feature (sim advances `delay`
# seconds per env step, only the final dt_env window is observed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delay_constructor_validation():
    # Not a multiple of dt_env
    with pytest.raises(ValueError, match="delay"):
        make_fast_pywake_env(delay=1.5)
    # Smaller than one env step
    with pytest.raises(ValueError, match="delay"):
        make_fast_pywake_env(delay=0.5)
    # Default: no delay
    env = make_fast_pywake_env()
    assert env.delay == env.dt_env
    env.close()


@pytest.mark.unit
def test_ignore_steps_validation():
    env = make_fast_pywake_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="ignore_steps"):
        env._advance_and_measure(2, ignore_steps=2)
    with pytest.raises(ValueError, match="ignore_steps"):
        env._advance_and_measure(2, ignore_steps=-1)
    env.close()


@pytest.mark.integration
def test_delay_advances_extra_sim_steps():
    """With delay = 2 * dt_env one env step advances the sim 2 * dt_env
    seconds (twice the sub-steps), and truncation accounts for delay-seconds
    per step so the sim never outruns time_max / the precomputed flow."""
    env = make_fast_pywake_env(delay=2)  # dt_sim = dt_env = 1 (pywake backend)
    env.reset(seed=0)
    t0 = float(env.fs.time)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, truncated, info = env.step(action)
    assert float(env.fs.time) - t0 == pytest.approx(env.delay)
    assert len(info["time_array"]) == 2  # steps_with_delay sub-steps recorded

    # Episode length: each step consumes `delay` sim-seconds, not dt_env
    expected_steps = int(np.ceil(env.time_max / env.delay))
    steps = 1
    while not truncated and steps < expected_steps + 10:  # safety bound
        _, _, _, truncated, _ = env.step(action)
        steps += 1
    assert truncated
    assert steps == expected_steps
    env.close()


@pytest.mark.integration
def test_delay_composes_with_sim_substeps_dynamiks():
    """dt_env > dt_sim and delay > dt_env together: one env step advances
    delay seconds = (delay / dt_env) * sim_steps_per_env_step sub-steps."""
    env = make_fast_pywake_env(
        backend="dynamiks", turbtype="None", dt_sim=1, dt_env=2, delay=4
    )
    obs0, _ = env.reset(seed=0)
    t0 = float(env.fs.time)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, _, _, _, info = env.step(action)
    assert float(env.fs.time) - t0 == pytest.approx(4.0)
    assert len(info["time_array"]) == 4  # 2 ignored + 2 observed sub-steps
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    env.close()


# ---------------------------------------------------------------------------
# max_time_steps: fixed episode length in env steps (parallel-env sync).
# time_max is always seconds; max_time_steps mode derives it as steps * delay.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_max_time_steps_truncates_after_exact_step_count():
    """Step counting starts only after reset's burn-in and sensor fill:
    with fill_window on (steps_on_reset > 1) the episode still runs exactly
    max_time_steps env steps after reset."""
    env = make_fast_pywake_env(max_time_steps=12, fill_window=True)
    env.reset(seed=0)
    assert env.time_max == 12  # steps * delay seconds (delay = dt_env = 1)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    truncated = False
    while not truncated and steps < 12 + 10:  # safety bound
        _, _, _, truncated, _ = env.step(action)
        steps += 1
    assert truncated
    assert steps == 12
    env.close()


@pytest.mark.integration
def test_max_time_steps_with_delay_sizes_time_max_in_seconds():
    """With delay > dt_env, time_max must cover steps * delay seconds so the
    wind-direction series spans the whole episode, while truncation still
    fires on the exact step count."""
    env = make_fast_pywake_env(max_time_steps=8, delay=2)
    env.reset(seed=0)
    assert env.time_max == 16  # 8 steps * 2 s, not 8

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    truncated = False
    while not truncated and steps < 8 + 10:  # safety bound
        _, _, _, truncated, _ = env.step(action)
        steps += 1
    assert truncated
    assert steps == 8
    env.close()


@pytest.mark.integration
def test_max_time_steps_disables_passthrough_method():
    """Setting max_time_steps disables the passthrough episode-length method:
    even a fixed length far below the ws-derived time_max (40 s in the fast
    config) runs exactly max_time_steps steps instead of truncating early."""
    env = make_fast_pywake_env(max_time_steps=5)
    assert env.n_passthrough == 999_999_999  # forced high at construction
    env.reset(seed=0)
    assert env.time_max == 5

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    truncated = False
    while not truncated and steps < 5 + 10:  # safety bound
        _, _, _, truncated, _ = env.step(action)
        steps += 1
    assert truncated
    assert steps == 5
    env.close()
