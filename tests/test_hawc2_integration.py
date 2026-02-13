# tests/test_hawc2_integration.py
import pytest
import numpy as np
import os
import tempfile
import yaml
import shutil
from unittest.mock import patch, MagicMock, PropertyMock

# Import classes to be tested or used in tests
from WindGym import WindFarmEnv, FarmEval, AgentEvalFast
from WindGym.Agents import ConstantAgent
from py_wake.examples.data.hornsrev1 import V80
from dynamiks.wind_turbines.hawc2_windturbine import HAWC2WindTurbines
from WindGym.utils.generate_layouts import generate_square_grid


# --- Fixtures for HAWC2 Testing ---


@pytest.fixture
def temp_yaml_file_factory():
    """Factory fixture to create temporary YAML files for tests."""
    created_files = []

    def _create_temp_yaml(content, name_suffix=""):
        if isinstance(content, dict):
            content_str = yaml.dump(content)
        else:
            content_str = str(content)

        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f"_{name_suffix}.yaml", encoding="utf-8"
        )
        tf.write(content_str)
        filepath = tf.name
        tf.close()
        created_files.append(filepath)
        return filepath

    yield _create_temp_yaml

    for f_path in created_files:
        if os.path.exists(f_path):
            os.remove(f_path)


@pytest.fixture
def temp_htc_file(tmp_path):
    """Creates a dummy HTC file path for testing. Its content doesn't matter."""
    htc_path = tmp_path / "dummy_turbine.htc"
    htc_path.touch()
    return str(htc_path)


@pytest.fixture
def hawc2_env_config_yaml(temp_yaml_file_factory):
    """Provides a complete YAML config suitable for HAWC2 tests, fixing the KeyError."""
    config = {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10,
            "ws_max": 10,
            "TI_min": 0.07,
            "TI_max": 0.07,
            "wd_min": 270,
            "wd_max": 270,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": True,
            "turb_power": True,
            "farm_ws": True,
            "farm_wd": True,
            "farm_TI": True,
            "farm_power": True,
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": False,
            "ws_history_N": 0,
            "ws_history_length": 1,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": False,
            "wd_history_N": 0,
            "wd_history_length": 1,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 0,
            "yaw_history_length": 1,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": False,
            "power_history_N": 0,
            "power_history_length": 1,
            "power_window_length": 1,
        },
    }
    return temp_yaml_file_factory(config, "hawc2_config")


# tests/test_hawc2_integration.py


@pytest.fixture
def mock_hawc2_wind_turbines():
    """
    Mocks HAWC2WindTurbines to handle multiple instantiations (agent and baseline)
    by using a factory function with side_effect.
    """

    def create_single_turbine_mock():
        # ... (this part is correct, no changes needed)
        single_mock = MagicMock()
        single_mock.ct.return_value = 0.8
        single_mock.diameter.return_value = np.array([V80().diameter()])
        single_mock.rotor_avg_ti.return_value = np.array([0.07])
        type(single_mock).rotor_avg_windspeed = PropertyMock(
            return_value=np.array([10.0, 0.0, 0.0])
        )

        def induction_side_effect(r_input):
            return np.full((len(r_input), 1), 0.1)

        single_mock.axisymetric_induction.side_effect = induction_side_effect

        return single_mock

    def mock_factory(*args, **kwargs):
        collection_mock = MagicMock()

        x_pos = args[0] if args else kwargs.get("x", [])
        y_pos = args[1] if len(args) > 1 else kwargs.get("y", [])
        num_turbines = len(x_pos)

        # --- Directly set data attributes ---
        z_pos = np.full_like(np.asarray(x_pos), 90.0)
        collection_mock.positions_east_north = np.array([x_pos, y_pos, z_pos])
        collection_mock.N = num_turbines
        collection_mock.yaw = np.zeros(num_turbines)
        collection_mock.step_handlers = []

        # --- Configure methods called on the collection ---
        collection_mock.diameter.return_value = np.full(num_turbines, V80().diameter())
        collection_mock.power.return_value = np.full(num_turbines, 1e6)
        collection_mock.add_sensor = MagicMock()
        collection_mock.yaw_tilt.return_value = (
            np.zeros(num_turbines),
            np.zeros(num_turbines),
        )

        # --- Configure properties called on the collection ---
        type(collection_mock).positions_xyz = PropertyMock(
            return_value=np.array([x_pos, y_pos, z_pos])
        )
        type(collection_mock).rotor_positions_xyz = PropertyMock(
            return_value=np.array([x_pos, y_pos, z_pos])
        )
        type(collection_mock).rotor_avg_windspeed = PropertyMock(
            return_value=np.zeros((num_turbines, 3))
        )

        # --- Configure nested mock objects ---
        collection_mock.h2 = MagicMock(close=MagicMock(), write_output=MagicMock())

        # FIX: Create a list of mock htc objects, one for each turbine
        htc_list = []
        for i in range(num_turbines):
            mock_htc_obj = MagicMock()
            # It's good practice to make mock filenames unique
            filename = f"mock_results_file_{i}"
            type(mock_htc_obj.output.filename).values = PropertyMock(
                return_value=[filename]
            )
            mock_htc_obj.modelpath = "/mock/path/"
            htc_list.append(mock_htc_obj)
        collection_mock.htc_lst = htc_list

        collection_mock.sensors = MagicMock()

        # --- Configure indexing to return the single-turbine mock ---
        collection_mock.__getitem__.return_value = create_single_turbine_mock()

        return collection_mock

    # Patch both locations where HAWC2WindTurbines is used
    with (
        patch(
            "WindGym.wind_farm_env.HAWC2WindTurbines", side_effect=mock_factory
        ) as mock_class_env,
        patch(
            "WindGym.core.baseline_manager.HAWC2WindTurbines", side_effect=mock_factory
        ) as mock_class_baseline,
    ):
        yield mock_class_env


@patch("shutil.rmtree")
@patch("WindGym.agent_eval.gtsdf.load")
def test_agent_eval_fast_with_hawc2(
    mock_gtsdf_load,
    mock_rmtree,
    hawc2_env_config_yaml,
    temp_htc_file,
    mock_hawc2_wind_turbines,
):
    """
    Tests the full AgentEvalFast workflow with a mocked HAWC2 environment.
    This covers the `elif env.HTC_path is not None:` block in `AgentEvalFast`.
    """
    with patch("dynamiks.flow_simulation.FlowSimulation.step"):
        # 1. ARRANGE
        dummy_time = np.arange(10)
        dummy_data = np.zeros((10, 113))
        mock_gtsdf_load.return_value = (dummy_time, dummy_data, {})

        x_pos, y_pos = generate_square_grid(V80(), nx=2, ny=1, xDist=7, yDist=7)

        env = FarmEval(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=hawc2_env_config_yaml,
            HTC_path=temp_htc_file,
            reset_init=True,
            turbtype="None",
        )
        model = ConstantAgent(yaw_angles=[10.0, 5.0])

        # 2. ACT
        AgentEvalFast(
            env,
            model,
            model_step=1,
            ws=10,
            ti=0.07,
            wd=270,
            t_sim=10,
            return_loads=True,  # This is crucial to trigger the HAWC2 logic
            cleanup=True,
        )

        # 3. ASSERT
        # Assert on the `env.wts` object that was used inside AgentEvalFast.
        # The cleanup logic does not delete `env.wts`, so it's still accessible.
        env.wts.h2.write_output.assert_called_once()
        mock_gtsdf_load.assert_called()
        env.wts.h2.close.assert_called_once()

        # Assert that the baseline turbine's resources were also closed.
        env.wts_baseline.h2.close.assert_called_once()

        # Assert that the temporary folders were removed.
        mock_rmtree.assert_called()

        env.close()


def test_cleanup_on_truncation(
    hawc2_env_config_yaml, temp_htc_file, mock_hawc2_wind_turbines
):
    """Tests that the environment cleans up HAWC2 files when an episode is truncated."""
    with patch("dynamiks.flow_simulation.FlowSimulation.step"):
        # 1. ARRANGE
        x_pos, y_pos = generate_square_grid(V80(), nx=2, ny=1, xDist=7, yDist=7)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=hawc2_env_config_yaml,
            HTC_path=temp_htc_file,
            reset_init=True,
            n_passthrough=0.1,
            burn_in_passthroughs=0.1,
            turbtype="None",
        )
        mock_h2_instance = env.wts.h2

        with patch.object(
            env, "_deleteHAWCfolder", wraps=env._deleteHAWCfolder
        ) as spy_delete:
            with patch("shutil.rmtree"):
                # 2. ACT
                terminated, truncated = False, False
                while not (terminated or truncated):
                    _, _, terminated, truncated, _ = env.step(env.action_space.sample())

                # 3. ASSERT
                mock_h2_instance.close.assert_called_once()
                spy_delete.assert_called_once()

        env.close()
