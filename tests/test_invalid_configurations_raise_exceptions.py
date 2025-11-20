import pytest
import yaml
import tempfile
import os
import re  # For regex matching
import copy  # For deepcopy
import numpy as np  # For dummy_action
from WindGym import WindFarmEnv
from py_wake.examples.data.hornsrev1 import V80  # Default turbine
from WindGym.utils.generate_layouts import generate_square_grid


# --- Helper to get a base, valid YAML dictionary ---
def get_base_valid_yaml_dict():
    return {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
        },
        "wind": {
            "ws_min": 8,
            "ws_max": 10,
            "TI_min": 0.05,
            "TI_max": 0.10,
            "wd_min": 268,
            "wd_max": 272,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {
            "Power_reward": "Baseline",
            "Power_avg": 50,
            "Power_scaling": 1.0,
        },
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
            "ws_rolling_mean": True,
            "ws_history_N": 1,
            "ws_history_length": 5,
            "ws_window_length": 2,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": 1,
            "wd_history_length": 5,
            "wd_window_length": 2,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": 1,
            "yaw_history_length": 5,
            "yaw_window_length": 2,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": 1,
            "power_history_length": 5,
            "power_window_length": 2,
        },
    }


@pytest.fixture
def env_config_for_exception_test(request):
    param_info = request.param
    yaml_config_overrides = param_info["yaml_config_kwargs"]  # Renamed for clarity
    test_id = param_info.get("id", "UnknownID")

    # Start with a fresh deep copy of the base config for each parameterization
    current_config_dict = copy.deepcopy(get_base_valid_yaml_dict())

    # Apply the overrides from yaml_config_overrides.
    # If a key in yaml_config_overrides is also a top-level key in current_config_dict,
    # its value in current_config_dict will be replaced.
    # This is critical: if yaml_config_overrides is {"mes_level": {"turb_ws": True}},
    # then current_config_dict["mes_level"] becomes exactly {"turb_ws": True}.
    for key, value in yaml_config_overrides.items():
        current_config_dict[key] = value

    # ---- START TARGETED DEBUG PRINTS ----
    if test_id == "MissingKeyInMesLevel":
        print(f"\nDEBUG INFO for test ID: {test_id}")
        print(
            f"  Initial current_config_dict['mes_level'] (from base): {get_base_valid_yaml_dict()['mes_level']}"
        )
        print(
            f"  yaml_config_overrides specific for this test: {yaml_config_overrides}"
        )
        print(
            f"  Value for 'mes_level' in yaml_config_overrides: {yaml_config_overrides.get('mes_level')}"
        )
        print(
            f"  current_config_dict['mes_level'] AFTER direct assignment: {current_config_dict.get('mes_level')}"
        )
        print(
            f"  Does current_config_dict['mes_level'] have 'turb_wd' key? {'turb_wd' in current_config_dict.get('mes_level', {})}"
        )
    elif test_id == "MissingKeyInWsMes":
        print(f"\nDEBUG INFO for test ID: {test_id}")
        print(
            f"  Initial current_config_dict['ws_mes'] (from base): {get_base_valid_yaml_dict()['ws_mes']}"
        )
        print(
            f"  yaml_config_overrides specific for this test: {yaml_config_overrides}"
        )
        print(
            f"  Value for 'ws_mes' in yaml_config_overrides: {yaml_config_overrides.get('ws_mes')}"
        )
        print(
            f"  current_config_dict['ws_mes'] AFTER direct assignment: {current_config_dict.get('ws_mes')}"
        )
        print(
            f"  Does current_config_dict['ws_mes'] have 'ws_current' key? {'ws_current' in current_config_dict.get('ws_mes', {})}"
        )
    # ---- END TARGETED DEBUG PRINTS ----

    temp_yaml_filepath = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml", encoding="utf-8"
        ) as tmp_yaml_file:
            # print(f"DEBUG: Dumping to YAML for test '{test_id}': {current_config_dict}") # General dump for all tests if needed
            yaml.dump(current_config_dict, tmp_yaml_file)
            temp_yaml_filepath = tmp_yaml_file.name

        yield {
            "yaml_path": temp_yaml_filepath,
            "expected_exception_info": param_info["expected_exception_info"],
            "id": test_id,
            "requires_step_to_trigger": param_info.get(
                "requires_step_to_trigger", False
            ),
        }
    finally:
        if temp_yaml_filepath and os.path.exists(temp_yaml_filepath):
            os.remove(temp_yaml_filepath)


class TestInvalidConfigurations:
    INVALID_CONFIG_PARAMS = [
        pytest.param(
            {
                "id": "InvalidPowerReward",
                "yaml_config_kwargs": {
                    "power_def": {
                        "Power_reward": "InvalidRewardType",
                        "Power_avg": 50,
                        "Power_scaling": 1.0,
                    }
                },
                "expected_exception_info": (
                    ValueError,
                    "The Power_reward must be either Baseline, Power_avg, None or Power_diff",
                ),
            },
            id="InvalidPowerReward",
        ),
        pytest.param(
            {
                "id": "InvalidBaseController",
                "yaml_config_kwargs": {"BaseController": "InvalidController"},
                "expected_exception_info": (
                    ValueError,
                    re.escape(
                        "BaseController must be one of: 'Local', 'Global', 'PyWake[_oracle|_local]'."
                    ),
                ),
            },
            id="InvalidBaseController",
        ),
        pytest.param(
            {
                "id": "InvalidActionMethod",
                "yaml_config_kwargs": {"ActionMethod": "InvalidAction"},
                "expected_exception_info": (
                    ValueError,
                    r"The ActionMethod must be yaw, wind or absolute",
                ),
                "requires_step_to_trigger": True,
            },
            id="InvalidActionMethod",
        ),
        pytest.param(
            {
                "id": "InvalidPowerDiffAvgTooSmall",
                "yaml_config_kwargs": {
                    "power_def": {
                        "Power_reward": "Power_diff",
                        "Power_avg": 10,
                        "Power_scaling": 1.0,
                    }
                },
                "expected_exception_info": (
                    ValueError,
                    "The Power_avg must be larger then 40 for the Power_diff reward",
                ),
            },
            id="InvalidPowerDiffAvgTooSmall",
        ),
        # pytest.param(
        #     {
        #         "id": "MissingFarmSection",
        #         "yaml_config_kwargs": {"farm": None},
        #         "expected_exception_info": (
        #             TypeError,
        #             r"'NoneType' object is not subscriptable",
        #         ),
        #     },
        #     id="MissingFarmSection",
        # ),
        # pytest.param(
        #     {
        #         "id": "MissingWindSection",
        #         "yaml_config_kwargs": {"wind": None},
        #         "expected_exception_info": (
        #             TypeError,
        #             r"'NoneType' object is not subscriptable",
        #         ),
        #     },
        #     id="MissingWindSection",
        # ),
        # pytest.param(
        #     {
        #         "id": "MissingPowerDefSection",
        #         "yaml_config_kwargs": {"power_def": None},
        #         "expected_exception_info": (
        #             TypeError,
        #             r"'NoneType' object is not subscriptable",
        #         ),
        #     },
        #     id="MissingPowerDefSection",
        # ),
        # pytest.param(
        #     {
        #         "id": "InvalidTypeForFarmNx",MissingFarmSection
        #         "yaml_config_kwargs": {
        #             "farm": {
        #                 "yaw_min": -30,
        #                 "yaw_max": 30,
        #             }
        #         },
        #         "expected_exception_info": (
        #             TypeError,
        #             r"'str' object cannot be interpreted as an integer",
        #         ),
        #     },
        #     id="InvalidTypeForFarmNx",
        # ),
        pytest.param(
            {
                "id": "MissingKeyInMesLevel",
                "yaml_config_kwargs": {
                    "mes_level": {"turb_ws": True}
                },  # Intends to make mes_level sparse
                "expected_exception_info": (KeyError, r"'turb_wd'"),
            },
            id="MissingKeyInMesLevel",
        ),
        pytest.param(
            {
                "id": "MissingKeyInWsMes",
                "yaml_config_kwargs": {
                    "ws_mes": {"ws_rolling_mean": True}
                },  # Intends to make ws_mes sparse
                "expected_exception_info": (KeyError, r"'ws_current'"),
            },
            id="MissingKeyInWsMes",
        ),
    ]

    @pytest.mark.parametrize(
        "env_config_for_exception_test", INVALID_CONFIG_PARAMS, indirect=True
    )
    def test_invalid_configurations_raise_exceptions(
        self, env_config_for_exception_test
    ):
        yaml_path = env_config_for_exception_test["yaml_path"]
        expected_exception_type, expected_msg_regex = env_config_for_exception_test[
            "expected_exception_info"
        ]
        requires_step = env_config_for_exception_test.get(
            "requires_step_to_trigger", False
        )
        test_id = env_config_for_exception_test["id"]

        with pytest.raises(
            expected_exception_type, match=expected_msg_regex
        ) as excinfo:
            env = None
            try:
                x_pos, y_pos = generate_square_grid(
                    turbine=V80(), nx=2, ny=1, xDist=5, yDist=3
                )
                env = WindFarmEnv(
                    turbine=V80(),
                    x_pos=x_pos,
                    y_pos=y_pos,
                    config=yaml_path,
                    seed=123,
                    reset_init=True,
                    turbtype="None",
                    n_passthrough=0.1,
                    burn_in_passthroughs=0.01,
                )
                if requires_step:
                    if env.action_space is None:
                        pytest.fail(
                            f"Test ID '{test_id}': Action space not initialized before attempting step for a 'requires_step_to_trigger' test."
                        )
                    dummy_action = np.zeros(
                        env.action_space.shape, dtype=env.action_space.dtype
                    )
                    env.step(dummy_action)
            finally:
                if env:
                    env.close()
