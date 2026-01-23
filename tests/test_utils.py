"""Shared utility functions for WindGym test suite."""


def get_base_yaml_dict(nx=2, ny=1, history_length=5, window_length=2, history_n=1):
    """
    Provides a base dictionary for YAML configuration.

    Default nx=2 to prevent time_max=0 issues with single turbine farms.

    Args:
        nx: Number of turbines in x direction (default 2)
        ny: Number of turbines in y direction (default 1)
        history_length: Length of measurement history (default 5)
        window_length: Length of rolling window (default 2)
        history_n: Number of history samples (default 1)

    Returns:
        dict: Base YAML configuration dictionary
    """
    return {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
            "xDist": 5,
            "yDist": 3,
            "nx": nx,
            "ny": ny,
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
            "Power_avg": 1,
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
            "ws_history_N": history_n,
            "ws_history_length": history_length,
            "ws_window_length": window_length,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": history_n,
            "wd_history_length": history_length,
            "wd_window_length": window_length,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": history_n,
            "yaw_history_length": history_length,
            "yaw_window_length": window_length,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": history_n,
            "power_history_length": history_length,
            "power_window_length": window_length,
        },
    }


def calculate_expected_obs_dim(config_dict, n_turbines):
    """
    Calculates the expected observation dimension based on the YAML configuration.

    Args:
        config_dict: Configuration dictionary with measurement settings
        n_turbines: Number of turbines in the farm

    Returns:
        int: Expected total observation dimension
    """
    mes_level = config_dict.get("mes_level", {})
    ws_mes_cfg = config_dict.get("ws_mes", {})
    wd_mes_cfg = config_dict.get("wd_mes", {})
    yaw_mes_cfg = config_dict.get("yaw_mes", {})
    power_mes_cfg = config_dict.get("power_mes", {})

    one_turb_mes_obs_actual = 0
    if mes_level.get("turb_ws", False):
        one_turb_mes_obs_actual += ws_mes_cfg.get("ws_current", False) + (
            ws_mes_cfg.get("ws_rolling_mean", False) * ws_mes_cfg.get("ws_history_N", 0)
        )
    if mes_level.get("turb_wd", False):
        one_turb_mes_obs_actual += wd_mes_cfg.get("wd_current", False) + (
            wd_mes_cfg.get("wd_rolling_mean", False) * wd_mes_cfg.get("wd_history_N", 0)
        )

    one_turb_mes_obs_actual += yaw_mes_cfg.get("yaw_current", False) + (
        yaw_mes_cfg.get("yaw_rolling_mean", False) * yaw_mes_cfg.get("yaw_history_N", 0)
    )

    if mes_level.get("turb_TI", False):
        one_turb_mes_obs_actual += 1

    if mes_level.get("turb_power", False):
        one_turb_mes_obs_actual += power_mes_cfg.get("power_current", False) + (
            power_mes_cfg.get("power_rolling_mean", False)
            * power_mes_cfg.get("power_history_N", 0)
        )

    total_turbine_level_obs = one_turb_mes_obs_actual * n_turbines

    farm_level_obs = 0
    if mes_level.get("farm_ws", False):
        farm_level_obs += ws_mes_cfg.get("ws_current", False) + (
            ws_mes_cfg.get("ws_rolling_mean", False) * ws_mes_cfg.get("ws_history_N", 0)
        )
    if mes_level.get("farm_wd", False):
        farm_level_obs += wd_mes_cfg.get("wd_current", False) + (
            wd_mes_cfg.get("wd_rolling_mean", False) * wd_mes_cfg.get("wd_history_N", 0)
        )
    if mes_level.get("farm_TI", False):
        farm_level_obs += 1
    if mes_level.get("farm_power", False):
        farm_level_obs += power_mes_cfg.get("power_current", False) + (
            power_mes_cfg.get("power_rolling_mean", False)
            * power_mes_cfg.get("power_history_N", 0)
        )

    return total_turbine_level_obs + farm_level_obs
