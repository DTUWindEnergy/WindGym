def test_env():
    """Make a simple test environment, which can be used during development."""
    from WindGym import WindFarmEnv
    from py_wake.examples.data.hornsrev1 import V80

    config = {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10,
            "ws_max": 10,
            "TI_min": 0.07,
            "TI_max": 0.07,
            "wd_min": 260,
            "wd_max": 280,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,  # Will be converted to deviation
            "turb_TI": False,
            "turb_power": True,  # Include power
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 3,  # History length
            "ws_history_length": 3,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": True,
            "wd_history_N": 3,
            "wd_history_length": 3,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 3,
            "yaw_history_length": 3,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": False,
            "power_history_N": 3,
            "power_history_length": 3,
            "power_window_length": 1,
        },
    }

    env = WindFarmEnv(
        turbine=V80(),
        x_pos=[0, 500, 1000],
        y_pos=[0, 0, 0],
        config=config,
        turbtype="Random",  # No turbulence for faster tests
        burn_in_passthroughs=1,  # Minimal burn-in for speed
    )

    return env
