"""
Predefined wind farm environment presets for quick experimentation and demonstrations.

Each factory function returns a fully configured :class:`WindFarmEnv` instance
using some defaults (dynamic backend, V80 turbines). All functions accept
``**kwargs`` that are forwarded to :class:`WindFarmEnv`, so you can override
any constructor parameter (e.g. ``backend="dynamiks"``, ``n_passthrough=10``, ...).

Example
-------
>>> from WindGym.presets import three_turbine_row
>>> env = three_turbine_row()
>>> obs, info = env.reset()
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from .wind_farm_env import WindFarmEnv
from .utils.generate_layouts import generate_square_grid

# ---------------------------------------------------------------------------
# Base configuration shared by all presets
# ---------------------------------------------------------------------------
_BASE_CONFIG: Dict[str, Any] = {
    "yaw_init": "Random",
    "noise": "None",
    "BaseController": "Local",
    "ActionMethod": "wind",
    "Track_power": False,
    # Power tracking (only used when Track_power is True; also requires
    # power_def.Power_reward "None"):
    # "track_def": {
    #     "Track_reward": "abs",  # or "gaussian"
    #     "track_sigma": 0.1,
    #     "track_ref_range": [0.2, 0.8],
    #     "track_obs_setpoint": True,
    #     "track_obs_error": True,
    #     "track_obs_preview": 0,
    # },
    "farm": {
        "yaw_min": -45,
        "yaw_max": 45,
    },
    "wind": {
        "ws_min": 7,
        "ws_max": 15,
        "TI_min": 0.02,
        "TI_max": 0.15,
        "wd_min": 255,
        "wd_max": 285,
    },
    "act_pen": {
        "action_penalty": 0.0,
        "action_penalty_type": "Change",
    },
    "power_def": {
        "Power_reward": "Baseline",
        "Power_avg": 10,
        "Power_scaling": 1.0,
    },
    "mes_level": {
        "turb_ws": True,
        "turb_wd": True,
        "turb_TI": False,
        "turb_power": False,
        "farm_ws": False,
        "farm_wd": False,
        "farm_TI": False,
        "farm_power": False,
    },
    "ws_mes": {
        "ws_current": True,
        "ws_rolling_mean": False,
        "ws_history_N": 1,
        "ws_history_length": 25,
        "ws_window_length": 25,
    },
    "wd_mes": {
        "wd_current": True,
        "wd_rolling_mean": False,
        "wd_history_N": 1,
        "wd_history_length": 20,
        "wd_window_length": 20,
    },
    "yaw_mes": {
        "yaw_current": True,
        "yaw_rolling_mean": False,
        "yaw_history_N": 1,
        "yaw_history_length": 10,
        "yaw_window_length": 10,
    },
    "power_mes": {
        "power_current": False,
        "power_rolling_mean": False,
        "power_history_N": 1,
        "power_history_length": 10,
        "power_window_length": 10,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into a deep copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _make_env(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    config_overrides: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> WindFarmEnv:
    """Build a :class:`WindFarmEnv` from base config + optional overrides."""
    from py_wake.examples.data.hornsrev1 import V80

    turbine = V80()
    config = _deep_merge(_BASE_CONFIG, config_overrides or {})
    kwargs.setdefault("turbtype", "Random")
    return WindFarmEnv(
        turbine=turbine,
        x_pos=x_pos,
        y_pos=y_pos,
        config=config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public preset functions
# ---------------------------------------------------------------------------


def three_turbine_row(**kwargs: Any) -> WindFarmEnv:
    """Create a 3-turbine row environment (V80s, 5D spacing).

    Ideal for quick-start tutorials and sanity checks.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`WindFarmEnv` (e.g. ``backend``, ``n_passthrough``,
        ``render_mode``).

    Returns
    -------
    WindFarmEnv
    """
    from py_wake.examples.data.hornsrev1 import V80

    turbine = V80()
    x_pos, y_pos = generate_square_grid(turbine, nx=3, ny=1, xDist=5, yDist=5)
    return _make_env(x_pos, y_pos, **kwargs)


def two_by_two_grid(**kwargs: Any) -> WindFarmEnv:
    """Create a 2x2 grid environment (V80s, 4D spacing).

    Good for wake steering research with a small, tractable farm.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`WindFarmEnv`.

    Returns
    -------
    WindFarmEnv
    """
    from py_wake.examples.data.hornsrev1 import V80

    turbine = V80()
    x_pos, y_pos = generate_square_grid(turbine, nx=2, ny=2, xDist=4, yDist=4)
    return _make_env(x_pos, y_pos, **kwargs)


def nine_turbine_grid(**kwargs: Any) -> WindFarmEnv:
    """Create a 3x3 grid environment (V80s, 5D spacing).

    Suitable for larger-scale wake steering experiments.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`WindFarmEnv`.

    Returns
    -------
    WindFarmEnv
    """
    from py_wake.examples.data.hornsrev1 import V80

    turbine = V80()
    x_pos, y_pos = generate_square_grid(turbine, nx=3, ny=3, xDist=5, yDist=5)
    return _make_env(x_pos, y_pos, **kwargs)
