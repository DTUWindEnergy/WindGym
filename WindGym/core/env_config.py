"""Environment configuration: loading, validation and the measurement class.

``WindFarmEnv`` accepts ``config`` as a dict, a YAML string or a path to a
YAML file. This module turns that input into a validated ``EnvSettings``
dataclass (one field per attribute the env exposes flat on itself) and builds
the ``FarmMes`` measurement object from those settings.

Error messages are part of the public behaviour (tests match on them), so
they are kept verbatim from the original ``WindFarmEnv._apply_config``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .mes_class import FarmMes


def load_config_dict(config) -> tuple:
    """Normalize the ``config`` argument to a dict.

    Returns ``(cfg, yaml_path)`` where ``yaml_path`` is the file path when
    ``config`` named an existing file, else ``None``.
    """
    if config is None:
        raise ValueError("A configuration must be provided via the `config` argument.")
    if isinstance(config, dict):  # If it is already a dict, then just return it
        return config, None
    if isinstance(config, (str, Path)):  #
        config_str = str(config)
        # Check if this looks like a file path (has .yaml/.yml extension or contains path separators)
        looks_like_file = (
            config_str.endswith((".yaml", ".yml"))
            or "/" in config_str
            or "\\" in config_str
        )

        if os.path.exists(config_str):  # treat as file
            with open(config_str, "r") as f:
                return (yaml.safe_load(f) or {}), config_str
        elif looks_like_file:
            # It looks like a file path but doesn't exist
            raise FileNotFoundError(
                f"Config file not found: {config_str}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Make sure the path is correct or provide an absolute path."
            )
        else:  # treat as YAML string content
            return (yaml.safe_load(str(config)) or {}), None
    raise TypeError("`config` must be a dict, YAML string, or path to a YAML file.")


@dataclass
class EnvSettings:
    """Validated config values. Every field becomes a flat attribute on the env."""

    # Top-level (optional)
    yaw_init: Any = None
    BaseController: Any = None
    ActionMethod: Any = None
    Track_power: Any = None
    # farm
    yaw_min: Any = None
    yaw_max: Any = None
    yaw_scaling_min: Any = None
    yaw_scaling_max: Any = None
    tilt: float = 0.0
    turb_memmap: bool = False
    dwm_setup: Any = None  # name from calibrated_values.DWM_SETUPS; None = default
    # wind
    ws_inflow_min: Any = None
    ws_inflow_max: Any = None
    TI_inflow_min: Any = None
    TI_inflow_max: Any = None
    wd_inflow_min: Any = None
    wd_inflow_max: Any = None
    veer_inflow_min: float = 0.0
    veer_inflow_max: float = 0.0
    # sections consumed by bare indexing later
    act_pen: dict = field(default_factory=dict)
    power_def: dict = field(default_factory=dict)
    mes_level: dict = field(default_factory=dict)
    ws_mes: dict = field(default_factory=dict)
    wd_mes: dict = field(default_factory=dict)
    yaw_mes: dict = field(default_factory=dict)
    power_mes: dict = field(default_factory=dict)
    # derived / convenience
    ti_sample_count: int = 30
    action_penalty: Any = None
    action_penalty_type: Any = None
    Power_scaling: Any = None
    power_avg: Any = None
    power_reward: Any = None
    tau: float = 0.02
    # derating
    derate_action: bool = False
    yaw_action: bool = True
    derate_min: float = 0.0
    derate_max: float = 1.0
    derate_penalty: float = 0.0
    derate_penalty_type: str = "change"
    derate_method: str = "absolute"
    derate_step_env: float = 0.1
    derate_step_sim: Optional[float] = None
    # power tracking
    track_reward_type: str = "abs"
    track_sigma: float = 0.1
    track_ref_range: Any = field(default_factory=lambda: [0.2, 0.8])
    track_obs_setpoint: bool = True
    track_obs_error: bool = True
    track_obs_preview: int = 0
    derate_reference: str = "available"
    derate_mes: dict = field(default_factory=dict)
    # probes (the ProbeManager itself is built by the env)
    probes_config: Any = field(default_factory=list)


def parse_env_settings(config: Dict[str, Any]) -> EnvSettings:
    """Validate the parsed config dict and map it to an ``EnvSettings``."""

    # helpers for clearer errors on missing/invalid sections/keys
    def require_section(name: str) -> Dict[str, Any]:
        section = config.get(name)
        if not isinstance(section, dict):
            raise ValueError(
                f"Config section '{name}' is required and must be a mapping."
            )
        return section

    def require_key(section: Dict[str, Any], key: str, section_name: str):
        if key not in section:
            raise ValueError(f"Key '{key}' is required in section '{section_name}'.")
        return section[key]

    s = EnvSettings()

    # Top-level fields (optional)
    s.yaw_init = config.get("yaw_init")
    s.BaseController = config.get("BaseController")
    s.ActionMethod = config.get("ActionMethod")
    s.Track_power = config.get("Track_power")

    # Farm section (required keys)
    farm = require_section("farm")
    s.yaw_min = require_key(farm, "yaw_min", "farm")
    s.yaw_max = require_key(farm, "yaw_max", "farm")
    s.yaw_scaling_min = s.yaw_min
    s.yaw_scaling_max = s.yaw_max
    # Optional fixed rotor tilt (deg, all turbines); positive deflects the
    # wake upward in DWM. Required for veer to produce a yaw-sign asymmetry.
    s.tilt = farm.get("tilt", 0.0)

    # turb_memmap: MannLoad opens boxes lazily (memmap) instead of reading
    # them into memory (see core.turbulence_manager).
    s.turb_memmap = bool(config.get("turb_memmap", False))

    # dwm_setup: DWM physics preset by name ("default", "les_calibrated"; see
    # core.calibrated_values). The env's dwm_setup= kwarg takes precedence.
    s.dwm_setup = config.get("dwm_setup")

    # Legacy box-spec keys died with the legacy DWM path: the Mann box comes
    # from the DWMSetup (core/dwm_defaults.py). A config still carrying them
    # would silently get that box instead of what it asks for — say so loudly.
    for _legacy_key in ("mann_nxyz", "mann_dxyz_over_D"):
        if _legacy_key in config:
            warnings.warn(
                f"config key '{_legacy_key}' is ignored: the Mann box spec "
                "comes from the env's DWMSetup (core/dwm_defaults.py, "
                "MannBoxSpec); the legacy configurable-box path was removed.",
                stacklevel=2,
            )

    # Wind section (required keys)
    wind = require_section("wind")
    s.ws_inflow_min = require_key(wind, "ws_min", "wind")
    s.ws_inflow_max = require_key(wind, "ws_max", "wind")
    s.TI_inflow_min = require_key(wind, "TI_min", "wind")
    s.TI_inflow_max = require_key(wind, "TI_max", "wind")
    s.wd_inflow_min = require_key(wind, "wd_min", "wind")
    s.wd_inflow_max = require_key(wind, "wd_max", "wind")
    # Optional veer range (deg per 100 m, 0 at hub height); defaults keep
    # existing configs byte-identical (no RNG draw when min == max).
    s.veer_inflow_min = wind.get("veer_min", 0.0)
    s.veer_inflow_max = wind.get("veer_max", 0.0)

    # Measurement & reward sections. These are consumed by bare [...]
    # indexing in build_farm_mes / RewardCalculator, so validate here to
    # get an actionable error instead of a KeyError deep in init.
    s.act_pen = config.get("act_pen", {}) or {}

    s.power_def = require_section("power_def")
    require_key(s.power_def, "Power_avg", "power_def")

    s.mes_level = require_section("mes_level")
    for key in (
        "turb_ws",
        "turb_wd",
        "turb_TI",
        "turb_power",
        "farm_ws",
        "farm_wd",
        "farm_TI",
        "farm_power",
    ):
        require_key(s.mes_level, key, "mes_level")

    s.ws_mes = require_section("ws_mes")
    s.wd_mes = require_section("wd_mes")
    s.yaw_mes = require_section("yaw_mes")
    s.power_mes = require_section("power_mes")
    for prefix, section in (
        ("ws", s.ws_mes),
        ("wd", s.wd_mes),
        ("yaw", s.yaw_mes),
        ("power", s.power_mes),
    ):
        for suffix in (
            "current",
            "rolling_mean",
            "history_N",
            "history_length",
            "window_length",
        ):
            require_key(section, f"{prefix}_{suffix}", f"{prefix}_mes")

    # Derived / convenience attributes with sensible fallbacks
    s.ti_sample_count = s.mes_level.get("ti_sample_count", 30)
    s.action_penalty = s.act_pen.get("action_penalty")
    s.action_penalty_type = s.act_pen.get("action_penalty_type")
    s.Power_scaling = s.power_def.get("Power_scaling")
    s.power_avg = s.power_def.get("Power_avg")
    s.power_reward = s.power_def.get("Power_reward")
    s.tau = s.power_def.get("tau", 0.02)

    # Derating action (optional, all default to off/zero)
    s.derate_action = config.get("derate_action", False)
    # yaw_action=False (with derate_action=True) gives a derate-only agent
    s.yaw_action = config.get("yaw_action", True)
    s.derate_min = config.get("derate_min", 0.0)
    s.derate_max = config.get("derate_max", 1.0)
    s.derate_penalty = config.get("derate_penalty", 0.0)
    s.derate_penalty_type = config.get("derate_penalty_type", "change")

    # How the derate action is applied:
    #   "absolute": action is the setpoint, mapped to [derate_min, derate_max]
    #   "step":     action is a delta, at most derate_step_env change per env step
    s.derate_method = str(config.get("derate_method", "absolute")).lower()
    if s.derate_method not in {"absolute", "step"}:
        raise ValueError("derate_method must be 'absolute' or 'step'")
    s.derate_step_env = config.get("derate_step_env", 0.1)
    # Optional slew limit toward the setpoint, per sim substep (mirrors
    # yaw_step_sim in the "wind" yaw method). None = setpoint applies
    # instantly, matching a power-reference command executing in seconds.
    s.derate_step_sim = config.get("derate_step_sim", None)
    if s.derate_step_sim is not None and s.derate_step_sim <= 0:
        raise ValueError("derate_step_sim must be positive (or None)")

    # Power tracking (optional section; only consumed when Track_power is
    # True). Track_reward selects the reward shape, track_sigma the width
    # of the gaussian form, track_ref_range the default sampler's fraction
    # range, and the track_obs_* keys toggle the farm-level observations.
    track_def = config.get("track_def", {}) or {}
    s.track_reward_type = track_def.get("Track_reward", "abs")
    s.track_sigma = track_def.get("track_sigma", 0.1)
    s.track_ref_range = track_def.get("track_ref_range", [0.2, 0.8])
    s.track_obs_setpoint = track_def.get("track_obs_setpoint", True)
    s.track_obs_error = track_def.get("track_obs_error", True)
    s.track_obs_preview = track_def.get("track_obs_preview", 0)
    if (
        len(s.track_ref_range) != 2
        or not 0 <= s.track_ref_range[0] <= s.track_ref_range[1]
    ):
        raise ValueError(
            "track_ref_range must be a (low, high) pair with 0 <= low <= high, "
            f"got {s.track_ref_range}"
        )
    if int(s.track_obs_preview) != s.track_obs_preview or (s.track_obs_preview < 0):
        raise ValueError(
            f"track_obs_preview must be a non-negative integer, "
            f"got {s.track_obs_preview}"
        )
    s.track_obs_preview = int(s.track_obs_preview)

    # What the derate command means:
    #   "available": fraction of locally available power (P = (1-d)*P_avail)
    #   "rated":     fraction of rated power, i.e. an absolute power cap,
    #                converted to the available-power fraction from the last
    #                measured power (one substep old). A cap above locally
    #                available power is a no-op (dead zone), matching a real
    #                power-reference controller.
    #   "setpoint":  fraction of rated power applied as an absolute cap on
    #                the turbine model itself (SetpointPyWakeWindTurbines:
    #                inertia-filtered available power, lagged Ct derate).
    #                dynamiks + PyWake turbines only.
    # Orthogonal to derate_method, which says how the command *evolves*.
    s.derate_reference = str(config.get("derate_reference", "available")).lower()
    if s.derate_reference not in {"available", "rated", "setpoint"}:
        raise ValueError("derate_reference must be 'available', 'rated' or 'setpoint'")

    # Derate observation (per turbine, mirrors yaw_mes). Defaults to
    # observing the current derate whenever the derate action is enabled.
    derate_mes = config.get("derate_mes") or {}
    s.derate_mes = {
        "derate_current": derate_mes.get("derate_current", s.derate_action),
        "derate_rolling_mean": derate_mes.get("derate_rolling_mean", False),
        "derate_history_N": derate_mes.get("derate_history_N", 1),
        "derate_history_length": derate_mes.get("derate_history_length", 10),
        "derate_window_length": derate_mes.get("derate_window_length", 10),
    }

    s.probes_config = config.get("probes", [])

    return s


def build_farm_mes(
    settings, n_turb: int, scaling: dict, maxturbpower: float
) -> FarmMes:
    """Build the ``FarmMes`` measurement object.

    ``settings`` is anything exposing the ``EnvSettings`` attributes (the env
    itself qualifies, since it carries them flat). ``scaling`` holds the
    observation scaling bounds: ``ws_min/ws_max``, ``wd_min/wd_max`` and
    ``TI_min/TI_max``; the yaw bounds come from
    ``settings.yaw_scaling_min/max``.
    """
    # TODO if history_length is 1, then we dont need to save the history, and we can just use the current values.
    # TODO is history_N is 1 or larger, then it is kinda implied that the rolling_mean is true.. Therefore we can change the if self.rolling_mean: check in the Mes() class, to be a if self.history_N >= 1 check... or something like that
    return FarmMes(
        n_turbines=n_turb,
        turb_ws=settings.mes_level["turb_ws"],
        turb_wd=settings.mes_level["turb_wd"],
        turb_TI=settings.mes_level["turb_TI"],
        turb_power=settings.mes_level["turb_power"],
        farm_ws=settings.mes_level["farm_ws"],
        farm_wd=settings.mes_level["farm_wd"],
        farm_TI=settings.mes_level["farm_TI"],
        farm_power=settings.mes_level["farm_power"],
        ws_current=settings.ws_mes["ws_current"],
        ws_rolling_mean=settings.ws_mes["ws_rolling_mean"],
        ws_history_N=settings.ws_mes["ws_history_N"],
        ws_history_length=settings.ws_mes["ws_history_length"],
        ws_window_length=settings.ws_mes["ws_window_length"],
        wd_current=settings.wd_mes["wd_current"],
        wd_rolling_mean=settings.wd_mes["wd_rolling_mean"],
        wd_history_N=settings.wd_mes["wd_history_N"],
        wd_history_length=settings.wd_mes["wd_history_length"],
        wd_window_length=settings.wd_mes["wd_window_length"],
        yaw_current=settings.yaw_mes["yaw_current"],
        yaw_rolling_mean=settings.yaw_mes["yaw_rolling_mean"],
        yaw_history_N=settings.yaw_mes["yaw_history_N"],
        yaw_history_length=settings.yaw_mes["yaw_history_length"],
        yaw_window_length=settings.yaw_mes["yaw_window_length"],
        derate_current=settings.derate_mes["derate_current"],
        derate_rolling_mean=settings.derate_mes["derate_rolling_mean"],
        derate_history_N=settings.derate_mes["derate_history_N"],
        derate_history_length=settings.derate_mes["derate_history_length"],
        derate_window_length=settings.derate_mes["derate_window_length"],
        power_current=settings.power_mes["power_current"],
        power_rolling_mean=settings.power_mes["power_rolling_mean"],
        power_history_N=settings.power_mes["power_history_N"],
        power_history_length=settings.power_mes["power_history_length"],
        power_window_length=settings.power_mes["power_window_length"],
        track_setpoint=bool(settings.Track_power) and settings.track_obs_setpoint,
        track_error=bool(settings.Track_power) and settings.track_obs_error,
        track_preview=settings.track_obs_preview if settings.Track_power else 0,
        ws_min=scaling["ws_min"],
        ws_max=scaling["ws_max"],
        # Max and min values for wind direction measurements   NOTE i have added 5 for some slack in the measurements. so the scaling is better.
        wd_min=scaling["wd_min"],
        wd_max=scaling["wd_max"],
        yaw_min=settings.yaw_scaling_min,
        yaw_max=settings.yaw_scaling_max,
        TI_min=scaling["TI_min"],
        TI_max=scaling["TI_max"],
        power_max=maxturbpower,
        ti_sample_count=settings.ti_sample_count,
    )
