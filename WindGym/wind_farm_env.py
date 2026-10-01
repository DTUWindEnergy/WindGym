from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os
import gc
import sys
import socket
import shutil
import math
from pathlib import Path


# Dynamiks imports
from dynamiks.views import XYView

from IPython import display

# WindGym imports
from . import utils
from .core.mes_class import FarmMes
from .core.reward_calculator import RewardCalculator
from .core.wind_manager import WindManager
from .core.turbulence_manager import TurbulenceManager
from .core.episode_timing import episode_time_parameters
from .core.env_config import load_config_dict, parse_env_settings, build_farm_mes
from .core.actuation import step_yaw_command, compute_derate
from .core.renderer import WindFarmRenderer
from .core.baseline_manager import BaselineManager
from .core.probe_manager import ProbeManager
from .core.power_tracking import PowerTrackingManager
from .core.dwm_defaults import make_wts, make_dwm, add_hawc2_yaw_sensor
from .core.calibrated_values import resolve_dwm_setup
from .core.derating import (
    add_hawc2_derate_sensor,
    check_htc_supports_derating,
    check_turbine_supports_derating,
)

from py_wake.wind_turbines import WindTurbines as WindTurbinesPW
from collections import deque, defaultdict
import yaml
from .backend.hawc2_adapter import (
    HAWC2WindTurbinesW,
    HubWindFreeWindCoupling,
    safe_close_h2,
    delete_case_folders,
)

# For live plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from WindGym.core.wind_probe import WindProbe

"""
This is the base for the wind farm environment. This is where the magic happens.
For now it only supports the PyWakeWindTurbines, but it should be easy to expand to other types of turbines.
"""

# TODO thrust coefficient control
# TODO for now I have just hardcoded this scaling value (1 and 25 for the wind_speed min and max). This is beacuse the wind speed is chosen from the normal distribution, but becasue of the wakes and the turbulence, we canhave cases where we go above or below these values.


class WindFarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # Optional steady-state operating-point table (see core.operating_point).
    # Class-level default so subclasses that don't forward kwargs (e.g.
    # WindFarmEnvMulti) still expose the attribute.
    op_lookup = None

    # Allowed keys for `dwm_params` (constructor) and `options["dwm_params"]`
    # (reset). Split so the reset path can partition closure overrides
    # (forwarded to ``make_dwm``) from Mann-box overrides (forwarded to
    # ``TurbulenceManager.create_sites``). Anything outside the union raises so
    # DR typos don't silently no-op.
    _CLOSURE_PARAM_KEYS = frozenset({"k1", "k2", "d_particle"})
    _MANN_PARAM_KEYS = frozenset({"mann_L", "mann_GAMMA", "mann_AE"})
    _DWM_PARAM_KEYS = _CLOSURE_PARAM_KEYS | _MANN_PARAM_KEYS

    @classmethod
    def _validate_dwm_keys(cls, params: dict, where: str = "") -> None:
        """Raise ValueError if any key in ``params`` is outside ``_DWM_PARAM_KEYS``.

        ``where`` is appended to the message (e.g. " in reset options") so the
        caller is identifiable from the traceback.
        """
        bad = set(params) - cls._DWM_PARAM_KEYS
        if bad:
            raise ValueError(
                f"Unknown dwm_params keys{where}: {sorted(bad)}. "
                f"Allowed: {sorted(cls._DWM_PARAM_KEYS)}"
            )

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        n_passthrough=5,
        ws_scaling_min: float = 0.0,
        ws_scaling_max: float = 30.0,
        wd_scaling_min: float = 0,
        wd_scaling_max: float = 360,
        ti_scaling_min: float = 0.0,
        ti_scaling_max: float = 1.0,
        yaw_scaling_min: float = -45,
        yaw_scaling_max: float = 45,
        TurbBox="Default",
        turbtype="Random",
        backend: str = "dynamiks",
        config=None,
        Baseline_comp=False,
        yaw_init=None,
        render_mode=None,
        fix_turbines=False,
        show_indices=True,
        fontsize=15,
        axes_lw=1.5,
        colorbar_vmax_step=2.0,
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        delay: float
        | None = None,  # Agent action interval in seconds. None -> dt_env (no delay)
        yaw_step_sim=1,  # How many degrees the yaw angles can change pr. simulation step
        yaw_step_env=None,  # How many degrees the yaw angles can change pr. environment step
        fill_window=True,
        sample_site=None,
        HTC_path=None,
        reset_init=True,
        burn_in_passthroughs=2,  # number of passthroughs before episode starts
        max_time_steps: int
        | None = None,  # fixed episode length in env steps; time_max becomes max_time_steps * delay seconds. None = use ws-derived time_max.
        cleanup_on_time_limit: bool = True,
        keep_hawc_results: bool = False,  # if True, never delete the HAWC2 res/htc/log folders
        hawc2_yaw_mode: str = "bearing2_slot",  # yaw sensor wiring, must match the htc servo DLL; see core.dwm_defaults.add_hawc2_yaw_sensor
        hawc2_yaw_slot: int = 1,  # HAWC2 general-variable index the servo DLL reads the yaw setpoint from
        wd_function=None,  # A function that takes in the timestep and returns the wind direction
        power_ref_function=None,  # A function (t_seconds, env) -> reference farm power in W. Only used when Track_power is True; None uses the default constant-setpoint sampler.
        max_turb_move=2,  # The maximum distance that the turbines can move in one timestep. This is used to avoid numerical issues with the DWM solver.
        op_lookup=None,  # Optional OperatingPointLookup: reports steady-state blade pitch / rotor RPM per turbine when derating.
        wd_est_tau=None,  # Sensor-derived wd estimator EWMA time constant (s). None = estimator off (env.wd_est unavailable).
        wd_est_consensus="median",  # Cross-turbine consensus for the wd estimator: median / mean / front.
        interpolation=None,  # Particle trajectory interpolation in the DWM solver: 'linear' (fast) or 'pchip' (cubic, original). None = dynamiks default (pchip). Setting it needs the dev dynamiks build.
        lateral_cutoff=None,  # Skip wake deficit evaluation beyond this factor times the deficit profile half-width (r_max*R) from the meandered wake centerline. None disables (original behavior). Setting it needs the dev dynamiks build.
        tilt: Optional[
            float
        ] = None,  # Fixed rotor tilt in deg for all turbines (positive deflects the wake upward in DWM). None -> use config `farm: tilt` (default 0). Needed for veer to create a yaw-sign asymmetry.
        dwm_setup=None,  # DWM physics: a core.dwm_defaults.DWMSetup or a preset name ("default", "les_calibrated"; see core.calibrated_values). None -> config `dwm_setup` key, else plain dynamiks.
        dwm_params: Optional[
            dict
        ] = None,  # Override DWM closure params (k1, k2, d_particle). Used for domain randomization; per-episode overrides go through reset(options={"dwm_params": ...}).
        **kwargs,
    ):
        """
        This is a steadystate environment. The environment only ever changes wind conditions at reset. Then the windconditions are constatnt for the rest of the episode
        Args:
            turbine: PyWakeWindTurbine: The wind turbine that is used in the environment
            n_passthrough: int: The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
            TI_min_mes: float: The minimum value for the turbulence intensity measurements. Used for internal scaling
            TI_max_mes: float: The maximum value for the turbulence intensity measurements. Used for internal scaling
            TurbBox: str: The path to the turbulence box files. If Default, then it will use the default turbulence box files.
            turbtype: str: The type of turbulence box that is used. Can be one of the following: MannLoad, MannGenerate, MannFixed, Random, None
            config (str | Path | dict): The environment configuration.
                - If dict: taken directly.
                - If str/Path to an existing file: loaded from file.
                - If str containing YAML (multi-line, not a file path): parsed as YAML.
            Baseline_comp: bool: If true, then the environment will compare the performance of the agent with a baseline farm. This is only used in the EnvEval class.
            yaw_init: str: The method for initializing the yaw angles of the turbines. If 'Random', then the yaw angles will be random. Else they will be zeros.
            render_mode: str: The render mode of the environment. If None, then nothing will be rendered. If human, then the environment will be rendered in a window. If rgb_array, then the environment will be rendered as an array.
            seed: int: The seed for the environment. If None, then the seed will be random.
            dt_sim: float: The simulation timestep in seconds. Can be used to speed up the simulation, if the DWM solver can take larger steps
            dt_env: float: The environment timestep in seconds. This is the timestep that the agent sees. The environment will run the simulation for dt_sim/dt_env steps pr. timestep.
            delay: float: The agent action interval in seconds. Each env step advances the simulation `delay` seconds, but the measurements averaged into the observation only cover the final `dt_env` window; the earlier sub-steps are simulated and discarded. Use delay > dt_env to let the agent act on a slower cadence than the sim, so slow wake propagation can reach downstream turbines between actions. Must be a positive multiple of dt_env. None (default) means delay = dt_env, i.e. no delay.
            yaw_step_sim: float: The step size for the yaw angles. How manny degress the yaw angles can change pr. step
            fill_window: bool: If True, then the measurements will be filled up at reset.
            sample_site: pywake site that includes information about the wind conditions. If None we sample uniformly from within the limits.
            HTC_path: str: The path to the high fidelity turbine model. If this is Not none, then we assume you want to use that instead of pywake turbines. Note you still need a pywake version of your turbine.
            reset_init: bool: If True, then the environment will be reset at initialization. This is used to save time for things that call the reset method anyways.
            max_time_steps: int: Fixed episode length in env steps; the episode truncates after exactly this many steps, so parallel envs stay in sync regardless of wind conditions. Step counting starts after reset's flow burn-in and sensor fill (as with the n_passthrough method). Setting this disables the passthrough episode-length method: n_passthrough is ignored (a note is printed at construction). Internally time_max is set to max_time_steps * delay seconds (time_max is always in seconds). None (default) = use the ws-derived time_max from n_passthrough.
            cleanup_on_time_limit: bool: If True, then the environment will clean up the HAWC2 files when the maximum time is reached. This is to avoid filling up the disk with files.
            keep_hawc_results: bool: If True, the HAWC2 res/htc/log folders are never deleted (overrides all cleanup), so they can be kept for later inspection. Default False.
            power_ref_function: callable(t_seconds, env) -> float: The farm power reference in watts at episode time t (t=0 is the first agent step; the callable is evaluated every `delay` seconds). May use env.np_random, env.ws, env.n_turb, env.rated_power. Only used when Track_power is True. None (default) samples a constant setpoint per episode, uniform in track_ref_range times the episode freestream farm power.
        """
        self.kwargs = locals()
        del self.kwargs["self"]  # Remove 'self' from the dictionary

        self.backend = backend.lower().strip()
        if self.backend not in {"dynamiks", "pywake"}:
            raise ValueError("backend must be 'dynamiks' or 'pywake'")
        # Check that x_pos and y_pos are the same length
        if len(x_pos) != len(y_pos):
            raise ValueError("x_pos and y_pos must be the same length")

        self.max_turb_move = max_turb_move
        self.wd_function = wd_function
        self.power_ref_function = power_ref_function

        # Predefined values
        self.wts = None
        self.wts_baseline = None
        self.burn_in_passthroughs = burn_in_passthroughs
        # Optional fixed episode length (env steps). When set, reset() overrides the
        # ws-derived time_max with this value so all parallel envs truncate (and autoreset)
        # on the same global step. None keeps the original ws-derived behavior.
        self.max_time_steps = max_time_steps
        self.cleanup_on_time_limit = cleanup_on_time_limit
        self.keep_hawc_results = keep_hawc_results
        self.hawc2_yaw_mode = hawc2_yaw_mode
        self.hawc2_yaw_slot = int(hawc2_yaw_slot)
        # The farm power reference (W) at the current env step and its preview,
        # maintained by _push_tracking when Track_power is True.
        self.power_setpoint = 0.0
        self.power_preview = np.zeros(0)
        self.act_var = (
            1  # number of actions pr. turbine. For now it is just the yaw angles
        )
        self.HTC_path = HTC_path
        self.fill_window = fill_window
        self.dt = dt_sim  # DWM simulation timestep
        self.dt_sim = dt_sim
        self.dt_env = dt_env  # Environment timestep
        self.sim_steps_per_env_step = int(self.dt_env / self.dt_sim)
        if self.dt_env % self.dt_sim != 0:
            raise ValueError("dt_env must be a multiple of dt_sim")

        # Agent action interval. Downstream projects may also overwrite
        # env.delay directly after construction instead of passing the kwarg.
        self.delay = dt_env if delay is None else delay
        # delay must be a positive multiple of dt_env (floor division below would
        # silently truncate otherwise) and can't be smaller than one env step.
        if self.delay < self.dt_env or self.delay % self.dt_env != 0:
            raise ValueError("delay must be a multiple of dt_env and >= dt_env")

        # If we use pywake as backend, then we need to make sure that the dt_sim and dt_env are the same. This is because pywake is a steady state solver, and therefore does not have a timestep.
        if self.backend == "pywake" and self.dt_env != self.dt_sim:
            raise ValueError(
                "When using pywake as backend, dt_env must be equal to dt_sim"
            )

        self.x_pos = x_pos
        self.y_pos = y_pos

        self.sample_site = sample_site
        self.yaw_start = 15.0  # This is the limit for the initialization of the yaw angles. This is used to make sure that the yaw angles are not too large at the start, but still not zero
        # Max power pr turbine. Used in the measurement class
        self.maxturbpower = max(turbine.power(np.arange(10, 25, 1)))
        self.baseline_wakes = True  # A flag that decides if we include the wakes in the baseline farm. For now always true.
        # The step size for the yaw angles. How manny degress the yaw angles can change pr. step
        self.interpolation = interpolation
        self.lateral_cutoff = lateral_cutoff
        self.turbtype = turbtype
        self.yaw_step_sim = yaw_step_sim  # How many degrees the yaw angles can change pr. simulation step

        if yaw_step_env is None:
            self.yaw_step_env = yaw_step_sim * self.sim_steps_per_env_step
        else:
            self.yaw_step_env = yaw_step_env

        # Saves to self
        self.ws_scaling_min = ws_scaling_min
        self.ws_scaling_max = ws_scaling_max
        self.wd_scaling_min = wd_scaling_min
        self.wd_scaling_max = wd_scaling_max
        self.ti_scaling_min = ti_scaling_min
        self.ti_scaling_max = ti_scaling_max
        self.yaw_scaling_min = yaw_scaling_min
        self.yaw_scaling_max = yaw_scaling_max
        self.seed = seed
        self.TurbBox = TurbBox
        self.turbine = turbine
        # Steady-state pitch/RPM lookup (None = feature off). The per-turbine
        # values are refreshed in _take_measurements each sim step.
        self.op_lookup = op_lookup
        self.current_pitch = None
        self.current_rpm = None
        # Sensor-derived wd estimator (None = feature off). Built lazily per
        # reset (needs n_turb / x_pos), updated in _take_measurements — zero
        # extra flow queries. See core/wd_estimator.py.
        self.wd_est_tau = wd_est_tau
        self.wd_est_consensus = wd_est_consensus
        self._wd_estimator = None
        # The maximum time of the simulation. This is used to make sure that the simulation doesnt run forever.
        self.time_max = 0
        # The number of times the flow passes through the farm. This is used to calculate the maximum simulation time.
        self.n_passthrough = n_passthrough
        if self.max_time_steps is not None:
            # Fixed episode length: the passthrough method is disabled. Force
            # n_passthrough so high that any code path still deriving an episode
            # length from it can never truncate before max_time_steps.
            self.n_passthrough = 999_999_999
            print(
                f"max_time_steps={self.max_time_steps}: episode length fixed to "
                f"{self.max_time_steps} env steps "
                f"({self.max_time_steps * self.delay} s of simulation); "
                f"n_passthrough is ignored."
            )
        self.timestep = 0

        # The initial yaw of the turbines. This is used if the yaw_init is "Defined"
        self.yaw_initial = [0]

        # --- Load config ---
        cfg = self._normalize_config_input(config)
        self._apply_config(cfg)

        # Constructor kwarg overrides the config `farm: tilt` value
        if tilt is not None:
            self.tilt = float(tilt)

        # DWM physics: constructor kwarg over config `dwm_setup`, else plain
        # dynamiks (opt in to the LES calibration via "les_calibrated").
        self.dwm_setup = resolve_dwm_setup(
            dwm_setup if dwm_setup is not None else self.dwm_setup
        )

        # Derating: extend the action space. The turbine itself must accept a
        # 'derate' input — the env only forwards per-turbine derate values.
        # With yaw_action=False the agent controls derating only (act_var=1).
        if self.derate_action:
            self.act_var = 2 if self.yaw_action else 1
            if self.HTC_path is not None:
                check_htc_supports_derating(self.HTC_path, self.derate_reference)
            else:
                check_turbine_supports_derating(turbine)
            if self.derate_reference == "rated":
                # Power-curve maximum, NOT self.rated_power (that one is the
                # power at the episode inflow ws, set per-reset for rewards).
                self.derate_rated_power = float(
                    np.max(self.turbine.power(np.arange(0.0, 30.01, 0.1)))
                )
        elif not self.yaw_action:
            raise ValueError("yaw_action=False requires derate_action=True")

        self.n_turb = len(x_pos)  # The number of turbines

        # Sets the yaw init method. If Random, then the yaw angles will be random. Else they will be zeros
        # Use yaw_init parameter if provided, otherwise use value from config
        yaw_init_method = yaw_init if yaw_init is not None else self.yaw_init
        self._yaw_init = self._create_yaw_initializer(yaw_init_method)

        # Initialize the reward calculator

        self.reward_calculator = RewardCalculator(
            power_reward_type=self.power_reward,
            track_power=self.Track_power,
            power_scaling=self.Power_scaling,
            action_penalty=self.action_penalty,
            action_penalty_type=self.action_penalty_type,
            power_window_size=self.power_avg,
            tau=self.tau,
            derate_penalty=self.derate_penalty,
            derate_penalty_type=self.derate_penalty_type,
            track_reward_type=self.track_reward_type,
            track_sigma=self.track_sigma,
        )

        # Initialize the wind manager
        self.wind_manager = WindManager(
            ws_min=self.ws_inflow_min,
            ws_max=self.ws_inflow_max,
            wd_min=self.wd_inflow_min,
            wd_max=self.wd_inflow_max,
            ti_min=self.TI_inflow_min,
            ti_max=self.TI_inflow_max,
            sample_site=sample_site,
            veer_min=self.veer_inflow_min,
            veer_max=self.veer_inflow_max,
        )

        # Initialize the power tracking manager (reference generation)
        self.power_tracking = (
            PowerTrackingManager(
                ref_function=self.power_ref_function,
                ref_range=tuple(self.track_ref_range),
                preview_steps=self.track_obs_preview,
            )
            if self.Track_power
            else None
        )

        # Initialize the turbulence manager
        self.turbulence_manager = TurbulenceManager(
            turbulence_type=turbtype,
            turbulence_box_path=TurbBox,
            max_turb_move=max_turb_move,
            memmap_boxes=self.turb_memmap,
            dwm_setup=self.dwm_setup,
        )

        # Initialize the renderer
        self.renderer = WindFarmRenderer(
            render_mode=render_mode,
            fix_turbines=fix_turbines,
            show_indices=show_indices,
            fontsize=fontsize,
            axes_lw=axes_lw,
            colorbar_vmax_step=colorbar_vmax_step,
        )

        # If we need to have a "baseline" farm, then we need to set up the baseline controller
        # This could be moved to the Power_reward check, but I have a feeling this will be expanded in the future, when we include damage.
        if self.power_reward in ("Baseline", "Wake_recovery") or Baseline_comp:
            self.Baseline_comp = True
        else:
            self.Baseline_comp = False

        # Initialize the baseline manager
        self.baseline_manager = None
        if self.Baseline_comp:
            self.baseline_manager = BaselineManager(
                baseline_controller_type=self.BaseController,
                x_pos=self.x_pos,
                y_pos=self.y_pos,
                turbine=turbine,
                yaw_max=self.yaw_max,
                yaw_min=self.yaw_min,
                yaw_step_env=self.yaw_step_env,
                yaw_step_sim=self.yaw_step_sim,
                htc_path=HTC_path,
                hawc2_yaw_mode=self.hawc2_yaw_mode,
                hawc2_yaw_slot=self.hawc2_yaw_slot,
                dwm_setup=self.dwm_setup,
            )

        # #Initializing the measurements class with the specified values.
        self._init_farm_mes()

        # The maximum history length of the measurements
        # self.hist_max = self.farm_measurements.max_hist()
        self.hist_max = max(self.power_avg, self.farm_measurements.max_hist())

        # Figure out the ammount of steps to do at the reset
        if self.fill_window is True:
            self.steps_on_reset = self.hist_max
        elif isinstance(self.fill_window, int) and self.fill_window >= 1:
            if self.fill_window > self.hist_max:
                self.fill_window = (
                    self.hist_max
                )  # fill_window cannot be larger then the max history length
            self.steps_on_reset = self.fill_window
        elif self.fill_window is False:
            self.steps_on_reset = 1
        else:
            raise ValueError("fill_window must be True or a non-negative integer")

        # Setting up the turbines:

        self.D = turbine.diameter()

        # Define the observation and action space
        self.obs_var = self.farm_measurements.observed_variables()

        self._init_spaces()

        # Base DWM closure-param overrides used at every reset unless an
        # episode supplies its own via reset(options={"dwm_params": ...}).
        # Unknown keys raise immediately.
        self._base_dwm_params: dict = dict(dwm_params) if dwm_params else {}
        self._validate_dwm_keys(self._base_dwm_params)
        self._active_dwm_params: dict = dict(self._base_dwm_params)

        if reset_init:
            # We should have this here, to set the seeding correctly
            self.reset(seed=seed)

        # Asserting that the render_mode is valid.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Keep for compatibility
        # Note: init_render() will be called lazily when first needed (after reset creates self.fs)

    @property
    def TF_files(self) -> list:
        """MannLoad turbulence box files (the turbulence manager's list)."""
        return self.turbulence_manager.turbulence_files

    @TF_files.setter
    def TF_files(self, files) -> None:
        self.turbulence_manager.set_turbulence_files(files)

    def _create_yaw_initializer(self, method: str):
        """
        Factory method for creating yaw initialization functions.

        Args:
            method: Initialization method ("Random", "Defined", or default to zeros)

        Returns:
            Callable that initializes yaw angles
        """
        if method == "Random":
            return lambda **kwargs: self.np_random.uniform(
                low=kwargs["min_val"], high=kwargs["max_val"], size=kwargs["n"]
            )
        elif method == "Defined":
            return lambda **kwargs: utils.defined_yaw(kwargs["yaws"], self.n_turb)
        else:
            return lambda **kwargs: np.zeros(kwargs["n"])

    def _init_wts(self):
        """
        Initialize the wind turbines.
        If the HTC path is given, then use hawc2 turbines, else use pywake turbines.
        Also is we have a baseline, then set that up also
        """
        self.wts = None
        self.wts_baseline = None

        if self.HTC_path is not None:  # pragma: no cover
            # TODO HTC stuff is not covered by the tests atm
            # If we have a high fidelity turbine model, then we need to load it in

            # We need to make a unique string, such that the results file doenst get overwritten.
            # The pid makes it collision-proof across vectorized env workers on
            # the same node (the random suffix alone can collide across 15 envs).
            node_string = socket.gethostname().split(".")[0]
            name_string = f"{node_string}_{os.getpid()}_{self.wd:.2f}_{self.ws:.2f}_{self.ti:.2f}_{self.np_random.integers(low=0, high=45000)}"
            name_string = name_string.replace(".", "p")

            # MultiH2Lib spawns one subprocess per turbine. Those children can only be
            # managed (closed / polled) from the process that created them, so remember
            # who that is and gate cleanup on it (see _safe_close_h2).
            self._h2_owner_pid = os.getpid()

            self.wts = HAWC2WindTurbinesW(  # power() normalized to W (native HAWC2 is kW)
                x=self.x_pos,
                y=self.y_pos,
                htc_lst=[self.HTC_path],
                case_name=name_string,  # subfolder name in the htc, res and log folders
                suppress_output=True,  # don't show hawc2 output in console
                # Feed the DTU WE controller its hub wind speed (general
                # variables 1-3); the dynamiks default leaves them at zero.
                coupling=HubWindFreeWindCoupling(),
            )
            # Yaw sensor wiring is htc-specific (constraint naming + which
            # general-variable slot the servo DLL reads), so it is configurable.
            add_hawc2_yaw_sensor(
                self.wts, mode=self.hawc2_yaw_mode, slot=self.hawc2_yaw_slot
            )
            if self.derate_action:
                # d <-> dr% mapping and sensor wiring live in core/derating.
                add_hawc2_derate_sensor(self.wts, self.n_turb)
        else:  # If we have no HTC path, use the pywake turbine
            # Rotor-average and TI model per the DWMSetup.
            self.wts = make_wts(
                x=self.x_pos,
                y=self.y_pos,  # x and y position of two wind turbines
                windTurbine=self.turbine,
                setup=self.dwm_setup,
            )
            if self.derate_action:
                self.wts.add_sensor("derate", expose=True)
                self.wts.sensors.derate = np.zeros(self.n_turb)

        # Initialize baseline turbines if needed
        if self.Baseline_comp and self.baseline_manager is not None:
            # Pass name_string if we have it (only created for HAWC2 turbines)
            baseline_name = name_string if self.HTC_path is not None else None
            self.wts_baseline = self.baseline_manager.initialize_baseline_turbines(
                name_string=baseline_name
            )
        else:
            self.wts_baseline = None

    def _normalize_config_input(self, config):
        """Normalize the config input to a dict (see ``core.env_config.load_config_dict``).

        Also records ``self.yaml_path`` (the file path, or None).
        """
        cfg, self.yaml_path = load_config_dict(config)
        return cfg

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Validate the config dict and expose every setting as a flat attribute.

        Parsing lives in ``core.env_config.parse_env_settings``; this method
        copies the resulting ``EnvSettings`` fields onto ``self`` (wrappers and
        downstream code read ``env.yaw_max``, ``env.mes_level`` etc. directly)
        and builds the probe manager.
        """
        self._settings = parse_env_settings(config)
        for k, v in vars(self._settings).items():
            setattr(self, k, v)

        # Initialize probe manager
        self.probe_manager = ProbeManager(probes_config=self.probes_config)

        # Keep references for backward compatibility
        self.probes = self.probe_manager.probes
        self.turbine_probes = self.probe_manager.turbine_probes

        # Set n_probes_per_turb now that probe_manager is initialized
        self.n_probes_per_turb = self.probe_manager.count_probes_per_turbine()

    def _init_farm_mes(self) -> None:
        """
        This function initializes the farm measurements class.
        This id done partly due to modularity, but also because we can delete it from memory later, as I suspect this might be the source of the memory leak
        """
        # Initializing the measurements class with the specified values.
        # The env carries every config setting flat on itself, so it serves
        # as the settings object.
        self.farm_measurements = build_farm_mes(
            self,
            n_turb=self.n_turb,
            scaling=dict(
                ws_min=self.ws_scaling_min,
                ws_max=self.ws_scaling_max,
                wd_min=self.wd_scaling_min,
                wd_max=self.wd_scaling_max,
                TI_min=self.ti_scaling_min,
                TI_max=self.ti_scaling_max,
            ),
            maxturbpower=self.maxturbpower,
        )

        # Deques that holds the power output of the farm and the baseline farm. This is used for the power reward
        self.farm_pow_deq = deque(maxlen=self.power_avg)
        self.base_pow_deq = deque(maxlen=self.power_avg)
        self.nowake_pow_deq = deque(maxlen=self.power_avg)
        self.power_len = self.power_avg

        for i, tm in enumerate(self.farm_measurements.turb_mes):
            # n_probes comes from the config counts (probe objects only exist
            # after reset creates the flow sim); the actual probe objects are
            # (re-)attached in reset() right after initialize_probes().
            tm.probes = self.turbine_probes.get(i, [])
            tm.n_probes = self.n_probes_per_turb.get(i, 0)
            tm.probe_min = self.ws_scaling_min
            tm.probe_max = self.ws_scaling_max

    def _init_spaces(self):
        """
        This function initializes the observation and action spaces.
        This is done in a seperate function, so we can replace it in the multi agent version of the environment
        """
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=((self.obs_var),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=((self.n_turb * self.act_var),), dtype=np.float32
        )

    def get_obs_dim_per_turbine(self) -> int:
        """Get observation dimension per turbine"""
        return self.farm_measurements.turb_mes[0].observed_variables()

    def init_render(self):
        """Initialize rendering - delegates to renderer."""
        self.renderer.init_render(self.fs, self.turbine)

    def _take_measurements(self) -> None:
        """
        Does the measurement and saves it to the self.
        """
        # Get the observation of the environment
        xyz_turbines = self.fs.windTurbines.rotor_positions_xyz

        self.current_ws = np.linalg.norm(
            self.fs.windTurbines.rotor_avg_windspeed, axis=1
        )

        self.current_wd = self.fs.get_wind_direction(
            xyz=xyz_turbines, include_wakes=True
        ).flatten()

        # Sensor-derived wd estimate: fold the measured local directions into
        # the circular EWMA every substep (zero extra flow queries).
        if self._wd_estimator is not None:
            self._wd_estimator.update(self.current_wd)

        # Copy: on dynamiks 0.0.7 the sensor getter returns the live array, so a
        # bare reference would keep changing inside already-returned info dicts.
        self.current_yaw = np.array(self.fs.windTurbines.yaw, dtype=float)
        self.current_powers = self.fs.windTurbines.power()  # The Power pr turbine

        if self.op_lookup is not None:
            # Same (ws, yaw, derate) triple the power surrogate sees, so the
            # reported operating point is consistent with current_powers.
            self.current_pitch, self.current_rpm = self.op_lookup.pitch_rpm(
                self.current_ws,
                self.current_yaw,
                getattr(self, "current_derate", np.zeros(self.n_turb)),
            )

    def _get_obs(self) -> np.ndarray:
        """
        Gets the sensordata from the farm_measurements class, and scales it to be between -1 and 1
        If you want to implement your own handling of the observations, then you can do that here by overwriting this function
        """

        values = self.farm_measurements.get_measurements(scaled=True)
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    @property
    def wd_est(self) -> float:
        """Sensor-derived consensus wd estimate (deg). Only available when the
        env was built with wd_est_tau; the trainer swaps this in for env.wd
        under --wd_source est."""
        if self._wd_estimator is None:
            raise AttributeError(
                "wd_est requires the env to be built with wd_est_tau "
                "(the sensor-derived estimator is off)"
            )
        return self._wd_estimator.estimate

    @property
    def wd_est_per_turbine(self) -> np.ndarray:
        """Per-turbine filtered wd (deg); see wd_est."""
        if self._wd_estimator is None:
            raise AttributeError(
                "wd_est_per_turbine requires the env to be built with "
                "wd_est_tau (the sensor-derived estimator is off)"
            )
        return self._wd_estimator.per_turbine

    def _nowake_power(self, fs) -> float:
        """Farm no-wake power in W for either turbine backend.

        HAWC2WindTurbines asserts on ``power(include_wakes=False)`` (the aero
        power sensor always sees the waked inflow), so for HAWC2 we evaluate
        the PyWake power curve at the free-stream rotor-average wind speed the
        DWM already computes every step (``rotor_avg_freestream``, set in
        dynamiks' ``update_wind_turbine_sensors``). This mirrors what
        PyWakeWindTurbines.power(include_wakes=False) does internally, minus
        the yaw-cosine kwargs — negligible for the greedy (≈0° yaw) baseline
        farm this feeds via ``nowake_pow_deq`` (the Wake_recovery headroom).
        Units are W in both branches (HAWC2 path goes through the PyWake curve;
        the waked HAWC2 path is W via HAWC2WindTurbinesW).
        """
        if self.HTC_path is None:
            return fs.windTurbines.power(include_wakes=False).sum()
        ws_free = np.asarray(fs.windTurbines.rotor_avg_freestream)[:, 0]
        return float(np.sum(self.turbine.power(ws_free)))

    def _build_dwm_fs(self, site, windTurbines, closure_overrides):
        """Construct the DWM flow simulation from ``self.dwm_setup``
        (dwm_defaults.make_dwm); closure_overrides (k1, k2, d_particle) are
        the per-episode DR draws (empty dict = the setup's values)."""
        return make_dwm(
            site=site,
            windTurbines=windTurbines,
            wind_direction=self.wd,
            dt=self.dt,
            addedTurbulenceModel=self.addedTurbulenceModel,
            setup=self.dwm_setup,
            interpolation=self.interpolation,
            lateral_cutoff=self.lateral_cutoff,
            **closure_overrides,
        )

    def _get_info(self) -> dict[str, Any]:
        """
        Return info dictionary.
        If we have a baseline comparison, then we also return the baseline values.
        """
        return_dict = {
            "yaw angles agent": self.current_yaw,
            "yaw angles measured": self.farm_measurements.get_yaw_turb(),
            "Wind speed Global": self.ws,
            "Wind speed at turbines": self.current_ws,
            "Wind speed at turbines measured": self.farm_measurements.get_ws_turb(),
            "Wind speed at farm measured": self.farm_measurements.get_ws_farm(),
            "Wind direction Global": self.wd,
            "Wind direction at turbines": self.current_wd,
            "Wind direction at turbines measured": self.farm_measurements.get_wd_turb(),
            "Wind direction at farm measured": self.farm_measurements.get_wd_farm(),
            "Turbulence intensity": self.ti,
            "Wind veer Global": self.veer,
            "Turbine tilt": self.tilt,
            "Power agent": self.fs.windTurbines.power().sum(),
            "Power agent nowake": self._nowake_power(self.fs),
            "Power pr turbine agent": self.fs.windTurbines.power(),
            "Turbine x positions": self.fs.windTurbines.positions_xyz[0],
            "Turbine y positions": self.fs.windTurbines.positions_xyz[1],
            "Turbulence intensity at turbines": self.farm_measurements.get_TI_turb(),
        }

        if self._wd_estimator is not None:
            return_dict["Wind direction estimate"] = self.wd_est
            return_dict["Wind direction estimate per turbine"] = self.wd_est_per_turbine

        if self.derate_action:
            return_dict["derate agent"] = self.current_derate
            return_dict["derate command"] = self.derate_command
            return_dict["derate measured"] = self.farm_measurements.get_derate_turb()

        if self.power_tracking is not None:
            return_dict["Power reference"] = self.power_setpoint
            # Instantaneous error at the current env step (the reward uses
            # window means of both sides instead).
            return_dict["Tracking error"] = self.farm_pow_deq[-1] - self.power_setpoint
            # Window-mean error — the exact quantity the reward normalizes
            # (matches the `tracking_error` breakdown entry). Both deques are
            # filled by reset's warm-up, so this runs after they are non-empty.
            return_dict["Tracking error window mean"] = float(
                np.mean(self.farm_pow_deq) - np.mean(self.power_tracking.ref_deque)
            )
            return_dict["Power reference preview"] = self.power_preview

        if self.Baseline_comp:
            return_dict["yaw angles base"] = np.array(
                self.fs_baseline.windTurbines.yaw, dtype=float
            )  # copy, see current_yaw
            return_dict["Power baseline"] = self.fs_baseline.windTurbines.power().sum()
            return_dict["Power pr turbine baseline"] = (
                self.fs_baseline.windTurbines.power()
            )
            # return_dict["Wind speed at turbines baseline"] = self.fs_baseline.windTurbines.rotor_avg_windspeed[:,0] #Just the largest component
            return_dict["Wind speed at turbines baseline"] = (
                self.fs_baseline.windTurbines.rotor_avg_windspeed[:, 0]
            )  # Just the largest component
        return return_dict

    def _set_windconditions(self) -> None:
        """
        Sets the global windconditions for the environment
        """
        wind_cond = self.wind_manager.sample_conditions()
        self.ws, self.wd, self.ti = wind_cond.unpack()
        self.veer = wind_cond.veer

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment. This is called at the start of every episode.
        - The wind conditions are sampled, and the site is set.
        - The flow simulation is run for the time it takes for the flow to develop.
        - The measurements are filled up with the initial values.

        The steps run in a fixed order because several of them draw from
        ``self.np_random`` (wind sampling, HAWC2 case name, power reference,
        turbulence box, yaw init); reordering them changes seeded episodes.

        Domain-randomization hook: pass
        ``options={"dwm_params": {"k1": ..., "mann_L": ..., ...}}`` to override
        DWM parameters for this episode only. Two parameter groups are
        supported:

        * **Closure** (``k1``, ``k2``, ``d_particle``) — forwarded to
          ``make_dwm``. Honored under any ``turbtype``.
        * **Mann box** (``mann_L``, ``mann_GAMMA``, ``mann_AE``) — forwarded to
          ``TurbulenceManager.create_sites`` and used by ``MannGenerate`` to
          rebuild the box per reset. **Only honored under
          ``turbtype="MannGenerate"``** — under other turbtypes the box is not
          regenerated from these statistics and the env raises rather than
          silently dropping them.

        Under DR with any Mann key set, ``self.ti`` (sampled by
        ``wind_manager``) is no longer authoritative for the box's ambient TI —
        ``mann_AE`` is. ``self.ti`` is kept as a nominal value for observations
        and normalization.
        """
        # Episode-level override of DWM params (domain randomization).
        # Done before any heavy work so a typo fails fast.
        closure_overrides, mann_overrides = self._resolve_episode_overrides(options)

        # Clean up previous episode resources FIRST
        self._soft_cleanup()

        # Seed the RNG used by this Env (sets self.np_random)
        super().reset(seed=seed)
        self.timestep = 0

        # Set random generators for managers
        self.wind_manager.np_random = self.np_random
        if self.power_tracking is not None:
            self.power_tracking.np_random = self.np_random

        # 1) Global wind conditions (first RNG draw of the episode)
        self._sample_episode_conditions()

        # 2) Fresh measurement buffers
        self._init_farm_mes()
        if hasattr(self, "farm_measurements") and self.farm_measurements is not None:
            self.farm_measurements.np_random = self.np_random
        else:
            print("WARNING: farm_measurements was not initialized before reset.")

        # Rated power at current ws (for reward scaling)
        self.rated_power = self.turbine.power(self.ws)

        # 3) Turbines (HAWC2 draws its case-name suffix here) + per-episode state
        self._init_wts()
        # Fresh estimator per episode: lazy warm-start at the first
        # _take_measurements (the burn-in holds base_wd, so the filter starts
        # converged — no cold-start transient).
        if self.wd_est_tau is not None:
            from .core.wd_estimator import WdEstimator

            self._wd_estimator = WdEstimator(
                n_turb=self.n_turb,
                dt=self.dt_sim,
                tau=self.wd_est_tau,
                consensus=self.wd_est_consensus,
                x_pos=self.x_pos,
            )
        self.current_derate = np.zeros(self.n_turb)
        # Commanded derate fraction (what the agent asked for). Differs from
        # current_derate (the applied available-power fraction) in "rated"
        # reference mode; identical in "available" mode.
        self.derate_command = np.zeros(self.n_turb)

        # Set random generator for turbulence manager
        self.turbulence_manager.np_random = self.np_random

        # Episode length (t_developed / time_max) from farm geometry
        self._compute_episode_timing()

        # Precompute the power reference trajectory for this episode (needs
        # the final time_max and self.rated_power, both set above).
        if self.power_tracking is not None:
            self.power_tracking.reset_episode(
                self,
                time_max=self.time_max,
                delay=self.delay,
                power_avg=self.power_avg,
            )
            self.power_setpoint = self.power_tracking.reference(0)
            self.power_preview = self.power_tracking.preview(0)

        # Generate wind direction list for the episode (backend-agnostic)
        wd_list = self.wind_manager.make_wind_direction_list(
            base_wd=self.wd,
            time_max=self.time_max,
            dt_sim=self.dt_sim,
            t_developed=self.t_developed,
            steps_on_reset=self.steps_on_reset,
            wd_function=self.wd_function,
        )

        # 3a) Main flow sim (turbulence box draw happens inside create_sites)
        self._build_agent_flow_sim(wd_list, closure_overrides, mann_overrides)

        # Initial yaw (yaw_init="Random" draws here) and tilt
        self._init_yaw_and_tilt()

        # Must init probes after fs
        self._attach_probes()

        # 3b) Baseline flow sim (optional)
        self._build_baseline_flow_sim(wd_list, closure_overrides)

        # 3c) Run the flow for the time it takes to develop
        self._run_burn_in()

        # 4) Fill measurement history window (and power deques)
        self._warm_up_measurements()

        # 5) Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Init render can now be called as fs needs to be created first
        if self.render_mode in ["human", "rgb_array"]:
            self.init_render()

        return observation, info

    # ------------------------------------------------------------------
    # reset() steps (kept in call order; see reset docstring)
    # ------------------------------------------------------------------

    def _resolve_episode_overrides(self, options) -> tuple:
        """Merge ``options["dwm_params"]`` over the constructor's DWM params.

        Returns ``(closure_overrides, mann_overrides)``, the two subsystems
        that consume these keys, and records ``self._active_dwm_params``.
        """
        episode_overrides = (options or {}).get("dwm_params") or {}
        self._validate_dwm_keys(episode_overrides, " in reset options")
        self._active_dwm_params = {**self._base_dwm_params, **episode_overrides}

        # Partition into the two subsystems that consume these keys.
        closure_overrides = {
            k: v
            for k, v in self._active_dwm_params.items()
            if k in self._CLOSURE_PARAM_KEYS
        }
        mann_overrides = {
            k: v
            for k, v in self._active_dwm_params.items()
            if k in self._MANN_PARAM_KEYS
        }

        # Per-episode Mann statistics only take effect when the turbulence
        # manager actually regenerates the box. Fail loud rather than silently
        # ignoring them under the other branches.
        if mann_overrides and self.turbtype != "MannGenerate":
            raise ValueError(
                f"dwm_params contains Mann keys {sorted(mann_overrides)} but "
                f"turbtype={self.turbtype!r} does not regenerate the box from "
                "those statistics. Set turbtype='MannGenerate' on env "
                "construction, or remove the Mann keys from the DR sampler."
            )
        return closure_overrides, mann_overrides

    def _sample_episode_conditions(self) -> None:
        """Sample ws/wd/ti/veer; a Precursor box then pins ws/wd/ti from its meta."""
        # wind_cond = self.wind_manager.sample_conditions()
        # self.ws, self.wd, self.ti = wind_cond.unpack()
        self._set_windconditions()

        # Precursor inflow dictates the wind conditions: ws is the LES
        # advection speed (drives rated_power, obs scaling and the time
        # parameters), wd is the box orientation, ti the measured hub-height
        # TI. Must run before rated_power.
        if self.turbtype == "Precursor":
            if self.veer:
                raise ValueError(
                    "turbtype='Precursor' carries the LES shear/veer in the box "
                    "itself; remove veer_min/veer_max from the wind config."
                )
            _meta = self.turbulence_manager.precursor_meta
            self.ws = float(_meta["advection_speed"])
            self.wd = 270.0
            self.ti = float(_meta["ti_hub"])

    def _compute_episode_timing(self) -> None:
        """Set ``self.t_developed`` and ``self.time_max`` for this episode."""
        turb_pos = np.stack([self.x_pos, self.y_pos]).T
        self.t_developed, self.time_max = episode_time_parameters(
            turbine_positions=turb_pos,
            rotor_diameter=self.D,
            ws=self.ws,
            n_passthrough=self.n_passthrough,
            burn_in_passthroughs=self.burn_in_passthroughs,
        )

        # Optional fixed episode length: replaces the ws-derived time_max (the
        # passthrough method is disabled at construction) so that all parallel envs
        # truncate (and therefore autoreset) on the same global step. This runs before
        # make_wind_direction_list, so the wind-direction series is sized to the
        # fixed length and the flow never runs past it. time_max stays in seconds:
        # each env step advances the sim `delay` seconds, so N steps need N * delay
        # seconds of wind-direction series. The step counting (for both episode-length
        # methods) only starts after reset's burn-in and sensor fill.
        if self.max_time_steps is not None:
            self.time_max = self.max_time_steps * self.delay

    def _build_pywake_fs(self, wd_list):
        """Steady-state PyWake flow simulation adapter for the current conditions."""
        from .backend.pywake_adapter import (
            PyWakeFlowSimulationAdapter,
        )  # or adjust import path

        return PyWakeFlowSimulationAdapter(
            x=np.asarray(self.x_pos, float),
            y=np.asarray(self.y_pos, float),
            windTurbine=self.turbine,  # py_wake WindTurbines definition
            ws=self.ws,
            wd=self.wd,
            ti=self.ti,
            dt=self.dt,
            wd_lst=wd_list,
        )

    def _build_agent_flow_sim(self, wd_list, closure_overrides, mann_overrides) -> None:
        """Create the sites (dynamiks) and ``self.fs`` for the agent farm."""
        if self.backend == "dynamiks":
            # --- ORIGINAL dynamic backend ---
            # Create sites and turbulence fields
            turb_pos = np.stack([self.x_pos, self.y_pos]).T
            (
                self.site,
                self.site_base,
                _,
                _,
                self.addedTurbulenceModel,
            ) = self.turbulence_manager.create_sites(
                ws=self.ws,
                wd=self.wd,
                ti=self.ti,
                wd_list=wd_list,
                dt_sim=self.dt_sim,
                turbine_positions=turb_pos,
                rotor_diameter=self.D,
                n_passthrough=self.n_passthrough,
                burn_in_passthroughs=self.burn_in_passthroughs,
                create_baseline=self.Baseline_comp,
                mann_overrides=mann_overrides or None,
                # User-facing veer is deg per 100 m; create_sites takes deg/m.
                veer_rate=self.veer / 100.0,
                veer_ref_height=float(self.turbine.hub_height()),
                # Total sim seconds this episode consumes (burn-in + sensor
                # fill + episode). Only the Precursor branch uses it, to bound
                # its random start-time window; passed from here because
                # time_max was overridden in _compute_episode_timing (max_time_steps).
                episode_time_budget_s=(
                    self.t_developed + self.steps_on_reset * self.delay + self.time_max
                ),
            )

            self.fs = self._build_dwm_fs(
                site=self.site,
                windTurbines=self.wts,
                closure_overrides=closure_overrides,
            )
            self.wd = self.fs._wind_direction  # Update to match wd_list first value
        else:
            # --- STEADY pywake_steady backend ---
            if self.HTC_path is not None:
                raise NotImplementedError(
                    "pywake_steady backend does not support HAWC2WindTurbines."
                )
            if self.veer != 0:
                raise NotImplementedError(
                    "pywake_steady backend does not support wind veer."
                )
            self.fs = self._build_pywake_fs(wd_list)

    def _init_yaw_and_tilt(self) -> None:
        """Initial yaw command (bounded by yaw_start) and optional fixed tilt."""
        # self.yaw_command is our authoritative commanded SETPOINT. We mutate it in plain
        # Python and only ever write it to windTurbines.yaw. Reading windTurbines.yaw back
        # to compute the next command is unsafe for HAWC2, whose getter returns the lagging
        # physical bearing while the setter writes a setpoint (see _adjust_yaws).
        self.yaw_command = np.asarray(
            self._yaw_init(
                min_val=-self.yaw_start,
                max_val=self.yaw_start,
                n=self.n_turb,
                yaws=self.yaw_initial,
            ),
            dtype=float,
        ).copy()
        self.fs.windTurbines.yaw = self.yaw_command

        # Fixed rotor tilt (deg). Only set when nonzero so default behavior is
        # untouched. The tilt sensor exists on the PyWake turbines only; HAWC2
        # models carry their own physical tilt.
        if self.tilt != 0:
            if self.backend != "dynamiks" or self.HTC_path is not None:
                raise NotImplementedError(
                    "tilt is only supported on the dynamiks backend with "
                    "PyWake turbines."
                )
            self.fs.windTurbines.tilt = np.full(self.n_turb, float(self.tilt))

    def _attach_probes(self) -> None:
        """Create this episode's probes on ``self.fs`` and attach them to the measurements."""
        self.probe_manager.initialize_probes(self.fs, self.fs.windTurbines.yaw)
        # Update references to point to probe_manager's collections
        self.probes = self.probe_manager.probes
        self.turbine_probes = self.probe_manager.turbine_probes
        # Re-attach the freshly created probes to the measurement classes
        # (the ones attached in _init_farm_mes belong to the previous episode's
        # flow simulation, or don't exist yet on the very first reset).
        for i, tm in enumerate(self.farm_measurements.turb_mes):
            tm.probes = self.turbine_probes.get(i, [])

    def _build_baseline_flow_sim(self, wd_list, closure_overrides) -> None:
        """Create ``self.fs_baseline`` (when Baseline_comp) with the agent's initial yaw/tilt."""
        if not self.Baseline_comp:
            return
        if self.backend == "dynamiks":
            # Note: addedTurbulenceModel is intentionally shared with the
            # agent sim. DWMFlowSimulation calls model.initialize(fs) at
            # construction, and every attribute that sets (transport
            # speed, Mann field, per-turbine offsets) is deterministic
            # from the model seed and the (deep-copied) site, so both
            # sims see identical added turbulence; __call__ is read-only.
            self.fs_baseline = self._build_dwm_fs(
                site=self.site_base,
                windTurbines=self.wts_baseline,
                closure_overrides=closure_overrides,
            )
        else:
            if self.HTC_path is not None:
                raise NotImplementedError(
                    "pywake_steady baseline does not support HAWC2WindTurbines."
                )
            self.fs_baseline = self._build_pywake_fs(wd_list)

        # Start baseline with same yaw (and tilt) as agent at reset
        self.fs_baseline.windTurbines.yaw = self.fs.windTurbines.yaw
        if self.tilt != 0:
            self.fs_baseline.windTurbines.tilt = np.full(self.n_turb, float(self.tilt))

    def _run_burn_in(self) -> None:
        """Advance both flow sims by ``t_developed`` and sync the baseline controller."""
        if self.backend == "dynamiks":
            self.fs.run(self.t_developed)
            if self.Baseline_comp:
                self.fs_baseline.run(self.t_developed)
        else:
            # Steady-state: nothing evolves, but advancing time keeps the
            # adapter's wd_list indexing aligned with the dynamiks time base
            self.fs.run(self.t_developed)
            if self.Baseline_comp:
                self.fs_baseline.run(self.t_developed)

        if self.Baseline_comp and self.baseline_manager is not None:
            # Update baseline manager wind conditions
            self.baseline_manager.update_wind_conditions(
                ws=self.ws, wd=self.wd, ti=self.ti
            )

    def _warm_up_measurements(self) -> None:
        """Fill the measurement history window (and power deques) with no action applied."""
        # Uses the unified inner loop; no action applied during reset.
        for _ in range(self.steps_on_reset):
            out = self._advance_and_measure(
                self.sim_steps_per_env_step,
                apply_agent_action=False,
                action=None,
                include_baseline=self.Baseline_comp,
            )

            # Push means into measurement buffers
            self.farm_measurements.add_measurements(
                out["mean_windspeed"],
                out["mean_winddir"],
                out["mean_yaw"],
                out["mean_power"],
                derates=self.current_derate,
            )
            # Power history (farm-level)
            self.farm_pow_deq.append(out["mean_power"].sum())
            # Warm-up fills the reference deque with the step-0 reference so it
            # stays sample-aligned with farm_pow_deq.
            self._push_tracking(0, out["mean_power"].sum())
            if self.Baseline_comp:
                self.base_pow_deq.append(out["baseline_power_mean"].sum())
                self.nowake_pow_deq.append(self._nowake_power(self.fs_baseline))

    def _push_tracking(self, step_idx: int, farm_power: float) -> None:
        """
        Advance the power-tracking state by one env step (no-op when tracking
        is off): push the reference at *step_idx* into the window deque and
        update the setpoint/preview state plus the tracking observations.

        Args:
            step_idx: Env-step index into the reference trajectory
            farm_power: Farm power (W) just appended to farm_pow_deq
        """
        if self.power_tracking is None:
            return
        setpoint = self.power_tracking.push(step_idx)
        self.power_setpoint = setpoint
        self.power_preview = self.power_tracking.preview(step_idx)
        self.farm_measurements.set_tracking(
            setpoint=setpoint,
            error=farm_power - setpoint,
            preview=self.power_preview,
        )

    def _advance_and_measure(
        self,
        n_sim_steps: int,
        *,
        apply_agent_action: bool = False,
        action: np.ndarray | None = None,
        include_baseline: bool = False,
        ignore_steps: int = 0,
    ):
        """
        Advance the simulation n_sim_steps times.
        Optionally apply the agent action each sim step (yaw or wind method).
        Optionally step baseline using its controller.
        ignore_steps: skip the first `ignore_steps` sub-step samples when
        averaging - used to model the action delay (see `delay` in __init__).

        Returns:
            dict with keys:
            - time_array: (n_sim_steps,)
            - windspeeds, winddirs, yaws, powers: (n_sim_steps, n_turb)
            - baseline_powers, yaws_baseline, windspeeds_baseline (if include_baseline): same shapes
            - mean_windspeed, mean_winddir, mean_yaw, mean_power: (n_turb,)
            - baseline_power_mean (if include_baseline): scalar (farm sum) or (n_turb,) – here we return (n_turb,)
        """
        T = n_sim_steps
        if ignore_steps < 0:
            raise ValueError("ignore_steps must be non-negative")
        if ignore_steps >= T:
            raise ValueError(
                "ignore_steps must be smaller than n_sim_steps, "
                "otherwise no samples remain to average"
            )
        n = self.n_turb
        time_array = np.zeros(T, dtype=np.float32)
        windspeeds = np.zeros((T, n), dtype=np.float32)
        winddirs = np.zeros((T, n), dtype=np.float32)
        yaws = np.zeros((T, n), dtype=np.float32)
        powers = np.zeros((T, n), dtype=np.float32)
        derates = np.zeros((T, n), dtype=np.float32)
        if self.op_lookup is not None:
            pitches = np.zeros((T, n), dtype=np.float32)
            rpms = np.zeros((T, n), dtype=np.float32)

        if include_baseline:
            baseline_powers = np.zeros((T, n), dtype=np.float32)
            yaws_baseline = np.zeros((T, n), dtype=np.float32)
            windspeeds_baseline = np.zeros((T, n), dtype=np.float32)

        # If in "yaw" mode, we have an action budget that spans the env step.
        # Only the first n_turb entries are yaw; the rest (if any) are derate.
        if apply_agent_action and self.yaw_action and self.ActionMethod == "yaw":
            self.action_remaining = (
                action[: self.n_turb] * self.yaw_step_env
            )  # total budget for this env step

        # Stepwise derating: snapshot the derate at env-step start so the
        # per-substep application is idempotent (setpoint = base + delta).
        if apply_agent_action and self.derate_action and self.derate_method == "step":
            self._derate_step_base = np.asarray(
                self.derate_command, dtype=np.float64
            ).copy()

        for j in range(T):
            # 1) Agent yaw update (if any)
            if apply_agent_action:
                if self.yaw_action:
                    self._adjust_yaws(action)
                if self.derate_action:
                    self._apply_derating(action)

            # 2) Step agent flow
            wd_old = self.fs.wind_direction
            self.fs.step()

            wd_new = self.fs.wind_direction
            delta_wd = wd_new - wd_old
            # Track the wind shift on our own command, not via read-modify-write of the
            # (physical, lagging for HAWC2) yaw getter, which would clobber the command.
            self.yaw_command = self.yaw_command + delta_wd
            self.fs.windTurbines.yaw = self.yaw_command

            # Update the winddirection to match the flow sim
            self.wd = self.fs.wind_direction
            # 3) Baseline, only if requested
            if include_baseline:
                if apply_agent_action:
                    new_baseline_yaws = self.baseline_manager.compute_baseline_action(
                        fs=self.fs_baseline, yaw_step=self.yaw_step_sim
                    )
                    self.fs_baseline.windTurbines.yaw = new_baseline_yaws
                wd_old_baseline = self.fs_baseline.wind_direction
                self.fs_baseline.step()

                wd_new_baseline = self.fs_baseline.wind_direction
                delta_wd_baseline = wd_new_baseline - wd_old_baseline
                self.fs_baseline.windTurbines.yaw += delta_wd_baseline

            # 4) Measurements at this sim step
            self._take_measurements()

            # HF TI buffering (if requested)
            if self.farm_measurements.turb_TI or self.farm_measurements.farm_TI:
                for i in range(self.n_turb):
                    self.farm_measurements.turb_mes[i].add_hf_ws(self.current_ws[i])
                if self.farm_measurements.farm_TI:
                    self.farm_measurements.farm_mes.add_hf_ws(np.mean(self.current_ws))

            # 5) Store arrays
            windspeeds[j] = self.current_ws
            winddirs[j] = self.current_wd
            yaws[j] = self.current_yaw
            powers[j] = self.current_powers
            derates[j] = self.current_derate
            if self.op_lookup is not None:
                pitches[j] = self.current_pitch
                rpms[j] = self.current_rpm
            time_array[j] = self.fs.time

            self.probe_manager.update_probe_positions(self.fs, yaws[j])

            if include_baseline:
                baseline_powers[j] = self.fs_baseline.windTurbines.power(
                    include_wakes=self.baseline_wakes
                )
                yaws_baseline[j] = self.fs_baseline.windTurbines.yaw
                windspeeds_baseline[j] = np.linalg.norm(
                    self.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
                )
                # update probe positions to follow turbine

        # 6) Aggregate to per-env-step means (circular mean for wind direction).
        # The first ignore_steps sub-step samples are discarded: with an action
        # delay, only the final dt_env window feeds the observation.
        mean_windspeed = np.mean(windspeeds[ignore_steps:, :], axis=0)
        mean_winddir = utils.circ_mean_deg(winddirs[ignore_steps:, :], axis=0)
        mean_yaw = np.mean(yaws[ignore_steps:, :], axis=0)
        mean_power = np.mean(powers[ignore_steps:, :], axis=0)  # per-turbine

        result = dict(
            time_array=time_array,
            windspeeds=windspeeds,
            winddirs=winddirs,
            yaws=yaws,
            powers=powers,
            derates=derates,
            mean_windspeed=mean_windspeed,
            mean_winddir=mean_winddir,
            mean_yaw=mean_yaw,
            mean_power=mean_power,
        )
        if self.op_lookup is not None:
            result.update(pitches=pitches, rpms=rpms)
        if include_baseline:
            result.update(
                baseline_powers=baseline_powers,
                yaws_baseline=yaws_baseline,
                windspeeds_baseline=windspeeds_baseline,
                baseline_power_mean=np.mean(baseline_powers, axis=0),  # per-turbine
            )
        return result

    def _adjust_yaws(self, action):
        """
        Heavily inspired from https://github.com/AlgTUDelft/wind-farm-env
        This function adjusts the yaw angles of the turbines, based on the actions given, but we now have differnt methods for the actions
        (the arithmetic lives in ``core.actuation.step_yaw_command``).
        """
        # When derate_action=True action is [yaw_0..yaw_n | derate_0..derate_n];
        # yaw logic only operates on the first n_turb entries.
        action = action[: self.n_turb]

        # self.action_remaining only exists under the "yaw" method (set per env
        # step in _advance_and_measure); the other methods ignore it.
        self.yaw_command, remaining = step_yaw_command(
            self.ActionMethod,
            action,
            self.yaw_command,
            getattr(self, "action_remaining", None),
            yaw_step_sim=self.yaw_step_sim,
            yaw_min=self.yaw_min,
            yaw_max=self.yaw_max,
            # This clip is only usefull for the pywake turbine model, as the
            # hawc2 model has inertia anyways
            rate_limit=self.HTC_path is None,
        )
        if self.ActionMethod == "yaw":
            self.action_remaining = remaining
        self.fs.windTurbines.yaw = self.yaw_command

    def _apply_derating(self, action):
        """Apply per-turbine derating from the last n_turb entries of *action*.

        Action layout when derate_action=True:
            [yaw_0 .. yaw_n-1 | derate_0 .. derate_n-1]   (yaw_action=True)
            [derate_0 .. derate_n-1]                       (yaw_action=False)

        The command semantics (absolute vs step method, available vs rated
        reference, derate_step_sim slew) live in ``core.actuation.compute_derate``.
        HAWC2 turbines skip the rated-reference conversion: the DTUWEC
        controller applies the rated-power cap (and its dead zone) natively.
        """
        derate_raw = action[self.n_turb :] if self.yaw_action else action[: self.n_turb]

        self.derate_command, derate = compute_derate(
            derate_raw,
            method=self.derate_method,
            # _derate_step_base is only set (and only read) under the "step"
            # method; see _advance_and_measure.
            step_base=self._derate_step_base if self.derate_method == "step" else None,
            step_env=self.derate_step_env,
            derate_min=self.derate_min,
            derate_max=self.derate_max,
            reference=self.derate_reference,
            rated_power=getattr(self, "derate_rated_power", None),
            current_powers=self.current_powers,
            current_derate=self.current_derate,
            step_sim=self.derate_step_sim,
            hawc2=self.HTC_path is not None,
        )
        self.current_derate = derate

        if self.backend == "dynamiks":
            self.wts.sensors.derate = derate
        else:
            self.fs._derate = derate

    def step(self, action):
        """
        The step function
        1. Adjust the yaw angles of the turbines
        2. Take a step in the flow simulation
        3. Update the measurements
        4. Calculate the reward
        5. Return the observation, reward, terminated, truncated and info

        """

        # Save the old yaw angles, so we can calculate the change in yaw angles
        self.old_yaws = copy.copy(self.fs.windTurbines.yaw)
        # Same for derate levels (used by the derate "change" penalty)
        self.old_derate = copy.copy(self.current_derate)

        # Advance the sim `delay` seconds per env step, but only average the final
        # dt_env window into the observation. With delay > dt_env the agent acts on
        # a slower cadence than the sim, giving wakes time to propagate downstream
        # between actions. delay == dt_env (default) makes this a plain env step.
        steps_with_delay = int(
            self.sim_steps_per_env_step
            + ((self.delay - self.dt_env) // self.dt_env) * self.sim_steps_per_env_step
        )
        ignore_steps = int(steps_with_delay - self.sim_steps_per_env_step)

        out = self._advance_and_measure(
            steps_with_delay,
            apply_agent_action=True,
            action=action,
            include_baseline=self.Baseline_comp,
            ignore_steps=ignore_steps,
        )

        # add to measurements/history
        self.farm_measurements.add_measurements(
            out["mean_windspeed"],
            out["mean_winddir"],
            out["mean_yaw"],
            out["mean_power"],
            derates=self.current_derate,
        )
        self.farm_pow_deq.append(out["mean_power"].sum())
        # timestep is incremented further down, so the step we just simulated
        # has reference index timestep + 1 (reset's warm-up pushed index 0).
        self._push_tracking(self.timestep + 1, out["mean_power"].sum())
        if self.Baseline_comp:
            self.base_pow_deq.append(out["baseline_power_mean"].sum())
            self.nowake_pow_deq.append(self._nowake_power(self.fs_baseline))

        if np.any(np.isnan(self.farm_pow_deq)):
            raise Exception("NaN Power")

        # Build observation / info
        observation = self._get_obs()
        info = self._get_info()
        info["time_array"] = out["time_array"]
        info["windspeeds"] = out["windspeeds"]
        info["winddirs"] = out["winddirs"]
        info["yaws"] = out["yaws"]
        info["powers"] = out["powers"]
        info["derates"] = out["derates"]
        if self.op_lookup is not None:
            info["pitches"] = out["pitches"]
            info["rpms"] = out["rpms"]
        if self.Baseline_comp:
            info["baseline_powers"] = out["baseline_powers"]
            info["yaws_baseline"] = out["yaws_baseline"]
            info["windspeeds_baseline"] = out["windspeeds_baseline"]

        # Calculate the reward using the reward calculator
        reward = self.reward_calculator.calculate_total_reward(
            farm_power_deque=self.farm_pow_deq,
            old_yaws=self.old_yaws,
            new_yaws=self.fs.windTurbines.yaw,
            yaw_max=self.yaw_max,
            baseline_power_deque=self.base_pow_deq if self.Baseline_comp else None,
            rated_power=self.rated_power,
            n_turbines=self.n_turb,
            nowake_power_deque=self.nowake_pow_deq if self.Baseline_comp else None,
            old_derates=self.old_derate if self.derate_action else None,
            new_derates=self.current_derate if self.derate_action else None,
            derate_max=self.derate_max,
            power_ref_deque=(
                self.power_tracking.ref_deque if self.power_tracking else None
            ),
            power_norm=self.maxturbpower * self.n_turb,
        )[0]  # [0] gets just the reward value, not the breakdown

        # If we are at the end of the simulation, we truncate the agents.
        # Note that this is not the same as terminating the agents.
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API#theory
        # https://arxiv.org/pdf/1712.00378
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        self.timestep += 1
        # time_max is always in seconds while timestep counts env steps of `delay`
        # seconds each. With max_time_steps set, time_max == max_time_steps * delay,
        # so this fires at exactly max_time_steps steps (same computation on both
        # sides — float-exact). Keeping the single check also keeps time_max as the
        # one truncation knob (FarmEval overwrites it to run "forever").
        if self.timestep * self.delay >= self.time_max:
            truncated = True
            # Clean up the flow simulation. This is to make sure that we dont have a memory leak.
            if self.cleanup_on_time_limit:
                self._cleanup_resources()
        else:
            truncated = False

        terminated = False

        return observation, reward, terminated, truncated, info

    def _safe_close_h2(self, wt) -> None:
        """Close a HAWC2 turbine's h2 connection defensively (owner process only).

        See ``backend.hawc2_adapter.safe_close_h2``.
        """
        safe_close_h2(wt, getattr(self, "_h2_owner_pid", None))

    def _soft_cleanup(self) -> None:
        """
        Clean up resources between episodes.
        Closes connections and clears references but doesn't delete files.
        """
        # Close HAWC2 connections if they exist
        if self.HTC_path is not None:
            self._safe_close_h2(self.wts)
            self._safe_close_h2(self.wts_baseline)

        # Clear references
        self.fs_baseline = None
        self.site_base = None
        self.wts_baseline = None
        self.fs = None
        self.site = None
        self.wts = None
        self.farm_measurements = None
        gc.collect()

    def _cleanup_resources(self) -> None:
        """
        Full cleanup including HAWC2 file deletion. Called on episode truncation.
        """
        # Delete HAWC folders BEFORE clearing references (needs self.wts for paths)
        if self.HTC_path is not None:
            # Close connections first
            self._safe_close_h2(self.wts)
            if self.Baseline_comp:
                self._safe_close_h2(self.wts_baseline)
            # Then delete folders (self.wts holds the paths; skip if already cleaned up)
            if self.wts is not None:
                self._deleteHAWCfolder()

        # Now clear all references
        self.fs_baseline = None
        self.site_base = None
        self.wts_baseline = None
        self.fs = None
        self.site = None
        self.wts = None
        self.farm_measurements = None
        gc.collect()

    def _deleteHAWCfolder(self):
        """
        This deletes the HAWC2 results folder from the directory.
        This is done to make sure we keep it nice and clean.

        Each turbine writes into res/<case>, htc/<case> and log/<case>; this removes all
        three for both the agent and (if present) the baseline turbines. Skipped entirely
        when keep_hawc_results is set, so folders can be retained for later inspection.

        Called from cleanup paths (truncation and close()), possibly more than once and
        during teardown, so it must not raise if a folder is already gone.
        """
        if self.keep_hawc_results or self.wts is None:
            return

        self._delete_case_folders(self.wts)
        if self.Baseline_comp and self.wts_baseline is not None:
            self._delete_case_folders(self.wts_baseline)

    def _delete_case_folders(self, wts):
        """Remove the res/, htc/ and log/ case subfolders for one set of HAWC2 turbines.

        See ``backend.hawc2_adapter.delete_case_folders``.
        """
        delete_case_folders(wts)

    def render(
        self,
        fix_turbines=False,
        show_indices=None,
        fontsize=None,
        axes_lw=None,
        colorbar_vmax_step=None,
    ):
        """Render the current environment state.

        Args:
            fix_turbines (bool): If True, the farm layout is fixed and the wind
                direction rotates (EastNorthView). If False (default), the wind
                always points right and the farm rotates with it (XYView).
            show_indices (bool): Whether to annotate turbines with their index
                numbers. Defaults to True (set at renderer construction).
            fontsize (int): Font size used for all text in the plot.
                Defaults to 15 (set at renderer construction).
            axes_lw (float): Line width for turbine/wake outline elements.
                Defaults to 1.0 (set at renderer construction).
            colorbar_vmax_step (float): Step size for the colorbar tick spacing.
                Defaults to 2.0 (set at renderer construction).

        Returns:
            np.ndarray | None: RGB array when render_mode='rgb_array',
                otherwise None (frame is displayed directly for 'human' mode).
        """
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        if show_indices is not None:
            self.renderer.show_indices = show_indices
        if fontsize is not None:
            self.renderer.fontsize = fontsize
        if axes_lw is not None:
            self.renderer.axes_lw = axes_lw
        if colorbar_vmax_step is not None:
            self.renderer.colorbar_vmax_step = colorbar_vmax_step
        return self.renderer.render(
            self.fs, fs_baseline, probes, self.turbine, fix_turbines
        )

    def _render_frame_for_human(self, baseline=False):
        """Render the environment and return an RGB frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        return self.renderer._render_frame(
            self.fs, fs_baseline, probes=probes, baseline=baseline, turbine=self.turbine
        )

    def _render_frame(self, baseline=False):
        """Renders the current environment state and returns the frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        probes = self.probes if hasattr(self, "probes") else None
        # NB: keyword args — the renderer signature has no ws parameter, and a
        # positional self.ws used to land in fix_turbines, silently forcing
        # EastNorthView for any nonzero wind speed.
        return self.renderer._render_frame(
            self.fs, fs_baseline, probes=probes, baseline=baseline, turbine=self.turbine
        )

    def close(self):
        """Close the environment and clean up resources.

        This is what runs on a normal stop (``envs.close()`` / Ctrl+C), so for HAWC2
        (level 3) it must also delete the htc/res/log folders. Previously folder deletion
        only happened on time-limit truncation, so stopping a job mid-episode left the
        HAWC2 folders behind on the node.

        Gymnasium's ``VectorEnv.__del__`` also calls this at interpreter shutdown for
        any env left unclosed. By then CPython has torn down module globals (``plt``,
        ``gc``, ``shutil`` -> None), so the cleanup below would raise a spurious
        "Exception ignored in __del__" TypeError. Skip it during finalization: the OS
        reclaims memory/handles anyway, and real cleanup already ran on the explicit
        ``close()`` path (where ``sys.is_finalizing()`` is False).
        """
        if sys.is_finalizing():
            return
        self.renderer.close()
        if getattr(self, "HTC_path", None) is not None:
            # Full cleanup: close h2 connections + delete HAWC2 folders + drop refs.
            self._cleanup_resources()
        else:
            if self.Baseline_comp:
                self.fs_baseline = None
                self.site_base = None
            self.fs = None
            self.site = None
            self.farm_measurements = None
        gc.collect()

    def plot_farm(self, baseline=False, fix_turbines=False):
        """Plot the entire farm layout - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer.plot_farm(
            self.fs, fs_baseline, self.turbine, baseline, fix_turbines
        )

    def _render_farm(self, baseline=False):
        """Internal farm rendering - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer._render_farm(self.fs, fs_baseline, baseline)

    def plot_frame(self, baseline=False):
        """Plot a single frame - delegates to renderer."""
        fs_baseline = self.fs_baseline if self.Baseline_comp else None
        self.renderer.plot_frame(self.fs, fs_baseline, self.turbine, baseline)

    def _get_num_raw_features(self):
        """Calculate based on YAML config - no hardcoding!"""
        turbine_level = ("turb_ws", "turb_wd", "turb_TI", "turb_power")
        farm_level = ("farm_ws", "farm_wd", "farm_TI", "farm_power")
        return sum(self.n_turb for k in turbine_level if self.mes_level[k]) + sum(
            1 for k in farm_level if self.mes_level[k]
        )

    @property
    def pywake_agent(self):
        """Expose pywake_agent from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager.pywake_agent
        return None

    @property
    def py_agent_mode(self):
        """Expose py_agent_mode from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager.py_agent_mode
        return None

    @property
    def _base_controller(self):
        """Expose _base_controller from baseline_manager for backward compatibility."""
        if self.baseline_manager is not None:
            return self.baseline_manager._base_controller
        return None

    def __del__(self):
        """Destructor to ensure cleanup.

        Runs during garbage collection / interpreter shutdown, possibly on a
        partially-constructed instance or in a non-owning process, so it must never
        raise (a raising ``__del__`` only prints "Exception ignored in __del__").
        """
        if getattr(self, "HTC_path", None) is None and not hasattr(self, "fs"):
            return
        try:
            self._soft_cleanup()
        except (AttributeError, ImportError, OSError, EOFError):
            # Destructor during GC / interpreter shutdown — never let teardown raise.
            pass
