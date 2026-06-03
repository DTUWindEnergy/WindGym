"""
Baseline controller management module for WindGym environments.

This module handles baseline controller setup, management, and execution
for comparing agent performance against baseline control strategies.
"""

from typing import Optional, Callable, TYPE_CHECKING
import numpy as np

from dynamiks.wind_turbines import PyWakeWindTurbines
from ..backend.hawc2_adapter import HAWC2WindTurbinesW
from ..BasicControllers import local_yaw_controller, global_yaw_controller

# Use TYPE_CHECKING to avoid circular import at runtime
# PyWakeAgent is imported lazily in _setup_pywake_baseline
if TYPE_CHECKING:
    from ..Agents.PyWakeAgent import PyWakeAgent


class BaselineManager:
    """
    Manages baseline controller setup and execution.

    Supports multiple baseline controller types:
    - Local: Local yaw controller
    - Global: Global yaw controller
    - PyWake: PyWake optimization-based agent (oracle or local mode)

    Also handles baseline turbine initialization for HAWC2 or PyWake turbines.
    """

    def __init__(
        self,
        baseline_controller_type: str,
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        turbine,
        yaw_max: float,
        yaw_min: float,
        yaw_step_env: float,
        yaw_step_sim: float,
        htc_path: Optional[str] = None,
    ):
        """
        Initialize the baseline manager.

        Args:
            baseline_controller_type: Type of baseline controller
                                     ("Local", "Global", "PyWake_oracle", "PyWake_local")
            x_pos: X positions of turbines
            y_pos: Y positions of turbines
            turbine: Turbine object
            yaw_max: Maximum yaw angle (degrees)
            yaw_min: Minimum yaw angle (degrees)
            yaw_step_env: Yaw step per environment step (degrees)
            yaw_step_sim: Yaw step per simulation step (degrees)
            htc_path: Optional path to HAWC2 HTC file
        """
        self.baseline_controller_type = baseline_controller_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.turbine = turbine
        self.yaw_max = yaw_max
        self.yaw_min = yaw_min
        self.yaw_step_env = yaw_step_env
        self.yaw_step_sim = yaw_step_sim
        self.htc_path = htc_path

        # Controller function
        self._base_controller: Optional[Callable] = None

        # PyWake agent (if using PyWake baseline)
        self.pywake_agent: Optional["PyWakeAgent"] = None
        self.py_agent_mode: Optional[str] = None
        self.pywake_wd: Optional[float] = None
        self.pywake_ws: Optional[float] = None

        # Baseline turbines
        self.wts_baseline = None

        # Environment references (set later)
        self.wd = None  # Current wind direction
        self.ti = None  # Current turbulence intensity

        # Initialize controller
        self._setup_baseline_controller()

    def _setup_baseline_controller(self):
        """
        Set up the baseline controller based on type.

        Raises:
            ValueError: If baseline controller type is invalid
        """
        base = (self.baseline_controller_type or "").strip()
        kind = base.split("_", 1)[0]

        if kind == "Local":
            self._base_controller = local_yaw_controller
        elif kind == "Global":
            self._base_controller = global_yaw_controller
        elif kind == "PyWake":
            self._setup_pywake_baseline(base)
        else:
            raise ValueError(
                "BaseController must be one of: 'Local', 'Global', 'PyWake[_oracle|_local]'."
            )

    def _setup_pywake_baseline(self, base: str):
        """
        Set up PyWake baseline controller.

        Args:
            base: Full baseline controller string (e.g., "PyWake_oracle")
        """
        # Lazy import to avoid circular dependency
        from ..Agents.PyWakeAgent import PyWakeAgent

        mode = base.split("_", 1)[1] if "_" in base else "oracle"
        if mode not in {"oracle", "local"}:
            raise ValueError(f"PyWake mode must be 'oracle' or 'local', got '{mode}'")
        self.py_agent_mode = mode

        # lookup_mode is True if local mode, False if oracle mode
        lookup_mode = self.py_agent_mode == "local"

        # Create fake environment for PyWakeAgent
        # (PyWakeAgent expects an environment object with certain attributes)
        class FakeEnv:
            """Minimal environment interface for PyWakeAgent."""

            def __init__(
                self,
                action_method="wind",
                yaw_max=45,
                yaw_min=-45,
                yaw_step_env=1,
                wd=270,
            ):
                self.ActionMethod = action_method
                self.unwrapped = self
                self.yaw_max = yaw_max
                self.yaw_min = yaw_min
                self.yaw_step_env = yaw_step_env
                self.wd = wd

        temp_env = FakeEnv(
            action_method="wind",
            yaw_max=self.yaw_max,
            yaw_min=self.yaw_min,
            yaw_step_env=self.yaw_step_env,
        )

        self.pywake_agent = PyWakeAgent(
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            turbine=self.turbine,
            yaw_max=self.yaw_max,
            yaw_min=self.yaw_min,
            env=temp_env,
            look_up=lookup_mode,
        )
        self._base_controller = self._pywake_agent_wrapper

    def initialize_baseline_turbines(self, name_string: Optional[str] = None):
        """
        Initialize baseline turbines (HAWC2 or PyWake).

        Args:
            name_string: Optional name string for HAWC2 case (required if htc_path is set)

        Returns:
            Baseline turbine object
        """
        if self.htc_path is not None:
            if name_string is None:
                raise ValueError(
                    "name_string is required when initializing HAWC2 baseline turbines"
                )
            # HAWC2 high-fidelity turbines
            self.wts_baseline = HAWC2WindTurbinesW(  # power() normalized to W (native HAWC2 is kW)
                x=self.x_pos,
                y=self.y_pos,
                htc_lst=[self.htc_path],
                case_name=name_string + "_baseline",
                suppress_output=True,
            )
            # Add yaw sensor
            self.wts_baseline.add_sensor(
                name="yaw_getter",
                getter="constraint bearing2 yaw_rot 1 only 1;",
                expose=False,
                ext_lst=["angle", "speed"],
            )
            self.wts_baseline.add_sensor(
                "yaw",
                getter=lambda wt: np.rad2deg(wt.sensors.yaw_getter[:, 0]),
                setter=lambda wt, value: wt.h2.set_variable_sensor_value(
                    1, np.deg2rad(value).tolist()
                ),
                expose=True,
            )
        else:
            # PyWake turbines
            self.wts_baseline = PyWakeWindTurbines(
                x=self.x_pos,
                y=self.y_pos,
                windTurbine=self.turbine,
            )

        return self.wts_baseline

    def compute_baseline_action(self, fs, yaw_step: float = 1.0) -> np.ndarray:
        """
        Compute baseline controller action.

        Args:
            fs: Flow simulation object (baseline)
            yaw_step: Yaw step size (degrees)

        Returns:
            np.ndarray: New yaw angles for baseline turbines
        """
        if self._base_controller is None:
            raise RuntimeError("Baseline controller not initialized")

        return self._base_controller(fs, yaw_step)

    def _pywake_agent_wrapper(self, fs, yaw_step: float = 1.0) -> np.ndarray:
        """
        Wrapper for PyWake agent to use as baseline controller.

        This adapts the PyWake agent interface to work like other baseline controllers.

        Args:
            fs: Flow simulation object (baseline)
            yaw_step: Yaw step size (degrees, not used but kept for interface compatibility)

        Returns:
            np.ndarray: New yaw angles
        """
        if self.pywake_agent is None:
            raise RuntimeError("PyWake agent not initialized")

        # Update wind conditions based on mode
        if self.py_agent_mode == "local":
            # Local mode: Use wind conditions at front turbine
            front_tb = np.argmin(fs.windTurbines.positions_xyz[0, :])
            ws_front = fs.windTurbines.get_rotor_avg_windspeed(include_wakes=True)[
                :, front_tb
            ]
            ws_use = np.linalg.norm(ws_front)
            wd_use = np.rad2deg(np.arctan2(ws_front[1], ws_front[0])) + self.wd

            # Polyak averaging for smooth updates
            tau = 0.05
            self.pywake_wd = (1 - tau) * self.pywake_wd + tau * wd_use
            self.pywake_ws = (1 - tau) * self.pywake_ws + tau * ws_use

            self.pywake_agent.update_wind(
                wind_speed=self.pywake_ws,
                wind_direction=self.pywake_wd,
                TI=self.ti,
            )

        # Oracle mode uses global wind conditions (already set in update_wind)

        # Get action from PyWake agent
        action = self.pywake_agent.predict()[0]

        # Unscale actions from [-1, 1] to [yaw_min, yaw_max]
        new_yaws = (action + 1.0) / 2.0 * (self.yaw_max - self.yaw_min) + self.yaw_min

        # Adjust for wind direction error
        # (PyWake agent internally handles this, but not when used as baseline)
        pywake_error = self.pywake_agent.wdir - self.wd
        actual_yaws = new_yaws - pywake_error

        # Clip to yaw rate limits
        yaw_max_step = fs.windTurbines.yaw + self.yaw_step_sim
        yaw_min_step = fs.windTurbines.yaw - self.yaw_step_sim

        new_yaws = np.clip(
            np.clip(actual_yaws, yaw_min_step, yaw_max_step),
            self.yaw_min,
            self.yaw_max,
        )

        return new_yaws

    def update_wind_conditions(self, ws: float, wd: float, ti: float):
        """
        Update wind conditions for baseline manager.

        This is needed for PyWake agent in oracle mode and for tracking
        current conditions.

        Args:
            ws: Wind speed (m/s)
            wd: Wind direction (degrees)
            ti: Turbulence intensity (fraction)
        """
        self.wd = wd
        self.ti = ti

        # Initialize PyWake agent wind conditions
        if self.pywake_agent is not None:
            if self.py_agent_mode == "oracle":
                # Oracle mode: always update with global wind conditions
                self.pywake_agent.update_wind(wind_speed=ws, wind_direction=wd, TI=ti)
                self.pywake_ws = ws
                self.pywake_wd = wd
            elif self.pywake_ws is None or self.pywake_wd is None:
                # Local mode: initialize once (updated via Polyak averaging later)
                self.pywake_ws = ws
                self.pywake_wd = wd
