"""Per-episode result arrays for ``eval_single_fast`` and their xarray form."""

from __future__ import annotations

import numpy as np
import xarray as xr

# Dataset dimension tuples. ``ws/wd/TI/turbbox/model_step/deterministic`` are
# all length-1 for a single evaluation so ``xr.merge`` can stack conditions.
FARM_DIMS = ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic")
TURB_DIMS = ("time", "turb") + FARM_DIMS[1:]
SCALAR_DIMS = FARM_DIMS[1:]


class EpisodeRecorder:
    """Sim-resolution arrays for one evaluation episode.

    ``_a`` is the agent and ``_b`` the baseline. Which optional groups exist
    is fixed at construction (``baseline``, ``op_mode``, ``log_derate``,
    ``tracking``); the figure writer and ``to_dataset`` read the same flags.
    """

    def __init__(
        self,
        n_time: int,
        n_turb: int,
        *,
        baseline: bool,
        op_mode: bool,
        log_derate: bool,
        tracking: bool,
    ):
        self.n_time = n_time
        self.n_turb = n_turb
        self.baseline = baseline
        self.op_mode = op_mode
        self.log_derate = log_derate
        self.tracking = tracking
        time = n_time

        # Initialize the arrays to store the results
        # _a is the agent and _b is the baseline
        self.powerF_a = np.zeros((time), dtype=np.float32)
        self.powerT_a = np.zeros((time, n_turb), dtype=np.float32)
        self.yaw_a = np.zeros((time, n_turb), dtype=np.float32)
        self.ws_a = np.zeros((time, n_turb), dtype=np.float32)
        # float: dt_sim may be sub-second (G1: 0.125 s); an int array would
        # repeat each second step_val times and break concatenation on time.
        self.time_plot = np.zeros((time), dtype=np.float64)
        self.rew_plot = np.zeros((time), dtype=np.float32)

        # Steady-state operating point (blade pitch / rotor RPM) is available when
        # the env carries an OperatingPointLookup; the derate control signal is
        # available on any derating env. Both are off for yaw-only envs.
        if op_mode:
            self.pitch_a = np.zeros((time, n_turb), dtype=np.float32)
            self.rpm_a = np.zeros((time, n_turb), dtype=np.float32)
        if log_derate:
            self.derate_a = np.zeros((time, n_turb), dtype=np.float32)

        if tracking:
            self.p_ref = np.zeros((time), dtype=np.float32)
            self.track_err = np.zeros((time), dtype=np.float32)

        if baseline:
            self.powerF_b = np.zeros((time), dtype=np.float32)
            self.powerT_b = np.zeros((time, n_turb), dtype=np.float32)
            self.yaw_b = np.zeros((time, n_turb), dtype=np.float32)
            self.ws_b = np.zeros((time, n_turb), dtype=np.float32)
            self.pct_inc = np.zeros((time), dtype=np.float32)

    # ------------------------------------------------------------------
    def record_initial(self, env) -> None:
        """Fill index 0 from the freshly reset env (no reward at t=0)."""
        # Put the initial values in the arrays
        self.powerF_a[0] = env.fs.windTurbines.power().sum()
        self.powerT_a[0] = env.fs.windTurbines.power()
        self.yaw_a[0] = env.fs.windTurbines.yaw
        self.ws_a[0] = np.linalg.norm(env.fs.windTurbines.rotor_avg_windspeed, axis=1)
        self.time_plot[0] = env.fs.time
        # There is no reward at the first time step, so we just set it to zero.
        self.rew_plot[0] = 0.0

        # reset()'s warm-up already ran _take_measurements, so these exist here.
        if self.op_mode:
            self.pitch_a[0] = env.current_pitch
            self.rpm_a[0] = env.current_rpm
        if self.log_derate:
            self.derate_a[0] = env.current_derate

        if self.tracking:
            self.p_ref[0] = env.power_setpoint
            self.track_err[0] = self.powerF_a[0] - self.p_ref[0]

        if self.baseline:
            self.powerF_b[0] = env.fs_baseline.windTurbines.power().sum()
            self.powerT_b[0] = env.fs_baseline.windTurbines.power()
            self.yaw_b[0] = env.fs_baseline.windTurbines.yaw
            self.ws_b[0] = np.linalg.norm(
                env.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1
            )
            # Percentage increase in power output. This should be zero (or close
            # to zero) at the first time step. Baseline power can be 0 (e.g.
            # below cut-in), so guard the division.
            self.pct_inc[0] = (
                ((self.powerF_a[0] - self.powerF_b[0]) / self.powerF_b[0]) * 100
                if self.powerF_b[0] != 0
                else 0.0
            )

    def record_step(self, i: int, step_val: int, info: dict, reward) -> None:
        """Store env step ``i`` (``step_val`` sim samples) from its ``info``."""
        sl = slice(i * step_val + 1, i * step_val + step_val + 1)

        # Put the values in the arrays
        self.powerF_a[sl] = info["powers"].sum(axis=1)
        self.powerT_a[sl] = info["powers"]
        self.yaw_a[sl] = info["yaws"]
        self.ws_a[sl] = info["windspeeds"]
        self.time_plot[sl] = info["time_array"]
        self.rew_plot[sl] = reward

        if self.op_mode:
            self.pitch_a[sl] = info["pitches"]
            self.rpm_a[sl] = info["rpms"]
        if self.log_derate:
            self.derate_a[sl] = info["derates"]

        if self.tracking:
            # The reference is per env step; the error is at sim resolution.
            self.p_ref[sl] = info["Power reference"]
            self.track_err[sl] = info["powers"].sum(axis=1) - info["Power reference"]

        if self.baseline:
            self.powerF_b[sl] = info["baseline_powers"].sum(axis=1)
            self.powerT_b[sl] = info["baseline_powers"]
            self.yaw_b[sl] = info["yaws_baseline"]
            self.ws_b[sl] = info["windspeeds_baseline"]

            # Percentage increase in power output. Guard against zero
            # baseline power (e.g. below cut-in) -> report 0 instead of inf.
            agent_farm_power = info["powers"].sum(axis=1)
            base_farm_power = info["baseline_powers"].sum(axis=1)
            self.pct_inc[sl] = (
                np.divide(
                    agent_farm_power - base_farm_power,
                    base_farm_power,
                    out=np.zeros_like(base_farm_power),
                    where=base_farm_power != 0,
                )
                * 100
            )

    # ------------------------------------------------------------------
    def _farm(self, a):
        return a.reshape(self.n_time, 1, 1, 1, 1, 1, 1)

    def _turb(self, a):
        return a.reshape(self.n_time, self.n_turb, 1, 1, 1, 1, 1, 1)

    def to_dataset(
        self, *, ws, wd, ti, turbbox, model_step, deterministic
    ) -> xr.Dataset:
        """Build the evaluation dataset (coordinates are stored as given, uncast)."""
        # Common data variables
        data_vars = {
            "powerF_a": (FARM_DIMS, self._farm(self.powerF_a)),
            "powerT_a": (TURB_DIMS, self._turb(self.powerT_a)),
            "yaw_a": (TURB_DIMS, self._turb(self.yaw_a)),
            "ws_a": (TURB_DIMS, self._turb(self.ws_a)),
            "reward": (FARM_DIMS, self._farm(self.rew_plot)),
        }

        # Add operating-point / derate variables if applicable
        if self.op_mode:
            data_vars["pitch_a"] = (TURB_DIMS, self._turb(self.pitch_a))
            data_vars["rpm_a"] = (TURB_DIMS, self._turb(self.rpm_a))
        if self.log_derate:
            data_vars["derate_a"] = (TURB_DIMS, self._turb(self.derate_a))

        # Add tracking variables if applicable
        if self.tracking:
            # Per-condition scalar (no time dim) so it merges across conditions
            # in eval_multiple like any other data variable.
            track_mae = np.full(
                (1, 1, 1, 1, 1, 1), np.abs(self.track_err).mean(), dtype=np.float32
            )
            data_vars["power_ref"] = (FARM_DIMS, self._farm(self.p_ref))
            data_vars["track_err"] = (FARM_DIMS, self._farm(self.track_err))
            data_vars["track_mae"] = (SCALAR_DIMS, track_mae)

        # Add baseline variables if applicable
        if self.baseline:
            data_vars["powerF_b"] = (FARM_DIMS, self._farm(self.powerF_b))
            data_vars["powerT_b"] = (TURB_DIMS, self._turb(self.powerT_b))
            data_vars["yaw_b"] = (TURB_DIMS, self._turb(self.yaw_b))
            data_vars["ws_b"] = (TURB_DIMS, self._turb(self.ws_b))
            data_vars["pct_inc"] = (FARM_DIMS, self._farm(self.pct_inc))

        # Common coordinates
        coords = {
            "ws": np.array([ws]),
            "wd": np.array([wd]),
            "turb": np.arange(self.n_turb),
            "time": self.time_plot,
            "TI": np.array([ti]),
            "turbbox": [turbbox],
            "model_step": np.array([model_step]),
            "deterministic": np.array([deterministic]),
        }

        return xr.Dataset(data_vars=data_vars, coords=coords)
