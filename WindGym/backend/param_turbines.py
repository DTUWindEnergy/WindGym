"""PyWake turbines with per-turbine model perturbations (turbine-parameter DR).

``ParamPyWakeWindTurbines`` subclasses dynamiks' ``PyWakeWindTurbines`` (no
dynamiks edits) and applies a ``TurbineParams`` vector:

* ``power`` is scaled by ``cp_gain`` and ``max(cos(yaw_actual), 0) ** yaw_exp``
  and, for the waked branch, passed through a first-order lag ``tau_power``;
* ``ct`` is scaled by ``ct_gain`` (reaches ``rotor_avg_induction`` and the
  Ainslie boundary particle; Ct is NOT lagged, documented trade-off);
* the ``yaw`` sensor is rerouted: the setter stores the COMMAND, the getter
  returns the ACTUAL yaw after a dead time ``delay_yaw`` (whole sim steps) and
  a first-order lag ``tau_yaw``. ``wts.yaw``, ``sensors.yaw``, ``yaw_tilt()``
  and ``get_kwargs()`` (the PyWake yaw kwarg) therefore all see the realised
  yaw, so wake deflection, power and the observed yaw are consistent;
* ``yaw_bias`` (a heading error): the PHYSICS reads the realised yaw minus the
  bias -- ``yaw_tilt()`` (the yaw the wake particles are emitted with),
  ``get_kwargs()`` (power and thrust) and the ``yaw_exp`` cosine -- while
  ``wts.yaw`` / ``sensors.yaw`` (the observation, the env's command logic and
  the greedy controllers) stay in the commanded frame. bias 0 is a bitwise
  identity.

Step-handler placement (``FlowSimulation.step_handlers`` =
``[update_particles] + site + windTurbines + user``):

* ``step_actuators`` is inserted at index 0 of the FLOW SIMULATION's handler
  list by ``attach_to(fs)``: it runs before ``update_particles`` so the
  particles of this step are emitted with the realised yaw.
* ``step_power_lag`` is index 0 of the TURBINE handler list (set in
  ``__init__``, copied into the flow simulation at construction): it runs
  after ``update_particles`` has refreshed ``rotor_avg_windspeed`` and after
  the site handlers, but before the ``sensors`` data dumper reads ``power``.

Accepted approximation: the site may change the wind direction inside step t
and the env applies the yaw frame shift (``shift_yaw_frame``) only after
``fs.step()`` returns, so the power-lag target computed in that step uses a
yaw that is ``delta_wd`` off. Only turbines with ``tau_power > 0`` are
affected, and only while wd is changing.
"""

from __future__ import annotations

import numpy as np
from dynamiks.wind_turbines import PyWakeWindTurbines

from WindGym.core.turbine_params import TurbineParams


def _get_yaw_actual(wts):
    return wts._yaw_act.copy()


def _set_yaw_command(wts, value):
    wts._yaw_cmd[:] = np.asarray(value, dtype=float)


class ParamPyWakeWindTurbines(PyWakeWindTurbines):
    """``PyWakeWindTurbines`` perturbed by a per-turbine ``TurbineParams``."""

    def __init__(self, x, y, windTurbine, params: TurbineParams, **kwargs):
        super().__init__(x=x, y=y, windTurbine=windTurbine, **kwargs)
        if not isinstance(params, TurbineParams):
            raise TypeError(
                "params must be a TurbineParams (see normalize_turbine_params)"
            )
        if params.n_turb != self.N:
            raise ValueError(f"params has {params.n_turb} turbines, farm has {self.N}")
        self.params = params
        n = self.N
        k_max = int(params.k_delay.max()) if n else 0
        # Actuator state. Row 0 of the history is the newest command.
        self._yaw_cmd = np.zeros(n)
        self._yaw_act = np.zeros(n)
        self._yaw_hist = np.zeros((k_max + 1, n))
        self._power_state = None  # lagged waked power (W); None until the first step
        self._attached_fs = None

        # Reroute the plain-storage yaw sensor: the exposed `yaw` property,
        # sensors.yaw, yaw_tilt() and get_kwargs() all look this pair up at
        # access time, so no further overrides are needed.
        self.sensors.dict["yaw"] = (_get_yaw_actual, _set_yaw_command)
        # Power lag runs after update_particles / site handlers, before the
        # sensors dumper (index 0 of the TURBINE handler list).
        self.step_handlers.insert(0, self.step_power_lag)

    # ------------------------------------------------------------ env API --
    @property
    def yaw_command(self) -> np.ndarray:
        """Copy of the commanded yaw (deg, wind-relative frame)."""
        return self._yaw_cmd.copy()

    def reset_yaw_state(self, yaw0) -> None:
        """Set command, actual and the dead-time buffer to ``yaw0``: no transient."""
        yaw0 = np.broadcast_to(np.asarray(yaw0, dtype=float), (self.N,))
        self._yaw_cmd[:] = yaw0
        self._yaw_act[:] = yaw0
        self._yaw_hist[:] = yaw0[None, :]

    def shift_yaw_frame(self, delta_wd: float) -> None:
        """Wind-direction frame shift: nacelles are fixed in the ground frame, so
        command, actual AND every buffered command move by ``delta_wd``."""
        self._yaw_cmd += delta_wd
        self._yaw_act += delta_wd
        self._yaw_hist += delta_wd

    def attach_to(self, fs) -> None:
        """Register ``step_actuators`` as the FIRST step handler of ``fs``."""
        assert float(fs.dt) == float(
            self.params.dt_sim
        ), f"flow simulation dt {fs.dt} != TurbineParams.dt_sim {self.params.dt_sim}"
        fs.step_handlers.insert(0, self.step_actuators)
        self._attached_fs = fs

    # ------------------------------------------------------ step handlers --
    def step_actuators(self, fs) -> None:
        """Dead time + first-order lag from command to actual yaw (once per sim step)."""
        hist = self._yaw_hist
        if hist.shape[0] > 1:
            hist[1:] = hist[:-1]
        hist[0] = self._yaw_cmd
        effective = hist[self.params.k_delay, np.arange(self.N)]
        a = self.params.a_yaw
        # np.where (not y + 1.0*(e-y)) so tau == 0 turbines stay bitwise exact.
        self._yaw_act = np.where(
            a >= 1.0, effective, self._yaw_act + a * (effective - self._yaw_act)
        )

    def step_power_lag(self, fs) -> None:
        """First-order lag on the waked power signal (after the sensors are current)."""
        if self._attached_fs is None:
            raise RuntimeError(
                "ParamPyWakeWindTurbines.step_power_lag called before attach_to(fs); "
                "make_dwm must attach the turbines to the flow simulation"
            )
        p_qs = self._static_power(include_wakes=True)
        if self._power_state is None:
            self._power_state = np.array(p_qs, dtype=float)
        else:
            a = self.params.a_power
            self._power_state = self._power_state + a * (p_qs - self._power_state)

    # ------------------------------------------------------------ physics --
    def _yaw_flow(self, idx=slice(None)):
        """The realised yaw the FLOW sees: the commanded frame minus ``yaw_bias``."""
        return self._yaw_act[idx] - self.params.values["yaw_bias"][idx]

    def yaw_tilt(self, idx=slice(None)):
        """The yaw the wake particles are emitted with: flow-relative (bias removed)."""
        return np.array([self._yaw_flow(idx), self.sensors.tilt[idx]])

    def get_kwargs(self, idx=slice(None)):
        """PyWake's power / Ct inputs, with the flow-relative yaw."""
        kw = super().get_kwargs(idx)
        if "yaw" in kw:
            kw["yaw"] = self._yaw_flow(idx)
        return kw

    def _static_power(self, include_wakes=True, idx=slice(None)):
        p = super().power(include_wakes=include_wakes, idx=idx)
        yaw = self._yaw_flow(idx)
        cos = np.maximum(np.cos(np.deg2rad(yaw)), 0.0)
        # 0.0 ** 0.0 == 1.0 and cp_gain 1.0: the defaults are a bitwise identity.
        return (
            p
            * self.params.values["cp_gain"][idx]
            * cos ** self.params.values["yaw_exp"][idx]
        )

    def power(self, include_wakes=True, idx=slice(None)):
        live = self._static_power(include_wakes=include_wakes, idx=idx)
        if not include_wakes or self._power_state is None:
            # No-wake reference (Wake_recovery headroom) stays quasi-static.
            return live
        return np.where(self.params.a_power[idx] >= 1.0, live, self._power_state[idx])

    def ct(self, idx=slice(None)):
        return super().ct(idx=idx) * self.params.values["ct_gain"][idx]
