"""PyWake turbines with an absolute per-turbine power cap on an inertia-filtered
available power ("setpoint mode", level B of the turbine-model ladder).

``SetpointPyWakeWindTurbines`` subclasses ``ParamPyWakeWindTurbines`` (yaw
actuator, Cp/Ct gains, ``attach_to``, ``shift_yaw_frame``, ``reset_yaw_state``
are inherited unchanged) and replaces its power lag by the following model,
evaluated per turbine once per sim step of length ``dt`` (``params.dt_sim``)::

    P_qs    = P_tab(U, yaw, d=0) * gains            quasi-steady available power
    P_avail = P_avail + a_P * (P_qs - P_avail)      a_P = 1 - exp(-dt / tau_power)
    P       = min(P_cap, P_avail)                   delivered power; the cap acts at once
    d_cmd   = clip(1 - P / P_avail, 0, d_table_max) effective derate (0 where P_avail == 0)
    d_eff   = d_eff + a_ct * (d_cmd - d_eff)        a_ct = 1 - exp(-dt / tau_ct)
    Ct      = Ct_tab(U, yaw, d_eff) * ct_gain       slow after a cap change, instant in U

``U`` is the waked rotor-average wind speed, ``yaw`` the realised yaw, ``P_cap``
the per-turbine cap in W (``+inf`` = uncapped). The derate axis of the table is
defined relative to the *yawed* available power (``P_tab(U, yaw, d) =
(1 - d) P_tab(U, yaw, 0)`` for the min-Ct DTU10MW surrogate), so ``d_cmd`` is
exactly the table coordinate that delivers ``P``.

Sensors (all per turbine):

``power_cap``
    plain storage, settable (``wts.power_cap = ...``), W, init ``+inf``.
``available_power``
    read-only getter: ``P_avail`` (filtered; quasi-steady before the first step).
``derate``
    read-only getter: ``d_eff``. dynamiks' ``get_kwargs`` feeds it to the
    power/Ct table (``ct()`` -> ``rotor_avg_induction`` -> wake particles), so
    the wake sees the lagged operating point. Assigning it raises: in
    setpoint mode the derate is derived from the cap.
``power``
    (inherited) ``wt.power()`` = ``min(P_cap, P_avail)``.

Handler order per sim step (``FlowSimulation.step_handlers`` =
``[step_actuators, update_particles] + site + [step_power_lag, sensors]``):
``step_actuators`` (realised yaw) -> ``update_particles`` (emits this step's
particles; reads ``sensors.derate`` = ``d_eff(t-1)``) -> site handlers ->
``step_power_lag`` (this model: refreshes ``P_avail``, ``d_eff``) -> the
``sensors`` dumper (reads ``power``, ``ct``, ``derate`` at ``t``).

Conventions: ``np.where(a >= 1, ...)`` keeps ``tau = 0`` bitwise identical to the
unfiltered value; the first handler call initialises both states at their
quasi-steady values (no start-up transient); before the first call (dynamiks
calls ``ct()`` in ``DWMFlowSimulation.__init__``, before ``attach_to``)
``ct()``/``power()``/``available_power`` use the quasi-steady values.
``power(include_wakes=False)`` keeps the Param class's quasi-static path (the
table at ``d_eff``; 0 on an uncapped farm, where the env's no-wake reference
uses it). With ``tau_power = tau_ct = 0`` the model equals the env's fraction
mode ``derate_reference="rated"`` without that path's one-step-old conversion.

Accepted approximations:

* particles emitted in step t see ``d_eff(t - 1)`` (one-substep lag, the same
  the Param class accepts for the yaw frame);
* constant ``tau_power`` / ``tau_ct`` instead of state-dependent ones (a real
  rotor at minimum speed, ws <= ~8 m/s for the DTU10MW, smooths less);
  ``tau(U, d)`` from the full table's rotor speed is a cheap later refinement;
* Ct from the table at the current ``U`` assumes the rotor speed has followed
  ``U`` (small Ct error during gusts);
* ``cp_gain > 1`` with ``P_cap = P_rated`` caps the turbine at nominal rated
  power while an uncapped baseline farm is not capped.

Level C (a 1-DOF rotor + controller on Cp/Ct(lambda, beta, yaw) surfaces)
replaces the two filter states by (omega, beta) dynamics behind the same
sensors and env interface.
"""

from __future__ import annotations

import numpy as np

from WindGym.core.turbine_params import TurbineParams

from .param_turbines import ParamPyWakeWindTurbines

READ_ONLY_DERATE_MSG = "derate is derived from power_cap in setpoint mode"


def _get_available_power(wts):
    return wts.available_power


def _get_effective_derate(wts):
    return wts.effective_derate


def _set_derate_forbidden(wts, value):
    raise ValueError(READ_ONLY_DERATE_MSG)


def _table_derate_max(windTurbine) -> float:
    """Upper end of the ``derate`` axis of the turbine's table (1.0 if none)."""
    pctf = windTurbine.powerCtFunction
    keys = list(getattr(pctf, "input_keys", []) or [])
    if "derate" not in keys:
        return 1.0
    try:
        grid = pctf.interp[0].x[keys.index("derate")]
        return float(np.max(grid))
    except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
        return 1.0


class SetpointPyWakeWindTurbines(ParamPyWakeWindTurbines):
    """``ParamPyWakeWindTurbines`` driven by an absolute power cap (see module doc)."""

    def __init__(self, x, y, windTurbine, params: TurbineParams, **kwargs):
        super().__init__(x=x, y=y, windTurbine=windTurbine, params=params, **kwargs)
        n = self.N
        self.d_table_max = _table_derate_max(self.windTurbine)
        self._avail_state = None  # filtered available power (W); None until the first step
        self._derate_state = None  # lagged effective derate; None until the first step
        # Settable cap (plain storage, exposed as wts.power_cap).
        self.add_sensor("power_cap", expose=True)
        self.sensors.power_cap = np.full(n, np.inf)
        # Read-only derived sensors. 'derate' is what get_kwargs feeds to the table.
        self.add_sensor("available_power", getter=_get_available_power)
        self.add_sensor("derate", getter=_get_effective_derate, setter=_set_derate_forbidden)

    # ------------------------------------------------------------ states --
    def _available_qs(self):
        """Quasi-steady available power: the table at derate 0, gains applied."""
        return np.asarray(self._static_power(include_wakes=True, derate=0.0), dtype=float)

    def _derate_command(self, power, available):
        """Table derate that delivers ``power`` out of ``available`` (0 where idle)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(available > 0.0, 1.0 - power / available, 0.0)
        return np.clip(d, 0.0, self.d_table_max)

    @property
    def available_power(self) -> np.ndarray:
        """Copy of the (filtered) available power, W; quasi-steady before the first step."""
        if self._avail_state is None:
            return self._available_qs()
        return self._avail_state.copy()

    @property
    def effective_derate(self) -> np.ndarray:
        """Copy of the lagged effective derate seen by the Cp/Ct table."""
        if self._derate_state is None:
            avail = self._available_qs()
            return self._derate_command(np.minimum(self.power_cap, avail), avail)
        return self._derate_state.copy()

    # ------------------------------------------------------ step handler --
    def step_power_lag(self, fs) -> None:
        """Advance ``P_avail`` and ``d_eff`` by one sim step (see module doc)."""
        if self._attached_fs is None:
            raise RuntimeError(
                "SetpointPyWakeWindTurbines.step_power_lag called before attach_to(fs); "
                "make_dwm must attach the turbines to the flow simulation"
            )
        p_qs = self._available_qs()
        if self._avail_state is None:
            avail = np.array(p_qs, dtype=float)
        else:
            a = self.params.a_power
            avail = np.where(a >= 1.0, p_qs, self._avail_state + a * (p_qs - self._avail_state))
        self._avail_state = avail
        d_cmd = self._derate_command(np.minimum(self.power_cap, avail), avail)
        if self._derate_state is None:
            self._derate_state = np.array(d_cmd, dtype=float)
        else:
            a = self.params.a_ct
            self._derate_state = np.where(
                a >= 1.0, d_cmd, self._derate_state + a * (d_cmd - self._derate_state)
            )

    # ------------------------------------------------------------ physics --
    def power(self, include_wakes=True, idx=slice(None)):
        if not include_wakes:
            # No-wake reference stays quasi-static (table at d_eff, like the Param class).
            return self._static_power(include_wakes=False, idx=idx)
        # atleast_1d: an int idx gives shape (1,), as the py_wake table does upstream.
        return np.atleast_1d(np.minimum(self.power_cap[idx], self.available_power[idx]))

    def ct(self, idx=slice(None)):
        # get_kwargs reads sensors.derate = d_eff; ct_gain applied by the Param class.
        return super().ct(idx=idx)
