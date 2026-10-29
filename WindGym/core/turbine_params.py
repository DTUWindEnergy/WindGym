"""Per-turbine turbine-model parameters for domain randomization (sim2real).

The DWM env's PyWake turbine is a quasi-static Cp/Ct curve with an instant,
exactly observed yaw. The deployment target (HAWC2 aeroelastic turbines) has a
yaw servo, rotor inertia and a controller, so power, thrust and the yaw
response all differ. ``TurbineParams`` is the per-episode, per-turbine vector
that ``WindGym.backend.param_turbines.ParamPyWakeWindTurbines`` applies on top
of the nominal turbine:

``yaw_exp``
    extra cosine exponent on power: ``P *= max(cos(yaw), 0) ** yaw_exp``
    (0 = the turbine's own yaw loss only).
``cp_gain`` / ``ct_gain``
    multiplicative gains on power and thrust coefficient. ``ct_gain`` reaches
    the wake strength through ``rotor_avg_induction = ct2a_madsen(ct)``;
    Ct values above ~1 are unphysical for ``ct2a_madsen`` so the upper bound
    is kept at 1.6 (the nominal IEA-22 Ct peaks near 0.8).
``tau_yaw`` / ``delay_yaw``
    first-order actuator lag (s) and dead time (s) between the yaw command and
    the realised yaw. The dead time is rounded to whole simulation steps;
    a lag below ~dt_sim/2 is invisible at the env's time resolution.
``tau_power``
    first-order lag (s) representing rotor inertia + controller settling. In
    fraction mode (``ParamPyWakeWindTurbines``) it lags the whole waked power
    signal; in setpoint mode (``SetpointPyWakeWindTurbines``) it lags the
    wind-driven *available* power, while the power cap acts at once. Thrust is
    NOT lagged by ``tau_power`` (documented trade-off).
``tau_ct``
    setpoint mode only: first-order lag (s) of the effective derate seen by
    the Ct table after a cap change (the rotor has to move to the new
    operating point). A no-op in fraction mode; the env refuses it there.

Bounds are validation limits (typos, unit mistakes), not priors: the training
ranges live in the trainer's ``--turb_dr`` spec. Pure numpy; no dynamiks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TurbineParamSpec:
    default: float
    lo: float
    hi: float
    doc: str


# Ordered: this is the order of the parameter VECTOR for RMA / estimation work.
TURBINE_PARAM_SPECS: dict[str, TurbineParamSpec] = {
    "yaw_exp": TurbineParamSpec(0.0, -3.0, 3.0, "extra cos exponent on power"),
    "cp_gain": TurbineParamSpec(1.0, 0.2, 2.0, "power gain"),
    "ct_gain": TurbineParamSpec(1.0, 0.2, 1.6, "thrust-coefficient gain"),
    "tau_yaw": TurbineParamSpec(0.0, 0.0, 600.0, "yaw actuator lag (s)"),
    "delay_yaw": TurbineParamSpec(0.0, 0.0, 300.0, "yaw dead time (s)"),
    "tau_power": TurbineParamSpec(0.0, 0.0, 600.0, "power response lag (s)"),
    "tau_ct": TurbineParamSpec(
        0.0, 0.0, 600.0, "thrust-coefficient lag after a cap change (s, setpoint mode)"
    ),
}
TURBINE_PARAM_KEYS = frozenset(TURBINE_PARAM_SPECS)


def _first_order_coeff(tau: np.ndarray, dt: float) -> np.ndarray:
    """``1 - exp(-dt/tau)``, exactly 1.0 where tau == 0 (no filtering)."""
    a = np.ones_like(tau)
    pos = tau > 0
    a[pos] = 1.0 - np.exp(-dt / tau[pos])
    return a


@dataclass(frozen=True)
class TurbineParams:
    """Validated per-turbine parameter arrays plus the derived filter constants.

    ``values[key]`` is a float64 array of shape ``(n_turb,)`` for every key in
    ``TURBINE_PARAM_SPECS``. ``a_yaw`` / ``a_power`` / ``a_ct`` are the first-order filter
    coefficients ``1 - exp(-dt_sim/tau)`` (exactly 1.0 for tau 0), ``k_delay``
    the dead time in whole sim steps. ``is_nominal`` is True when every value
    equals its default, in which case the env uses the plain dynamiks turbine
    class and the nominal code path is bit-identical to pre-DR WindGym.
    """

    values: Mapping[str, np.ndarray]
    dt_sim: float
    a_yaw: np.ndarray
    a_power: np.ndarray
    a_ct: np.ndarray
    k_delay: np.ndarray
    is_nominal: bool

    @property
    def n_turb(self) -> int:
        return int(self.values["cp_gain"].shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        """Copies of the parameter arrays, keyed by name (env turbine order)."""
        return {k: np.array(v, copy=True) for k, v in self.values.items()}


def normalize_turbine_params(params, n_turb: int, dt_sim: float) -> TurbineParams:
    """Validate ``params`` (name -> scalar or array) into a ``TurbineParams``.

    * scalars broadcast to ``(n_turb,)``;
    * arrays with ``len >= n_turb`` are sliced to ``[:n_turb]`` (MultiLayoutEnv
      samplers draw ``max_turbines`` values and the smaller layouts use the
      leading ones);
    * ``len < n_turb``, ``ndim > 1``, unknown keys, non-finite values and
      values outside the spec bounds raise ``ValueError`` naming the key;
    * missing keys take their defaults.
    """
    params = dict(params or {})
    unknown = set(params) - TURBINE_PARAM_KEYS
    if unknown:
        raise ValueError(
            f"Unknown turbine param(s) {sorted(unknown)}; "
            f"allowed: {list(TURBINE_PARAM_SPECS)}"
        )
    n_turb = int(n_turb)
    dt_sim = float(dt_sim)
    if n_turb < 1 or dt_sim <= 0:
        raise ValueError(f"n_turb must be >= 1 and dt_sim > 0 (got {n_turb}, {dt_sim})")

    values: dict[str, np.ndarray] = {}
    nominal = True
    for key, spec in TURBINE_PARAM_SPECS.items():
        if key not in params:
            values[key] = np.full(n_turb, spec.default, dtype=np.float64)
            continue
        arr = np.asarray(params[key], dtype=np.float64)
        if arr.ndim > 1:
            raise ValueError(f"turbine param {key!r}: ndim {arr.ndim} > 1 (shape {arr.shape})")
        if arr.ndim == 0:
            arr = np.full(n_turb, float(arr), dtype=np.float64)
        elif arr.shape[0] < n_turb:
            raise ValueError(
                f"turbine param {key!r}: len {arr.shape[0]} < n_turb {n_turb}"
            )
        else:
            arr = np.array(arr[:n_turb], dtype=np.float64, copy=True)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"turbine param {key!r} must be finite, got {arr}")
        if np.any(arr < spec.lo) or np.any(arr > spec.hi):
            raise ValueError(
                f"turbine param {key!r} outside [{spec.lo}, {spec.hi}]: {arr}"
            )
        arr.flags.writeable = False
        values[key] = arr
        if np.any(arr != spec.default):
            nominal = False

    for v in values.values():
        v.flags.writeable = False
    return TurbineParams(
        values=values,
        dt_sim=dt_sim,
        a_yaw=_first_order_coeff(values["tau_yaw"], dt_sim),
        a_power=_first_order_coeff(values["tau_power"], dt_sim),
        a_ct=_first_order_coeff(values["tau_ct"], dt_sim),
        k_delay=np.rint(values["delay_yaw"] / dt_sim).astype(int),
        is_nominal=nominal,
    )
