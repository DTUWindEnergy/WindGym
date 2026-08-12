"""Sensor-derived wind-direction estimator (WD-estimation ladder, Phase 6).

Replaces the privileged, noiseless, instantaneous ``env.wd`` scalar with an
estimate computed from the MEASURED per-turbine local wind directions the env
already collects every sim substep (``WindFarmEnv._take_measurements`` ->
``current_wd``) — zero extra flow queries.

Structure (fixed by the T1 offline probe, wd_estimation/t1_probe.py, which
evaluates exactly this filter on recorded traces so the winning tau/consensus
transfers verbatim):

  * per-turbine circular EWMA on unit vectors:
        z_i <- (1 - alpha) z_i + alpha * e^{i theta_i},
        alpha = 1 - exp(-dt_sim / tau)
    Unit-vector filtering is what makes the average circular-safe: filtering
    raw degrees would tear at the 0/360 wrap.
  * consensus across turbines: circular MEDIAN of arg(z_i) by default —
    robust to waked turbines whose local flow is deflected off the free
    stream — with "mean" and "front" (upwind-most turbine, argmin x) as
    T1-selectable options.
  * lazy warm-start: the filter state is initialized from the FIRST update's
    measurements. The env's burn-in holds the per-reset base_wd, so the
    filter starts converged and there is no cold-start transient.

SENSOR-MODEL ASSUMPTION: on dynamiks, ``current_wd`` embeds the prescribed
wd_slow frame exactly plus the resolved turbulence's local tilt; a physical
vane would add rotor distortion and sensor noise (see t1_probe.py docstring).
pywake's adapter hard-codes v = w = 0, so atan2(v, u) == 0 there — a measured
local wd DOES NOT EXIST on the pywake backend and the trainer hard-errors on
wd_source=est + backend=pywake.
"""
from __future__ import annotations

import numpy as np


class WdEstimator:
    """Per-turbine circular EWMA + cross-turbine consensus.

    Args:
        n_turb: number of turbines (state size).
        dt: seconds between ``update()`` calls (the env's dt_sim).
        tau: EWMA time constant in seconds. Larger = smoother but laggier;
            pick from the T1 probe against the T0 error budget.
        consensus: "median" (default, robust), "mean", or "front".
        x_pos: turbine x positions; required for consensus="front"
            (front = argmin x, the upwind-most turbine at westerly winds).
    """

    CONSENSUS = ("median", "mean", "front")

    def __init__(self, n_turb: int, dt: float, tau: float,
                 consensus: str = "median", x_pos=None):
        if tau <= 0:
            raise ValueError(f"wd_est_tau must be > 0, got {tau}")
        if consensus not in self.CONSENSUS:
            raise ValueError(f"wd_est_consensus must be one of "
                             f"{self.CONSENSUS}, got {consensus!r}")
        if consensus == "front" and x_pos is None:
            raise ValueError("consensus='front' needs x_pos")
        self.n_turb = int(n_turb)
        self.dt = float(dt)
        self.tau = float(tau)
        self.alpha = 1.0 - float(np.exp(-self.dt / self.tau))
        self.consensus = consensus
        self._front = int(np.argmin(x_pos)) if x_pos is not None else None
        self._z = None  # complex (n_turb,); None = lazy warm-start pending

    def reset(self) -> None:
        """Forget all state; the next update() warm-starts the filter."""
        self._z = None

    def update(self, wd_deg) -> float:
        """Fold one substep of measured per-turbine wd (deg) into the filter
        and return the consensus estimate (deg, [0, 360))."""
        wd = np.asarray(wd_deg, dtype=float)
        if wd.shape != (self.n_turb,):
            raise ValueError(f"expected ({self.n_turb},) measurements, "
                             f"got {wd.shape}")
        u = np.exp(1j * np.deg2rad(wd))
        if self._z is None:
            self._z = u.astype(complex)  # warm-start: converged at sample 1
        else:
            self._z = (1.0 - self.alpha) * self._z + self.alpha * u
        return self.estimate

    @property
    def per_turbine(self) -> np.ndarray:
        """(n_turb,) per-turbine filtered wd (deg). Warm-start pending -> nan."""
        if self._z is None:
            return np.full(self.n_turb, np.nan)
        return np.rad2deg(np.angle(self._z)) % 360.0

    @property
    def estimate(self) -> float:
        """Consensus wd estimate (deg, [0, 360))."""
        per = self.per_turbine
        if np.isnan(per).any():
            return float("nan")
        if self.consensus == "front":
            return float(per[self._front])
        if self.consensus == "mean":
            rad = np.deg2rad(per)
            return float(np.rad2deg(np.arctan2(
                np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
        # circular median: the sample minimizing summed abs circular distance
        # (exact O(n^2); n_turb <= ~25 in every campaign)
        d = np.abs((per[:, None] - per[None, :] + 180.0) % 360.0 - 180.0)
        return float(per[d.sum(axis=1).argmin()])
