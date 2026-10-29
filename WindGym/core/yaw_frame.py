"""Yaw-frame conversion between a controller's wind-direction belief and the
simulator's yaw frame.

Yaw in WindGym is wind-relative and the nacelle is fixed in the ground frame:
``WindFarmEnv._shift_yaw_frame`` adds every change of the simulator's wind
direction to the yaw command, so "yaw 0" always means "aligned with the wind
the SIMULATOR knows". A deployed controller does not know that wind; it
believes the wind is at ``phi_ctrl`` (a sensor-derived estimate, or a fixed
nominal direction) and reads / commands yaw in THAT frame. The two frames
differ by the signed angle between the two directions:

    yaw_ctrl = yaw_sim + wrap180(phi_ctrl - wd_sim)
    yaw_sim  = yaw_ctrl - wrap180(phi_ctrl - wd_sim)

The sign is the one ``pywake_lut.LutYawController.target_yaw`` has used since
Stage 9 (target_grid = gamma(phi) - wrap180(phi - wd_frame)) and matches
``_shift_yaw_frame`` (a +5 deg true-wd change with a fixed nacelle reads +5
deg in the new frame). Three callers share this module so they cannot drift:
``WindFarmEnv`` (``yaw_frame_source="est"``, training and the DWM harness),
``LESRL/harness_frame.py`` (re-export) and the LES deployment script, whose
simulator frame is the fixed 270-deg grid.
"""
from __future__ import annotations

import numpy as np


def wrap180(a):
    """Wrap an angle (deg) into [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def frame_offset(phi_ctrl: float, wd_sim: float) -> float:
    """Signed angle (deg) to ADD to a simulator-frame yaw to express it in the
    controller frame: wrap180(phi_ctrl - wd_sim)."""
    return float(wrap180(float(phi_ctrl) - float(wd_sim)))


def sim_to_ctrl(yaw_sim, phi_ctrl: float, wd_sim: float):
    """Simulator-frame yaw -> controller-frame yaw (what the agent observes)."""
    off = frame_offset(phi_ctrl, wd_sim)
    if np.ndim(yaw_sim) == 0:
        return float(yaw_sim) + off
    return np.asarray(yaw_sim, dtype=float) + off


def ctrl_to_sim(yaw_ctrl, phi_ctrl: float, wd_sim: float):
    """Controller-frame yaw target -> simulator-frame yaw (what the actuator
    gets). On the LES grid wd_sim = 270: target_grid = target - wrap180(phi - 270)."""
    off = frame_offset(phi_ctrl, wd_sim)
    if np.ndim(yaw_ctrl) == 0:
        return float(yaw_ctrl) - off
    return np.asarray(yaw_ctrl, dtype=float) - off


def slew_toward(phi: float, target: float, max_step: float) -> float:
    """Move the frame direction ``phi`` toward ``target`` by at most
    ``max_step`` deg (the shortest way round), unwrapped so consecutive values
    stay continuous. A non-finite target (estimator not warm yet) holds."""
    if max_step < 0:
        raise ValueError(f"max_step must be >= 0, got {max_step}")
    phi = float(phi)
    if not np.isfinite(target):
        return phi
    d = float(wrap180(float(target) - phi))
    return phi + float(np.clip(d, -max_step, max_step))


def residual_target(a, prior_target, delta: float, yaw_min: float, yaw_max: float):
    """Residual-on-prior yaw target: ``prior_target + a * delta`` clipped to the
    yaw band (plan 2026-10-29 WP3.5, the lut3 action prior). ``a`` in [-1, 1]
    per turbine; ``prior_target`` in the agent's frame."""
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    return np.clip(np.asarray(prior_target, dtype=float) + a * float(delta), yaw_min, yaw_max)


def wind_action_from_target(target, yaw_min: float, yaw_max: float):
    """Inverse of the "wind" ActionMethod decode (core.actuation.step_yaw_command:
    target = (a + 1) / 2 * (yaw_max - yaw_min) + yaw_min), clipped to [-1, 1]."""
    t = np.clip(np.asarray(target, dtype=float), yaw_min, yaw_max)
    return np.clip((t - yaw_min) / (yaw_max - yaw_min) * 2.0 - 1.0, -1.0, 1.0)
