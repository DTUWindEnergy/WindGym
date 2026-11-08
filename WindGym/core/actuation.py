"""Pure actuation helpers: per-sim-step yaw and derate command updates.

``WindFarmEnv._adjust_yaws`` / ``_apply_derating`` slice the agent action,
call these functions and write the results to the turbine sensors. Keeping
the arithmetic here makes the command semantics testable without a flow
simulation. The numerics (dtype casts, clip order) are kept exactly as they
were inside the env so seeded episodes stay byte-identical.
"""

from __future__ import annotations

import numpy as np


def step_yaw_command(
    method: str,
    action,
    yaw_command: np.ndarray,
    action_remaining,
    *,
    yaw_step_sim: float,
    yaw_min: float,
    yaw_max: float,
    rate_limit: bool,
    deadband: float = 0.0,
) -> tuple:
    """Advance the yaw command by one sim step.

    Heavily inspired from https://github.com/AlgTUDelft/wind-farm-env.

    Args:
        method: ``ActionMethod`` ("yaw" | "wind" | "absolute").
        action: Per-turbine yaw action in [-1, 1] (only used by "wind").
        yaw_command: The current commanded setpoint (authoritative; never
            read back from the turbines, whose HAWC2 getter lags).
        action_remaining: Remaining yaw budget for this env step ("yaw"
            method only; anything else is passed back untouched).
        yaw_step_sim: Max yaw change per sim step (deg).
        yaw_min / yaw_max: Yaw bounds (deg).
        rate_limit: Rate-limit the "wind" method toward the setpoint
            (``HTC_path is None`` in the env: HAWC2 has inertia anyway).
        deadband: "wind" method only: a turbine whose target is less than
            ``deadband`` deg from its current command keeps the command (no
            yaw motion for small target changes). 0 disables it.

    Returns:
        ``(new_command, new_action_remaining)``
    """
    if method == "yaw":
        # The new yaw angles are the old yaw angles + the action, scaled with the yaw_step
        # 0 action means no change
        # the new yaw angles are the old yaw angles + the action, scaled with the yaw_step

        # This is how much the yaw can change pr sim step
        yaw_change = np.clip(
            action_remaining,
            -yaw_step_sim,
            yaw_step_sim,
            dtype=np.float32,
        )

        # Accumulate on our own command (clipped to bounds), then write it once.
        # Never read windTurbines.yaw back here: for HAWC2 the getter returns the
        # lagging physical bearing, so a read-modify-write erases the command.
        yaw_command = np.clip(yaw_command + yaw_change, yaw_min, yaw_max)

        action_remaining = action_remaining - yaw_change
        return yaw_command, action_remaining

    elif method == "wind":
        # The new yaw angles are the action, scaled to be between the min and max yaw angles
        # 0 action means to move to 0 yaw angle, and 1 action means to move to the max yaw angle
        new_yaws = (action + 1.0) / 2.0 * (yaw_max - yaw_min) + yaw_min
        if deadband > 0:
            # Small target changes are ignored: hold the current command.
            new_yaws = np.where(
                np.abs(new_yaws - yaw_command) < deadband, yaw_command, new_yaws
            )

        if rate_limit:  # This clip is only usefull for the pywake turbine model, as the hawc2 model has inertia anyways
            # Rate-limit relative to our own command, not the (physical) readback.
            hi = yaw_command + yaw_step_sim
            lo = yaw_command - yaw_step_sim

            # The new yaw angles are the new yaw angles, but clipped to be between the yaw_max and yaw_min
            yaw_command = np.clip(np.clip(new_yaws, lo, hi), yaw_min, yaw_max)

        else:
            # The new yaw angles are the new yaw angles, but clipped to be between the yaw_min and yaw_max
            yaw_command = np.clip(new_yaws, yaw_min, yaw_max)

        return yaw_command, action_remaining

    elif method == "absolute":
        raise NotImplementedError("The absolute method is not implemented yet")

    else:
        raise ValueError("The ActionMethod must be yaw, wind or absolute")


def compute_derate(
    derate_raw,
    *,
    method: str,
    step_base,
    step_env: float,
    derate_min: float,
    derate_max: float,
    reference: str,
    rated_power: float,
    current_powers,
    current_derate,
    step_sim,
    hawc2: bool,
) -> tuple:
    """Turn the raw per-turbine derate action into (command, applied derate).

    ``method="absolute"``: each value in [-1, 1] is affine-mapped to a
    setpoint in [derate_min, derate_max].
    ``method="step"``: each value in [-1, 1] is a delta of at most
    ``step_env`` per env step, added to ``step_base`` (the command at
    env-step start).

    ``reference="rated"`` reinterprets the commanded fraction as a fraction
    of ``rated_power`` (an absolute cap) and converts it to the
    available-power fraction the turbine model expects; commands above
    locally available power apply no derating. HAWC2 turbines
    (``hawc2=True``) skip that conversion: the DTUWEC controller applies the
    rated-power cap natively, so the command passes straight through.
    ``reference="setpoint"`` also passes the command through (fraction of
    rated): the env turns it into the turbine model's absolute ``power_cap``
    and the model derives the effective derate itself.

    If ``step_sim`` is set, the applied derate slews toward the setpoint by
    at most ``step_sim`` per sim substep.

    Returns:
        ``(command, applied)`` both float64 arrays.
    """
    # float64 so the derate_step_env/derate_step_sim bounds hold exactly
    # (agent actions arrive as float32)
    derate_raw = np.asarray(derate_raw, dtype=np.float64)

    if method == "step":
        delta = np.clip(derate_raw, -1.0, 1.0) * step_env
        cmd = np.clip(step_base + delta, derate_min, derate_max).astype(np.float64)
    else:
        # Affine map [-1, 1] → [derate_min, derate_max] so the full action
        # range is useful even when derate_max < 1 (no saturated dead zone).
        frac = np.clip((derate_raw + 1.0) / 2.0, 0.0, 1.0)
        cmd = (derate_min + frac * (derate_max - derate_min)).astype(np.float64)

    if reference == "rated" and not hawc2:
        # cmd is a fraction of rated power → absolute target. Convert to
        # the equivalent available-power fraction using the invariant
        # P = (1 - d) * P_avail, so P_avail = current_power / (1 - d).
        # A target above available power clips to d = 0 (dead zone).
        p_target = (1.0 - cmd) * rated_power
        p_avail = current_powers / np.maximum(1.0 - current_derate, 1e-6)
        derate = np.clip(1.0 - p_target / np.maximum(p_avail, 1e-6), 0.0, derate_max)
    else:
        derate = cmd

    if step_sim is not None:
        prev = np.asarray(current_derate, dtype=np.float64)
        derate = np.clip(derate, prev - step_sim, prev + step_sim)

    return cmd, derate


def shares_from_action(derate_raw) -> np.ndarray:
    """Normalised deficit shares from the derate part of the action (``derate_method="share"``).

    ``u = (clip(a, -1, 1) + 1) / 2``, ``w = u / sum(u)``; equal shares when
    ``sum(u) < 1e-6`` (every action at -1).
    """
    u = (np.clip(np.asarray(derate_raw, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0
    total = u.sum()
    if total < 1e-6:
        return np.full(u.shape, 1.0 / u.size)
    return u / total


def allocate_deficit(p_hat, shares, deficit: float, floor_frac: float):
    """Per-turbine power reductions that sum to the farm deficit (water-filling).

    The inner loop of ``derate_method="share"``: turbine ``i`` gives up
    ``w_i * deficit`` of its available power ``p_hat_i``, at most
    ``(1 - floor_frac) * p_hat_i`` (the turbine model cannot deliver less than
    ``floor_frac * p_hat_i``). The part a saturated turbine cannot give goes to
    the unsaturated ones in proportion to their shares; once every turbine with
    a positive share is saturated, the rest goes to the zero-share turbines in
    proportion to their remaining headroom, so the deficit is met whenever the
    farm can meet it. A deficit beyond the farm's total headroom saturates
    every turbine.

    Args:
        p_hat: Available power per turbine (W).
        shares: Non-negative shares (normalised here).
        deficit: ``sum(p_hat) - P_ref`` (W).
        floor_frac: Lowest deliverable fraction of the available power.

    Returns:
        Reductions (W) per turbine, or ``None`` when ``deficit <= 0`` (no caps).
    """
    if deficit <= 0:
        return None
    p_hat = np.maximum(np.asarray(p_hat, dtype=np.float64), 0.0)
    w = np.asarray(shares, dtype=np.float64)
    headroom = (1.0 - floor_frac) * p_hat
    red = np.zeros_like(p_hat)
    remaining = float(deficit)
    tol = 1e-9 * max(float(p_hat.sum()), 1.0)
    for _ in range(2 * p_hat.size + 1):
        free = red < headroom - tol
        if remaining <= tol or not free.any():
            break
        weight = np.where(free, w, 0.0)
        if weight.sum() <= 0:  # only zero-share turbines left: by headroom
            weight = np.where(free, headroom - red, 0.0)
        new = np.minimum(red + remaining * weight / weight.sum(), headroom)
        remaining -= float((new - red).sum())
        red = new
    return red
