"""Hill-vortex particle motion with the wake-deflection strength exposed.

dynamiks hardcodes the self-induction coefficient ``0.4`` inside
``HillVortexParticleMotion.__call__``, with no constructor argument. That
constant sets how hard a yawed rotor pushes its own wake sideways, and so sets
the whole wake-steering gain. ``ScaledHillVortexParticleMotion`` makes it a
parameter; everything else in ``__call__`` is copied verbatim from the pinned
dynamiks, so at ``c = DEFLECTION_C`` it is identical to upstream
(``tests/test_particle_motion.py`` asserts that against a live simulation).

If dynamiks' ``HillVortexParticleMotion.__call__`` changes, re-copy it here.
"""

from __future__ import annotations

import numpy as np
from dynamiks.dwm.particle_motion_models import (
    HillVortexParticleMotion,
    ParticleMotionModel,
)

DEFLECTION_C = 0.4  # dynamiks' hardcoded value


class ScaledHillVortexParticleMotion(HillVortexParticleMotion):
    """``HillVortexParticleMotion`` with the self-induction coefficient ``c`` exposed.

    ``c_pos`` (default ``None`` = ``c``) is the coefficient for particles emitted
    at a POSITIVE flow-relative yaw (the turbines' ``yaw`` sensor, the sign the
    campaigns use); ``c`` then serves negative yaw. A yawed wake's deflection is
    not symmetric in the yaw sign (the wake's rotation adds to or opposes the
    yaw-induced vortex pair), and the G1 data show it: a split c is its
    first-order representation.
    """

    def __init__(self, c: float = DEFLECTION_C, c_pos: float | None = None,
                 drift_deg: float = 0.0, **kwargs):
        HillVortexParticleMotion.__init__(self, **kwargs)
        self.c = float(c)
        self.c_pos = self.c if c_pos is None else float(c_pos)
        # A mean lateral drift of every wake: the particles move sideways at
        # tan(drift) x their streamwise speed, toward +y (the flow frame's
        # lateral axis; wind_direction 270 = flow along +x). The TUM tunnel's
        # unyawed wake at the 5 D column's lane drifts ~0.5 deg toward the
        # centreline (SingleWake); the inflow map carries no mean lateral
        # velocity (the CTA mapping removes it), so the env's wakes ran
        # straight. Boundary particles (yaw None) do not move.
        self.drift = float(np.tan(np.deg2rad(float(drift_deg))))

    def __call__(self, position_xip, velocity_uip, wt_idx):
        position, new_velocity_uip = ParticleMotionModel.__call__(
            self, position_xip, velocity_uip, wt_idx=wt_idx
        )

        fs = self.flowSimulation
        x_p = position_xip[0, wt_idx]
        yaw_abs_ip, tilt_ip = np.moveaxis(
            fs.windTurbinesParticles[wt_idx].get_yaw_tilt_abs(), 0, 1
        )
        m = [y != None for y in yaw_abs_ip]  # noqa: E711
        theta_yaw_rel_im = [
            np.deg2rad(
                yaw[yaw != None].astype(float) - self.flowSimulation.wind_direction  # noqa: E711
            )
            for yaw in yaw_abs_ip
        ]
        theta_tilt_im = [
            np.deg2rad(tilt[tilt != None].astype(float))  # noqa: E711
            for tilt in tilt_ip
        ]
        delta_U_iu = fs.windTurbinesParticles[wt_idx].deficit_norm_magnitude(
            list(x_p), list(m)
        )

        for iwt, i in enumerate(wt_idx):
            y_m, t_m = [np.atleast_1d(v[i]) for v in [theta_yaw_rel_im, theta_tilt_im]]
            # ONLY CHANGE FROM UPSTREAM: `.4` -> the coefficient, per particle by
            # the sign of its yaw. theta = yaw_abs - wind_direction = -yaw_rel
            # (DWMFlowSimulation.update_particles), so theta < 0 is positive
            # flow-relative yaw.
            c_m = np.where(y_m < 0, self.c_pos, self.c)
            self_induc_um = (
                delta_U_iu[iwt]
                * c_m
                * [-np.cos(y_m) * np.cos(t_m), np.cos(t_m) * np.sin(y_m), np.sin(t_m)]
            )
            position[:, i, m[iwt]] = (
                position_xip[:, i, m[iwt]]
                + (new_velocity_uip[:, i, m[iwt]] + self_induc_um) * self.dt
            )
            if self.drift:
                position[1, i, m[iwt]] += self.drift * new_velocity_uip[0, i, m[iwt]] * self.dt
        return position, new_velocity_uip
