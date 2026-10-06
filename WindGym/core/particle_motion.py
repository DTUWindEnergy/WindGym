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
    """``HillVortexParticleMotion`` with the self-induction coefficient ``c`` exposed."""

    def __init__(self, c: float = DEFLECTION_C, **kwargs):
        HillVortexParticleMotion.__init__(self, **kwargs)
        self.c = float(c)

    def __call__(self, position_xip, velocity_uip, wt_idx):
        position, new_velocity_uip = ParticleMotionModel.__call__(self, position_xip, velocity_uip, wt_idx=wt_idx)

        fs = self.flowSimulation
        x_p = position_xip[0, wt_idx]
        yaw_abs_ip, tilt_ip = np.moveaxis(fs.windTurbinesParticles[wt_idx].get_yaw_tilt_abs(), 0, 1)
        m = [y != None for y in yaw_abs_ip]  # noqa: E711
        theta_yaw_rel_im = [np.deg2rad(yaw[yaw != None].astype(float) - self.flowSimulation.wind_direction)  # noqa: E711
                            for yaw in yaw_abs_ip]
        theta_tilt_im = [np.deg2rad(tilt[tilt != None].astype(float)) for tilt in tilt_ip]  # noqa: E711
        delta_U_iu = fs.windTurbinesParticles[wt_idx].deficit_norm_magnitude(list(x_p), list(m))

        for iwt, i in enumerate(wt_idx):
            y_m, t_m = [np.atleast_1d(v[i]) for v in [theta_yaw_rel_im, theta_tilt_im]]
            # ONLY CHANGE FROM UPSTREAM: `.4` -> `self.c`
            self_induc_um = delta_U_iu[iwt] * self.c * [-np.cos(y_m) * np.cos(t_m),
                                                        np.cos(t_m) * np.sin(y_m),
                                                        np.sin(t_m)]
            position[:, i, m[iwt]] = position_xip[:, i, m[iwt]] + \
                (new_velocity_uip[:, i, m[iwt]] + self_induc_um) * self.dt
        return position, new_velocity_uip
