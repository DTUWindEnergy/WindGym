"""dwm_params wake_drift_deg: a mean lateral drift of every wake."""

import numpy as np

from WindGym.core.particle_motion import ScaledHillVortexParticleMotion


def test_drift_is_the_tangent():
    m = ScaledHillVortexParticleMotion(c=0.14, drift_deg=0.5)
    assert np.isclose(m.drift, np.tan(np.deg2rad(0.5)))
    assert ScaledHillVortexParticleMotion(c=0.14).drift == 0.0
