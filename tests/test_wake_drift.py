"""dwm_params wake_drift_deg: a mean lateral drift of every wake."""

import numpy as np

from WindGym.core.particle_motion import ScaledHillVortexParticleMotion


def test_drift_is_the_tangent():
    m = ScaledHillVortexParticleMotion(c=0.14, drift_deg=0.5)
    assert np.isclose(m.drift, np.tan(np.deg2rad(0.5)))
    assert ScaledHillVortexParticleMotion(c=0.14).drift == 0.0


def test_zero_and_negative_values_validate():
    from WindGym.core.dwm_defaults import validate_dwm_params
    validate_dwm_params({"yaw_ct_exp": 0.0, "yaw_width_exp": 0.0, "meandering_radius_scale": 0.0,
                         "wake_drift_deg": 0.0})
    validate_dwm_params({"wake_drift_deg": -0.5})
    import pytest
    with pytest.raises(ValueError):
        validate_dwm_params({"yaw_ct_exp": -1.0})
    with pytest.raises(ValueError):
        validate_dwm_params({"gauss_k": 0.0})
