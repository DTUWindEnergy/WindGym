"""LES-calibrated DWM setup (opt-in).

These values were calibrated by aligning DWM output against LES data; do not
change them casually. They are not WindGym's default. Opt in with::

    from WindGym.core.calibrated_values import LES_CALIBRATED
    env = WindFarmEnv(..., dwm_setup=LES_CALIBRATED)

or ``dwm_setup: les_calibrated`` in the env config. The calibration ran on the
dev dynamiks build: ``TISensor``, ``BranlardScaling`` and ``MixedSum`` only
exist there, so on PyPI dynamiks this setup raises an ImportError pointing to
WindGym's ``dev_dynamiks`` branch. The DWM speed-ups (env ``interpolation`` /
``lateral_cutoff``) are not part of the calibration, which ran pchip with no
cutoff.
"""

from __future__ import annotations

import dataclasses

from .dwm_defaults import DWMSetup, MannBoxSpec

# Mann turbulence box (turbtype MannGenerate / MannFixed).
#
# Under the non-DR path the raw box at αε=1 is renormalised by
# `tf.scale_TI(TI=self.ti, U=ws)` to the env's nominal TI, so alphaepsilon is
# effectively cosmetic *in that path*.
#
# Under domain randomization (Mann keys present in `dwm_params`), the
# turbulence manager skips `scale_TI` and αε directly controls the box's
# ambient TI (matching `calibration/simulator.py:_build_site`). At
# (L=29.4, Γ=3.9, WS=9), raw σ_u ≈ 3.63 → TI ≈ 0.40·√αε, so the calibrated
# αε ≈ 0.0056 corresponds to TI ≈ 3%. If you ever switch the default branch
# to skip `scale_TI` unconditionally, lower alphaepsilon accordingly.
#
# Box extent: 1024*3.2 = 3.28 km streamwise. Under Bounds.Repeat a long
# episode re-samples the box many times (a Stage-7 10,300 s episode at ws 9-11
# advects 90-110 km ~ 30 wraps). This is calibration-faithful (the LES/SBI
# calibration and the LESRL trainings ran the same spec) but it is a
# deliberate trade against the old wdest (4096, ...) anti-recycle box; under
# DR every episode regenerates a fresh box, so recycling is within-episode
# only.
LES_CALIBRATED_MANN = MannBoxSpec(
    L=29.4,
    gamma=3.9,
    alphaepsilon=1.0,
    nxyz=(1024, 256, 128),
    dxyz=(3.2, 3.2, 3.2),
)

LES_CALIBRATED = DWMSetup(
    # Ainslie closure
    viscosity_model="keck",
    k1=0.0914,  # keck ambient eddy-viscosity coefficient
    k2=0.0216,  # keck wake-shear eddy-viscosity coefficient
    ainslie_r_max=3,
    ainslie_n_r=52,
    ainslie_dx=0.1,
    projection="none",
    d_particle=0.48,
    # Particle motion: particles advect with their own speed and feel the
    # other wakes, rotor-averaged over a CGI(9) kernel; no temporal filter.
    particle_x_speed="particle",
    particle_cutoff_d=None,
    particle_include_wakes=True,
    particle_spatial_avg_n=9,
    # Turbines: CGI(21) inflow kernel + 600 s transport-speed TI sensor
    rotor_avg_n=21,
    ti_sensor_window_s=600,
    # Root-sum-square on the u-deficit, linear on v/w
    superposition="mixed",
    added_turbulence_scaling="branlard",
    mann=LES_CALIBRATED_MANN,
)

# DWM re-calibration v2 (LESRL calibration/RECAL_V2.md, Stage-3 MAP 3.8,
# confirmed in the surrogate and HAWC2 cells 3.9 / 3.11): the Ainslie wake-shear
# coefficient k2 and the Hill-vortex self-induction factor refit against the
# LES rollouts, d_particle fixed at 0.2 (unidentified, flat posterior). The
# inflow is the FITTED Mann box (``mann_fit.npz``: L 72.7 m, Gamma 2.0,
# ``ae_for_ti`` 0.0129 at the nominal TI), same grid as LES_CALIBRATED_MANN.
# One-line revert to the Keck inflow: ``mann=LES_CALIBRATED_MANN``.
LES_V2_MANN = MannBoxSpec(
    L=72.7,
    gamma=2.0,
    alphaepsilon=0.0129,
    nxyz=(1024, 256, 128),
    dxyz=(3.2, 3.2, 3.2),
)

LES_V2 = dataclasses.replace(
    LES_CALIBRATED,
    k2=0.020,
    d_particle=0.2,
    hill_vortex_factor=0.30,
    mann=LES_V2_MANN,
)

# Names accepted by ``dwm_setup:`` in the env config / ``dwm_setup=`` strings.
DWM_SETUPS = {
    "default": DWMSetup(),
    "les_calibrated": LES_CALIBRATED,
    "les_v2": LES_V2,
}


def resolve_dwm_setup(value) -> DWMSetup:
    """Map ``None`` / a registered name / a ``DWMSetup`` to a ``DWMSetup``."""
    if value is None:
        return DWM_SETUPS["default"]
    if isinstance(value, DWMSetup):
        return value
    try:
        return DWM_SETUPS[value]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown dwm_setup {value!r}: expected a DWMSetup or one of "
            f"{sorted(DWM_SETUPS)}"
        ) from None
