"""DWM physics defaults for WindGym environments.

Single source of truth for the dynamic wake meandering (DWM) configuration
used by `WindFarmEnv`.

The constants here describe the *physics* of the simulator (closure model,
particle setup, Mann turbulence box, sensor averaging). Per-episode quantities
(wind speed, TI, wind direction, turbine layout, dt) are passed in by the env
and stay variable.

The values were calibrated by aligning DWM output against LES data; do not
change them casually. ``n_particles`` is intentionally NOT pinned here — it is
left to dynamiks to auto-compute from the farm extent and ``D_PARTICLE``, so
that larger farms get more particles automatically.
"""
from __future__ import annotations

from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion, XSpeed
from dynamiks.dwm.projection_models import NoProjection
# from dynamiks.dwm.superposition import rss_superposition
# from dynamiks.utils.data_dumper import runningAverageSensor
from dynamiks.wind_turbines import PyWakeWindTurbines
# from dynamiks.wind_turbines.ti_model import RunningAverageSensorTIModel
from jDWM.EddyViscosityModel import keck
from jDWM.Solvers import implicit
from py_wake.rotor_avg_models import CGIRotorAvg

from dynamiks.dwm.superposition import MixedSum #MixedSum instead of rss
from dynamiks.wind_turbines.ti_models import TISensor, MeanMethod


# === DWM closure / particle setup ===========================================
K1 = 0.0914                  # keck ambient eddy-viscosity coefficient
K2 = 0.0216                  # keck wake-shear eddy-viscosity coefficient
D_PARTICLE = 0.48            # streamwise particle spacing in rotor diameters

AINSLIE_R_MAX = 3            # radial domain extent (rotor diameters)
AINSLIE_N_R = 52             # radial grid points
AINSLIE_DX = 0.1             # axial step (rotor diameters)

ROTOR_AVG_N = 21             # CGI rotor-average kernel for the WT inflow sensor
PARTICLE_SPATIAL_AVG_N = 9   # CGI rotor-averaged spatial sampling for particle motion
TI_RUNNING_AVG_S = 600       # RunningAverageSensorTIModel window in seconds


# === Mann turbulence box ====================================================
# Used by `TurbulenceManager._generate_mann_generate` and `_generate_mann_fixed`.
#
# Under the legacy non-DR path the raw box at αε=1 is renormalised by
# `tf.scale_TI(TI=self.ti, U=ws)` to the env's nominal TI, so MANN_AE is
# effectively cosmetic *in that path*.
#
# Under domain randomization (Mann keys present in `dwm_params`), the
# turbulence manager skips `scale_TI` and αε directly controls the box's
# ambient TI (matching `calibration/simulator.py:_build_site`). At
# (L=29.4, Γ=3.9, WS=9), raw σ_u ≈ 3.63 → TI ≈ 0.40·√αε, so the calibrated
# αε ≈ 0.0056 corresponds to TI ≈ 3%. If you ever switch the default branch
# to skip `scale_TI` unconditionally, lower MANN_AE accordingly.
MANN_L = 29.4
MANN_AE = 1.0                # alphaepsilon — see docstring above re: scale_TI
MANN_GAMMA = 3.9
MANN_NXYZ = (4096, 512, 128)
MANN_DXYZ = (3.2, 3.2, 3.2)


def make_wts(x, y, windTurbine) -> PyWakeWindTurbines:
    """PyWakeWindTurbines wired with the calibrated rotor-average + TI sensor."""
    return PyWakeWindTurbines(
        x=x,
        y=y,
        windTurbine=windTurbine,
        rotorAvgModel=CGIRotorAvg(ROTOR_AVG_N),
        turbulenceIntensityModel = TISensor(mean_method=MeanMethod.TURBULENCE_TRANSPORT_SPEED, T=600),
    )


def make_dwm(
    *,
    site,
    windTurbines,
    wind_direction,
    dt,
    addedTurbulenceModel,
    k1: float = K1,
    k2: float = K2,
    d_particle: float = D_PARTICLE,
) -> DWMFlowSimulation:
    """Assemble a DWMFlowSimulation under the calibrated setup.

    The caller drives it via ``fs.step()`` in a time loop. ``n_particles`` is
    not passed — dynamiks auto-computes it from farm extent and ``d_particle``.

    The three closure knobs (``k1``, ``k2``, ``d_particle``) default to the
    module constants, so existing call sites stay unchanged. Override them at
    episode reset to do domain randomization.
    """
    deficit_gen = jDWMAinslieGenerator(
        viscosity_model=keck(TI=1.0, dudz_abl=1.0, k1=k1, k2=k2),
        solver=implicit(),
        projectionModel=NoProjection(),
        r_max=AINSLIE_R_MAX,
        n_r=AINSLIE_N_R,
        dx=AINSLIE_DX,
    )

    particle_motion = HillVortexParticleMotion(
        x_speed=XSpeed.Particle,
        temporal_filter=None,
        spatial_filter=CGIRotorAvg(PARTICLE_SPATIAL_AVG_N),
        include_wakes=True,
        include_own_wake=False,
    )

    return DWMFlowSimulation(
        site=site,
        windTurbines=windTurbines,
        particleDeficitGenerator=deficit_gen,
        particleMotionModel=particle_motion,
        d_particle=d_particle,
        addedTurbulenceModel=addedTurbulenceModel,
        superpositionModel=MixedSum(),
        wind_direction=wind_direction,
        dt=dt,
    )
