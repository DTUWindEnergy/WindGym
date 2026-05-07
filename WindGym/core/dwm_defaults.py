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
from dynamiks.dwm.particle_motion_models import CutOffFrq, HillVortexParticleMotion, XSpeed
from dynamiks.dwm.projection_models import NoProjection
from dynamiks.dwm.superposition import rss_superposition
from dynamiks.utils.data_dumper import runningAverageSensor
from dynamiks.wind_turbines import PyWakeWindTurbines
from dynamiks.wind_turbines.ti_model import RunningAverageSensorTIModel
from jDWM.EddyViscosityModel import keck
from jDWM.Solvers import implicit
from py_wake.rotor_avg_models import CGIRotorAvg


# === DWM closure / particle setup ===========================================
K1 = 0.0914                  # keck ambient eddy-viscosity coefficient
K2 = 0.0216                  # keck wake-shear eddy-viscosity coefficient
TI_W = 1.0                   # keck dimensionless TI weight (NOT physical TI)
SHEAR_W = 1.0                # keck dimensionless dudz_abl weight
D_PARTICLE = 0.48            # streamwise particle spacing in rotor diameters

AINSLIE_R_MAX = 3            # radial domain extent (rotor diameters)
AINSLIE_N_R = 52             # radial grid points
AINSLIE_DX = 0.1             # axial step (rotor diameters)

ROTOR_AVG_N = 21             # CGI rotor-average kernel for the WT inflow sensor
PARTICLE_SPATIAL_AVG_N = 9   # CGI rotor-averaged spatial sampling for particle motion
TI_RUNNING_AVG_S = 600       # RunningAverageSensorTIModel window in seconds


# === Mann turbulence box ====================================================
# Used by `TurbulenceManager._generate_mann_generate` and `_generate_mann_fixed`.
MANN_L = 29.4
MANN_AE = 1.0                # alphaepsilon (cosmetic — scale_TI rescales after)
MANN_GAMMA = 3.9
MANN_NXYZ = (1024, 128, 32)
MANN_DXYZ = (3.2, 3.2, 3.2)


def make_wts(x, y, windTurbine) -> PyWakeWindTurbines:
    """PyWakeWindTurbines wired with the calibrated rotor-average + TI sensor."""
    return PyWakeWindTurbines(
        x=x,
        y=y,
        windTurbine=windTurbine,
        rotorAvgModel=CGIRotorAvg(ROTOR_AVG_N),
        tiModel=RunningAverageSensorTIModel(
            runningAverageSensor=runningAverageSensor(
                TI_RUNNING_AVG_S, TI_RUNNING_AVG_S, None
            ),
            eval_fct=None,
        ),
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
    ti_w: float = TI_W,
    shear_w: float = SHEAR_W,
    d_particle: float = D_PARTICLE,
    d_meander: float | None = None,
) -> DWMFlowSimulation:
    """Assemble a DWMFlowSimulation under the calibrated setup.

    The caller drives it via ``fs.step()`` in a time loop. ``n_particles`` is
    not passed — dynamiks auto-computes it from farm extent and ``d_particle``.

    All closure-model knobs are kwargs that default to the module constants,
    so existing call sites stay unchanged. Override them at episode reset to
    do domain randomization. ``d_meander`` defaults to None (no temporal
    meandering filter), matching pre-randomization behaviour.
    """
    deficit_gen = jDWMAinslieGenerator(
        viscosity_model=keck(TI=ti_w, dudz_abl=shear_w, k1=k1, k2=k2),
        solver=implicit(),
        projectionModel=NoProjection(),
        r_max=AINSLIE_R_MAX,
        n_r=AINSLIE_N_R,
        dx=AINSLIE_DX,
    )

    particle_motion = HillVortexParticleMotion(
        x_speed=XSpeed.Particle,
        temporal_filter=CutOffFrq(d_meander) if d_meander is not None else None,
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
        superpositionModel=rss_superposition(),
        wind_direction=wind_direction,
        dt=dt,
    )
