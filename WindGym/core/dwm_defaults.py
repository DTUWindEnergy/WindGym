"""DWM physics defaults for WindGym environments.

Single source of truth for the dynamic wake meandering (DWM) configuration
used by `WindFarmEnv`.

The constants here describe the *physics* of the simulator (closure model,
particle setup, Mann turbulence box, sensor averaging). Per-episode quantities
(wind speed, TI, wind direction, turbine layout, dt) are passed in by the env
and stay variable.

The values were calibrated by aligning DWM output against LES data; do not
change them casually. ``n_particles`` is intentionally NOT pinned here — it is
computed from the farm extent and ``d_particle`` like dynamiks does, but with
a 15D-downstream floor (see ``make_dwm``), so that larger farms get more
particles automatically and side-by-side layouts still carry wakes.
"""
from __future__ import annotations

import numpy as np

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
from dynamiks.utils.geometry import get_xyz


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
# NOTE on box extent: 1024*3.2 = 3.28 km streamwise. Under Bounds.Repeat a
# long episode re-samples the box many times (a Stage-7 10,300 s episode at
# ws 9-11 advects 90-110 km ~ 30 wraps). This is calibration-faithful — the
# LES/SBI calibration and the LESRL trainings ran the same spec — but it is a
# deliberate trade against the old wdest (4096, ...) anti-recycle box; under
# DR every episode regenerates a fresh box, so recycling is within-episode
# only.
MANN_NXYZ = (1024, 256, 128)
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
    interpolation: str = "pchip",
    lateral_cutoff=None,
) -> DWMFlowSimulation:
    """Assemble a DWMFlowSimulation under the calibrated setup.

    The caller drives it via ``fs.step()`` in a time loop.

    The three closure knobs (``k1``, ``k2``, ``d_particle``) default to the
    module constants, so existing call sites stay unchanged. Override them at
    episode reset to do domain randomization.
    """
    # Particle count: dynamiks auto-computes ceil(farm_size_x*1.2/d_particle)
    # with a floor of 10 particles, which degenerates for layouts where all
    # turbines share one downwind x (side-by-side farms, or a single row at
    # wd 0/180): 10 particles ~ 4.8D of wake and everything beyond silently
    # vanishes. Keep the wdest fix: cover at least 15D downstream. For
    # extended farms (farm_x*1.2 >= 15D, e.g. the les_3x3 layouts) this
    # matches the dynamiks auto-compute the calibration ran with.
    # (rotor_positions_xyz needs a bound flow simulation, so rotate the
    # east/north positions into the wd frame here; the extent is invariant
    # to the center_offset translation.)
    try:
        _en = np.asarray(windTurbines.rotor_positions_east_north, dtype=float)
    except Exception:  # HAWC2 variants expose it as a property that may need fs
        _en = np.asarray(windTurbines.positions_east_north, dtype=float)
    _x = get_xyz(_en, wind_direction)[0]
    _D = np.atleast_1d(windTurbines.diameter()).astype(float)
    _desired = max(float(_x.max() - _x.min()) * 1.2, 15.0 * float(_D.max()))
    n_particles = max(int(np.ceil(_desired / (d_particle * float(_D.min())))), 10)

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
        n_particles=n_particles,
        addedTurbulenceModel=addedTurbulenceModel,
        superpositionModel=MixedSum(),
        wind_direction=wind_direction,
        dt=dt,
        # Speedups, not part of the LES calibration (which ran pchip / no
        # cutoff, the defaults here): linear centerline interpolation and the
        # lateral interaction cutoff. WindFarmEnv passes its own settings
        # (default linear / 1.5, as in the Stage-6 sweeps).
        interpolation=interpolation,
        lateral_cutoff=lateral_cutoff,
    )


def add_hawc2_yaw_sensor(wts, mode: str = "bearing2_slot", slot: int = 1):
    """Attach the exposed ``yaw`` sensor pair to a HAWC2WindTurbines object.

    The wiring must match the htc's yaw-servo DLL, so it is configurable:

    - ``"bearing2_slot"`` (legacy WindGym default): read the yaw bearing via a
      ``constraint bearing2 yaw_rot`` output sensor and write the setpoint
      into HAWC2 general-variable ``slot``. Matches the DTU10MW/IEA22MW_yaw
      htc files (slot 1, ``bearing2 yaw_rot`` constraint).
    - ``"yaw_tilt"``: read via ``wt.yaw_tilt()[0]`` (generic h2lib rotor
      orientation in degrees — no dependence on the htc's constraint naming,
      HAWC2->dynamiks sign flip already applied) and write the setpoint into
      general-variable ``slot``. Validated against
      ``LEShawc2files/htc/input_hawc_yaw_actuator_tipcorr.htc`` (slot 4,
      positive sign).

    Getter returns MEASURED yaw in degrees; the setter writes a SETPOINT in
    radians to the servo DLL. The two are intentionally asymmetric — see the
    ``yaw_command`` invariant in ``wind_farm_env.py``.
    """
    if mode == "bearing2_slot":
        wts.add_sensor(
            name="yaw_getter",
            getter="constraint bearing2 yaw_rot 1 only 1;",
            expose=False,
            ext_lst=["angle", "speed"],
        )
        wts.add_sensor(
            "yaw",
            getter=lambda wt: np.rad2deg(wt.sensors.yaw_getter[:, 0]),
            setter=lambda wt, value: wt.h2.set_variable_sensor_value(
                slot, np.deg2rad(value).tolist()
            ),
            expose=True,
        )
    elif mode == "yaw_tilt":
        wts.add_sensor(
            "yaw",
            getter=lambda wt: wt.yaw_tilt()[0],
            setter=lambda wt, value: wt.h2.set_variable_sensor_value(
                slot, np.deg2rad(value).tolist()
            ),
            expose=True,
        )
    else:
        raise ValueError(
            f"Unknown hawc2_yaw_mode: {mode!r} "
            "(expected 'bearing2_slot' or 'yaw_tilt')"
        )
