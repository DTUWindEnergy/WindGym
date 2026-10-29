"""DWM flow-simulation setup for WindGym environments.

``DWMSetup`` collects the physics choices of the dynamic wake meandering (DWM)
simulation that ``WindFarmEnv`` builds: wake-deficit model, particle motion,
turbine inflow sensing, superposition, added turbulence and the Mann box.
Per-episode quantities (wind speed, TI, wind direction, layout, dt) are passed
in by the env and stay variable.

The defaults are plain dynamiks, i.e. what ``DWMFlowSimulation`` and its
components do out of the box, plus WindGym's 15D particle floor (see
``make_dwm``). The LES-calibrated setup is opt-in: pass
``dwm_setup=LES_CALIBRATED`` (``WindGym.core.calibrated_values``) to the env, or
set ``dwm_setup: les_calibrated`` in the env config. Parts of it need the dev
dynamiks build (WindGym's ``dev_dynamiks`` branch) and raise on PyPI dynamiks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.added_turbulence_models import (
    SynchronizedAutoScalingIsotropicMannTurbulence,
)
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import (
    CutOffFrq,
    HillVortexParticleMotion,
    XSpeed,
)
from dynamiks.dwm.projection_models import FullProjection, NoProjection
from dynamiks.utils.geometry import get_xyz
from dynamiks.wind_turbines import PyWakeWindTurbines
from jDWM import BoundaryCondition
from jDWM.EddyViscosityModel import keck, madsen
from jDWM.Solvers import implicit
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.superposition_models import LinearSum

from .dynamiks_features import (
    HAS_BRANLARD,
    HAS_INTERP,
    HAS_MIXED_SUM,
    HAS_TI_SENSOR,
    require,
)

_VISCOSITY_MODELS = {"madsen": madsen, "keck": keck}
# Ainslie near-wake boundary condition (jDWM.BoundaryCondition classes; the
# generator asserts a class and instantiates it). "madsen" is the dynamiks default.
_BOUNDARY_CONDITIONS = {
    "madsen": BoundaryCondition.madsen,
    "iec": BoundaryCondition.IEC,
    "keck": BoundaryCondition.keck,
    "none": BoundaryCondition.none,
}
_PROJECTIONS = {"full": FullProjection, "none": NoProjection}
# dynamiks hard-codes the Hill-vortex self-induction coefficient at 0.4
HILL_VORTEX_DEFAULT = 0.4
_X_SPEEDS = {"global": XSpeed.Global, "particle": XSpeed.Particle}


@dataclass(frozen=True)
class MannBoxSpec:
    """Mann turbulence box generated for turbtype MannGenerate / MannFixed."""

    L: float = 33.6  # length scale (m)
    gamma: float = 3.9  # anisotropy
    alphaepsilon: float = (
        0.1  # dissipation; rescaled away by scale_TI (see TurbulenceManager)
    )
    # Grid points (x, y, z). x wraps (Bounds.Repeat), so long episodes
    # recycle the box.
    nxyz: tuple[int, int, int] = (1024, 512, 64)
    # Grid spacing in metres. None = rotor diameter / d_over_dxyz.
    dxyz: tuple[float, float, float] | None = None
    d_over_dxyz: tuple[float, float, float] = (20, 10, 10)

    def spacing(self, rotor_diameter: float) -> tuple[float, float, float]:
        if self.dxyz is not None:
            return tuple(self.dxyz)
        return tuple(rotor_diameter / n for n in self.d_over_dxyz)


@dataclass(frozen=True)
class DWMSetup:
    """Physics of the DWM flow simulation. Defaults are plain dynamiks."""

    # Ainslie wake-deficit generator (jDWMAinslieGenerator)
    viscosity_model: str = "madsen"  # "madsen" or "keck" (jDWM.EddyViscosityModel)
    k1: float | None = None  # None = the viscosity model's default
    k2: float | None = None
    ainslie_r_max: float = 3  # radial domain extent (rotor radii)
    ainslie_n_r: int = 51  # radial grid points
    ainslie_dx: float = 0.1  # axial step
    projection: str = "full"  # "full" (FullProjection) or "none" (NoProjection)
    boundary_condition: str = "madsen"  # Ainslie near-wake boundary condition: "madsen" (dynamiks default), "iec", "keck", "none"

    d_particle: float = 0.2  # streamwise particle spacing in rotor diameters

    # Particle motion (HillVortexParticleMotion)
    particle_x_speed: str = "global"  # "global" or "particle" (XSpeed)
    particle_cutoff_d: float | None = 4  # CutOffFrq temporal filter; None = no filter
    particle_include_wakes: bool = False  # particles also feel the other wakes
    particle_spatial_avg_n: int | None = None  # CGIRotorAvg spatial filter; None = off
    # Hill-vortex self-induction coefficient (wake-deflection strength). 0.4 =
    # dynamiks' hard-coded value (plain HillVortexParticleMotion, bit-identical);
    # any other value selects WindGym's ScaledHillVortexParticleMotion.
    hill_vortex_factor: float = HILL_VORTEX_DEFAULT

    # PyWake turbines
    rotor_avg_n: int | None = None  # CGIRotorAvg inflow kernel; None = dynamiks default
    ti_sensor_window_s: float | None = None  # TISensor window; None = ambient site TI

    superposition: str = "linear"  # "linear" (LinearSum) or "mixed" (MixedSum)
    added_turbulence_scaling: str = "default"  # "default" or "branlard"

    mann: MannBoxSpec = MannBoxSpec()


def make_wts(
    x, y, windTurbine, setup: DWMSetup, turbine_params=None, power_setpoint: bool = False
) -> PyWakeWindTurbines:
    """PyWakeWindTurbines with the rotor-average and TI model of ``setup``.

    ``turbine_params`` (a ``core.turbine_params.TurbineParams``) selects the
    perturbed subclass ``backend.param_turbines.ParamPyWakeWindTurbines``
    (per-turbine Cp/Ct gains, yaw actuator lag, power lag) for turbine-parameter
    domain randomization. ``None`` or an all-default vector returns the plain
    dynamiks class, so the nominal path is unchanged.

    ``power_setpoint=True`` (the env's ``derate_reference="setpoint"``) returns
    ``backend.setpoint_turbines.SetpointPyWakeWindTurbines`` regardless of
    ``is_nominal`` (it needs a ``TurbineParams`` for its filter constants).
    """
    kwargs = {}
    if setup.rotor_avg_n is not None:
        kwargs["rotorAvgModel"] = CGIRotorAvg(setup.rotor_avg_n)
    if setup.ti_sensor_window_s is not None:
        require(HAS_TI_SENSOR, "DWMSetup.ti_sensor_window_s (TISensor)")
        from dynamiks.wind_turbines.ti_models import MeanMethod, TISensor

        kwargs["turbulenceIntensityModel"] = TISensor(
            mean_method=MeanMethod.TURBULENCE_TRANSPORT_SPEED,
            T=setup.ti_sensor_window_s,
        )
    if power_setpoint:
        if turbine_params is None:
            raise ValueError("power_setpoint=True requires turbine_params (filter constants)")
        from ..backend.setpoint_turbines import SetpointPyWakeWindTurbines  # lazy

        return SetpointPyWakeWindTurbines(
            x=x, y=y, windTurbine=windTurbine, params=turbine_params, **kwargs
        )
    if turbine_params is None or turbine_params.is_nominal:
        return PyWakeWindTurbines(x=x, y=y, windTurbine=windTurbine, **kwargs)
    from ..backend.param_turbines import ParamPyWakeWindTurbines  # lazy: dynamiks subclass

    return ParamPyWakeWindTurbines(
        x=x, y=y, windTurbine=windTurbine, params=turbine_params, **kwargs
    )


def make_added_turbulence_model(setup: DWMSetup):
    """Synchronized Mann added-turbulence model with the scaling of ``setup``."""
    if setup.added_turbulence_scaling == "branlard":
        require(HAS_BRANLARD, "DWMSetup.added_turbulence_scaling='branlard'")
        from dynamiks.dwm.added_turbulence_models import BranlardScaling

        return SynchronizedAutoScalingIsotropicMannTurbulence(
            scaling=BranlardScaling(), cache_field=False
        )
    if setup.added_turbulence_scaling != "default":
        raise ValueError(
            f"Unknown added_turbulence_scaling {setup.added_turbulence_scaling!r} "
            "(expected 'default' or 'branlard')"
        )
    return SynchronizedAutoScalingIsotropicMannTurbulence(cache_field=False)


def _superposition_model(setup: DWMSetup):
    if setup.superposition == "mixed":
        require(HAS_MIXED_SUM, "DWMSetup.superposition='mixed' (MixedSum)")
        from dynamiks.dwm.superposition import MixedSum

        return MixedSum()
    if setup.superposition != "linear":
        raise ValueError(
            f"Unknown superposition {setup.superposition!r} "
            "(expected 'linear' or 'mixed')"
        )
    return LinearSum()


def make_dwm(
    *,
    site,
    windTurbines,
    wind_direction,
    dt,
    addedTurbulenceModel,
    setup: DWMSetup,
    k1: float | None = None,
    k2: float | None = None,
    d_particle: float | None = None,
    interpolation: str | None = None,
    lateral_cutoff: float | None = None,
    hill_vortex_factor: float | None = None,
) -> DWMFlowSimulation:
    """Assemble a DWMFlowSimulation from ``setup``.

    The caller drives it via ``fs.step()`` in a time loop.

    ``k1``, ``k2``, ``d_particle`` and ``hill_vortex_factor`` override the
    setup's values for one episode (domain randomization); None keeps the
    setup's value.

    ``interpolation`` / ``lateral_cutoff`` are dev-dynamiks speed-ups; None
    leaves them out (dynamiks default: pchip, no cutoff).

    Turbines that expose ``attach_to`` (``ParamPyWakeWindTurbines``) are
    attached to the new flow simulation here, so both the agent and the
    baseline farm get their actuator step handler through this one builder.
    """
    k1 = setup.k1 if k1 is None else k1
    k2 = setup.k2 if k2 is None else k2
    d_particle = setup.d_particle if d_particle is None else d_particle
    hill_vortex_factor = (
        setup.hill_vortex_factor if hill_vortex_factor is None else float(hill_vortex_factor)
    )

    # Particle count: dynamiks auto-computes ceil(farm_size_x*1.2/d_particle)
    # with a floor of 10 particles, which degenerates for layouts where all
    # turbines share one downwind x (side-by-side farms, or a single row at
    # wd 0/180): 10 particles ~ 4.8D of wake and everything beyond silently
    # vanishes. Keep the wdest fix: cover at least 15D downstream. For
    # extended farms (farm_x*1.2 >= 15D, e.g. the les_3x3 layouts) this
    # matches the dynamiks auto-compute.
    # (rotor_positions_xyz needs a bound flow simulation, so rotate the
    # east/north positions into the wd frame here; the extent is invariant
    # to the center_offset translation.)
    try:
        _en = np.asarray(windTurbines.rotor_positions_east_north, dtype=float)
    except AttributeError:  # HAWC2 variants expose it as a property that may need fs
        _en = np.asarray(windTurbines.positions_east_north, dtype=float)
    _x = get_xyz(_en, wind_direction, np.zeros(3))[0]
    _D = np.atleast_1d(windTurbines.diameter()).astype(float)
    _desired = max(float(_x.max() - _x.min()) * 1.2, 15.0 * float(_D.max()))
    n_particles = max(int(np.ceil(_desired / (d_particle * float(_D.min())))), 10)

    closure = {k: v for k, v in (("k1", k1), ("k2", k2)) if v is not None}
    try:
        boundary = _BOUNDARY_CONDITIONS[setup.boundary_condition]
    except KeyError:
        raise ValueError(
            f"Unknown boundary_condition {setup.boundary_condition!r} "
            f"(expected one of {sorted(_BOUNDARY_CONDITIONS)})"
        ) from None
    deficit_gen = jDWMAinslieGenerator(
        viscosity_model=_VISCOSITY_MODELS[setup.viscosity_model](TI=1.0, **closure),
        solver=implicit(),
        projectionModel=_PROJECTIONS[setup.projection](),
        boundaryConditionModel=boundary,
        r_max=setup.ainslie_r_max,
        n_r=setup.ainslie_n_r,
        dx=setup.ainslie_dx,
    )

    motion_kwargs = dict(
        x_speed=_X_SPEEDS[setup.particle_x_speed],
        temporal_filter=(
            None
            if setup.particle_cutoff_d is None
            else CutOffFrq(setup.particle_cutoff_d)
        ),
        spatial_filter=(
            None
            if setup.particle_spatial_avg_n is None
            else CGIRotorAvg(setup.particle_spatial_avg_n)
        ),
        include_wakes=setup.particle_include_wakes,
    )
    if float(hill_vortex_factor) == HILL_VORTEX_DEFAULT:
        particle_motion = HillVortexParticleMotion(**motion_kwargs)
    else:
        from .particle_motion import ScaledHillVortexParticleMotion

        particle_motion = ScaledHillVortexParticleMotion(
            c=float(hill_vortex_factor), **motion_kwargs
        )

    # Speedups, not part of the LES calibration (which ran pchip / no cutoff,
    # the dynamiks defaults): linear centerline interpolation and the lateral
    # interaction cutoff. Only forwarded when set, since PyPI dynamiks lacks them.
    speedups = {}
    if interpolation is not None:
        speedups["interpolation"] = interpolation
    if lateral_cutoff is not None:
        speedups["lateral_cutoff"] = lateral_cutoff
    if speedups:
        require(HAS_INTERP, "DWM interpolation/lateral_cutoff")

    fs = DWMFlowSimulation(
        site=site,
        windTurbines=windTurbines,
        particleDeficitGenerator=deficit_gen,
        particleMotionModel=particle_motion,
        d_particle=d_particle,
        n_particles=n_particles,
        addedTurbulenceModel=addedTurbulenceModel,
        superpositionModel=_superposition_model(setup),
        wind_direction=wind_direction,
        dt=dt,
        **speedups,
    )
    if hasattr(windTurbines, "attach_to"):
        windTurbines.attach_to(fs)
    return fs


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
