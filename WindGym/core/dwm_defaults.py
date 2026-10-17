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

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynamiks.dwm import DWMFlowSimulation
from dynamiks.dwm.particle_deficit_profiles.ainslie import jDWMAinslieGenerator
from dynamiks.dwm.particle_motion_models import (
    CutOffFrq,
    HillVortexParticleMotion,
    XSpeed,
)
from dynamiks.dwm.projection_models import NoProjection
from dynamiks.dwm.secondary_steering import DECAY_D as SS_DECAY_D
from dynamiks.dwm.secondary_steering import SecondarySteeringGenerator

# from dynamiks.dwm.superposition import rss_superposition
# from dynamiks.utils.data_dumper import runningAverageSensor
from dynamiks.wind_turbines import PyWakeWindTurbines

# from dynamiks.wind_turbines.ti_model import RunningAverageSensorTIModel
from jDWM import BoundaryCondition
from jDWM.EddyViscosityModel import IEC, keck, larsen, madsen
from jDWM.Solvers import implicit
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.superposition_models import LinearSum

from dynamiks.dwm.superposition import MixedSum  # MixedSum instead of rss
from dynamiks.wind_turbines.ti_models import TISensor, MeanMethod
from dynamiks.utils.geometry import get_xyz

from .particle_motion import DEFLECTION_C, ScaledHillVortexParticleMotion


# === DWM closure / particle setup ===========================================
K1 = 0.0914  # keck ambient eddy-viscosity coefficient
K2 = 0.0216  # keck wake-shear eddy-viscosity coefficient
D_PARTICLE = 0.48  # streamwise particle spacing in rotor diameters

AINSLIE_R_MAX = 3  # radial domain extent (rotor diameters)
AINSLIE_N_R = 52  # radial grid points
AINSLIE_DX = 0.1  # axial step (rotor diameters)

ROTOR_AVG_N = 21  # CGI rotor-average kernel for the WT inflow sensor
PARTICLE_SPATIAL_AVG_N = 9  # CGI rotor-averaged spatial sampling for particle motion
TI_RUNNING_AVG_S = 600  # RunningAverageSensorTIModel window in seconds


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
MANN_AE = 1.0  # alphaepsilon — see docstring above re: scale_TI
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


# === Model-structure registries ==============================================
# Categorical ``dwm_params`` keys resolve through these tables (case-insensitive).
@dataclass(frozen=True)
class _ViscositySpec:
    """How to build one jDWM eddy-viscosity model from ``(k1, k2)``.

    ``k1_name`` is the model's own name for the ambient coefficient (``kamb``
    for larsen, ``k1`` otherwise); ``pinned`` are WindGym's calibrated
    coefficients, applied when the caller leaves ``k1``/``k2`` at ``None``.
    Models without a pinned entry fall back to their jDWM defaults.
    """

    cls: type
    k1_name: str
    pinned: dict = field(default_factory=dict)


VISCOSITY_MODELS: dict[str, _ViscositySpec] = {
    "keck": _ViscositySpec(keck, "k1", {"k1": K1, "k2": K2}),
    "madsen": _ViscositySpec(madsen, "k1"),
    "larsen": _ViscositySpec(larsen, "kamb"),
    "iec": _ViscositySpec(IEC, "k1"),
}
# PyWake's SquaredSum asserts on the signed v/w deficits DWM produces, so it
# is deliberately not offered here.
SUPERPOSITION_MODELS: dict[str, type] = {"mixed": MixedSum, "linear": LinearSum}
X_SPEEDS: dict[str, XSpeed] = {
    "particle": XSpeed.Particle,
    "global": XSpeed.Global,
    "rotor": XSpeed.Rotor,
}
# Near-wake boundary condition of the Ainslie solver. "madsen" is
# jDWMAinslieGenerator's own default, i.e. what the calibrated setup runs.
# Entries must be BoundaryCondition *classes* (jDWMAinslieGenerator asserts
# issubclass at construction); jDWM's ``rotor_area_mean`` is a helper function,
# not a boundary condition, so it is deliberately not offered.
BOUNDARY_CONDITIONS: dict[str, type] = {
    "madsen": BoundaryCondition.madsen,
    "iec": BoundaryCondition.IEC,
    "keck": BoundaryCondition.keck,
    "none": BoundaryCondition.none,
}

# The wake deficit each particle carries. "ainslie" is jDWM's eddy-viscosity
# solver (boundary_condition, viscosity_model, k1/k2, r_max/n_r/dx apply);
# "bastankhah" is py_wake's BastankhahGaussianDeficit wrapped by dynamiks'
# PyWakeDeficitGenerator: a self-similar Gaussian of width ceps + k x/D (those
# Ainslie keys are then ignored). Its literature k fits time-averaged wakes,
# which already contain meandering; DWM meanders the particles explicitly, so
# under DWM k must be fitted in the wake-fixed frame, not taken from the paper.
DEFICIT_MODELS: tuple[str, ...] = ("ainslie", "bastankhah")
BASTANKHAH_K = 0.0324555  # py_wake's default (Bastankhah & Porte-Agel 2014)
BASTANKHAH_CEPS = 0.2

# How a turbine turns the wind over its rotor into power; the classes live in
# backend.rotor_power (imported lazily, like the turbine-parameter DR class).
# "mean" is dynamiks' own arithmetic rotor average.
ROTOR_POWER_CHOICES: tuple[str, ...] = ("mean", "cubic")


def _lookup(registry: dict, value, name: str):
    """Resolve a categorical ``dwm_params`` value; raise listing the choices."""
    key = value.lower() if isinstance(value, str) else None
    if key not in registry:
        raise ValueError(f"Unknown {name} {value!r}. Choices: {sorted(registry)}")
    return registry[key]


def _make_viscosity_model(viscosity_model, k1, k2):
    """Instantiate the selected jDWM viscosity model at TI=1 (dynamiks rescales TI)."""
    spec = _lookup(VISCOSITY_MODELS, viscosity_model, "viscosity_model")
    kwargs = dict(spec.pinned)
    if k1 is not None:
        kwargs[spec.k1_name] = float(k1)
    if k2 is not None:
        kwargs["k2"] = float(k2)
    return spec.cls(TI=1.0, **kwargs)


# === dwm_params specification ================================================
# TODO: every physics value below should be overridable through dwm_params, as
# the evaluation code expects (full flexibility, each key defaulting to the
# current value). Still hardcoded:
#   - make_wts: rotor-average kernel CGIRotorAvg(ROTOR_AVG_N), and the TI model
#     TISensor(mean_method, T=600) (TI_RUNNING_AVG_S is defined but unused).
#     Built at env construction, not per reset, so these need a
#     constructor-only group.
#   - make_dwm motion model: PARTICLE_SPATIAL_AVG_N (and None = no spatial
#     filter), include_wakes (True: downstream particles are advected by the
#     superposed wake field, which is what lets secondary_steering act),
#     include_own_wake.
#   - make_dwm deficit generator: solver (implicit), projectionModel
#     (NoProjection), scale_with_freestream.
#   - make_dwm: the 15 D downstream floor on n_particles.
#   - TurbulenceManager._added_turb_model: the Branlard added-turbulence scaling.
@dataclass(frozen=True)
class DWMParamSpec:
    """Validation metadata for one ``dwm_params`` key.

    ``kind`` is ``"float"``, ``"int"``, ``"bool"``, ``"choice"``, or ``"int3"`` /
    ``"float3"`` (a 3-sequence, e.g. a Mann grid); ``group`` says which
    subsystem consumes the key (``"make_dwm"``, ``"mann"`` and ``"wat"``,
    routed to ``TurbulenceManager.create_sites``, or ``"make_wts"``, the
    turbine objects of both farms).
    """

    kind: str
    default: Any
    group: str
    choices: tuple = ()
    nullable: bool = False
    doc: str = ""


DWM_PARAM_SPEC: dict[str, DWMParamSpec] = {
    "k1": DWMParamSpec(
        "float",
        None,
        "make_dwm",
        nullable=True,
        doc="Ambient eddy-viscosity coefficient (larsen: kamb). "
        "None = the selected viscosity model's own value "
        f"(keck: calibrated {K1}).",
    ),
    "k2": DWMParamSpec(
        "float",
        None,
        "make_dwm",
        nullable=True,
        doc="Wake-shear eddy-viscosity coefficient. None = the "
        f"selected model's own value (keck: calibrated {K2}).",
    ),
    "d_particle": DWMParamSpec(
        "float", D_PARTICLE, "make_dwm", doc="Streamwise particle spacing [D]."
    ),
    "viscosity_model": DWMParamSpec(
        "choice",
        "keck",
        "make_dwm",
        choices=tuple(VISCOSITY_MODELS),
        doc="jDWM eddy-viscosity closure.",
    ),
    "superposition": DWMParamSpec(
        "choice",
        "mixed",
        "make_dwm",
        choices=tuple(SUPERPOSITION_MODELS),
        doc="Wake deficit superposition (PyWake model).",
    ),
    "x_speed": DWMParamSpec(
        "choice",
        "particle",
        "make_dwm",
        choices=tuple(X_SPEEDS),
        doc="Particle streamwise advection speed. global/rotor "
        "change the dynamics materially vs the calibrated "
        "particle setting.",
    ),
    "r_max": DWMParamSpec(
        "float",
        AINSLIE_R_MAX,
        "make_dwm",
        doc="Ainslie radial domain extent [R]. Also rescales the "
        "lateral_cutoff radius (cutoff = lateral_cutoff*r_max*R).",
    ),
    "n_r": DWMParamSpec(
        "int",
        AINSLIE_N_R,
        "make_dwm",
        doc="Ainslie radial grid points. dynamiks warns when "
        "dr = r_max/(n_r-1) > 0.2 (implicit solver stability).",
    ),
    "dx": DWMParamSpec(
        "float",
        AINSLIE_DX,
        "make_dwm",
        doc="Ainslie axial step [D]. dynamiks warns when dx < 25*dr**2.",
    ),
    "lateral_cutoff": DWMParamSpec(
        "float",
        None,
        "make_dwm",
        nullable=True,
        doc="Lateral wake-interaction cutoff in units of "
        "r_max*R; None = no cutoff. Overrides the env's "
        "constructor value for the episode.",
    ),
    "boundary_condition": DWMParamSpec(
        "choice",
        "madsen",
        "make_dwm",
        choices=tuple(BOUNDARY_CONDITIONS),
        doc="Ainslie near-wake boundary condition (jDWM.BoundaryCondition).",
    ),
    "deflection_c": DWMParamSpec(
        "float",
        DEFLECTION_C,
        "make_dwm",
        doc="Hill-vortex self-induction coefficient: how hard "
        "a yawed rotor deflects its own wake. dynamiks "
        f"hardcodes {DEFLECTION_C}; see core.particle_motion.",
    ),
    "deflection_c_pos": DWMParamSpec(
        "float",
        None,
        "make_dwm",
        nullable=True,
        doc="Wake-deflection strength for wakes emitted at POSITIVE flow-relative "
        "yaw (the yaw sensor's sign); deflection_c then serves negative yaw. "
        "None (calibrated) = deflection_c for both signs. The G1's steering "
        "gain is not symmetric in the yaw sign.",
    ),
    "meandering_d": DWMParamSpec(
        "float",
        None,
        "make_dwm",
        nullable=True,
        doc="Meandering low-pass filter CutOffFrq(d): cut-off "
        "fc = U/(d*D), so d is in rotor diameters and scales "
        "with the turbine. None (calibrated) = no temporal "
        "filter; 4 is dynamiks' own default (Lio 2021), "
        "fitted with x_speed=global. NOTE: under the "
        "calibrated x_speed='particle' the same filter also "
        "low-passes the streamwise particle speed (dynamiks "
        "only resets u for global/rotor), so it changes wake "
        "advection, not just lateral meandering. For pure "
        "meandering damping pair it with x_speed='global' "
        "or re-fit the closure with it on.",
    ),
    "superposition_p": DWMParamSpec(
        "float",
        2.0,
        "make_dwm",
        doc="The exponent the streamwise wake deficits combine with, "
        "(sum d^p)^(1/p): 2 = MixedSum's root-sum-square (the calibrated "
        "setup), 1 = linear. Only with superposition='mixed'; v and w stay "
        "linear.",
    ),
    "wake_drift_deg": DWMParamSpec(
        "float",
        0.0,
        "make_dwm",
        doc="A mean lateral drift of every wake, degrees toward +y in the flow "
        "frame: the particles move sideways at tan(drift) x their streamwise "
        "speed. For a flow whose mean direction differs from the inflow map's "
        "(which carries no lateral velocity). 0 = none.",
    ),
    "meandering_radius_scale": DWMParamSpec(
        "float",
        1.0,
        "make_dwm",
        doc="Meandering amplitude: the disk a particle's advection velocity is "
        "averaged over, as a fraction of the local wake radius (CGI nodes). "
        "1.0 = the whole wake (dynamiks' and WindGym's setup); 0 = the point "
        "at the particle. A smaller disk passes more of the ambient lateral "
        "turbulence to the wake, so it meanders more.",
    ),
    "deficit_model": DWMParamSpec(
        "choice",
        "ainslie",
        "make_dwm",
        choices=DEFICIT_MODELS,
        doc="Particle wake deficit: 'ainslie' (jDWM eddy-viscosity "
        "solver) or 'bastankhah' (py_wake's Gaussian, width "
        "gauss_ceps + gauss_k x/D; the Ainslie keys are then "
        "ignored). See DEFICIT_MODELS.",
    ),
    "gauss_k": DWMParamSpec(
        "float",
        BASTANKHAH_K,
        "make_dwm",
        doc="Bastankhah wake growth rate k (width per unit x/D). "
        "Only with deficit_model='bastankhah'. The literature "
        "value includes meandering; fit it in the wake-fixed frame.",
    ),
    "gauss_ceps": DWMParamSpec(
        "float",
        BASTANKHAH_CEPS,
        "make_dwm",
        doc="Bastankhah initial wake width coefficient ceps (the "
        "width at x = 0 is ceps*sqrt(beta) D). Only with "
        "deficit_model='bastankhah'.",
    ),
    "yaw_ct_exp": DWMParamSpec(
        "float",
        0.0,
        "make_dwm",
        doc="Yawed Gaussian: extra cos(yaw)**yaw_ct_exp on the thrust in the "
        "particle deficit, on top of the turbine's own Ct(yaw) (momentum "
        "theory puts the streamwise deficit at cos^3 in all). 0 = off, the "
        "stock profile. Only with deficit_model='bastankhah'.",
    ),
    "yaw_width_exp": DWMParamSpec(
        "float",
        0.0,
        "make_dwm",
        doc="Yawed Gaussian: the lateral onset width scales with "
        "cos(yaw)**yaw_width_exp (Bastankhah & Porte-Agel 2016: 1, the "
        "rotor's frontal projection); the vertical width does not. 0 = off. "
        "Only with deficit_model='bastankhah'.",
    ),
    "secondary_steering": DWMParamSpec(
        "bool",
        False,
        "make_dwm",
        doc="Secondary steering: each yawed turbine's particles carry the "
        "GCH counter-rotating vortex pair's lateral velocity "
        "(dynamiks.dwm.secondary_steering), which the include_wakes "
        "particle advection turns into a deflection of downstream wakes. "
        "False (calibrated) = off. The yawed turbine's own wake is "
        "unaffected either way.",
    ),
    "ss_gain": DWMParamSpec(
        "float",
        1.0,
        "make_dwm",
        doc="Secondary steering: scale on the vortex circulation "
        "(1.0 = FLORIS GCH). Only with secondary_steering=True.",
    ),
    "ss_decay_D": DWMParamSpec(
        "float",
        SS_DECAY_D,
        "make_dwm",
        doc="Secondary steering: downstream decay length of the vortex "
        "field in rotor diameters, 1/(1 + x/L). The default is FLORIS's "
        "mixing-length viscosity at the TUM tunnel's hub shear. Only with "
        "secondary_steering=True.",
    ),
    "wat_gain": DWMParamSpec(
        "float",
        1.0,
        "wat",
        doc="Wake-added turbulence: factor on the Branlard scaling's weight "
        "(1.0 = the paper's calibration). Routed to TurbulenceManager, "
        "honoured at every reset.",
    ),
    "wat_length_scale_D": DWMParamSpec(
        "float",
        1.0 / 16.0,
        "wat",
        doc="Wake-added turbulence: the box's Mann length scale in rotor "
        "diameters (dynamiks: 1/16; Branlard 2024 calibrated his weights "
        "with 1). Routed to TurbulenceManager, honoured at every reset.",
    ),
    "rotor_power": DWMParamSpec(
        "choice",
        "mean",
        "make_wts",
        choices=ROTOR_POWER_CHOICES,
        doc="How power is computed from the wind over the rotor: "
        "'mean' evaluates the power curve at dynamiks' "
        "arithmetic rotor average, 'cubic' at the cube-mean "
        "(sum w u^3)^(1/3) over the same nodes (FLORIS v2.3), "
        "which a partial wake needs. Thrust keeps the "
        "arithmetic mean. See backend.rotor_power.",
    ),
    "mann_L": DWMParamSpec("float", MANN_L, "mann", doc="Mann length scale [m]."),
    "mann_GAMMA": DWMParamSpec("float", MANN_GAMMA, "mann", doc="Mann anisotropy Γ."),
    "mann_AE": DWMParamSpec(
        "float",
        MANN_AE,
        "mann",
        nullable=True,
        doc="Mann αε. Required whenever any Mann key is active. A "
        "number makes αε authoritative: σ_u is then fixed, so the "
        "box's TI drifts with the episode's ws (a 5-6 m/s band is "
        "a 20% TI swing). None = generate at the default αε and "
        "rescale the box to the episode's TI per reset "
        "(scale_TI(TI, U=ws)), as the non-DR path does; this is "
        "the right choice when the ambient TI is held fixed while "
        "ws varies (a wind tunnel), and it needs no fitted αε.",
    ),
    "mann_Nxyz": DWMParamSpec(
        "int3", MANN_NXYZ, "mann", doc="Mann box grid points (nx, ny, nz)."
    ),
    "mann_dxyz": DWMParamSpec(
        "float3",
        MANN_DXYZ,
        "mann",
        doc="Mann box grid spacing (dx, dy, dz) [m]. dy/dx sets "
        "the inertial slope of the streamwise spectrum.",
    ),
}


def _check_numeric(key: str, spec: DWMParamSpec, value, where: str) -> None:
    if isinstance(value, (bool, str)):
        raise ValueError(f"dwm_params[{key!r}]{where} must be a number, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"dwm_params[{key!r}]{where} must be a number, got {value!r}"
        ) from None
    if not math.isfinite(f) or f <= 0:
        raise ValueError(
            f"dwm_params[{key!r}]{where} must be finite and > 0, got {value!r}"
        )
    if spec.kind == "int" and f != int(f):
        raise ValueError(
            f"dwm_params[{key!r}]{where} must be an integer, got {value!r}"
        )


def validate_dwm_params(params: dict, where: str = "") -> None:
    """Raise ``ValueError`` unless every entry of ``params`` is a valid ``dwm_params`` key/value.

    ``where`` is appended to the message (e.g. " in reset options") so the
    caller is identifiable from the traceback. The "Unknown dwm_params keys"
    prefix is relied on by existing tests.
    """
    bad = set(params) - set(DWM_PARAM_SPEC)
    if bad:
        raise ValueError(
            f"Unknown dwm_params keys{where}: {sorted(bad)}. "
            f"Allowed: {sorted(DWM_PARAM_SPEC)}"
        )
    for key, value in params.items():
        spec = DWM_PARAM_SPEC[key]
        if value is None:
            if not spec.nullable:
                raise ValueError(f"dwm_params[{key!r}]{where} may not be None")
            continue
        if spec.kind == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(  # noqa: TRY004  (ValueError is this function's contract)
                    f"dwm_params[{key!r}]{where} must be a bool, got {value!r}"
                )
        elif spec.kind == "choice":
            if not isinstance(value, str) or value.lower() not in spec.choices:
                raise ValueError(
                    f"dwm_params[{key!r}]{where} must be one of {list(spec.choices)}, "
                    f"got {value!r}"
                )
        elif spec.kind in ("int3", "float3"):
            if (
                isinstance(value, (str, bytes))
                or not hasattr(value, "__len__")
                or len(value) != 3
            ):
                raise ValueError(
                    f"dwm_params[{key!r}]{where} must be a sequence of 3 numbers, got {value!r}"
                )
            element = DWMParamSpec(spec.kind[:-1], None, spec.group)
            for i, v in enumerate(value):
                _check_numeric(key, element, v, f"[{i}]{where}")
        else:
            _check_numeric(key, spec, value, where)


def make_wts(
    x, y, windTurbine, turbine_params=None, rotor_power="mean"
) -> PyWakeWindTurbines:
    """PyWakeWindTurbines wired with the calibrated rotor-average + TI sensor.

    ``turbine_params`` (a ``core.turbine_params.TurbineParams``) selects the
    perturbed subclass ``backend.param_turbines.ParamPyWakeWindTurbines``
    (per-turbine Cp/Ct gains, yaw actuator lag, power lag) for turbine-parameter
    domain randomization. ``rotor_power`` (``ROTOR_POWER_CHOICES``) selects how
    power is computed from the rotor nodes; the two compose
    (``backend.rotor_power.ROTOR_POWER_CLASSES``). ``None`` or an all-default
    vector with ``rotor_power="mean"`` returns the plain dynamiks class, so the
    nominal path is unchanged.
    """
    _lookup(dict.fromkeys(ROTOR_POWER_CHOICES), rotor_power, "rotor_power")
    kwargs = dict(
        rotorAvgModel=CGIRotorAvg(ROTOR_AVG_N),
        turbulenceIntensityModel=TISensor(
            mean_method=MeanMethod.TURBULENCE_TRANSPORT_SPEED, T=600
        ),
    )
    nominal = turbine_params is None or turbine_params.is_nominal
    if nominal and rotor_power.lower() == "mean":
        return PyWakeWindTurbines(x=x, y=y, windTurbine=windTurbine, **kwargs)
    from ..backend.rotor_power import ROTOR_POWER_CLASSES  # lazy: dynamiks subclasses

    plain, perturbed = ROTOR_POWER_CLASSES[rotor_power.lower()]
    if nominal:
        return plain(x=x, y=y, windTurbine=windTurbine, **kwargs)
    return perturbed(x=x, y=y, windTurbine=windTurbine, params=turbine_params, **kwargs)


# Set once a dropped-kwargs warning has been emitted, so we only warn once per
# process even though make_dwm() runs once per episode reset.
_dwm_speed_kwargs_warned = False


def _dwm_speed_kwargs(interpolation, lateral_cutoff) -> dict:
    """Build the DWMFlowSimulation kwargs for the proj/fast speedups.

    The installed `dynamiks` submodule may predate the `interpolation` /
    `lateral_cutoff` speedups (they live on a separate `dynamiks` branch that
    cannot currently be merged into the pinned one). Only pass kwargs that
    `DWMFlowSimulation.__init__` actually accepts, so WindGym keeps working
    against an older dynamiks; warn once per process about anything dropped.
    """
    global _dwm_speed_kwargs_warned

    candidates = {"interpolation": interpolation, "lateral_cutoff": lateral_cutoff}
    supported = inspect.signature(DWMFlowSimulation.__init__).parameters
    kwargs = {name: value for name, value in candidates.items() if name in supported}
    dropped = [name for name in candidates if name not in supported]

    if dropped and not _dwm_speed_kwargs_warned:
        _dwm_speed_kwargs_warned = True
        warnings.warn(
            f"The installed dynamiks predates the proj/fast DWM speedups; "
            f"dropping unsupported DWMFlowSimulation kwarg(s): {', '.join(dropped)}.",
            stacklevel=2,
        )

    return kwargs


class _PowerSum(MixedSum):
    """dynamiks' MixedSum with the streamwise exponent exposed: the u-deficits
    combine as (sum d^p)^(1/p), v and w linearly as in MixedSum. p = 2 is
    MixedSum's root-sum-square, p = 1 the linear sum. Small negative
    u-deficits (wake-edge speedups) are clipped to 0 as MixedSum does."""

    def __init__(self, p: float = 2.0):
        MixedSum.__init__(self)
        self.p = float(p)
        if not 1.0 <= self.p <= 4.0:
            raise ValueError(f"superposition_p must be in [1, 4], got {p!r}")

    def __call__(self, value_jxxx):
        u = value_jxxx[:, 0]
        min_u = u.min() if u.size else 0.0
        if min_u < 0:
            assert min_u > -self.CLIP_SPEEDUP_LIMIT, (
                f"PowerSum only works for deficits - not speedups (min u-deficit {min_u:.3f} m/s)")
            np.maximum(u, 0, out=u)
        return np.array([np.sum(np.power(u, self.p), 0) ** (1.0 / self.p),
                         np.sum(value_jxxx[:, 1], 0), np.sum(value_jxxx[:, 2], 0)])


def _superposition_model(superposition: str, p: float):
    """The superposition model instance: the named one, or with superposition
    'mixed' and p != 2, MixedSum's exponent changed (_PowerSum)."""
    cls = _lookup(SUPERPOSITION_MODELS, superposition, "superposition")
    if cls is MixedSum and float(p) != 2.0:
        return _PowerSum(float(p))
    return cls()


def _particle_spatial_filter(radius_scale: float):
    """The disk a particle's advection velocity is averaged over: the CGI nodes
    over ``radius_scale`` x the local wake radius. 1.0 is dynamiks' and
    WindGym's setup (the whole wake: the wake moves with the mean flow over
    its own extent, Larsen 2008); 0 collapses the nodes onto the particle (a
    point sample). dynamiks multiplies the node coordinates by the wake
    radius, so scaling the nodes scales the disk."""
    s = float(radius_scale)
    if not 0.0 <= s <= 2.0:
        raise ValueError(f"meandering_radius_scale must be in [0, 2], got {radius_scale!r}")
    f = CGIRotorAvg(PARTICLE_SPATIAL_AVG_N)
    f.nodes_x = np.asarray(f.nodes_x, float) * s
    f.nodes_y = np.asarray(f.nodes_y, float) * s
    return f


def _ScalarTIPyWakeDeficitGenerator(*args, **kwargs):
    """dynamiks' PyWakeDeficitGenerator for an axisymmetric Gaussian deficit, with
    two changes:

    * the turbine's TI is read as jDWMAinslieGenerator reads it
      (``rotor_avg_ti()[0]``). dynamiks' TISensor returns every turbine's TI
      whatever the index, so the stock wrapper's reshape to (1, 1, 1) fails for
      more than one turbine; the Ainslie generator takes element 0 -- the first
      turbine's TI for every wake -- and this does the same. (Bastankhah's
      deficit does not use TI.)
    * the radial derivative dUdr, which dynamiks needs at every particle
      evaluation, is analytic -- d(deficit)/dr = -(r / sigma^2) deficit for a
      Gaussian -- instead of autograd through py_wake, which took ~2/3 of an
      env step.
    * the YAWED Gaussian (``yaw_ct_exp``, ``yaw_width_exp``; both 0 = off, the
      stock axisymmetric profile): the thrust in the deficit takes an extra
      cos(yaw)**yaw_ct_exp on top of the turbine's own Ct(yaw), and the lateral
      onset width takes cos(yaw)**yaw_width_exp (Bastankhah & Porte-Agel 2016:
      the rotor's frontal projection, exponent 1) while the vertical width does
      not -- an elliptical Gaussian, deficit_centre = 2 a(ct D^2 / (8 sy sz)).
      py_wake's axisymmetric profile keeps a yawed wake as strong as an
      unyawed one (the Ct-dependent onset width cancels the cos(yaw) thrust
      drop); the TUM G1's wake at 30 deg is 25 % weaker and 30 % narrower.
      The yaw is the particle's own at emission (yaw_ilk).
    """
    from dynamiks.dwm.particle_deficit_profiles.pywake_deficit_wrapper import (
        PyWakeDeficitGenerator,
        PyWakeDeficitProfile,
    )

    class _Profile(PyWakeDeficitProfile):
        def _yawed(self):
            g = self.generator
            return g.yaw_ct_exp != 0.0 or g.yaw_width_exp != 0.0

        def _ellipse(self, x):
            """The yawed Gaussian at downstream distance x [m]: the thrust in the
            deficit and the lateral / vertical sigma [m]."""
            g, dm = self.generator, self.generator.deficitModel
            c = float(np.cos(np.deg2rad(float(np.ravel(self.kwargs["yaw_ilk"])[0]))))
            ct = float(np.ravel(self.kwargs["ct_ilk"])[0]) * c**g.yaw_ct_exp
            D = float(self.kwargs["D_src_il"][0, 0])
            eps = float(np.ravel(dm.epsilon_ilk(np.reshape(ct, (1, 1, 1))))[0]) * D
            k = float(np.ravel(dm.k_ilk())[0])
            return ct, D, k * x + eps * c**g.yaw_width_exp, k * x + eps

        def _get_deficit_norm(self, rel_x, rel_y, rel_z):
            if not self._yawed():
                return PyWakeDeficitProfile._get_deficit_norm(self, rel_x, rel_y, rel_z)
            x, y, z = (np.asarray(v, dtype=float) for v in (rel_x, rel_y, rel_z))
            ct, D, sy, sz = self._ellipse(x)
            centre = np.minimum(1.0, 2.0 * self.generator.deficitModel.ct2a(ct * D**2 / (8.0 * sy * sz)))
            d = centre * np.exp(-0.5 * (y / sy) ** 2 - 0.5 * (z / sz) ** 2)
            return np.array([d, np.zeros_like(d), np.zeros_like(d)])

        def _get_UVW_dUdr(self, rel_x, rel_y, rel_z, compute_gradient=True):
            deficit = self._get_deficit_norm(rel_x, rel_y, rel_z)
            if not compute_gradient:
                return deficit, None
            x = np.asarray(rel_x, dtype=float)
            y, z = np.asarray(rel_y, float), np.asarray(rel_z, float)
            if self._yawed():
                _, _, sy, sz = self._ellipse(x)
                # |grad| of the elliptical Gaussian [1/m], times R as dynamiks normalises it
                return deficit, np.sqrt((y / sy**2) ** 2 + (z / sz**2) ** 2) * deficit[0] * self.R
            sigma = self.generator.deficitModel.sigma_ijlk(
                dw_ijlk=x.reshape(1, -1, 1, 1), **self.kwargs
            ).reshape(x.shape)
            r = np.sqrt(y**2 + z**2)
            # |grad| of the normalised deficit [1/m], times R as dynamiks normalises it
            return deficit, r / sigma**2 * deficit[0] * self.R

    class _Generator(PyWakeDeficitGenerator):
        def __init__(self, *a, yaw_ct_exp=0.0, yaw_width_exp=0.0, **kw):
            PyWakeDeficitGenerator.__init__(self, *a, **kw)
            self.yaw_ct_exp = float(yaw_ct_exp)
            self.yaw_width_exp = float(yaw_width_exp)

        def get_kwargs_func(self):
            stock = PyWakeDeficitGenerator.get_kwargs_func(self)

            def get_kwargs(windTurbine):
                kw = stock(windTurbine)
                kw["TI"] = np.atleast_1d(kw["TI"])[0]
                return kw

            return get_kwargs

        def new_particle_deficit(self, *a, **kw):
            profile = PyWakeDeficitGenerator.new_particle_deficit(self, *a, **kw)
            profile.__class__ = _Profile  # same state, analytic dUdr
            return profile

    return _Generator(*args, **kwargs)


def make_dwm(
    *,
    site,
    windTurbines,
    wind_direction,
    dt,
    addedTurbulenceModel,
    k1: float | None = None,
    k2: float | None = None,
    d_particle: float = D_PARTICLE,
    interpolation: str = "pchip",
    lateral_cutoff=None,
    viscosity_model: str = "keck",
    superposition: str = "mixed",
    x_speed: str = "particle",
    r_max: float = AINSLIE_R_MAX,
    n_r: int = AINSLIE_N_R,
    dx: float = AINSLIE_DX,
    boundary_condition: str = "madsen",
    deflection_c: float = DEFLECTION_C,
    deflection_c_pos: float | None = None,
    meandering_d: float | None = None,
    meandering_radius_scale: float = 1.0,
    wake_drift_deg: float = 0.0,
    superposition_p: float = 2.0,
    deficit_model: str = "ainslie",
    gauss_k: float = BASTANKHAH_K,
    gauss_ceps: float = BASTANKHAH_CEPS,
    yaw_ct_exp: float = 0.0,
    yaw_width_exp: float = 0.0,
    secondary_steering: bool = False,
    ss_gain: float = 1.0,
    ss_decay_D: float = SS_DECAY_D,
) -> DWMFlowSimulation:
    """Assemble a DWMFlowSimulation under the calibrated setup.

    The caller drives it via ``fs.step()`` in a time loop.

    Every keyword after ``addedTurbulenceModel`` defaults to the calibrated
    setup, so existing call sites stay unchanged. Override them at episode
    reset to do domain randomization or robustness evaluation:

    - ``k1``/``k2`` (``None`` = the selected viscosity model's coefficients;
      keck -> the calibrated ``K1``/``K2``), ``d_particle``
    - ``viscosity_model`` / ``superposition`` / ``x_speed``: categorical,
      resolved through ``VISCOSITY_MODELS`` / ``SUPERPOSITION_MODELS`` /
      ``X_SPEEDS``
    - ``r_max`` / ``n_r`` / ``dx``: Ainslie solver grid
    - ``boundary_condition``: Ainslie near-wake boundary condition, resolved
      through ``BOUNDARY_CONDITIONS``
    - ``deflection_c``: wake-deflection strength (dynamiks hardcodes 0.4);
      ``deflection_c_pos``: the same for positive flow-relative yaw (``None`` =
      symmetric)
    - ``meandering_d``: ``CutOffFrq(d)`` meandering filter; ``None`` = none;
      ``meandering_radius_scale``: the fraction of the wake radius a particle's
      advection velocity is averaged over (1.0 = the whole wake)
    - ``deficit_model`` (``"ainslie"`` / ``"bastankhah"``), ``gauss_k``,
      ``gauss_ceps``: the particle wake deficit; see ``DEFICIT_MODELS``;
      ``yaw_ct_exp``, ``yaw_width_exp``: the yawed Gaussian (weaker and
      laterally narrower with yaw; 0 = off)
    - ``secondary_steering`` (bool), ``ss_gain``, ``ss_decay_D``: wrap the
      deficit generator in dynamiks' ``SecondarySteeringGenerator`` so yawed
      wakes carry the vortex pair's lateral velocity and deflect downstream
      wakes; off by default (the calibrated setup)

    See ``DWM_PARAM_SPEC`` for the per-key documentation.

    Turbines that expose ``attach_to`` (``ParamPyWakeWindTurbines``) are
    attached to the new flow simulation here, so both the agent and the
    baseline farm get their actuator step handler through this one builder.
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
    # HAWC2: the property reads _rotor_position, unset before HAWC2 starts
    except AttributeError:
        _en = np.asarray(windTurbines.positions_east_north, dtype=float)
    _x = get_xyz(_en, wind_direction)[0]
    _D = np.atleast_1d(windTurbines.diameter()).astype(float)
    _desired = max(float(_x.max() - _x.min()) * 1.2, 15.0 * float(_D.max()))
    n_particles = max(int(np.ceil(_desired / (d_particle * float(_D.min())))), 10)

    deficit_model = str(deficit_model).lower()
    if deficit_model not in DEFICIT_MODELS:
        raise ValueError(
            f"deficit_model must be one of {list(DEFICIT_MODELS)}, got {deficit_model!r}"
        )
    if deficit_model == "bastankhah":
        from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit

        # Deficit relative to the rotor's own (possibly waked) inflow, as the
        # Ainslie generator scales it; NoProjection as the Ainslie path.
        deficit_gen = _ScalarTIPyWakeDeficitGenerator(
            BastankhahGaussianDeficit(
                k=float(gauss_k), ceps=float(gauss_ceps), use_effective_ws=True
            ),
            scale_with_freestream=False,
            projectionModel=NoProjection(),
            yaw_ct_exp=float(yaw_ct_exp),
            yaw_width_exp=float(yaw_width_exp),
        )
    else:
        deficit_gen = None
    deficit_gen = deficit_gen or jDWMAinslieGenerator(
        boundaryConditionModel=_lookup(
            BOUNDARY_CONDITIONS, boundary_condition, "boundary_condition"
        ),
        viscosity_model=_make_viscosity_model(viscosity_model, k1, k2),
        solver=implicit(),
        projectionModel=NoProjection(),
        r_max=float(r_max),
        n_r=int(n_r),
        dx=float(dx),
    )
    if not isinstance(secondary_steering, (bool, np.bool_)):
        raise TypeError(
            f"secondary_steering must be a bool, got {secondary_steering!r}"
        )
    if secondary_steering:
        # Wraps either generator; r_max / projectionModel read through to it.
        deficit_gen = SecondarySteeringGenerator(
            deficit_gen, gain=float(ss_gain), decay_D=float(ss_decay_D)
        )

    motion_kwargs = dict(
        x_speed=_lookup(X_SPEEDS, x_speed, "x_speed"),
        temporal_filter=None
        if meandering_d is None
        else CutOffFrq(float(meandering_d)),
        spatial_filter=_particle_spatial_filter(meandering_radius_scale),
        include_wakes=True,
        include_own_wake=False,
    )
    # The stock class at dynamiks' own coefficient, so the default path is
    # upstream's code rather than a copy of it.
    if float(deflection_c) == DEFLECTION_C and deflection_c_pos is None and not float(wake_drift_deg):
        particle_motion = HillVortexParticleMotion(**motion_kwargs)
    else:
        particle_motion = ScaledHillVortexParticleMotion(
            c=deflection_c,
            c_pos=None if deflection_c_pos is None else float(deflection_c_pos),
            drift_deg=float(wake_drift_deg),
            **motion_kwargs,
        )

    fs = DWMFlowSimulation(
        site=site,
        windTurbines=windTurbines,
        particleDeficitGenerator=deficit_gen,
        particleMotionModel=particle_motion,
        d_particle=d_particle,
        n_particles=n_particles,
        addedTurbulenceModel=addedTurbulenceModel,
        superpositionModel=_superposition_model(superposition, superposition_p),
        wind_direction=wind_direction,
        dt=dt,
        # Speedups, not part of the LES calibration (which ran pchip / no
        # cutoff, the defaults here): linear centerline interpolation and the
        # lateral interaction cutoff. WindFarmEnv passes its own settings
        # (default linear / 1.5, as in the Stage-6 sweeps). Only forwarded
        # when the installed dynamiks accepts them; see _dwm_speed_kwargs.
        **_dwm_speed_kwargs(interpolation, lateral_cutoff),
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
            f"Unknown hawc2_yaw_mode: {mode!r} (expected 'bearing2_slot' or 'yaw_tilt')"
        )
