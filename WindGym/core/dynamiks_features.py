"""Runtime probes for dynamiks features beyond the PyPI release.

WindGym runs on PyPI ``dynamiks==0.0.7``. Some features (LES-precursor
inflow, memmapped Mann boxes, wind veer, the DWM interpolation/lateral-cutoff
speed-ups) need a newer dynamiks build, which the ``dev_dynamiks`` branch of
WindGym pins. Each flag below probes the installed dynamiks for one feature;
``require`` turns a missing feature into a clear error at the point of use.
"""

from __future__ import annotations

import importlib.util
import inspect

from dynamiks.dwm import DWMFlowSimulation
from dynamiks.sites._site import MetmastSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField


def _has_kwarg(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


# Memory-saving box loading: MannTurbulenceField.from_netcdf(memmap=True) =
# read-only np.memmap of a float32 .npy sidecar next to the .nc (created once)
# + lazy TI scaling (hipersim >= 0.1.18).
HAS_MEMMAP = _has_kwarg(MannTurbulenceField.from_netcdf, "memmap")

# LES-precursor inflow (turbtype="Precursor").
HAS_PRECURSOR = importlib.util.find_spec("dynamiks.sites.precursor") is not None

# Height-dependent wind veer carried by MetmastSite.
HAS_VEER = _has_kwarg(MetmastSite.__init__, "veer_rate")

# DWMFlowSimulation(interpolation=, lateral_cutoff=) speed-ups.
HAS_INTERP = _has_kwarg(DWMFlowSimulation.__init__, "interpolation")

# Pieces of the LES-calibrated DWM setup (see core/calibrated_values.py).
HAS_MIXED_SUM = importlib.util.find_spec("dynamiks.dwm.superposition") is not None
HAS_TI_SENSOR = importlib.util.find_spec("dynamiks.wind_turbines.ti_models") is not None
HAS_BRANLARD = hasattr(
    importlib.import_module("dynamiks.dwm.added_turbulence_models"), "BranlardScaling"
)


def require(flag: bool, feature: str) -> None:
    """Raise a clear ImportError if the installed dynamiks lacks ``feature``."""
    if not flag:
        raise ImportError(
            f"{feature} needs the dev dynamiks build, which the installed "
            "dynamiks does not provide. Use WindGym's dev_dynamiks branch, "
            "which pins it (see Docs/docs/installation.md)."
        )
