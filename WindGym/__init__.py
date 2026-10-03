# netCDF4 must be imported before h5py (pulled in below via agent_eval ->
# wetb.gtsdf). In envs where both are pip wheels with their own bundled
# libhdf5 (LUMI packed env: netCDF4 1.7.4 + h5py 3.15.1, both HDF5 1.14.6)
# the copy loaded second cannot open files and every netCDF4 read fails with
# "NetCDF: HDF error". Until windgym dfc0e17 an unused py_wake import in
# agent_eval.py happened to enforce this order (Stage-9 job 22497558, 2026-10-02).
import netCDF4  # noqa: F401  isort: skip

from .agent_eval import AgentEval
from .agent_eval import eval_single_fast as AgentEvalFast
from .Agents import PyWakeAgent
from .farm_eval import FarmEval
from .wind_env_multi import WindFarmEnvMulti
from .wind_farm_env import WindFarmEnv
from . import presets

try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
