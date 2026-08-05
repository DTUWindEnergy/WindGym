"""Derating helpers: validation and HAWC2 sensor wiring.

Everything the env needs to know about derating as a *domain* lives here:

- ``check_turbine_supports_derating`` / ``check_htc_supports_derating`` —
  constructor-time validation that the turbine model (PyWake) or htc file
  (HAWC2/DTUWEC) can actually take runtime derate commands.
- ``add_hawc2_derate_sensor`` — wires the exposed ``derate`` sensor onto a
  HAWC2 turbine set, absorbing the d <-> dr% = (1 - d) * 100 mapping between
  the env's derate fraction and the DTUWEC controller's percentage channel.

The contract for PyWake-side turbines is simple: the turbine's
``powerCtFunction`` must accept a ``derate`` input, e.g. a surrogate that
tabulates power/ct over (ws, yaw, derate) directly — a HAWCStab2-generated
PowerCtNDTabular with a 'derate' dimension (see hawcpowerctcurvegenerator).
Such turbines handle derating natively; this module only validates that the
input exists and wires the HAWC2 derate sensor.
"""

from __future__ import annotations
import os
import numpy as np


def check_htc_supports_derating(htc_path, derate_reference):
    """Validate that the htc at *htc_path* can take runtime derate commands.

    Looks for the type2_dll section whose filename contains "derate" (the
    DTUWEC derate controller) and checks its init constants:

    - constant 79 (derate strategy) must be nonzero, else derating is off.
    - constant 80 (derate percentage) must be negative: negative activates
      runtime derating via controller input 18 ("general variable 2");
      a fixed percentage would ignore the env's commands.
    - constant 104 (derate reference mode) must match *derate_reference*:
      1 = fraction of currently available power ("available"),
      0 = fraction of rated power ("rated"). Absent means 0, the
      controller default.
    """
    from wetb.hawc2.htc_file import HTCFile

    # explicit modelpath skips wetb's autodetection, which requires the
    # model's input folders on disk — irrelevant for parsing constants
    htc = HTCFile(str(htc_path), modelpath=os.path.dirname(str(htc_path)))
    dll = htc.get("dll")
    constants = None
    if dll is not None:
        for key in dll.keys():
            if not key.startswith("type2_dll"):
                continue
            sec = dll[key]
            fn = sec.get("filename")
            fname = str(fn.values[0]) if fn is not None else ""
            if "derate" not in fname.lower():
                continue
            init = sec.get("init")
            constants = {}
            if init is not None:
                for ckey in init.keys():
                    if ckey.startswith("constant"):
                        vals = init[ckey].values
                        constants[int(vals[0])] = float(vals[1])
            break
    if constants is None:
        raise ValueError(
            "derate_action=True with a HAWC2 turbine requires an htc whose "
            "dll section loads the DTUWEC derate controller (a type2_dll "
            "with 'derate' in its filename), e.g. the shipped "
            "examples/HawcFiles/htc/DTU10mw_derate.htc. "
            f"No such controller found in {htc_path}."
        )
    if constants.get(79, 0.0) == 0.0:
        raise ValueError(
            "The derate controller in the htc has constant 79 = 0 "
            "(derating disabled). Set it to a derate strategy (e.g. 2)."
        )
    if constants.get(80, 0.0) >= 0.0:
        raise ValueError(
            "The derate controller in the htc has constant 80 >= 0 (fixed "
            "derate percentage). Runtime derate commands need constant 80 "
            "< 0, which activates reading controller input 18 every step."
        )
    mode = constants.get(104, 0.0)  # absent = 0, the controller default
    expected = 1.0 if derate_reference == "available" else 0.0
    if mode != expected:
        raise ValueError(
            f"derate_reference='{derate_reference}' needs htc constant 104 "
            f"= {expected:.0f}, but the htc has {mode:.0f} "
            "(0 = fraction of rated power, 1 = fraction of available "
            "power). Use a matching htc or change derate_reference."
        )


def check_turbine_supports_derating(turbine):
    """Validate that *turbine* can be derated (powerCtFunction accepts 'derate')."""
    pctf = turbine.powerCtFunction
    inputs = list(getattr(pctf, "required_inputs", [])) + list(
        getattr(pctf, "optional_inputs", [])
    )
    if "derate" not in inputs:
        raise ValueError(
            "derate_action=True requires a turbine whose powerCtFunction "
            "accepts a 'derate' input, e.g. built from a derating surrogate "
            "(PowerCtNDTabular with a 'derate' dimension, generated with "
            "hawcpowerctcurvegenerator — see 'examples/Example 5 Derating "
            "PyWake surrogate.ipynb')."
        )


def add_hawc2_derate_sensor(wts, n_turb):
    """Register the exposed 'derate' sensor on a HAWC2 turbine set *wts*.

    The DTUWEC derate controller reads "general variable 2" as the derate
    percentage dr% (100 = no derating), so the sensor absorbs the
    d <-> dr% = (1 - d) * 100 mapping: env-side the derate is a fraction d
    in [0, 1], same as the PyWake turbine.
    """
    wts.add_sensor(
        name="derate_getter",
        getter="general variable 2 100;",
        expose=False,
    )
    wts.add_sensor(
        "derate",
        getter=lambda wt: 1.0 - np.ravel(wt.sensors.derate_getter) / 100.0,
        # .tolist(): MultiH2Lib only distributes plain lists across
        # its per-turbine subprocesses (same as the yaw setter).
        setter=lambda wt, value: wt.h2.set_variable_sensor_value(
            2,
            (100.0 * (1.0 - np.asarray(value, dtype=np.float64))).tolist(),
        ),
        expose=True,
    )
    wts.sensors.derate = np.zeros(n_turb)
