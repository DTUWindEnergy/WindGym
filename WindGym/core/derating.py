"""Derating helpers: validation, HAWC2 sensor wiring, and the legacy wrapper.

Everything the env needs to know about derating as a *domain* lives here:

- ``check_turbine_supports_derating`` / ``check_htc_supports_derating`` —
  constructor-time validation that the turbine model (PyWake) or htc file
  (HAWC2/DTUWEC) can actually take runtime derate commands.
- ``add_hawc2_derate_sensor`` — wires the exposed ``derate`` sensor onto a
  HAWC2 turbine set, absorbing the d <-> dr% = (1 - d) * 100 mapping between
  the env's derate fraction and the DTUWEC controller's percentage channel.
- ``add_derating_to_turbine`` — LEGACY power-curve-inversion retrofit for
  tabular turbines (see below).

Legacy derating via power-curve inversion
-----------------------------------------
``add_derating_to_turbine`` retrofits derating onto tabular (PowerCtTabular)
turbines by inverting the power curve: P(ws_op) = (1 - derate) * P(ws). It
approximates the derated Ct as Ct(ws_op), i.e. it assumes the turbine tracks
its normal operating curve — it does NOT know how a real controller would
re-optimise pitch/TSR at reduced power.

Preferred approach: build the turbine from a derating surrogate that tabulates
power/ct over (ws, yaw, derating) directly — e.g. a HAWCStab2-generated
PowerCtNDTabular with a 'derate' dimension (see hawcpowerctcurvegenerator).
Such turbines accept 'derate' natively and need nothing from this module;
WindFarmEnv only requires that the powerCtFunction accepts a 'derate' input.

Use the legacy wrapper only when no surrogate exists for the turbine (e.g.
SWT23):

    from WindGym.core.derating import add_derating_to_turbine
    turbine = add_derating_to_turbine(SWT23())
"""

from __future__ import annotations
import os
import numpy as np
from py_wake.wind_turbines.power_ct_functions import AdditionalModel
from py_wake.utils.model_utils import fix_shape


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
            "(PowerCtNDTabular with a 'derate' dimension). Alternatively, "
            "retrofit a tabular turbine with the legacy power-curve-inversion "
            "wrapper: WindGym.core.derating.add_derating_to_turbine(turbine)."
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


class DeratingModel(AdditionalModel):
    """Below-rated pitch derating via power-curve inversion.

    Adds 'derate' as an optional input to any PowerCtTabular-based turbine.
    When derate[i] > 0, the operating wind speed ws_op[i] is found by solving:

        P(ws_op) = (1 - derate) * P(ws)

    so that both power AND Ct follow the turbine's own aerodynamics rather
    than an arbitrary scaling.  This is physically correct because ws_op < ws
    moves the turbine to a lower point on the P(ws) and Ct(ws) curves,
    producing a smaller DWM deficit downstream.

    Parameters
    ----------
    ws_arr : array_like
        Wind speed values from the turbine's power curve (m/s).
    power_arr : array_like
        Corresponding power values (arbitrary unit — only the curve shape
        matters for the inversion, not the absolute scale).
    """

    def __init__(self, ws_arr: np.ndarray, power_arr: np.ndarray):
        AdditionalModel.__init__(
            self,
            input_keys=["ws", "derate"],
            optional_inputs=["derate"],
            output_keys=["power", "ct"],
        )
        self._build_inverse(np.asarray(ws_arr, float), np.asarray(power_arr, float))

    def _build_inverse(self, ws: np.ndarray, power: np.ndarray) -> None:
        p_max = power.max()
        # Keep only the strictly-increasing portion up to the first rated point.
        # Above-rated, power is flat so there is no unique inverse.
        first_rated = int(np.argmax(power >= p_max))
        self._ws_inc = ws[: first_rated + 1].copy()
        self._pow_inc = power[: first_rated + 1].copy()

    def __call__(self, f, ws, derate=None, **kwargs):
        if derate is None:
            return f(ws, **kwargs)

        derate = np.clip(
            np.asarray(fix_shape(derate, ws), dtype=float),
            0.0,
            1.0,
        )

        if np.all(derate == 0.0):
            return f(ws, **kwargs)

        ws_arr = np.asarray(ws, dtype=float)
        frac = 1.0 - derate

        # Target power in raw tabular units (unit-agnostic — scale cancels in ratio)
        target_p = frac * np.interp(
            ws_arr.ravel(),
            self._ws_inc,
            self._pow_inc,
        ).reshape(ws_arr.shape)

        # Invert power curve: target_p → ws_op
        ws_op = np.interp(
            target_p.ravel(),
            self._pow_inc,
            self._ws_inc,
        ).reshape(ws_arr.shape)

        # derate == 0: keep original ws (avoids rated-plateau ambiguity where
        #              the inverse would return ws_rated instead of ws).
        # derate == 1: force ws_op = 0 so Ct = 0 (full shutdown, no wake).
        #              Without this, np.interp clamps to ws_inc[0] = cut-in
        #              speed, which still produces a non-zero Ct and a wake.
        ws_op = np.where(derate == 0.0, ws_arr, ws_op)
        ws_op = np.where(derate >= 1.0, 0.0, ws_op)

        return f(ws_op, **kwargs)


def add_derating_to_turbine(turbine):
    """Attach a DeratingModel to *turbine*'s PowerCtTabular powerCtFunction.

    Modifies *turbine* in-place and returns it (for chaining, e.g.
    ``WindFarmEnv(turbine=add_derating_to_turbine(SWT23()), ...)``).
    After this call the turbine accepts an optional 'derate' kwarg in all
    power / ct computations.  Passing derate=0 (default) leaves behaviour
    identical to the unmodified turbine.

    Turbines whose powerCtFunction already accepts a 'derate' input (e.g. a
    surrogate-based PowerCtNDTabular with a derate dimension) are left
    untouched — they handle derating natively.

    Raises TypeError if the turbine's powerCtFunction is not a PowerCtTabular.
    """
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

    pctf = turbine.powerCtFunction
    if "derate" in (list(pctf.required_inputs) + list(pctf.optional_inputs)):
        return turbine
    if not isinstance(pctf, PowerCtTabular):
        raise TypeError(
            f"add_derating_to_turbine requires a PowerCtTabular powerCtFunction, "
            f"got {type(pctf).__name__}"
        )
    # Guard: don't add twice (e.g. across multiple env resets sharing turbine)
    if any(isinstance(m, DeratingModel) for m in pctf.model_lst):
        return turbine
    dm = DeratingModel(pctf.ws_tab, pctf.power_ct_tab[0])
    pctf.model_lst.append(dm)
    pctf.add_inputs(dm.required_inputs, dm.optional_inputs)
    return turbine
