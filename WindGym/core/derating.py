"""LEGACY derating via power-curve inversion.

This module retrofits derating onto tabular (PowerCtTabular) turbines by
inverting the power curve: P(ws_op) = (1 - derate) * P(ws). It approximates
the derated Ct as Ct(ws_op), i.e. it assumes the turbine tracks its normal
operating curve — it does NOT know how a real controller would re-optimise
pitch/TSR at reduced power.

Preferred approach: build the turbine from a derating surrogate that tabulates
power/ct over (ws, yaw, derating) directly — e.g. a HAWCStab2-generated
PowerCtNDTabular with a 'derate' dimension (see hawcpowerctcurvegenerator).
Such turbines accept 'derate' natively and need nothing from this module;
WindFarmEnv only requires that the powerCtFunction accepts a 'derate' input.

Use this module only when no surrogate exists for the turbine (e.g. SWT23):

    from WindGym.core.derating import add_derating_to_turbine
    turbine = add_derating_to_turbine(SWT23())
"""

from __future__ import annotations
import numpy as np
from py_wake.wind_turbines.power_ct_functions import AdditionalModel
from py_wake.utils.model_utils import fix_shape


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
