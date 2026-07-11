"""HAWC2-backend derating check: 2 DTU 10 MW inline, upwind turbine derated.

Layout (wd = 270 -> flow along +x, D = 178.3 m):

    WT0 front (0, 0)  d = 0 -> 0.5 -> 0      WT1 back (5D, 0)  in WT0's wake

WindFarmEnv runs the shipped DTUWEC derate model (HawcFiles/htc/DTU10mw_derate.htc:
strategy 2, runtime derating via "general variable 2", avail-mode dr) through the
dynamiks HAWC2 backend. WT0 is derated to 50% of available power for 350-600 s via
the env's derate action; being below rated, the hold power should sit near 50% of
WT0's own pre-derate power while WT1 gains from the weakened wake.

Phase timing (from the validated hawc2rosco minimal example): WT1's deep wake needs
~100 s advection (5D at 10 m/s) + ~200 s to (re)settle after any change at WT0, so
each phase gets 350 s and only the settled tail is judged. Unlike that laminar
check, the inflow here is turbulent (TI 6%): "available power" floats with the
wind (P ~ V^3), so WT0's hold is judged against a band around the 0.5 ideal and
the noise-dominated deep-wake WT1 quantities are REPORTed rather than FAILed.

NOTE: the shipped controller/servo binaries in examples/HawcFiles/control/ are
Linux-only .so files (fork build of DTU BasicDTUController, branch
"runtime-derating"); there is no derate .dll for Windows.

NOTE: h2lib uses the multiprocessing *spawn* start method, so all env usage MUST
stay behind `if __name__ == "__main__":` (each child re-imports this module).

Run (writes hawc2_derating_2wt.png + hawc2_derating_2wt.npz; takes ~10-30 min):

    python examples/hawc2_derating_2wt.py [--replot]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

# The HAWC2 shared library needs the Intel Fortran runtime shipped inside the
# active environment (site lib dir); h2lib does not add it to the loader path.
os.environ["LD_LIBRARY_PATH"] = (
    sys.prefix + "/lib" + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)

import numpy as np  # noqa: E402

HTC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "HawcFiles", "htc", "DTU10mw_derate.htc"
)
SPACING_D = 5.0  # streamwise spacing [D]
WSP = 10.0  # mean wind speed [m/s] (below rated: WT1 has headroom to respond)
TI = 0.06
DERATE = 0.5  # WT0 derate fraction during the hold
T_ON, T_OFF, T_END = 350, 600, 950  # env steps (1 s each)
WINDOW = 150  # steady tail [s] of each phase used by the checks
# WT0's hold as a fraction of its own undisturbed power: ideally 1 - DERATE, but
# avail-mode holds a fraction of *instantaneous* available power, which floats
# with the turbulent wind -- a single seed only supports a band around the ideal
HOLD_BAND = (0.35, 0.70)
RECOVERY_TOL = 0.15  # WT0 (unwaked) recovery tolerance
PLOT_PATH = "hawc2_derating_2wt.png"
DATA_PATH = "hawc2_derating_2wt.npz"

CONFIG = {
    "yaw_action": False,  # derate-only agent
    "derate_action": True,
    "derate_reference": "available",
    "derate_method": "absolute",
    "derate_min": 0.0,
    "derate_max": 1.0,
    "ActionMethod": "wind",
    "farm": {"yaw_min": -30, "yaw_max": 30},
    "wind": {
        "ws_min": WSP,
        "ws_max": WSP,
        "TI_min": TI,
        "TI_max": TI,
        "wd_min": 270.0,
        "wd_max": 270.0,
    },
    "act_pen": {"action_penalty": 0.0, "action_penalty_type": "change"},
    "power_def": {"Power_reward": "Power_avg", "Power_avg": 5, "Power_scaling": 1.0},
    "mes_level": {
        "turb_ws": True,
        "turb_wd": False,
        "turb_TI": False,
        "turb_power": True,
        "farm_ws": False,
        "farm_wd": False,
        "farm_TI": False,
        "farm_power": False,
    },
    "ws_mes": {
        "ws_current": True,
        "ws_rolling_mean": False,
        "ws_history_N": 1,
        "ws_history_length": 10,
        "ws_window_length": 10,
    },
    "wd_mes": {
        "wd_current": True,
        "wd_rolling_mean": False,
        "wd_history_N": 1,
        "wd_history_length": 10,
        "wd_window_length": 10,
    },
    "yaw_mes": {
        "yaw_current": True,
        "yaw_rolling_mean": False,
        "yaw_history_N": 1,
        "yaw_history_length": 10,
        "yaw_window_length": 10,
    },
    "power_mes": {
        "power_current": True,
        "power_rolling_mean": False,
        "power_history_N": 1,
        "power_history_length": 10,
        "power_window_length": 10,
    },
}


def derate_schedule(t):
    """WT0 derate fraction commanded at env step *t* (1 s per step)."""
    return DERATE if T_ON <= t < T_OFF else 0.0


def run():
    from py_wake.examples.data.dtu10mw import DTU10MW

    from WindGym import WindFarmEnv

    turbine = DTU10MW()
    d = turbine.diameter()
    env = WindFarmEnv(
        turbine=turbine,  # pywake surrogate still required alongside HTC_path
        x_pos=[0.0, SPACING_D * d],
        y_pos=[0.0, 0.0],
        config=CONFIG,
        HTC_path=HTC_PATH,
        backend="dynamiks",
        dt_sim=1,
        dt_env=1,
        max_time_steps=T_END,
        burn_in_passthroughs=1,
        # MannGenerate makes the box at reset; if that is too slow / RAM-heavy on
        # your machine, pre-generate boxes and use turbtype="MannLoad" + TurbBox=...
        turbtype="MannGenerate",
        reset_init=False,
        seed=0,
    )

    t_hist, power, cmd, measured = [], [], [], []
    readback = {}
    try:
        print(f">>> reset: spawning 2 HAWC2 processes ({HTC_PATH})")
        env.reset(seed=0)
        print(f">>> running 0-{T_END} s (WT0 at d={DERATE:g} for {T_ON}-{T_OFF} s)")
        for t in range(T_END):
            d0 = derate_schedule(t)
            # absolute derate_method maps action a in [-1, 1] affinely onto
            # [derate_min, derate_max] = [0, 1], so the inverse is a = 2 d - 1
            action = np.array([2.0 * d0 - 1.0, -1.0], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)

            t_hist.append(t)
            power.append(np.asarray(info["Power pr turbine agent"], dtype=float))
            cmd.append(np.asarray(info["derate agent"], dtype=float))
            measured.append(np.asarray(info["derate measured"], dtype=float))
            # getter-readback: the derate sensor re-reads "general variable 2"
            # from HAWC2 and maps it back to a fraction
            if t in (T_ON - WINDOW // 2, T_OFF - WINDOW // 2, T_END - WINDOW // 2):
                readback[t] = np.ravel(np.asarray(env.wts.sensors.derate, dtype=float))
                print(f"    t={t:4d} s  sensor readback d = {readback[t]}")
            if t % 50 == 0:
                p = power[-1] / 1e6
                print(
                    f"    t={t:4d} s  d(WT0)={d0:4.2f}  Pel [MW]: "
                    + "  ".join(f"{v:5.2f}" for v in p)
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    return (
        np.asarray(t_hist),
        np.asarray(power) / 1e6,
        np.asarray(cmd),
        np.asarray(measured),
        readback,
    )


def summarize(t, power, cmd, measured, readback):
    """Steady-state table + PASS/FAIL. The wake gain at WT1 is a finding -> REPORT."""
    windows = {
        "pre-derate": (T_ON - WINDOW, T_ON),
        "derate hold": (T_OFF - WINDOW, T_OFF),
        "recovery": (T_END - WINDOW, T_END),
    }
    P = {
        name: np.array([power[(t >= w0) & (t < w1), i].mean() for i in range(2)])
        for name, (w0, w1) in windows.items()
    }

    print(f"\n>>> mean electrical power [MW] (last {WINDOW} s of each phase):")
    print(f"    {'window':<12}  {'WT0':>6}  {'WT1':>6}")
    for name, p in P.items():
        print(f"    {name:<12}  {p[0]:6.2f}  {p[1]:6.2f}")

    all_ok = True

    def check(label, ok):
        nonlocal all_ok
        all_ok &= ok
        print(f"    {label}: {'PASS' if ok else 'FAIL'}")

    print("\n>>> checks:")
    pre, hold, rec = P["pre-derate"], P["derate hold"], P["recovery"]
    # avail-mode dr below rated: available ~ what WT0 would be producing, so the
    # hold should sit near (1 - d) of WT0's own undisturbed power (NOT of rated)
    baseline0 = 0.5 * (pre[0] + rec[0])
    ratio = hold[0] / baseline0
    check(
        f"WT0 hold / undisturbed power = {ratio:.2f} within "
        f"[{HOLD_BAND[0]:.2f}, {HOLD_BAND[1]:.2f}] (ideal {1 - DERATE:.2f})",
        HOLD_BAND[0] <= ratio <= HOLD_BAND[1],
    )
    check(
        f"WT0 recovers to pre-derate power after {T_OFF} s (within {RECOVERY_TOL:.0%})",
        abs(rec[0] - pre[0]) / pre[0] < RECOVERY_TOL,
    )
    in_hold = (t >= T_OFF - WINDOW) & (t < T_OFF)
    check(
        "'derate measured' tracks the command during the hold",
        np.allclose(measured[in_hold, 0], DERATE, atol=0.02)
        and np.allclose(cmd[in_hold, 0], DERATE, atol=1e-6),
    )
    hold_rb = readback.get(T_OFF - WINDOW // 2)
    check(
        "sensor getter reads the commanded derate back from HAWC2",
        hold_rb is not None and np.allclose(hold_rb, [DERATE, 0.0], atol=0.02),
    )
    dP1 = hold[1] - pre[1]
    print(
        f"    REPORT: WT1 {'gains' if dP1 > 0 else 'loses'} {dP1:+.2f} MW "
        f"({dP1 / pre[1]:+.1%}) during the hold (expect a gain: weaker wake)"
    )
    print(
        f"    REPORT: WT1 recovery {rec[1]:.2f} vs pre-derate {pre[1]:.2f} MW "
        "(deep-wake signal, noise-dominated at operating TI -> not a check)"
    )
    return all_ok


def plot(t, power, cmd, measured):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    for i, (label, color) in enumerate(
        [("WT0 front (derated)", "#D55E00"), ("WT1 back (waked)", "#0072B2")]
    ):
        axes[0].plot(t, power[:, i], color=color, lw=1.5, label=label)
    axes[1].plot(t, cmd[:, 0], color="0.3", ls=":", lw=1.6, label="WT0 derate command")
    axes[1].plot(
        t, measured[:, 0], color="#D55E00", lw=1.5, label="WT0 derate measured"
    )
    for ax in axes:
        ax.axvspan(T_ON, T_OFF, color="0.85", alpha=0.5, zorder=0)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0)
    axes[0].set_ylabel("Electrical power [MW]")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Derate fraction d [-]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(frameon=False)
    fig.suptitle(
        f"2x DTU 10 MW inline through WindFarmEnv (HAWC2 backend) -- WT0 derated "
        f"to {1 - DERATE:.0%} of available power for {T_ON}-{T_OFF} s "
        f"({WSP:g} m/s, {SPACING_D:g}D)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f">>> plot saved to {PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replot",
        action="store_true",
        help="skip the simulation; summarize/plot from the saved .npz",
    )
    args = parser.parse_args()

    if args.replot:
        saved = np.load(DATA_PATH)
        t, power, cmd, measured = (
            saved["t"],
            saved["power"],
            saved["cmd"],
            saved["measured"],
        )
        readback = dict(zip(saved["readback_t"].tolist(), saved["readback_v"]))
    else:
        t, power, cmd, measured, readback = run()
        np.savez(
            DATA_PATH,
            t=t,
            power=power,
            cmd=cmd,
            measured=measured,
            readback_t=np.array(list(readback.keys())),
            readback_v=np.array(list(readback.values())),
        )
        print(f">>> data saved to {DATA_PATH} (replot with --replot)")

    all_ok = summarize(t, power, cmd, measured, readback)
    plot(t, power, cmd, measured)
    print(f"\n>>> RESULT: {'ALL PASS' if all_ok else 'FAILURES -- see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
