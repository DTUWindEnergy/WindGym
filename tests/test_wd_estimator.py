"""Unit tests for WindGym/core/wd_estimator.py.

The reference implementations here re-derive the filter in plain numpy with
the SAME math as the offline T1 probe (paper-derating
wd_estimation/t1_probe.py: circ_ewma_deg + circ_median_deg/circ_mean_deg), so
these tests pin the online estimator to the offline reference: same trace in
-> same estimate out. If either side drifts, the T1-selected (tau, consensus)
no longer transfers to training and this file must fail.
"""
import numpy as np
import pytest

from WindGym.core.wd_estimator import WdEstimator


# --- offline reference (t1_probe.py math, reimplemented independently) -----

def ref_circ_ewma(deg_t, dt, tau):
    alpha = 1.0 - np.exp(-dt / tau)
    rad = np.deg2rad(np.asarray(deg_t, dtype=float))
    z = np.exp(1j * rad[0]).astype(complex)
    out = [np.rad2deg(np.angle(z)) % 360.0]
    for t in range(1, rad.shape[0]):
        z = (1.0 - alpha) * z + alpha * np.exp(1j * rad[t])
        out.append(np.rad2deg(np.angle(z)) % 360.0)
    return np.stack(out)


def ref_circ_median(deg):
    deg = np.asarray(deg, dtype=float)
    d = np.abs((deg[:, None] - deg[None, :] + 180.0) % 360.0 - 180.0)
    return float(deg[d.sum(axis=1).argmin()])


def ref_circ_mean(deg):
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(),
                                       np.cos(rad).mean())) % 360.0)


def make_trace(rng, n_steps=80, n_turb=9, base=270.0):
    """Wd trace with a ramp + per-turbine noise, crossing nothing weird."""
    t = np.arange(n_steps)
    ramp = np.clip((t - 20) * 1.5, 0.0, 45.0)
    return (base + ramp[:, None]
            + rng.normal(0.0, 4.0, size=(n_steps, n_turb))) % 360.0


# --- equivalence with the offline reference --------------------------------

@pytest.mark.parametrize("tau", [5.0, 30.0, 120.0])
@pytest.mark.parametrize("consensus", ["median", "mean", "front"])
def test_online_matches_offline_reference(tau, consensus):
    rng = np.random.default_rng(0)
    trace = make_trace(rng)
    n_turb = trace.shape[1]
    x_pos = np.arange(n_turb, dtype=float)[::-1]  # front = last index

    est = WdEstimator(n_turb=n_turb, dt=10.0, tau=tau, consensus=consensus,
                      x_pos=x_pos)
    online = np.array([est.update(row) for row in trace])

    smoothed = ref_circ_ewma(trace, dt=10.0, tau=tau)
    if consensus == "median":
        offline = np.array([ref_circ_median(r) for r in smoothed])
    elif consensus == "mean":
        offline = np.array([ref_circ_mean(r) for r in smoothed])
    else:
        offline = smoothed[:, int(np.argmin(x_pos))]

    err = (online - offline + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(err, 0.0, atol=1e-9)


def test_warm_start_returns_first_consensus():
    """The first update must return the consensus of the raw first sample —
    no cold-start transient from a zero/arbitrary init."""
    est = WdEstimator(n_turb=3, dt=10.0, tau=60.0)
    first = est.update([270.0, 280.0, 290.0])
    assert first == pytest.approx(280.0)  # circular median of the sample


def test_wraparound_safe():
    """Angles straddling 0/360 must average across the wrap, not through 180."""
    est = WdEstimator(n_turb=2, dt=10.0, tau=1e9, consensus="mean")
    # tau -> inf: alpha ~ dt/tau ~ 0, estimate ~ the warm-start sample
    out = est.update([359.0, 1.0])
    assert min(abs(out - 0.0), abs(out - 360.0)) < 1e-6


def test_static_convergence_and_lag():
    """On a step change the filter approaches the new value like 1-exp(-t/tau)."""
    tau, dt = 60.0, 10.0
    est = WdEstimator(n_turb=1, dt=dt, tau=tau, consensus="mean")
    est.update([270.0])
    n = 12
    vals = [est.update([300.0]) for _ in range(n)]
    # after n steps: err_ratio = exp(-n*dt/tau) on the unit circle (small
    # angles: linear is a good approximation)
    expected = 300.0 - 30.0 * np.exp(-n * dt / tau)
    assert vals[-1] == pytest.approx(expected, abs=0.5)
    assert vals[0] < vals[-1] < 300.0  # monotone approach, lagging


def test_reset_forgets_state():
    est = WdEstimator(n_turb=2, dt=10.0, tau=60.0)
    est.update([100.0, 100.0])
    est.reset()
    assert np.isnan(est.per_turbine).all()
    assert est.update([200.0, 200.0]) == pytest.approx(200.0)


def test_per_turbine_shape_and_nan_before_update():
    est = WdEstimator(n_turb=4, dt=10.0, tau=60.0)
    assert est.per_turbine.shape == (4,)
    assert np.isnan(est.per_turbine).all()
    assert np.isnan(est.estimate)


def test_validation_errors():
    with pytest.raises(ValueError):
        WdEstimator(n_turb=2, dt=10.0, tau=0.0)
    with pytest.raises(ValueError):
        WdEstimator(n_turb=2, dt=10.0, tau=60.0, consensus="mode")
    with pytest.raises(ValueError):
        WdEstimator(n_turb=2, dt=10.0, tau=60.0, consensus="front")  # no x_pos
    est = WdEstimator(n_turb=3, dt=10.0, tau=60.0)
    with pytest.raises(ValueError):
        est.update([1.0, 2.0])  # wrong shape


def test_median_robust_to_waked_outlier():
    """One deeply-waked turbine (deflected local flow) must not drag the
    consensus — the reason median is the default."""
    est_med = WdEstimator(n_turb=5, dt=10.0, tau=60.0, consensus="median")
    est_mean = WdEstimator(n_turb=5, dt=10.0, tau=60.0, consensus="mean")
    sample = [270.0, 271.0, 269.0, 270.5, 300.0]  # last one waked
    assert abs(est_med.update(sample) - 270.0) < 1.5
    assert abs(est_mean.update(sample) - 270.0) > 4.0
