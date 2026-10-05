# test_power_tracking.py
"""Tests for the farm power tracking feature (Track_power).

Covers the PowerTrackingManager reference generation, the tracking reward in
RewardCalculator, the tracking observations, and the env/AgentEval wiring.
Uses the fast pywake backend throughout.
"""

from collections import deque

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular

from WindGym import WindFarmEnv, FarmEval, AgentEvalFast, utils
from WindGym.Agents import ConstantAgent
from WindGym.core.measurement_manager import (
    MeasurementManager,
    MeasurementType,
    NoisyWindFarmEnv,
    WhiteNoiseModel,
)
from WindGym.core.power_tracking import PowerTrackingManager
from WindGym.core.reward_calculator import RewardCalculator
from WindGym.wind_env_multi import WindFarmEnvMulti

RATED_POWER = 2.0e6  # W (synthetic derating turbine)


def make_env_config(**overrides):
    """Minimal env config for a fast pywake-backend tracking env."""
    config = {
        "ActionMethod": "wind",
        "Track_power": True,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 9.0,
            "ws_max": 9.0,
            "TI_min": 0.06,
            "TI_max": 0.06,
            "wd_min": 270.0,
            "wd_max": 270.0,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "change"},
        "power_def": {
            "Power_reward": "None",
            "Power_avg": 5,
            "Power_scaling": 1.0,
        },
        "mes_level": {
            "turb_ws": True,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": False,
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
            "power_current": False,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 10,
            "power_window_length": 10,
        },
    }
    config.update(overrides)
    return config


def make_env(
    n_turb=3, spacing_d=5.0, config=None, env_cls=WindFarmEnv, turbine=None, **kwargs
):
    turbine = V80() if turbine is None else turbine
    d = turbine.diameter()
    return env_cls(
        turbine=turbine,
        x_pos=np.arange(n_turb) * spacing_d * d,
        y_pos=np.zeros(n_turb),
        config=config if config is not None else make_env_config(),
        backend="pywake",
        reset_init=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def derating_turbine():
    """Synthetic turbine whose powerCtFunction accepts a 'derate' input
    (same construction as in test_derating.py)."""
    ws = np.linspace(0.0, 30.0, 61)
    derate = np.linspace(0.0, 0.8, 17)

    p_avail = RATED_POWER * np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 3
    power = p_avail[:, None] * (1.0 - derate[None, :])
    ct = 0.8 * (1.0 - derate[None, :]) ** 1.5 * np.ones_like(ws)[:, None]

    pctf = PowerCtNDTabular(
        input_keys=["ws", "derate"],
        value_lst=[ws, derate],
        power_arr=power,
        power_unit="W",
        ct_arr=ct,
        default_value_dict={"derate": 0.0},
    )
    for gi in pctf.interp:
        gi.bounds = "limit"

    return WindTurbine(
        name="SyntheticDerating",
        diameter=80.0,
        hub_height=70.0,
        powerCtFunction=pctf,
    )


# ---------------------------------------------------------------------------
# RewardCalculator unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_track_power_mutually_exclusive_with_power_reward():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RewardCalculator(power_reward_type="Baseline", track_power=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RewardCalculator(power_reward_type="Power_avg", track_power=True)
    # Valid tracking configuration
    RewardCalculator(power_reward_type="None", track_power=True)


@pytest.mark.unit
def test_tracking_reward_abs_math():
    rc = RewardCalculator(power_reward_type="None", track_power=True)
    reward = rc.calculate_power_reward(
        farm_power_deque=deque([80.0]),
        power_ref_deque=deque([100.0]),
        power_norm=200.0,
    )
    assert abs(reward - (-0.1)) < 1e-12

    # Symmetric in the sign of the error
    reward = rc.calculate_power_reward(
        farm_power_deque=deque([120.0]),
        power_ref_deque=deque([100.0]),
        power_norm=200.0,
    )
    assert abs(reward - (-0.1)) < 1e-12


@pytest.mark.unit
def test_tracking_reward_uses_window_means():
    rc = RewardCalculator(power_reward_type="None", track_power=True)
    # mean(farm) = 90, mean(ref) = 100 -> err = -10, norm 100 -> -0.1
    reward = rc.calculate_power_reward(
        farm_power_deque=deque([80.0, 100.0]),
        power_ref_deque=deque([100.0, 100.0]),
        power_norm=100.0,
    )
    assert abs(reward - (-0.1)) < 1e-12


@pytest.mark.unit
def test_tracking_reward_gaussian_math():
    rc = RewardCalculator(
        power_reward_type="None",
        track_power=True,
        track_reward_type="gaussian",
        track_sigma=0.1,
    )
    # Zero error -> reward 1
    reward = rc.calculate_power_reward(
        farm_power_deque=deque([100.0]),
        power_ref_deque=deque([100.0]),
        power_norm=200.0,
    )
    assert abs(reward - 1.0) < 1e-12

    # err = sigma * norm -> exp(-1)
    reward = rc.calculate_power_reward(
        farm_power_deque=deque([120.0]),
        power_ref_deque=deque([100.0]),
        power_norm=200.0,
    )
    assert abs(reward - np.exp(-1)) < 1e-12


@pytest.mark.unit
def test_tracking_reward_invalid_config_raises():
    with pytest.raises(ValueError, match="track_reward_type"):
        RewardCalculator(
            power_reward_type="None", track_power=True, track_reward_type="quadratic"
        )
    with pytest.raises(ValueError, match="track_sigma"):
        RewardCalculator(power_reward_type="None", track_power=True, track_sigma=0.0)
    # Invalid tracking options are ignored when tracking is off
    RewardCalculator(power_reward_type="Baseline", track_reward_type="quadratic")


@pytest.mark.unit
def test_tracking_reward_missing_inputs_raise():
    rc = RewardCalculator(power_reward_type="None", track_power=True)
    with pytest.raises(ValueError, match="power_ref_deque"):
        rc.calculate_power_reward(farm_power_deque=deque([100.0]), power_norm=200.0)
    with pytest.raises(ValueError, match="power_ref_deque"):
        rc.calculate_power_reward(
            farm_power_deque=deque([100.0]),
            power_ref_deque=deque(),
            power_norm=200.0,
        )
    with pytest.raises(ValueError, match="power_norm"):
        rc.calculate_power_reward(
            farm_power_deque=deque([100.0]), power_ref_deque=deque([100.0])
        )
    with pytest.raises(ValueError, match="power_norm"):
        rc.calculate_power_reward(
            farm_power_deque=deque([100.0]),
            power_ref_deque=deque([100.0]),
            power_norm=0.0,
        )


@pytest.mark.unit
def test_tracking_total_reward_breakdown():
    rc = RewardCalculator(power_reward_type="None", track_power=True)
    total, breakdown = rc.calculate_total_reward(
        farm_power_deque=deque([80.0]),
        old_yaws=np.zeros(2),
        new_yaws=np.zeros(2),
        yaw_max=30.0,
        power_ref_deque=deque([100.0]),
        power_norm=200.0,
    )
    assert abs(total - (-0.1)) < 1e-12
    assert "power_reward" in breakdown
    assert "scaled_power_reward" in breakdown
    assert "action_penalty" in breakdown
    assert "total_reward" in breakdown
    assert abs(breakdown["tracking_error"] - (-20.0)) < 1e-12


# ---------------------------------------------------------------------------
# PowerTrackingManager unit tests
# ---------------------------------------------------------------------------


class _FakeEnv:
    rated_power = 1.0e6
    n_turb = 3


@pytest.mark.unit
def test_manager_validation():
    with pytest.raises(ValueError, match="ref_range"):
        PowerTrackingManager(ref_range=(0.5,))
    with pytest.raises(ValueError, match="ref_range"):
        PowerTrackingManager(ref_range=(0.8, 0.2))
    with pytest.raises(ValueError, match="ref_range"):
        PowerTrackingManager(ref_range=(-0.1, 0.5))
    with pytest.raises(ValueError, match="preview_steps"):
        PowerTrackingManager(preview_steps=-1)


@pytest.mark.unit
def test_manager_requires_rng_for_default_sampler():
    manager = PowerTrackingManager()
    with pytest.raises(RuntimeError, match="np_random"):
        manager.reset_episode(_FakeEnv(), time_max=10, delay=1, power_avg=5)


@pytest.mark.unit
def test_manager_default_sampler_constant_in_range_deterministic():
    farm_free = _FakeEnv.rated_power * _FakeEnv.n_turb

    def sample(seed):
        manager = PowerTrackingManager(ref_range=(0.2, 0.8))
        manager.np_random = np.random.default_rng(seed)
        manager.reset_episode(_FakeEnv(), time_max=20, delay=2, power_avg=5)
        return manager

    m1, m2, m3 = sample(0), sample(0), sample(1)

    # Constant over the episode and inside the configured range
    assert np.all(m1.trajectory == m1.trajectory[0])
    assert 0.2 * farm_free <= m1.trajectory[0] <= 0.8 * farm_free
    # Same seed -> same setpoint; different seed -> different setpoint
    assert m1.trajectory[0] == m2.trajectory[0]
    assert m1.trajectory[0] != m3.trajectory[0]


@pytest.mark.unit
def test_manager_callable_trajectory_values_and_length():
    manager = PowerTrackingManager(
        ref_function=lambda t, env: 100.0 + t, preview_steps=2
    )
    manager.reset_episode(_FakeEnv(), time_max=10, delay=2, power_avg=5)

    # ceil(10/2) + 1 = 6 episode entries + 2 preview tail
    assert len(manager.trajectory) == 8
    np.testing.assert_allclose(manager.trajectory, 100.0 + 2.0 * np.arange(8))
    assert manager.reference(3) == 106.0


@pytest.mark.unit
def test_manager_clamps_past_horizon_without_callable():
    manager = PowerTrackingManager()
    manager.np_random = np.random.default_rng(0)
    manager.reset_episode(_FakeEnv(), time_max=10, delay=1, power_avg=5)
    setpoint = manager.reference(0)
    assert manager.reference(10_000_000) == setpoint
    # Clamping does not grow the trajectory
    assert len(manager.trajectory) == 11


@pytest.mark.unit
def test_manager_lazily_extends_with_callable():
    calls = []

    def ref(t, env):
        calls.append(t)
        return 2.0 * t

    manager = PowerTrackingManager(ref_function=ref)
    manager.reset_episode(_FakeEnv(), time_max=4, delay=1, power_avg=5)
    n_initial = len(manager.trajectory)

    assert manager.reference(10) == 20.0
    assert len(manager.trajectory) == 11
    # Each time evaluated exactly once, in order
    assert calls == [float(t) for t in range(n_initial)] + [
        float(t) for t in range(n_initial, 11)
    ]
    # Re-reading a cached index does not re-evaluate
    n_calls = len(calls)
    assert manager.reference(7) == 14.0
    assert len(calls) == n_calls


@pytest.mark.unit
def test_manager_clamps_far_extension_with_callable():
    """A far index clamps instead of materializing a pathological range.

    Mirrors test_manager_clamps_past_horizon_without_callable but for a
    callable manager: this is what defuses the eval memory-cleanup step, which
    sets timestep = time_max (~1e7). A modest jump must still lazily extend.
    """
    calls = []

    def ref(t, env):
        calls.append(t)
        return 2.0 * t

    manager = PowerTrackingManager(ref_function=ref)
    manager.reset_episode(_FakeEnv(), time_max=4, delay=1, power_avg=5)

    # A modest jump still lazily extends the trajectory (contract preserved).
    assert manager.reference(10) == 20.0
    assert len(manager.trajectory) == 11

    # A jump past MAX_TRAJECTORY_STEPS clamps to the last value: the trajectory
    # does not grow and the callable is not evaluated for the sentinel range.
    n_before = len(manager.trajectory)
    n_calls_before = len(calls)
    assert manager.reference(2_000_000) == manager.trajectory[-1]
    assert len(manager.trajectory) == n_before
    assert len(calls) == n_calls_before


@pytest.mark.unit
def test_manager_preview_length_and_values():
    manager = PowerTrackingManager(
        ref_function=lambda t, env: 10.0 * t, preview_steps=3
    )
    manager.reset_episode(_FakeEnv(), time_max=6, delay=2, power_avg=5)

    np.testing.assert_allclose(manager.preview(0), [20.0, 40.0, 60.0])
    # Preview at the last episode step reaches into the precomputed tail
    last_idx = 3  # ceil(6/2) = 3
    np.testing.assert_allclose(manager.preview(last_idx), [80.0, 100.0, 120.0])


@pytest.mark.unit
def test_manager_push_fills_deque():
    manager = PowerTrackingManager(ref_function=lambda t, env: 5.0 + t)
    manager.reset_episode(_FakeEnv(), time_max=10, delay=1, power_avg=3)
    assert manager.push(0) == 5.0
    assert manager.push(1) == 6.0
    assert manager.push(2) == 7.0
    assert manager.push(3) == 8.0  # deque holds only the last 3
    assert list(manager.ref_deque) == [6.0, 7.0, 8.0]


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tracking_env_happy_path():
    env = make_env()
    obs, info = env.reset(seed=1)

    assert env.power_tracking is not None
    assert "Power reference" in info
    assert "Tracking error" in info
    assert "Tracking error window mean" in info
    assert "Power reference preview" in info

    farm_free = env.rated_power * env.n_turb
    assert 0.2 * farm_free <= info["Power reference"] <= 0.8 * farm_free

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert obs.shape == env.observation_space.shape
    # abs reward is bounded: -|err|/P_norm with |err| <= P_norm-ish
    assert reward <= 0.0

    # The window-mean info key is exactly the quantity the reward normalizes:
    # mean(farm_pow_deq) - mean(ref_deque). Hand-compute it from the deques.
    window_err = float(
        np.mean(env.farm_pow_deq) - np.mean(env.power_tracking.ref_deque)
    )
    assert info["Tracking error window mean"] == pytest.approx(window_err)

    # With the default "abs" form and no penalties configured (action penalty
    # 0, no derate) at Power_scaling 1.0, the reward is exactly -|err|/P_norm.
    norm = env.maxturbpower * env.n_turb
    assert reward == pytest.approx(-abs(window_err) / norm)
    env.close()


@pytest.mark.integration
def test_tracking_env_mutual_exclusion_raises():
    config = make_env_config(
        power_def={"Power_reward": "Baseline", "Power_avg": 5, "Power_scaling": 1.0}
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(config=config)


@pytest.mark.integration
def test_tracking_env_invalid_track_def_raises():
    with pytest.raises(ValueError, match="track_ref_range"):
        make_env(config=make_env_config(track_def={"track_ref_range": [0.8, 0.2]}))
    with pytest.raises(ValueError, match="track_obs_preview"):
        make_env(config=make_env_config(track_def={"track_obs_preview": -1}))
    with pytest.raises(ValueError, match="track_reward_type"):
        make_env(config=make_env_config(track_def={"Track_reward": "quadratic"}))


@pytest.mark.integration
@pytest.mark.parametrize(
    "track_def, extra_obs",
    [
        ({"track_obs_setpoint": False, "track_obs_error": False}, 0),
        ({"track_obs_setpoint": True, "track_obs_error": False}, 1),
        ({"track_obs_setpoint": True, "track_obs_error": True}, 2),
        (
            {
                "track_obs_setpoint": True,
                "track_obs_error": True,
                "track_obs_preview": 4,
            },
            6,
        ),
    ],
    ids=["none", "setpoint", "setpoint+error", "setpoint+error+preview4"],
)
def test_tracking_obs_toggles_and_scaling(track_def, extra_obs):
    """Each toggle adds exactly its entries, appended (scaled) at the end of
    the observation vector."""
    env_off = make_env(
        config=make_env_config(
            Track_power=False,
            power_def={
                "Power_reward": "Power_avg",
                "Power_avg": 5,
                "Power_scaling": 1.0,
            },
        )
    )
    env = make_env(config=make_env_config(track_def=track_def))

    assert (
        env.observation_space.shape[0] == env_off.observation_space.shape[0] + extra_obs
    )

    if extra_obs > 0:
        obs, _ = env.reset(seed=3)
        obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        norm = env.maxturbpower * env.n_turb
        expected = []
        if track_def.get("track_obs_setpoint"):
            expected.append(utils.scale_val(env.power_setpoint, 0, norm))
        if track_def.get("track_obs_error"):
            expected.append(
                utils.scale_val(env.farm_pow_deq[-1] - env.power_setpoint, -norm, norm)
            )
        if track_def.get("track_obs_preview", 0) > 0:
            expected.extend(utils.scale_val(np.asarray(env.power_preview), 0, norm))

        np.testing.assert_allclose(
            obs[-extra_obs:], np.clip(expected, -1.0, 1.0), rtol=1e-5, atol=1e-6
        )

    env.close()
    env_off.close()


@pytest.mark.integration
def test_tracking_obs_pass_through_noise_unchanged():
    """Tracking observations are commands, not sensors, so measurement noise
    must never touch them. `MeasurementManager` builds specs only for the
    sensor block, and the noise models write into `spec.index_range` slices —
    the tracking tail has no spec, so it survives verbatim. This test pins
    that contract: the last 4 entries (setpoint, error, 2 preview) must be
    bit-equal to the clean obs, while the sensor block actually changes.
    """
    # 4 tracking tail entries: setpoint + error (both default True) + 2 preview.
    env = make_env(config=make_env_config(track_def={"track_obs_preview": 2}))
    n_track = 4

    mm = MeasurementManager(env, seed=42)
    mm.set_noise_model(WhiteNoiseModel({MeasurementType.WIND_SPEED: 0.5}))
    # The lambda reuses the already-built env instance (same pattern as the
    # measurement-manager smoke tests) rather than constructing a fresh one.
    wrapped = NoisyWindFarmEnv(
        base_env_class=lambda **kwargs: env, measurement_manager=mm
    )

    def assert_passthrough(noisy_obs, info):
        clean_obs = info["clean_obs"]
        # Tracking tail is passed through untouched (bit-equal).
        np.testing.assert_array_equal(noisy_obs[-n_track:], clean_obs[-n_track:])
        # The sensor block did get noised — so the passthrough assertion above
        # is not vacuously true (noise actually fired this call).
        assert not np.array_equal(noisy_obs[:-n_track], clean_obs[:-n_track])

    noisy_obs, info = wrapped.reset(seed=7)
    assert_passthrough(noisy_obs, info)

    noisy_obs, _, _, _, info = wrapped.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert_passthrough(noisy_obs, info)

    wrapped.close()


@pytest.mark.integration
def test_tracking_preview_near_truncation():
    """Preview stays in-bounds when the episode runs to (and past) time_max."""
    env = make_env(
        config=make_env_config(track_def={"track_obs_preview": 3}),
        max_time_steps=4,
    )
    obs, info = env.reset(seed=0)
    truncated = False
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert len(info["Power reference preview"]) == 3
        assert np.all(np.isfinite(info["Power reference preview"]))
    env.close()


@pytest.mark.integration
def test_tracking_seed_determinism():
    def rollout(seed):
        env = make_env(config=make_env_config(track_def={"track_obs_preview": 2}))
        obs, info = env.reset(seed=seed)
        setpoint = info["Power reference"]
        obs2, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        env.close()
        return setpoint, obs, obs2

    sp_a, obs_a, obs2_a = rollout(42)
    sp_b, obs_b, obs2_b = rollout(42)
    sp_c, obs_c, _ = rollout(43)

    assert sp_a == sp_b
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(obs2_a, obs2_b)
    assert sp_a != sp_c


@pytest.mark.integration
def test_tracking_sampler_range_over_reseeds():
    env = make_env(config=make_env_config(track_def={"track_ref_range": [0.3, 0.5]}))
    farm_free = None
    setpoints = []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        farm_free = env.rated_power * env.n_turb  # constant (ws fixed at 9)
        setpoints.append(info["Power reference"])
    env.close()

    setpoints = np.array(setpoints)
    assert np.all(setpoints >= 0.3 * farm_free)
    assert np.all(setpoints <= 0.5 * farm_free)
    # The setpoints actually vary across episodes
    assert len(np.unique(setpoints)) > 1


@pytest.mark.integration
@pytest.mark.parametrize("reward_type", ["abs", "gaussian"])
def test_tracking_reward_sanity(reward_type):
    """Reference equal to the measured greedy power gives a (near-)perfect
    reward; an unreachable reference gives the worst-case reward."""
    ref_holder = [1.0]

    def make(ref_value):
        ref_holder[0] = ref_value
        env = make_env(
            config=make_env_config(track_def={"Track_reward": reward_type}),
            power_ref_function=lambda t, env: ref_holder[0],
        )
        return env

    # Measure the greedy (zero yaw) farm power with a probe episode
    env = make(1.0)
    env.reset(seed=7)
    for _ in range(3):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    greedy_power = float(np.mean(env.farm_pow_deq))
    env.close()

    # Same seed + reference == measured greedy power -> near-perfect tracking
    env = make(greedy_power)
    env.reset(seed=7)
    for _ in range(3):
        _, reward, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    if reward_type == "abs":
        assert abs(reward) < 1e-2
    else:
        assert reward > 0.99
    env.close()

    # Unreachable reference (10x nameplate) -> worst-case reward
    env = make(10.0 * env.maxturbpower * env.n_turb)
    env.reset(seed=7)
    _, reward, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    if reward_type == "abs":
        assert reward < -5.0
    else:
        assert reward < 1e-6
    env.close()


@pytest.mark.integration
def test_tracking_env_kwargs_clone_roundtrip():
    """env.kwargs must capture power_ref_function so cloned envs track too."""
    ramp = lambda t, env: 1.0e6 + 100.0 * t  # noqa: E731

    env = make_env(config=make_env_config(), power_ref_function=ramp)
    assert env.kwargs["power_ref_function"] is ramp

    clone_kwargs = {k: v for k, v in env.kwargs.items() if k != "kwargs"}
    clone_kwargs.update(env.kwargs.get("kwargs", {}))
    clone = WindFarmEnv(**clone_kwargs)

    clone.reset(seed=0)
    assert clone.power_setpoint == 1.0e6  # ramp evaluated at t=0
    np.testing.assert_allclose(
        clone.power_tracking.trajectory[:3],
        1.0e6 + 100.0 * clone.delay * np.arange(3),
    )
    clone.close()
    env.close()


@pytest.mark.integration
def test_tracking_multi_env_supported():
    """Constructing a tracking WindFarmEnvMulti no longer raises, and its
    obs_var grows by exactly the tracking-tail width."""
    track_env = make_env(env_cls=WindFarmEnvMulti, config=make_env_config())
    assert track_env.Track_power

    plain_env = make_env(
        env_cls=WindFarmEnvMulti,
        config=make_env_config(Track_power=False),
    )
    n_track = len(track_env.farm_measurements.get_tracking())
    assert n_track > 0
    assert track_env.obs_var == plain_env.obs_var + n_track
    track_env.close()
    plain_env.close()


@pytest.mark.integration
def test_tracking_multi_obs_includes_reference():
    """Each agent's obs matches its observation_space, and the trailing
    tracking slice is identical across agents and reflects the pushed
    reference (moves after a step under a ramp)."""
    ramp = lambda t, env: 1.0e6 + 500.0 * t  # noqa: E731
    env = make_env(
        env_cls=WindFarmEnvMulti, config=make_env_config(), power_ref_function=ramp
    )
    observations, _ = env.reset(seed=0)

    n_track = len(env.farm_measurements.get_tracking())
    tails = {}
    for a in env.agents:
        obs = observations[a]
        assert obs.shape[0] == env.observation_space(a).shape[0] == env.obs_var
        tails[a] = obs[-n_track:]

    # The tracking tail is the same farm-level command for every agent.
    first = env.agents[0]
    for a in env.agents[1:]:
        np.testing.assert_array_equal(tails[a], tails[first])

    # It matches the scaled reference the parent pushed onto FarmMes.
    np.testing.assert_array_equal(
        tails[first], env.farm_measurements.get_tracking(scaled=True).astype(np.float32)
    )

    # And it moves after a step because the ramp keeps rising.
    actions = {a: np.zeros(env.action_space(a).shape, np.float32) for a in env.agents}
    obs_after, *_ = env.step(actions)
    assert not np.array_equal(obs_after[first][-n_track:], tails[first])
    env.close()


@pytest.mark.integration
def test_tracking_multi_reward_shared():
    """After a step, all agents receive the same finite (shared) reward."""
    env = make_env(env_cls=WindFarmEnvMulti, config=make_env_config())
    env.reset(seed=0)
    actions = {a: np.zeros(env.action_space(a).shape, np.float32) for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    values = list(rewards.values())
    assert all(np.isfinite(v) for v in values)
    assert all(v == values[0] for v in values)
    env.close()


@pytest.mark.integration
def test_tracking_multi_info_keys():
    """Every agent's info dict carries all four tracking keys, at reset and
    after a step."""
    keys = {
        "Power reference",
        "Tracking error",
        "Tracking error window mean",
        "Power reference preview",
    }
    env = make_env(env_cls=WindFarmEnvMulti, config=make_env_config())
    _, infos = env.reset(seed=0)
    for a in env.agents:
        assert keys.issubset(infos[a].keys())

    actions = {a: np.zeros(env.action_space(a).shape, np.float32) for a in env.agents}
    _, _, _, _, infos = env.step(actions)
    for a in env.agents:
        assert keys.issubset(infos[a].keys())
    env.close()


@pytest.mark.integration
def test_tracking_multi_custom_reference():
    """A ramp power_ref_function reaches the multi env and drives a
    time-varying setpoint across steps."""
    ramp = lambda t, env: 1.0e6 + 500.0 * t  # noqa: E731
    env = make_env(
        env_cls=WindFarmEnvMulti, config=make_env_config(), power_ref_function=ramp
    )
    assert env.kwargs["power_ref_function"] is ramp

    env.reset(seed=0)
    actions = {a: np.zeros(env.action_space(a).shape, np.float32) for a in env.agents}
    setpoints = []
    for _ in range(3):
        _, _, _, _, infos = env.step(actions)
        setpoints.append(infos[env.agents[0]]["Power reference"])

    # The ramp climbs, so consecutive setpoints strictly increase.
    assert setpoints[0] < setpoints[1] < setpoints[2]
    env.close()


@pytest.mark.integration
def test_multi_non_tracking_obs_len_unchanged():
    """A non-tracking multi env keeps obs_var / per-agent obs length equal to
    turbine + farm blocks only (0-width tracking tail)."""
    env = make_env(env_cls=WindFarmEnvMulti, config=make_env_config(Track_power=False))
    turbine_obs_var = env.farm_measurements.turb_mes[0].observed_variables()
    farm_obs_var = env.farm_measurements.farm_mes.observed_variables()
    assert env.obs_var == turbine_obs_var + farm_obs_var
    assert len(env.farm_measurements.get_tracking()) == 0

    observations, _ = env.reset(seed=0)
    for a in env.agents:
        assert observations[a].shape[0] == env.obs_var
    env.close()


@pytest.mark.integration
def test_tracking_composes_with_derating(derating_turbine):
    """Tracking reward + derate action + derate penalty work together."""
    config = make_env_config(
        derate_action=True,
        derate_min=0.0,
        derate_max=0.8,
        derate_method="absolute",
        derate_penalty=0.5,
        derate_penalty_type="total",
    )
    env = make_env(turbine=derating_turbine, config=config)
    env.reset(seed=2)

    # Full derate command on all turbines (last n_turb action entries)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[env.n_turb :] = 1.0
    _, reward, *_ = env.step(action)
    assert np.isfinite(reward)

    # The breakdown still contains the derate penalty next to the tracking reward
    _, breakdown = env.reward_calculator.calculate_total_reward(
        farm_power_deque=env.farm_pow_deq,
        old_yaws=env.old_yaws,
        new_yaws=env.fs.windTurbines.yaw,
        yaw_max=env.yaw_max,
        old_derates=env.old_derate,
        new_derates=env.current_derate,
        derate_max=env.derate_max,
        power_ref_deque=env.power_tracking.ref_deque,
        power_norm=env.maxturbpower * env.n_turb,
    )
    assert breakdown["derate_penalty"] > 0.0
    assert "tracking_error" in breakdown
    env.close()


# ---------------------------------------------------------------------------
# AgentEval integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_agent_eval_tracking_metrics():
    """AgentEvalFast on a tracking env reports power_ref/track_err/track_mae."""
    turbine = V80()
    d = turbine.diameter()
    env = FarmEval(
        turbine=turbine,
        x_pos=np.arange(2) * 5 * d,
        y_pos=np.zeros(2),
        config=make_env_config(),
        backend="pywake",
        turbtype="None",
        reset_init=False,
    )
    agent = ConstantAgent(yaw_angles=np.zeros(2), yaw_max=30, yaw_min=-30)

    ds = AgentEvalFast(env, agent, ws=9.0, ti=0.06, wd=270.0, t_sim=10)

    assert "power_ref" in ds.data_vars
    assert "track_err" in ds.data_vars
    assert "track_mae" in ds.data_vars

    track_err = ds.track_err.values
    track_mae = float(ds.track_mae.values.squeeze())
    assert np.isclose(track_mae, np.abs(track_err).mean())
    # The reference is constant per episode (default sampler), matching info
    power_ref = ds.power_ref.values.flatten()
    assert np.all(power_ref > 0)
    np.testing.assert_allclose(power_ref, power_ref[0])


@pytest.mark.integration
def test_agent_eval_tracking_custom_reference():
    """AgentEvalFast tracks a time-varying custom power_ref_function (ramp).

    Also implicitly proves the eval memory-cleanup step (timestep = time_max)
    no longer explodes for callable references. The cleanup jump is bounded
    both by FarmEval capping time_max at 10_000 and by the defensive
    MAX_TRAJECTORY_STEPS clamp; before either, that step materialized ~10M
    trajectory entries and called the ramp ~10M times.
    """
    ramp = lambda t, env: 1.0e6 + 100.0 * t  # noqa: E731

    turbine = V80()
    d = turbine.diameter()
    env = FarmEval(
        turbine=turbine,
        x_pos=np.arange(2) * 5 * d,
        y_pos=np.zeros(2),
        config=make_env_config(),
        power_ref_function=ramp,
        backend="pywake",
        turbtype="None",
        reset_init=False,
    )
    # The kwarg reached WindFarmEnv and is captured for clone/roundtrip.
    assert env.kwargs["power_ref_function"] is ramp

    agent = ConstantAgent(yaw_angles=np.zeros(2), yaw_max=30, yaw_min=-30)

    ds = AgentEvalFast(env, agent, ws=9.0, ti=0.06, wd=270.0, t_sim=10)

    assert "power_ref" in ds.data_vars
    assert "track_err" in ds.data_vars
    assert "track_mae" in ds.data_vars

    # Negative control vs. the constant-sampler test above: a ramp reference is
    # time-varying, not constant.
    power_ref = ds.power_ref.values.flatten()
    assert not np.allclose(power_ref, power_ref[0])
    assert power_ref[0] == pytest.approx(1.0e6)  # ramp evaluated at t=0
    assert np.all(np.diff(power_ref) >= -1e-3)  # monotonically increasing ramp
    # Distinct references step up by the ramp slope per env step (100 * delay).
    steps = np.diff(np.unique(power_ref))
    np.testing.assert_allclose(steps, 100.0 * env.delay, rtol=1e-4)

    # track_mae stays consistent with |track_err|.
    track_err = ds.track_err.values
    track_mae = float(ds.track_mae.values.squeeze())
    assert np.isclose(track_mae, np.abs(track_err).mean())


@pytest.mark.integration
def test_agent_eval_non_tracking_unchanged():
    """A non-tracking eval dataset has no tracking variables."""
    turbine = V80()
    d = turbine.diameter()
    env = FarmEval(
        turbine=turbine,
        x_pos=np.arange(2) * 5 * d,
        y_pos=np.zeros(2),
        config=make_env_config(
            Track_power=False,
            power_def={
                "Power_reward": "Power_avg",
                "Power_avg": 5,
                "Power_scaling": 1.0,
            },
        ),
        backend="pywake",
        turbtype="None",
        reset_init=False,
    )
    agent = ConstantAgent(yaw_angles=np.zeros(2), yaw_max=30, yaw_min=-30)

    ds = AgentEvalFast(env, agent, ws=9.0, ti=0.06, wd=270.0, t_sim=10)

    assert "power_ref" not in ds.data_vars
    assert "track_err" not in ds.data_vars
    assert "track_mae" not in ds.data_vars


@pytest.mark.integration
def test_agent_eval_raises_on_truncation():
    """An eval that overruns the env horizon fails loudly, not silently.

    Truncation triggers _cleanup_resources() (frees the flow sim), so the eval
    loop cannot continue past it. Guards the footgun of a too-small time_max:
    force a tiny horizon and check the run raises instead of stepping into
    freed state and returning corrupt results.
    """
    turbine = V80()
    d = turbine.diameter()
    env = FarmEval(
        turbine=turbine,
        x_pos=np.arange(2) * 5 * d,
        y_pos=np.zeros(2),
        config=make_env_config(),
        backend="pywake",
        turbtype="None",
        reset_init=False,
    )
    agent = ConstantAgent(yaw_angles=np.zeros(2), yaw_max=30, yaw_min=-30)

    # Shrink the sandbox horizon after reset so t_sim=10 overruns it.
    base_reset = env.reset

    def tiny_horizon_reset(seed=None, options=None):
        obs, info = base_reset(seed=seed, options=options)
        env.time_max = 3  # truncate after ~3 env steps (delay=1)
        return obs, info

    env.reset = tiny_horizon_reset

    with pytest.raises(RuntimeError, match="truncated during evaluation"):
        AgentEvalFast(env, agent, ws=9.0, ti=0.06, wd=270.0, t_sim=10)
