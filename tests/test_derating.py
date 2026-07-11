# test_derating.py
"""Tests for the per-turbine derating action.

Uses a small synthetic PowerCtNDTabular turbine with a 'derate' dimension so
the tests do not depend on the shipped DTU10MW surrogate file.
"""

import numpy as np
import pytest
from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular

from WindGym import WindFarmEnv
from WindGym.core.reward_calculator import RewardCalculator
from WindGym.wind_env_multi import WindFarmEnvMulti

RATED_POWER = 2.0e6  # W


@pytest.fixture(scope="module")
def derating_turbine():
    """Small synthetic turbine whose powerCtFunction accepts a 'derate' input.

    power = (1 - derate) * P_available(ws)
    ct    = 0.8 * (1 - derate)**1.5   (thrust sheds faster than power,
                                       like the min-ct HAWCStab2 surrogate)
    """
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


def make_env_config(**overrides):
    """Minimal env config for a fast pywake-backend derating env."""
    config = {
        "derate_action": True,
        "derate_min": 0.0,
        "derate_max": 0.8,
        "derate_method": "absolute",
        "ActionMethod": "wind",
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
            "Power_reward": "Power_avg",
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
    turbine, n_turb=3, spacing_d=5.0, config=None, env_cls=WindFarmEnv, **kwargs
):
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


# ---------------------------------------------------------------------------
# Turbine-level behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_power_and_ct_decrease_monotonically_with_derate(derating_turbine):
    derate = np.linspace(0.0, 0.8, 9)
    ws = np.full_like(derate, 9.0)

    power = derating_turbine.power(ws, derate=derate)
    ct = derating_turbine.ct(ws, derate=derate)

    assert np.all(np.diff(power) < 0)
    assert np.all(np.diff(ct) < 0)


@pytest.mark.unit
def test_zero_derate_matches_baseline(derating_turbine):
    ws = np.array([5.0, 9.0, 15.0])

    p_default = derating_turbine.power(ws)
    p_zero = derating_turbine.power(ws, derate=np.zeros_like(ws))

    np.testing.assert_allclose(p_zero, p_default)
    np.testing.assert_allclose(
        derating_turbine.ct(ws, derate=np.zeros_like(ws)), derating_turbine.ct(ws)
    )


# ---------------------------------------------------------------------------
# Env integration (pywake backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_action_layout_and_wake_response(derating_turbine):
    """[yaw | derate] layout; derating the front turbine sheds its power and
    lets the waked turbine recover."""
    env = make_env(derating_turbine)
    obs, info = env.reset(seed=0)

    n = env.n_turb
    assert env.action_space.shape == (2 * n,)

    # No derating (raw -1 maps to derate_min = 0)
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = -1.0
    _, _, _, _, info = env.step(action)
    p_base = info["Power pr turbine agent"].copy()

    # Derate T0 to 0.4 (raw 0 is the midpoint of [0, 0.8])
    action_derated = action.copy()
    action_derated[n] = 0.0
    _, _, _, _, info = env.step(action_derated)
    p_derated = info["Power pr turbine agent"].copy()

    np.testing.assert_allclose(info["derate agent"], [0.4] + [0.0] * (n - 1), atol=1e-6)
    np.testing.assert_allclose(
        info["derate measured"], [0.4] + [0.0] * (n - 1), atol=1e-6
    )

    assert p_derated[0] < p_base[0]  # front turbine sheds power
    assert p_derated[1] > p_base[1]  # waked turbine recovers
    env.close()


@pytest.mark.integration
def test_derate_only_action_space(derating_turbine):
    env = make_env(derating_turbine, config=make_env_config(yaw_action=False))
    env.reset(seed=0)
    assert env.action_space.shape == (env.n_turb,)

    # Full derate on T0 only
    action = np.full(env.n_turb, -1.0, dtype=np.float32)
    action[0] = 1.0
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(
        info["derate agent"], [0.8] + [0.0] * (env.n_turb - 1), atol=1e-6
    )
    env.close()


@pytest.mark.integration
def test_step_mode_bounded_by_derate_step_env(derating_turbine):
    env = make_env(
        derating_turbine,
        config=make_env_config(derate_method="step", derate_step_env=0.1),
    )
    env.reset(seed=0)

    n = env.n_turb
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = 1.0  # max increase on all turbines

    previous = env.current_derate.copy()
    for _ in range(3):
        env.step(action)
        change = env.current_derate - previous
        assert np.all(change <= 0.1 + 1e-9)
        assert np.all(change >= 0.0)
        previous = env.current_derate.copy()

    # Never above derate_max no matter how long we push up
    for _ in range(12):
        env.step(action)
    assert np.all(env.current_derate <= 0.8 + 1e-9)
    env.close()


@pytest.mark.integration
def test_derate_step_sim_slews_toward_setpoint(derating_turbine):
    """With derate_step_sim set, an absolute setpoint is approached gradually
    instead of applying instantly."""
    env = make_env(
        derating_turbine,
        config=make_env_config(derate_step_sim=0.05),
    )
    env.reset(seed=0)

    n = env.n_turb
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = -1.0
    action[n] = 0.0  # T0 setpoint = 0.4

    env.step(action)
    first = env.current_derate.copy()
    assert 0.0 < first[0] < 0.4  # partway there, not instant

    # Keep holding the setpoint: monotone convergence to 0.4
    previous = first
    for _ in range(20):
        env.step(action)
        assert env.current_derate[0] >= previous[0] - 1e-12
        previous = env.current_derate.copy()
    np.testing.assert_allclose(env.current_derate[0], 0.4, atol=1e-6)
    env.close()


@pytest.mark.integration
def test_multi_env_action_packing(derating_turbine):
    """Per-agent [yaw, derate] actions map onto the grouped parent layout."""
    env = make_env(derating_turbine, env_cls=WindFarmEnvMulti)
    env.reset(seed=0)

    agents = env.possible_agents
    assert env.action_space(agents[0]).shape == (2,)

    # Distinct derate raw value per agent: -1, 0, 1 -> derate 0, 0.4, 0.8
    raw = np.linspace(-1.0, 1.0, len(agents))
    actions = {
        a: np.array([0.0, raw[i]], dtype=np.float32) for i, a in enumerate(agents)
    }
    _, _, _, _, infos = env.step(actions)

    np.testing.assert_allclose(env.current_derate, (raw + 1.0) / 2.0 * 0.8, atol=1e-6)
    for i, a in enumerate(agents):
        np.testing.assert_allclose(
            infos[a]["derate agent"], (raw[i] + 1.0) / 2.0 * 0.8, atol=1e-6
        )
    env.close()


# ---------------------------------------------------------------------------
# Rated-power reference (derate_reference="rated")
# ---------------------------------------------------------------------------

# Fixture turbine at the 9 m/s test inflow: P_avail = 2 MW * ((9-3)/9)^3
AVAIL_AT_9MS = RATED_POWER * (6.0 / 9.0) ** 3  # ~0.593 MW


@pytest.mark.integration
def test_rated_reference_dead_zone(derating_turbine):
    """A rated-power cap above locally available power is a no-op: the
    applied derate stays 0 and the front turbine keeps producing, while the
    info dict still reports the commanded fraction."""
    env = make_env(derating_turbine, config=make_env_config(derate_reference="rated"))
    env.reset(seed=0)

    n = env.n_turb
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = -1.0  # cmd 0 everywhere
    _, _, _, _, info = env.step(action)
    p_base = info["Power pr turbine agent"].copy()

    # cmd 0.5 -> target 1.0 MW, well above the ~0.593 MW available at 9 m/s
    action_capped = action.copy()
    action_capped[n] = 0.25  # affine map: raw 0.25 -> cmd 0.5 on [0, 0.8]
    _, _, _, _, info = env.step(action_capped)

    np.testing.assert_allclose(
        info["derate command"], [0.5] + [0.0] * (n - 1), atol=1e-6
    )
    np.testing.assert_allclose(info["derate agent"], np.zeros(n), atol=1e-9)
    np.testing.assert_allclose(info["Power pr turbine agent"][0], p_base[0], rtol=0.1)
    env.close()


@pytest.mark.integration
def test_rated_reference_binding_cap(derating_turbine):
    """A rated-power cap below available power binds: the front turbine
    converges to the absolute target, not to a fraction of available."""
    env = make_env(derating_turbine, config=make_env_config(derate_reference="rated"))
    env.reset(seed=0)

    n = env.n_turb
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = -1.0
    action[n] = 1.0  # cmd 0.8 -> target (1-0.8)*2 MW = 0.4 MW < available

    for _ in range(3):
        _, _, _, _, info = env.step(action)

    p_target = 0.2 * RATED_POWER
    np.testing.assert_allclose(info["Power pr turbine agent"][0], p_target, rtol=0.05)
    # equivalent available-power fraction: 1 - 0.4/0.593 ~ 0.33
    expected_applied = 1.0 - p_target / AVAIL_AT_9MS
    np.testing.assert_allclose(info["derate agent"][0], expected_applied, atol=0.05)
    np.testing.assert_allclose(info["derate command"][0], 0.8, atol=1e-6)
    env.close()


@pytest.mark.integration
def test_rated_reference_step_mode_composes(derating_turbine):
    """Step method + rated reference: the command integrates by at most
    derate_step_env per env step, and applied derate stays 0 while the
    implied cap is above available power."""
    env = make_env(
        derating_turbine,
        config=make_env_config(
            derate_reference="rated", derate_method="step", derate_step_env=0.1
        ),
    )
    env.reset(seed=0)

    n = env.n_turb
    action = np.zeros(2 * n, dtype=np.float32)
    action[n:] = 1.0  # max increase on all turbines

    previous = env.derate_command.copy()
    for _ in range(3):
        env.step(action)
        change = env.derate_command - previous
        assert np.all(change <= 0.1 + 1e-9)
        assert np.all(change >= 0.0)
        previous = env.derate_command.copy()

    # After 3 steps cmd = 0.3 -> target 1.4 MW > available: still a no-op
    # on the (unwaked) front turbine.
    np.testing.assert_allclose(env.derate_command, 0.3, atol=1e-9)
    assert env.current_derate[0] == pytest.approx(0.0, abs=1e-9)

    # Push until the cap binds (cmd > 1 - avail/rated ~ 0.70 for T0)
    for _ in range(5):
        env.step(action)
    np.testing.assert_allclose(env.derate_command, 0.8, atol=1e-9)
    assert env.current_derate[0] > 0.2
    env.close()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_derating_turbine_raises(derating_turbine):
    with pytest.raises(ValueError, match="derate"):
        make_env(V80(), config=make_env_config())


@pytest.mark.unit
def test_yaw_action_false_requires_derate_action(derating_turbine):
    with pytest.raises(ValueError, match="yaw_action"):
        make_env(
            derating_turbine,
            config=make_env_config(derate_action=False, yaw_action=False),
        )


@pytest.mark.unit
def test_invalid_derate_method_raises(derating_turbine):
    with pytest.raises(ValueError, match="derate_method"):
        make_env(derating_turbine, config=make_env_config(derate_method="bogus"))


@pytest.mark.unit
def test_invalid_derate_reference_raises(derating_turbine):
    with pytest.raises(ValueError, match="derate_reference"):
        make_env(derating_turbine, config=make_env_config(derate_reference="bogus"))


@pytest.mark.unit
def test_negative_derate_step_sim_raises(derating_turbine):
    with pytest.raises(ValueError, match="derate_step_sim"):
        make_env(derating_turbine, config=make_env_config(derate_step_sim=-0.1))


# ---------------------------------------------------------------------------
# Derate penalty (reward calculator)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_derate_penalty_change():
    rc = RewardCalculator(
        power_reward_type="None", derate_penalty=2.0, derate_penalty_type="change"
    )
    pen = rc.calculate_derate_penalty(
        old_derates=np.array([0.0, 0.0]), new_derates=np.array([0.4, 0.0])
    )
    assert pen == pytest.approx(2.0 * 0.2)  # mean |change| = 0.2


@pytest.mark.unit
def test_derate_penalty_total():
    rc = RewardCalculator(
        power_reward_type="None", derate_penalty=1.0, derate_penalty_type="total"
    )
    pen = rc.calculate_derate_penalty(
        old_derates=np.zeros(2), new_derates=np.array([0.4, 0.4]), derate_max=0.8
    )
    assert pen == pytest.approx(0.5)  # mean derate / derate_max


@pytest.mark.unit
def test_derate_penalty_reduces_total_reward():
    rc = RewardCalculator(
        power_reward_type="None", derate_penalty=1.0, derate_penalty_type="change"
    )
    reward, breakdown = rc.calculate_total_reward(
        farm_power_deque=[1.0],
        old_yaws=np.zeros(2),
        new_yaws=np.zeros(2),
        yaw_max=30.0,
        old_derates=np.zeros(2),
        new_derates=np.array([0.4, 0.0]),
        derate_max=0.8,
    )
    assert breakdown["derate_penalty"] == pytest.approx(0.2)
    assert reward == pytest.approx(-0.2)


@pytest.mark.unit
def test_invalid_derate_penalty_type_raises():
    with pytest.raises(ValueError, match="derate_penalty_type"):
        RewardCalculator(
            power_reward_type="None", derate_penalty=1.0, derate_penalty_type="bogus"
        )


# ---------------------------------------------------------------------------
# Backend / agent interaction with derating
# ---------------------------------------------------------------------------


def test_adapter_nowake_power_honours_derate(derating_turbine):
    """The pywake adapter's no-wake power must reflect the derate levels."""
    from WindGym.backend.pywake_adapter import PyWakeFlowSimulationAdapter

    fs = PyWakeFlowSimulationAdapter(
        x=np.array([0.0, 400.0]),
        y=np.array([0.0, 0.0]),
        windTurbine=derating_turbine,
        ws=9.0,
        wd=270.0,
        ti=0.06,
    )
    p_full = fs.windTurbines.power(include_wakes=False).copy()

    fs._derate = np.array([0.5, 0.0])
    fs._compute_steady_state()
    p_derated = fs.windTurbines.power(include_wakes=False)

    assert p_derated[0] == pytest.approx(0.5 * p_full[0], rel=1e-3)
    assert p_derated[1] == pytest.approx(p_full[1], rel=1e-6)


def test_pywake_agent_rejects_derating_env(derating_turbine):
    """PyWakeAgent only optimizes yaw; it must refuse derating envs instead
    of silently emitting a wrong-sized action."""
    from WindGym.Agents.PyWakeAgent import PyWakeAgent

    env = make_env(derating_turbine, n_turb=2)
    d = derating_turbine.diameter()
    agent = PyWakeAgent(x_pos=np.arange(2) * 5.0 * d, y_pos=np.zeros(2), env=env)
    with pytest.raises(ValueError, match="derate_action"):
        agent.predict(None)
    env.close()
