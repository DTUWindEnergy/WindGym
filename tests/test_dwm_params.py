"""Tests for per-reset DWM parameter override (the domain-randomization hook).

Covers both validation paths (init and reset) and a regression check that the
override actually changes the DWM simulation rather than being silently
swallowed somewhere.
"""
from pathlib import Path

import numpy as np
import pytest
from dynamiks.sites.turbulence_fields import MannTurbulenceField
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid


CONFIG = Path("examples/EnvConfigs/2turb.yaml")


@pytest.fixture(scope="session")
def mann_turbulence_field():
    return MannTurbulenceField.generate(
        alphaepsilon=0.1,
        L=33.6,
        Gamma=3.9,
        Nxyz=(1024, 128, 32),
        dxyz=(3.0, 3.0, 3.0),
        seed=1234,
    )


@pytest.fixture
def env_kwargs():
    turbine = V80()
    x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=4, yDist=4)
    # n_passthrough=1.5 gives enough sim time for the upstream wake to reach
    # the downstream turbine (4D ≈ 32 steps at WS=10 m/s, dt=1s).
    # turbtype="Random" (rather than "None") provides ambient TI so keck's
    # k1·TI eddy-viscosity term is non-zero — without it, k1 changes have no
    # effect. The Mann-box generator is monkeypatched in the env fixture so
    # this is still fast.
    return dict(
        turbine=turbine,
        x_pos=x_pos,
        y_pos=y_pos,
        n_passthrough=1.5,
        config=CONFIG,
        turbtype="Random",
        burn_in_passthroughs=0.0001,
    )


@pytest.fixture
def env(env_kwargs, mann_turbulence_field, monkeypatch):
    """A ready-to-use WindFarmEnv with the Mann box mocked for speed."""
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate",
        lambda *a, **kw: mann_turbulence_field,
    )
    e = WindFarmEnv(**env_kwargs)
    yield e
    e.close()


def test_dwm_params_init_unknown_key_raises(env_kwargs):
    """A typo in the constructor's dwm_params should fail loudly."""
    with pytest.raises(ValueError, match="Unknown dwm_params keys"):
        WindFarmEnv(
            **env_kwargs,
            dwm_params={"not_a_real_param": 1.0},
            reset_init=False,  # don't run reset; we only need to hit validation
        )


def test_dwm_params_reset_unknown_key_raises(env):
    """A typo in reset(options=) should fail before any heavy DWM work."""
    with pytest.raises(ValueError, match="Unknown dwm_params keys in reset options"):
        env.reset(options={"dwm_params": {"k_seven": 1.0}})


def test_dwm_params_reset_changes_simulation(env, mann_turbulence_field, monkeypatch):
    """Very different k1 must produce different observations after wake propagation.

    Same seed and (mocked) Mann box on both sides; only the DWM closure constant
    differs. We step the env long enough for the upstream-turbine wake to reach
    the downstream turbine (~4D / 10 m/s ≈ 32 steps at dt=1s for V80), since k1
    only affects wake *recovery* and is invisible until a wake is observed.
    """
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate",
        lambda *a, **kw: mann_turbulence_field,
    )
    n_steps = 40
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def rollout(theta):
        env.reset(seed=0, options={"dwm_params": theta})
        obs_seq = []
        for _ in range(n_steps):
            obs, _, term, trunc, _ = env.step(zero_action)
            obs_seq.append(obs)
            if term or trunc:
                break
        return np.concatenate(obs_seq)

    obs_low = rollout({"k1": 0.005, "k2": 0.005})
    obs_high = rollout({"k1": 0.20, "k2": 0.005})

    assert not np.allclose(obs_low, obs_high), (
        "Identical observations across k1=0.005 vs k1=0.20 — the override is "
        "likely not being plumbed into make_dwm()."
    )


def test_d_particle_changes_simulation(env, mann_turbulence_field, monkeypatch):
    """Very different d_particle must produce different observations after wake propagation.

    Mirrors test_dwm_params_reset_changes_simulation but exercises the
    streamwise particle-spacing knob now that it's a calibrated parameter.
    """
    monkeypatch.setattr(
        "dynamiks.sites.turbulence_fields.MannTurbulenceField.generate",
        lambda *a, **kw: mann_turbulence_field,
    )
    n_steps = 40
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def rollout(theta):
        env.reset(seed=0, options={"dwm_params": theta})
        obs_seq = []
        for _ in range(n_steps):
            obs, _, term, trunc, _ = env.step(zero_action)
            obs_seq.append(obs)
            if term or trunc:
                break
        return np.concatenate(obs_seq)

    obs_dense = rollout({"d_particle": 0.2})
    obs_sparse = rollout({"d_particle": 1.0})

    assert not np.allclose(obs_dense, obs_sparse), (
        "Identical observations across d_particle=0.2 vs d_particle=1.0 — the "
        "override is likely not being plumbed into make_dwm()."
    )
