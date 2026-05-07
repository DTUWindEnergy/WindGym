"""Per-reset DWM parameter randomization (generic).

Knows only that the wrapped env supports
``reset(options={"dwm_params": {<key>: <value>, ...}})`` (the channel added by
the Phase 1 windgym work). The actual *distribution* the params are drawn
from is supplied by the caller as a ``sampler`` callable, so this wrapper is
reusable for posterior-driven domain randomization, uniform priors, scripted
schedules, etc.
"""
from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np


SamplerFn = Callable[[np.random.Generator], dict]


class DWMRandomizationWrapper(gym.Wrapper):
    """Draw DWM closure parameters from a sampler on each ``reset()``.

    Parameters
    ----------
    env : gym.Env
        Underlying env. Must accept ``reset(options={"dwm_params": ...})``;
        any wrapper above ``WindFarmEnv`` that forwards ``options`` is fine.
    sampler : Callable[[np.random.Generator], dict]
        Each call returns a dict of DWM params for one episode. Receives this
        wrapper's RNG so draws are reproducible from a single seed.
    seed : int, optional
        Seeds the wrapper's internal RNG. If ``None`` the RNG is seeded
        non-deterministically (numpy default).

    Behaviour
    ---------
    - ``reset(seed=...)`` reseeds the wrapper's internal RNG if a seed is
      passed, so end-to-end reproducibility chains through.
    - Caller-supplied ``options["dwm_params"]`` overrides individual keys
      drawn by the sampler (useful for evaluation against a fixed parameter
      vector while otherwise keeping training-time wrappers identical).
    """

    def __init__(
        self,
        env: gym.Env,
        sampler: SamplerFn,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if not callable(sampler):
            raise TypeError("sampler must be callable")
        self._sampler = sampler
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # A caller-supplied seed deterministically pins the full param
            # stream from this point — useful for evaluation and debugging.
            self._rng = np.random.default_rng(seed)

        theta = self._sampler(self._rng)
        if not isinstance(theta, dict):
            raise TypeError(
                f"sampler must return dict, got {type(theta).__name__}"
            )

        merged_options = dict(options or {})
        caller_overrides = merged_options.get("dwm_params") or {}
        # Caller's keys win — they're the explicit override path.
        merged_options["dwm_params"] = {**theta, **caller_overrides}

        return self.env.reset(seed=seed, options=merged_options)
