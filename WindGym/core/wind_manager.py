"""
Wind condition management module for WindGym environments.

This module handles wind speed, wind direction, and turbulence intensity sampling,
including support for site-based sampling using PyWake sites.
"""

from typing import Optional, Callable
from dataclasses import dataclass
import inspect
import numpy as np
import math


@dataclass
class WindConditions:
    """Container for wind conditions."""

    wind_speed: float  # m/s
    wind_direction: float  # degrees
    turbulence_intensity: float  # fraction (0-1)
    veer: float = 0.0  # linear veer rate (deg per 100 m), 0 at hub height

    def unpack(self):
        """Unpack wind conditions as tuple (veer accessed explicitly via .veer)."""
        return self.wind_speed, self.wind_direction, self.turbulence_intensity


class WindManager:
    """
    Manages wind condition sampling and wind direction time series generation.

    Supports two sampling modes:
    1. Uniform sampling within specified ranges
    2. Site-based sampling using PyWake site data (Weibull distributions)

    Also handles generation of time-varying wind direction sequences.
    """

    def __init__(
        self,
        ws_min: float,
        ws_max: float,
        wd_min: float,
        wd_max: float,
        ti_min: float,
        ti_max: float,
        sample_site: Optional[object] = None,
        veer_min: float = 0.0,
        veer_max: float = 0.0,
    ):
        """
        Initialize the wind manager.

        Args:
            ws_min: Minimum wind speed (m/s)
            ws_max: Maximum wind speed (m/s)
            wd_min: Minimum wind direction (degrees)
            wd_max: Maximum wind direction (degrees)
            ti_min: Minimum turbulence intensity (fraction)
            ti_max: Maximum turbulence intensity (fraction)
            sample_site: Optional PyWake site for realistic wind sampling
            veer_min: Minimum veer rate (deg per 100 m, positive = wd
                increases with height, 0 at hub height)
            veer_max: Maximum veer rate (deg per 100 m)
        """
        self.ws_min = ws_min
        self.ws_max = ws_max
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.ti_min = ti_min
        self.ti_max = ti_max
        self.sample_site = sample_site
        self.veer_min = veer_min
        self.veer_max = veer_max

        # Random number generator (set by environment)
        self.np_random = None

    def sample_conditions(self) -> WindConditions:
        """
        Sample wind speed, direction, and turbulence intensity.

        Returns:
            WindConditions: Sampled wind conditions
        """
        if self.np_random is None:
            raise RuntimeError(
                "np_random must be set before sampling. "
                "Call env.reset() to initialize the random generator."
            )

        if self.sample_site is None:
            return self._sample_uniform()
        else:
            return self._sample_from_site()

    def _sample_uniform(self) -> WindConditions:
        """
        Sample wind conditions uniformly from specified ranges.

        Returns:
            WindConditions: Uniformly sampled wind conditions
        """
        ws = self._random_uniform(self.ws_min, self.ws_max)
        wd = self._random_uniform(self.wd_min, self.wd_max)
        ti = self._random_uniform(self.ti_min, self.ti_max)
        veer = self._sample_veer()

        return WindConditions(
            wind_speed=ws, wind_direction=wd, turbulence_intensity=ti, veer=veer
        )

    def _sample_from_site(self) -> WindConditions:
        """
        Sample wind conditions from a PyWake site using Weibull distributions.

        The site provides sector frequencies and Weibull parameters for
        realistic wind condition sampling.

        Returns:
            WindConditions: Site-based sampled wind conditions
        """
        # Wind resource from site
        dirs = np.arange(0, 360, 1)  # wind directions (degrees)
        ws = np.arange(3, 25, 1)  # wind speeds (m/s)

        # Get local wind characteristics
        local_wind = self.sample_site.local_wind(x=0, y=0, wd=dirs, ws=ws)
        freqs = local_wind.Sector_frequency_ilk[0, :, 0]
        As = local_wind.Weibull_A_ilk[0, :, 0]  # Weibull A parameter
        ks = local_wind.Weibull_k_ilk[0, :, 0]  # Weibull k parameter

        # Sample wind direction and speed
        wd, ws = self._sample_weibull_wind(dirs, As, ks, freqs)

        # Clip to specified ranges
        wd = np.clip(wd, self.wd_min, self.wd_max)
        ws = np.clip(ws, self.ws_min, self.ws_max)

        # TI is still uniformly sampled (not provided by site)
        ti = self._random_uniform(self.ti_min, self.ti_max)
        veer = self._sample_veer()

        return WindConditions(
            wind_speed=ws, wind_direction=wd, turbulence_intensity=ti, veer=veer
        )

    def _sample_veer(self) -> float:
        """
        Sample the veer rate (deg per 100 m).

        When the interval is degenerate (min == max, the default 0/0) no RNG
        draw is made, so pre-veer seeded runs keep an identical random stream.
        """
        if self.veer_min == self.veer_max:
            return float(self.veer_min)
        return self._random_uniform(self.veer_min, self.veer_max)

    def _sample_weibull_wind(self, dirs, As, ks, freqs):
        """
        Sample wind direction and speed from Weibull distributions.

        Args:
            dirs: Array of wind directions (degrees)
            As: Weibull A parameters for each sector
            ks: Weibull k parameters for each sector
            freqs: Sector frequencies (probabilities)

        Returns:
            tuple: (wind_direction, wind_speed)
        """
        # Sample direction sector based on frequency
        idx = self.np_random.choice(np.arange(dirs.size), 1, p=freqs)

        # Get direction and Weibull parameters for selected sector
        wd = dirs[idx]
        A = As[idx]
        k = ks[idx]

        # Sample wind speed from Weibull distribution
        ws = A * self.np_random.weibull(k)

        return wd.item(), ws.item()

    def make_wind_direction_list(
        self,
        base_wd: float,
        time_max: float,
        dt_sim: float,
        t_developed: float,
        steps_on_reset: int,
        wd_function: Optional[Callable[[float], float]] = None,
    ) -> list:
        """
        Generate a time series of wind directions for an episode.

        The wind direction list has two phases:
        1. Burn-in/steady-state period: Constant wind direction
        2. Episode period: Either constant or time-varying (if wd_function provided)

        Args:
            base_wd: Base wind direction to start with (degrees)
            time_max: Maximum simulation time for the episode (seconds)
            dt_sim: Simulation timestep (seconds)
            t_developed: Time for flow to develop (seconds)
            steps_on_reset: Number of environment steps during reset
            wd_function: Optional callable for time-varying wind. Either
                ``f(t) -> wd`` (absolute schedule) or ``f(t, base_wd) -> wd``
                (relative schedule, so the episode's randomized base direction
                is preserved). The arity is detected automatically.

        Returns:
            list: Wind direction for each simulation timestep
        """
        # Calculate total number of simulation steps
        num_sim_steps = math.ceil(time_max / dt_sim) + 1

        # Calculate steady-state period (burn-in + reset steps)
        steady_state_steps = math.ceil(t_developed / dt_sim) + steps_on_reset

        wd_list = []

        # Phase 1: Burn-in period with constant wind direction
        wd_list.extend([base_wd] * int(steady_state_steps))

        # Phase 2: Episode period
        if wd_function is None:
            # Constant wind direction
            wd_list.extend([base_wd] * num_sim_steps)
        else:
            # Time-varying wind direction. Relative schedules take base_wd as a
            # second argument so they can return base_wd + delta(t) and leave
            # the per-episode wd randomization intact; absolute schedules take
            # time alone. Detect which we were handed (works for plain
            # functions and for callable objects, whose signature resolves
            # through __call__ with self already bound).
            wants_base_wd = self._wd_function_takes_base_wd(wd_function)
            for i in range(num_sim_steps):
                t = i * dt_sim
                if wants_base_wd:
                    wd_list.append(wd_function(t, base_wd))
                else:
                    wd_list.append(wd_function(t))

        # Ensure first value matches base_wd for consistency
        wd_list[0] = base_wd

        return wd_list

    @staticmethod
    def _wd_function_takes_base_wd(wd_function: Callable) -> bool:
        """True if ``wd_function`` accepts a second positional ``base_wd`` argument.

        Falls back to the single-argument convention when the signature cannot
        be introspected (builtins, C callables), which is the pre-existing
        behaviour for every registered evaluation schedule.
        """
        try:
            params = [
                p for p in inspect.signature(wd_function).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            return False
        return len(params) >= 2

    def _random_uniform(self, min_val: float, max_val: float) -> float:
        """
        Generate a random value uniformly distributed between min and max.

        Args:
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            float: Random value in [min_val, max_val]
        """
        if self.np_random is None:
            raise RuntimeError("np_random not set")
        return float(self.np_random.uniform(min_val, max_val))
