"""
Tests for WindManager module.

These tests cover wind condition sampling, exception paths, and wind direction
time series generation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from WindGym.core.wind_manager import WindManager, WindConditions


class TestWindConditions:
    """Tests for WindConditions dataclass."""

    def test_wind_conditions_creation(self):
        """Test creating WindConditions."""
        wc = WindConditions(
            wind_speed=10.0, wind_direction=270.0, turbulence_intensity=0.07
        )
        assert wc.wind_speed == 10.0
        assert wc.wind_direction == 270.0
        assert wc.turbulence_intensity == 0.07

    def test_wind_conditions_unpack(self):
        """Test unpacking WindConditions."""
        wc = WindConditions(
            wind_speed=10.0, wind_direction=270.0, turbulence_intensity=0.07
        )
        ws, wd, ti = wc.unpack()
        assert ws == 10.0
        assert wd == 270.0
        assert ti == 0.07


class TestWindManagerInitialization:
    """Tests for WindManager initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with bounds."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        assert wm.ws_min == 5.0
        assert wm.ws_max == 15.0
        assert wm.wd_min == 250.0
        assert wm.wd_max == 290.0
        assert wm.ti_min == 0.05
        assert wm.ti_max == 0.10
        assert wm.np_random is None
        assert wm.sample_site is None

    def test_initialization_with_site(self):
        """Test initialization with PyWake site."""
        mock_site = MagicMock()
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
            sample_site=mock_site,
        )
        assert wm.sample_site is mock_site


class TestSampleConditionsExceptions:
    """Tests for exception paths in sampling."""

    def test_sample_without_np_random_raises(self):
        """Test that sampling without np_random raises RuntimeError."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        # np_random is None

        with pytest.raises(RuntimeError, match="np_random must be set before sampling"):
            wm.sample_conditions()

    def test_random_uniform_without_np_random_raises(self):
        """Test that _random_uniform without np_random raises RuntimeError."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        with pytest.raises(RuntimeError, match="np_random not set"):
            wm._random_uniform(0.0, 1.0)


class TestUniformSampling:
    """Tests for uniform wind condition sampling."""

    def test_sample_uniform(self):
        """Test uniform sampling within bounds."""
        wm = WindManager(
            ws_min=8.0,
            ws_max=12.0,
            wd_min=260.0,
            wd_max=280.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        wm.np_random = np.random.default_rng(seed=42)

        # Sample multiple times and verify bounds
        for _ in range(10):
            wc = wm.sample_conditions()
            assert 8.0 <= wc.wind_speed <= 12.0
            assert 260.0 <= wc.wind_direction <= 280.0
            assert 0.05 <= wc.turbulence_intensity <= 0.10

    def test_sample_uniform_fixed_values(self):
        """Test sampling when min == max (fixed values)."""
        wm = WindManager(
            ws_min=10.0,
            ws_max=10.0,  # Fixed wind speed
            wd_min=270.0,
            wd_max=270.0,  # Fixed wind direction
            ti_min=0.07,
            ti_max=0.07,  # Fixed TI
        )
        wm.np_random = np.random.default_rng(seed=42)

        wc = wm.sample_conditions()
        assert wc.wind_speed == 10.0
        assert wc.wind_direction == 270.0
        assert wc.turbulence_intensity == 0.07

    def test_sample_reproducibility(self):
        """Test that same seed produces same samples."""
        wm1 = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        wm1.np_random = np.random.default_rng(seed=123)

        wm2 = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        wm2.np_random = np.random.default_rng(seed=123)

        wc1 = wm1.sample_conditions()
        wc2 = wm2.sample_conditions()

        assert wc1.wind_speed == wc2.wind_speed
        assert wc1.wind_direction == wc2.wind_direction
        assert wc1.turbulence_intensity == wc2.turbulence_intensity


class TestSiteSampling:
    """Tests for site-based wind condition sampling."""

    def test_sample_from_site(self):
        """Test sampling from PyWake site."""
        wm = WindManager(
            ws_min=3.0,
            ws_max=25.0,
            wd_min=0.0,
            wd_max=360.0,
            ti_min=0.05,
            ti_max=0.10,
            sample_site=MagicMock(),
        )
        wm.np_random = np.random.default_rng(seed=42)

        # Mock the site's local_wind method
        mock_local_wind = MagicMock()
        # Create realistic Weibull parameters
        n_dirs = 360
        mock_local_wind.Sector_frequency_ilk = np.ones((1, n_dirs, 1)) / n_dirs
        mock_local_wind.Weibull_A_ilk = np.full((1, n_dirs, 1), 9.0)  # A parameter
        mock_local_wind.Weibull_k_ilk = np.full((1, n_dirs, 1), 2.0)  # k parameter
        wm.sample_site.local_wind.return_value = mock_local_wind

        wc = wm.sample_conditions()

        # Values should be clipped to bounds
        assert 3.0 <= wc.wind_speed <= 25.0
        assert 0.0 <= wc.wind_direction <= 360.0
        # TI is still uniform even with site
        assert 0.05 <= wc.turbulence_intensity <= 0.10


class TestWindDirectionList:
    """Tests for wind direction time series generation."""

    def test_make_wind_direction_list_constant(self):
        """Test generating constant wind direction list."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        wd_list = wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=10.0,
            dt_sim=1.0,
            t_developed=2.0,
            steps_on_reset=1,
            wd_function=None,  # Constant
        )

        # All values should be 270.0
        assert all(wd == 270.0 for wd in wd_list)
        # Length should be burn-in + episode steps
        assert len(wd_list) > 0

    def test_make_wind_direction_list_varying(self):
        """Test generating time-varying wind direction list."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        # Define a simple time-varying function
        def wd_function(t):
            return 270.0 + t  # Increase by 1 deg per second

        wd_list = wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=10.0,
            dt_sim=1.0,
            t_developed=2.0,
            steps_on_reset=1,
            wd_function=wd_function,
        )

        # First value should be base_wd
        assert wd_list[0] == 270.0

        # After burn-in, values should vary according to function
        # Burn-in steps = ceil(t_developed/dt_sim) + steps_on_reset = 3
        burn_in_steps = 3
        # Episode values start after burn-in
        for i, wd in enumerate(wd_list[burn_in_steps:]):
            t = i * 1.0  # dt_sim = 1.0
            expected = wd_function(t)
            assert wd == expected, f"At index {i}, expected {expected}, got {wd}"

    def test_make_wind_direction_list_first_value(self):
        """Test that first value is always base_wd."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        def wd_function(t):
            return 300.0  # Different from base_wd

        wd_list = wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=5.0,
            dt_sim=1.0,
            t_developed=1.0,
            steps_on_reset=0,
            wd_function=wd_function,
        )

        # First value should always be base_wd, not wd_function(0)
        assert wd_list[0] == 270.0

    def test_make_wind_direction_list_short_episode(self):
        """Test with very short episode."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        wd_list = wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=1.0,  # Very short
            dt_sim=1.0,
            t_developed=0.0,
            steps_on_reset=0,
            wd_function=None,
        )

        assert len(wd_list) >= 1
        assert wd_list[0] == 270.0

    def test_two_arg_wd_function_receives_base_wd(self):
        """A ``wd_function(t, base_wd)`` gets the episode's base direction.

        Randomized TRAINING schedules are relative -- they return
        ``base_wd + delta(t)`` -- so that a time-varying wd composes with the
        env's per-episode wd domain randomization instead of replacing it.
        """
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        def wd_function(t, base_wd):
            return base_wd + t  # relative: delta(0) == 0

        wd_list = wm.make_wind_direction_list(
            base_wd=263.0,
            time_max=5.0,
            dt_sim=1.0,
            t_developed=1.0,
            steps_on_reset=0,
            wd_function=wd_function,
        )

        burn_in_steps = 1  # ceil(t_developed/dt_sim) + steps_on_reset
        for i, wd in enumerate(wd_list[burn_in_steps:]):
            assert wd == 263.0 + i * 1.0, f"index {i}: got {wd}"

    def test_two_arg_callable_object_receives_base_wd(self):
        """Training schedules are callable *objects*, not plain functions.

        ``inspect.signature`` must resolve through ``__call__`` (and drop
        ``self``), otherwise the real schedules fall back to the 1-arg path and
        raise at call time.
        """

        class Schedule:
            def __call__(self, t, base_wd):
                return base_wd + 2.0 * t

        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        wd_list = wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=3.0,
            dt_sim=1.0,
            t_developed=0.0,
            steps_on_reset=0,
            wd_function=Schedule(),
        )

        assert wd_list[0] == 270.0
        assert wd_list[1] == 272.0
        assert wd_list[2] == 274.0

    def test_wd_function_is_called_from_t_zero_once_per_list(self):
        """Training schedules re-draw their episode on the t == 0.0 call, so the
        list build must hit t=0 exactly once and then sweep upward."""
        seen = []

        def wd_function(t, base_wd):
            seen.append(t)
            return base_wd

        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )

        wm.make_wind_direction_list(
            base_wd=270.0,
            time_max=10.0,
            dt_sim=1.0,
            t_developed=2.0,
            steps_on_reset=1,
            wd_function=wd_function,
        )

        assert seen.count(0.0) == 1
        assert seen == sorted(seen)


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_narrow_bounds(self):
        """Test with very narrow sampling bounds."""
        wm = WindManager(
            ws_min=9.99,
            ws_max=10.01,
            wd_min=269.9,
            wd_max=270.1,
            ti_min=0.069,
            ti_max=0.071,
        )
        wm.np_random = np.random.default_rng(seed=42)

        wc = wm.sample_conditions()
        assert 9.99 <= wc.wind_speed <= 10.01
        assert 269.9 <= wc.wind_direction <= 270.1
        assert 0.069 <= wc.turbulence_intensity <= 0.071

    def test_full_wind_direction_range(self):
        """Test sampling over full wind direction range."""
        wm = WindManager(
            ws_min=10.0,
            ws_max=10.0,
            wd_min=0.0,
            wd_max=360.0,
            ti_min=0.07,
            ti_max=0.07,
        )
        wm.np_random = np.random.default_rng(seed=42)

        # Sample many times to cover range
        directions = [wm.sample_conditions().wind_direction for _ in range(100)]

        # Should have variation across the range
        assert min(directions) < 90
        assert max(directions) > 270

    def test_sample_conditions_returns_windconditions(self):
        """Test that sample_conditions returns WindConditions object."""
        wm = WindManager(
            ws_min=5.0,
            ws_max=15.0,
            wd_min=250.0,
            wd_max=290.0,
            ti_min=0.05,
            ti_max=0.10,
        )
        wm.np_random = np.random.default_rng(seed=42)

        result = wm.sample_conditions()
        assert isinstance(result, WindConditions)
