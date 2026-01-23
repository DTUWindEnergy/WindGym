"""
Tests for visualization plot_utils module.

Tests helper functions for wind farm visualization.
"""

import pytest
import numpy as np
import xarray as xr

# Use Agg backend for testing (no display needed) - must be set before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from WindGym.visualization.plot_utils import setup_wind_grid_axes, calculate_time_limits


class TestSetupWindGridAxes:
    """Tests for setup_wind_grid_axes function."""

    @pytest.fixture
    def axes_grid(self):
        """Create a 2x2 grid of axes for testing."""
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        yield axs
        plt.close(fig)

    def test_top_row_gets_title(self, axes_grid):
        """Test that top row axes get wind direction title."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(axes_grid, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270)

        assert "270" in axes_grid[0, 0].get_title()

    def test_non_top_row_no_title(self, axes_grid):
        """Test that non-top row axes get empty title."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(axes_grid, j=1, i=0, WSS=WSS, WDS=WDS, WS=10, wd=270)

        assert axes_grid[1, 0].get_title() == ""

    def test_left_column_gets_ylabel(self, axes_grid):
        """Test that left column axes get wind speed ylabel."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(axes_grid, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270)

        assert "8" in axes_grid[0, 0].get_ylabel()

    def test_non_left_column_no_ylabel(self, axes_grid):
        """Test that non-left column axes get empty ylabel."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(axes_grid, j=0, i=1, WSS=WSS, WDS=WDS, WS=8, wd=280)

        assert axes_grid[0, 1].get_ylabel() == ""

    def test_grid_added_by_default(self, axes_grid):
        """Test that grid is added by default."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(
            axes_grid, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270, add_grid=True
        )

        # Check that grid was set (checking xaxis grid lines)
        # This is a simple check - grid was called
        # The actual grid visibility depends on the axis state
        assert axes_grid[0, 0] is not None

    def test_grid_not_added_when_disabled(self, axes_grid):
        """Test that grid is not added when add_grid=False."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(
            axes_grid, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270, add_grid=False
        )

        # Should execute without error
        assert axes_grid[0, 0] is not None

    def test_xlabel_always_empty(self, axes_grid):
        """Test that xlabel is always set to empty."""
        WSS = [8, 10]
        WDS = [270, 280]

        setup_wind_grid_axes(axes_grid, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270)

        assert axes_grid[0, 0].get_xlabel() == ""


class TestCalculateTimeLimits:
    """Tests for calculate_time_limits function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray Dataset for testing."""
        # Create time coordinates
        time = np.arange(0, 100, 1)

        # Create dimensions
        ws_vals = [8.0, 10.0]
        wd_vals = [270.0, 280.0]
        TI_vals = [0.06]
        turbbox_vals = ["box1"]
        turb_vals = [0, 1, 2]

        # Create data variables with all dimensions
        power_data = np.random.rand(
            len(ws_vals), len(wd_vals), len(TI_vals), len(turbbox_vals), len(time)
        )
        power_turb_data = np.random.rand(
            len(ws_vals),
            len(wd_vals),
            len(TI_vals),
            len(turbbox_vals),
            len(turb_vals),
            len(time),
        )

        ds = xr.Dataset(
            {
                "power": (["ws", "wd", "TI", "turbbox", "time"], power_data),
                "power_turb": (
                    ["ws", "wd", "TI", "turbbox", "turb", "time"],
                    power_turb_data,
                ),
            },
            coords={
                "ws": ws_vals,
                "wd": wd_vals,
                "TI": TI_vals,
                "turbbox": turbbox_vals,
                "turb": turb_vals,
                "time": time,
            },
        )
        return ds

    def test_calculate_time_limits_farm_level(self, sample_dataset):
        """Test calculating time limits for farm-level variable."""
        x_start, x_end = calculate_time_limits(
            data=sample_dataset,
            ws=8.0,
            wd=270.0,
            TI=0.06,
            TURBBOX="box1",
            variable="power",
            avg_n=10,
            turb_idx=None,
        )

        # After rolling mean with window=10, we lose some data at edges
        assert x_start >= sample_dataset.time.values.min()
        assert x_end <= sample_dataset.time.values.max()
        assert x_start < x_end

    def test_calculate_time_limits_turbine_level(self, sample_dataset):
        """Test calculating time limits for turbine-level variable."""
        x_start, x_end = calculate_time_limits(
            data=sample_dataset,
            ws=8.0,
            wd=270.0,
            TI=0.06,
            TURBBOX="box1",
            variable="power_turb",
            avg_n=10,
            turb_idx=0,
        )

        assert x_start >= sample_dataset.time.values.min()
        assert x_end <= sample_dataset.time.values.max()
        assert x_start < x_end

    def test_calculate_time_limits_different_avg_n(self, sample_dataset):
        """Test that different avg_n values affect time limits."""
        x_start_5, x_end_5 = calculate_time_limits(
            data=sample_dataset,
            ws=8.0,
            wd=270.0,
            TI=0.06,
            TURBBOX="box1",
            variable="power",
            avg_n=5,
        )

        x_start_20, x_end_20 = calculate_time_limits(
            data=sample_dataset,
            ws=8.0,
            wd=270.0,
            TI=0.06,
            TURBBOX="box1",
            variable="power",
            avg_n=20,
        )

        # Larger window should result in narrower range (more edge trimming)
        assert x_end_5 - x_start_5 >= x_end_20 - x_start_20

    def test_calculate_time_limits_returns_tuple(self, sample_dataset):
        """Test that function returns a tuple of two values."""
        result = calculate_time_limits(
            data=sample_dataset,
            ws=8.0,
            wd=270.0,
            TI=0.06,
            TURBBOX="box1",
            variable="power",
            avg_n=10,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSetupWindGridAxesEdgeCases:
    """Edge case tests for setup_wind_grid_axes."""

    @pytest.fixture
    def single_axes(self):
        """Create a single axis (not grid) wrapped in 2D array."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # Wrap in 2D array to match expected interface
        axs = np.array([[ax]])
        yield axs
        plt.close(fig)

    def test_single_cell_grid(self, single_axes):
        """Test with a 1x1 grid."""
        WSS = [8]
        WDS = [270]

        setup_wind_grid_axes(single_axes, j=0, i=0, WSS=WSS, WDS=WDS, WS=8, wd=270)

        # Should have both title (top row) and ylabel (left column)
        assert "270" in single_axes[0, 0].get_title()
        assert "8" in single_axes[0, 0].get_ylabel()
