"""
Tests for WindFarmRenderer.

Tests rendering functionality including RGB array generation and
various rendering modes.
"""

import pytest
import numpy as np
from pathlib import Path

# Use non-interactive backend for testing - must be set before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid
from WindGym.core.renderer import WindFarmRenderer


@pytest.fixture
def base_example_data_path():
    """Provides path to the example configuration directory."""
    return Path("examples/EnvConfigs")


@pytest.fixture
def renderer_none():
    """Create a renderer with no render mode."""
    return WindFarmRenderer(render_mode=None)


@pytest.fixture
def renderer_rgb():
    """Create a renderer for RGB array output."""
    return WindFarmRenderer(render_mode="rgb_array")


@pytest.fixture
def renderer_human():
    """Create a renderer for human viewing."""
    return WindFarmRenderer(render_mode="human")


@pytest.fixture
def env_with_rendering(base_example_data_path):
    """Create an environment for rendering tests."""
    yaml_path = base_example_data_path / Path("Env1.yaml")
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=2, xDist=5, yDist=5)
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=yaml_path,
        render_mode="rgb_array",
    )
    env.reset()
    yield env
    env.close()


class TestRendererInitialization:
    """Tests for renderer initialization."""

    def test_renderer_init_none_mode(self, renderer_none):
        """Test renderer initializes with None mode."""
        assert renderer_none.render_mode is None
        assert renderer_none.figure is None
        assert renderer_none.ax is None
        assert renderer_none.view is None

    def test_renderer_init_rgb_mode(self, renderer_rgb):
        """Test renderer initializes with rgb_array mode."""
        assert renderer_rgb.render_mode == "rgb_array"

    def test_renderer_init_human_mode(self, renderer_human):
        """Test renderer initializes with human mode."""
        assert renderer_human.render_mode == "human"


class TestRendererRenderMethod:
    """Tests for the main render method."""

    def test_render_returns_none_for_none_mode(self, env_with_rendering):
        """Test that render returns None when render_mode is None."""
        renderer = WindFarmRenderer(render_mode=None)
        result = renderer.render(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )
        assert result is None

    def test_render_rgb_returns_array(self, env_with_rendering):
        """Test that rgb_array mode returns a numpy array."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        result = renderer.render(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3  # Height x Width x Channels
        assert result.shape[2] == 3  # RGB channels

    def test_render_human_returns_none(self, env_with_rendering):
        """Test that human mode returns None (displays instead)."""
        # With Agg backend, plt.show() and plt.pause() are no-ops
        renderer = WindFarmRenderer(render_mode="human")
        result = renderer.render(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )
        assert result is None
        plt.close("all")


class TestRendererInitRender:
    """Tests for init_render method."""

    def test_init_render_creates_figure(self, env_with_rendering):
        """Test that init_render creates matplotlib objects."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        assert renderer.figure is not None
        assert renderer.ax is not None
        assert renderer.view is not None
        assert renderer.a is not None
        assert renderer.b is not None

        plt.close("all")


class TestRendererFrameRendering:
    """Tests for frame rendering methods."""

    def test_render_frame_returns_rgb_array(self, env_with_rendering):
        """Test that _render_frame returns valid RGB array."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frame = renderer._render_frame(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        # Frame should have reasonable dimensions
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

        plt.close("all")

    def test_render_frame_for_human_returns_rgb(self, env_with_rendering):
        """Test that _render_frame_for_human returns valid RGB array."""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frame = renderer._render_frame_for_human(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

        plt.close("all")

    def test_render_frame_lazy_init(self, env_with_rendering):
        """Test that _render_frame initializes lazily if needed."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        # Don't call init_render - should be done lazily
        assert renderer.view is None

        frame = renderer._render_frame(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )

        # View should now be initialized
        assert renderer.view is not None
        assert isinstance(frame, np.ndarray)

        plt.close("all")

    def test_render_frame_raises_without_turbine(self, env_with_rendering):
        """Test that _render_frame raises if view not initialized and no turbine."""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        with pytest.raises(RuntimeError, match="Renderer not initialized"):
            renderer._render_frame(env_with_rendering.fs, turbine=None)


class TestRendererPlotMethods:
    """Tests for plot_farm and plot_frame methods."""

    def test_plot_farm(self, env_with_rendering):
        """Test plot_farm method executes without error."""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        # Should not raise
        renderer.plot_farm(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )

        plt.close("all")

    def test_plot_frame(self, env_with_rendering):
        """Test plot_frame method executes without error."""
        renderer = WindFarmRenderer(render_mode="rgb_array")

        # Should not raise
        renderer.plot_frame(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
        )

        plt.close("all")


class TestRendererClose:
    """Tests for renderer close method."""

    def test_close_clears_objects(self, env_with_rendering):
        """Test that close clears matplotlib objects."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        # Objects should be set
        assert renderer.figure is not None
        assert renderer.ax is not None
        assert renderer.view is not None

        renderer.close()

        # Objects should be cleared
        assert renderer.figure is None
        assert renderer.ax is None
        assert renderer.view is None


class TestRendererWithBaseline:
    """Tests for rendering with baseline comparison."""

    def test_render_frame_with_baseline(self, env_with_rendering):
        """Test rendering with baseline flow simulation."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        # Render with baseline flag
        frame = renderer._render_frame(
            env_with_rendering.fs,
            fs_baseline=env_with_rendering.fs_baseline,
            baseline=True,
            turbine=env_with_rendering.turbine,
        )

        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3

        plt.close("all")


class TestRendererEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_render_with_wind_speed_scale(self, env_with_rendering):
        """Test rendering with explicit wind speed for color scale."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frame = renderer._render_frame(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
            ws=10.0,  # Explicit wind speed for color scale
        )

        assert isinstance(frame, np.ndarray)

        plt.close("all")

    def test_multiple_renders(self, env_with_rendering):
        """Test that multiple renders work correctly."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frames = []
        for _ in range(3):
            frame = renderer._render_frame(
                env_with_rendering.fs,
                turbine=env_with_rendering.turbine,
            )
            frames.append(frame)

        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape[2] == 3

        plt.close("all")
