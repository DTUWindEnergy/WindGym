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

from dynamiks.views import XYView, EastNorthView
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

    def test_init_render_creates_view(self, env_with_rendering):
        """Test that init_render creates the view and grid extents."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

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

    def test_render_frame_with_probes_returns_rgb(self, env_with_rendering):
        """Test that _render_frame with probes returns valid RGB array."""
        renderer = WindFarmRenderer(render_mode="human")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frame = renderer._render_frame(
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
        """Test that close clears the view."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        assert renderer.view is not None

        renderer.close()

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


class TestEnvRenderArguments:
    """Tests for env.render() argument forwarding and defaults."""

    def test_fix_turbines_default_is_false(self, env_with_rendering):
        """env.render() should default to fix_turbines=False."""
        result = env_with_rendering.render()
        assert isinstance(result, np.ndarray)
        plt.close("all")

    def test_fix_turbines_true_returns_array(self, env_with_rendering):
        """env.render(fix_turbines=True) should produce a valid RGB array."""
        result = env_with_rendering.render(fix_turbines=True)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3
        plt.close("all")

    def test_fix_turbines_false_returns_array(self, env_with_rendering):
        """env.render(fix_turbines=False) should produce a valid RGB array."""
        result = env_with_rendering.render(fix_turbines=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3
        plt.close("all")

    def test_show_indices_updates_renderer(self, env_with_rendering):
        """env.render(show_indices=...) should update renderer.show_indices."""
        env_with_rendering.render(show_indices=False)
        assert env_with_rendering.renderer.show_indices is False
        env_with_rendering.render(show_indices=True)
        assert env_with_rendering.renderer.show_indices is True
        plt.close("all")

    def test_fontsize_updates_renderer(self, env_with_rendering):
        """env.render(fontsize=...) should update renderer.fontsize."""
        env_with_rendering.render(fontsize=20)
        assert env_with_rendering.renderer.fontsize == 20
        plt.close("all")

    def test_axes_lw_updates_renderer(self, env_with_rendering):
        """env.render(axes_lw=...) should update renderer.axes_lw."""
        env_with_rendering.render(axes_lw=2.5)
        assert env_with_rendering.renderer.axes_lw == 2.5
        plt.close("all")

    def test_colorbar_vmax_step_updates_renderer(self, env_with_rendering):
        """env.render(colorbar_vmax_step=...) should update renderer.colorbar_vmax_step."""
        env_with_rendering.render(colorbar_vmax_step=5.0)
        assert env_with_rendering.renderer.colorbar_vmax_step == 5.0
        plt.close("all")

    def test_unset_args_do_not_change_renderer_defaults(self, env_with_rendering):
        """Calling render() without optional args should leave renderer state unchanged."""
        renderer = env_with_rendering.renderer
        original = (
            renderer.show_indices,
            renderer.fontsize,
            renderer.axes_lw,
            renderer.colorbar_vmax_step,
        )
        env_with_rendering.render()
        assert (
            renderer.show_indices,
            renderer.fontsize,
            renderer.axes_lw,
            renderer.colorbar_vmax_step,
        ) == original
        plt.close("all")


class TestFixTurbinesView:
    """Tests that fix_turbines selects the correct view type in _build_view."""

    def test_fix_turbines_false_uses_xyview(self, env_with_rendering):
        """fix_turbines=False should build an XYView."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)
        view, *_ = renderer._build_view(env_with_rendering.fs, fix_turbines=False)
        assert isinstance(view, XYView)

    def test_fix_turbines_true_uses_eastnorthview(self, env_with_rendering):
        """fix_turbines=True should build an EastNorthView."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)
        view, *_ = renderer._build_view(env_with_rendering.fs, fix_turbines=True)
        assert isinstance(view, EastNorthView)


class TestRendererEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_render_with_wind_speed_scale(self, env_with_rendering):
        """Test rendering with explicit wind speed for color scale."""
        renderer = WindFarmRenderer(render_mode="rgb_array")
        renderer.init_render(env_with_rendering.fs, env_with_rendering.turbine)

        frame = renderer._render_frame(
            env_with_rendering.fs,
            turbine=env_with_rendering.turbine,
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
