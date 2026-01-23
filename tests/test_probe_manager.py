"""
Tests for ProbeManager module.

These tests cover probe initialization, positioning, and reading functionality
for both free placement and turbine-relative probe modes.
"""

import pytest
import numpy as np
import math
from unittest.mock import MagicMock, patch

from WindGym.core.probe_manager import ProbeManager


@pytest.fixture
def mock_fs():
    """Create a mock flow simulation object."""
    fs = MagicMock()

    # Mock wind turbine positions (2 turbines)
    fs.windTurbines.positions_xyz = np.array(
        [
            [0.0, 500.0],  # x positions
            [0.0, 0.0],  # y positions
            [90.0, 90.0],  # z positions (hub height)
        ]
    )
    fs.windTurbines.rotor_positions_xyz = fs.windTurbines.positions_xyz

    # Mock wind speed reading
    fs.get_windspeed.return_value = np.array([10.0, 0.5, 0.0])
    fs.get_turbulence_intensity.return_value = 0.10

    return fs


@pytest.fixture
def mock_env():
    """Create a mock environment object for free placement probes."""
    env = MagicMock()
    env.fs = MagicMock()
    return env


class TestProbeManagerInitialization:
    """Tests for ProbeManager initialization."""

    def test_empty_initialization(self):
        """Test initialization with no probes."""
        pm = ProbeManager()
        assert pm.probes == []
        assert pm.probes_config == []
        assert len(pm.turbine_probes) == 0

    def test_initialization_with_config(self):
        """Test initialization with probe configurations."""
        config = [
            {"position": [100, 50, 90], "probe_type": "WS", "name": "probe_1"},
            {"position": [200, 50, 90], "probe_type": "TI", "name": "probe_2"},
        ]
        pm = ProbeManager(probes_config=config)
        assert pm.probes_config == config
        assert len(pm.probes) == 0  # Probes not initialized yet

    def test_count_probes_per_turbine(self):
        """Test counting probes per turbine."""
        config = [
            {"turbine_index": 0, "relative_position": [50, 0, 0], "probe_type": "WS"},
            {"turbine_index": 0, "relative_position": [100, 0, 0], "probe_type": "WS"},
            {"turbine_index": 1, "relative_position": [50, 0, 0], "probe_type": "WS"},
            {
                "position": [500, 500, 90],
                "probe_type": "TI",
            },  # Free placement, no turbine
        ]
        pm = ProbeManager(probes_config=config)
        counts = pm.count_probes_per_turbine()

        assert counts[0] == 2, "Turbine 0 should have 2 probes"
        assert counts[1] == 1, "Turbine 1 should have 1 probe"
        assert len(counts) == 2, "Only turbines with probes should be counted"


class TestFreePlacementProbes:
    """Tests for free placement probe initialization."""

    def test_initialize_probes_free_placement(self, mock_env):
        """Test initializing probes with absolute positions."""
        config = [
            {"position": [100, 50, 90], "probe_type": "WS", "name": "ws_probe"},
            {
                "position": [200, 100, 90],
                "probe_type": "TI",
                "name": "ti_probe",
                "include_wakes": False,
            },
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            mock_probe = MagicMock()
            MockWindProbe.return_value = mock_probe

            probes = pm.initialize_probes_free_placement(mock_env)

            # Verify probes were created
            assert len(probes) == 2
            assert MockWindProbe.call_count == 2

            # Verify first probe call
            first_call = MockWindProbe.call_args_list[0]
            assert first_call[1]["position"] == (100, 50, 90)
            assert first_call[1]["include_wakes"] is True  # default

            # Verify second probe call
            second_call = MockWindProbe.call_args_list[1]
            assert second_call[1]["include_wakes"] is False

    def test_free_placement_probe_names(self, mock_env):
        """Test that probes get correct names."""
        config = [
            {"position": [100, 50, 90], "name": "custom_name"},
            {"position": [200, 100, 90]},  # No name, should get default
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            # Create separate mock instances for each probe
            mock_probes = [MagicMock(), MagicMock()]
            MockWindProbe.side_effect = mock_probes

            probes = pm.initialize_probes_free_placement(mock_env)

            # Verify names were set on probes (the code sets probe.name after creation)
            assert len(probes) == 2
            # The name attribute is set directly on the mock after creation
            assert mock_probes[0].name == "custom_name"
            assert mock_probes[1].name == "probe_1"


class TestTurbineRelativeProbes:
    """Tests for turbine-relative probe initialization."""

    def test_initialize_probes_turbine_relative(self, mock_fs):
        """Test initializing probes relative to turbines."""
        config = [
            {
                "turbine_index": 0,
                "relative_position": [50, 0, 0],
                "probe_type": "WS",
                "name": "turb0_probe",
            },
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            mock_probe = MagicMock()
            MockWindProbe.return_value = mock_probe

            yaw_angles = np.array([0.0, 0.0])  # Zero yaw
            probes = pm.initialize_probes(mock_fs, yaw_angles)

            assert len(probes) == 1
            # With zero yaw, position should be (turbine_x + 50, turbine_y + 0, turbine_z + 0)
            call_kwargs = MockWindProbe.call_args[1]
            expected_position = (0.0 + 50, 0.0 + 0, 90.0 + 0)
            assert call_kwargs["position"] == expected_position

    def test_turbine_relative_with_yaw(self, mock_fs):
        """Test probe positioning with non-zero yaw."""
        config = [
            {
                "turbine_index": 0,
                "relative_position": [50, 0, 0],  # 50m upstream
                "probe_type": "WS",
            },
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            mock_probe = MagicMock()
            MockWindProbe.return_value = mock_probe

            # 90 degree yaw - probe should rotate from (50,0) to (0,50) relative
            yaw_angles = np.array([90.0, 0.0])
            pm.initialize_probes(mock_fs, yaw_angles)

            call_kwargs = MockWindProbe.call_args[1]
            position = call_kwargs["position"]

            # With 90 deg yaw, relative (50, 0) should become approximately (0, 50)
            expected_x = 0.0 + 0  # turbine_x + rotated_rel_x
            expected_y = 0.0 + 50  # turbine_y + rotated_rel_y
            assert abs(position[0] - expected_x) < 1e-6
            assert abs(position[1] - expected_y) < 1e-6

    def test_scalar_yaw_angle(self, mock_fs):
        """Test that scalar yaw angle is broadcast to all turbines."""
        config = [
            {"turbine_index": 0, "relative_position": [50, 0, 0], "probe_type": "WS"},
            {"turbine_index": 1, "relative_position": [50, 0, 0], "probe_type": "WS"},
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe"):
            # Pass scalar yaw - should work for both turbines
            probes = pm.initialize_probes(mock_fs, yaw_angles=0.0)
            assert len(probes) == 2

    def test_turbine_probes_grouping(self, mock_fs):
        """Test that probes are correctly grouped by turbine index."""
        config = [
            {"turbine_index": 0, "relative_position": [50, 0, 0], "name": "t0_p1"},
            {"turbine_index": 0, "relative_position": [100, 0, 0], "name": "t0_p2"},
            {"turbine_index": 1, "relative_position": [50, 0, 0], "name": "t1_p1"},
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            # Create separate mock instances for each probe with proper attribute setting
            mock_probes = []
            for _ in range(3):
                mock_probe = MagicMock()
                # Initialize turbine_index as None - will be set during initialize_probes
                mock_probe.turbine_index = None
                mock_probes.append(mock_probe)

            MockWindProbe.side_effect = mock_probes

            pm.initialize_probes(mock_fs, np.array([0.0, 0.0]))

            # Verify turbine_index was set correctly on each probe
            assert mock_probes[0].turbine_index == 0
            assert mock_probes[1].turbine_index == 0
            assert mock_probes[2].turbine_index == 1

            # Check turbine_probes grouping
            assert 0 in pm.turbine_probes
            assert 1 in pm.turbine_probes
            assert len(pm.turbine_probes[0]) == 2
            assert len(pm.turbine_probes[1]) == 1


class TestProbePositionUpdates:
    """Tests for updating probe positions when turbines yaw."""

    def test_update_probe_positions(self, mock_fs):
        """Test updating probe positions after yaw change."""
        config = [
            {
                "turbine_index": 0,
                "relative_position": [50, 0, 0],
                "probe_type": "WS",
                "name": "test_probe",
            },
        ]
        pm = ProbeManager(probes_config=config)

        # Create a real-ish probe mock
        mock_probe = MagicMock()
        mock_probe.turbine_index = 0
        mock_probe.name = "test_probe"
        mock_probe.position = (50, 0, 90)
        mock_probe.yaw_angle = 0.0

        pm.probes = [mock_probe]

        # Update with 45 degree yaw
        new_yaw = np.array([45.0, 0.0])
        pm.update_probe_positions(mock_fs, new_yaw)

        # Check position was updated
        # With 45 deg, relative (50, 0) -> (50*cos(45), 50*sin(45)) ≈ (35.4, 35.4)
        new_position = mock_probe.position
        expected_x = 50 * np.cos(np.radians(45))
        expected_y = 50 * np.sin(np.radians(45))

        assert abs(new_position[0] - expected_x) < 1e-3
        assert abs(new_position[1] - expected_y) < 1e-3

    def test_update_skips_free_placement_probes(self, mock_fs):
        """Test that free placement probes are not updated."""
        config = [
            {"position": [100, 50, 90], "name": "free_probe"},
        ]
        pm = ProbeManager(probes_config=config)

        mock_probe = MagicMock()
        mock_probe.turbine_index = None  # Free placement
        mock_probe.name = "free_probe"
        mock_probe.position = (100, 50, 90)

        pm.probes = [mock_probe]

        # Try to update
        pm.update_probe_positions(mock_fs, np.array([45.0, 0.0]))

        # Position should not change
        assert mock_probe.position == (100, 50, 90)


class TestProbeReadings:
    """Tests for getting probe readings."""

    def test_get_probe_readings(self):
        """Test getting readings from all probes."""
        pm = ProbeManager()

        # Create mock probes with readings
        mock_probe1 = MagicMock()
        mock_probe1.read.return_value = 10.5
        mock_probe2 = MagicMock()
        mock_probe2.read.return_value = 8.3

        pm.probes = [mock_probe1, mock_probe2]

        readings = pm.get_probe_readings()

        assert len(readings) == 2
        assert readings[0] == 10.5
        assert readings[1] == 8.3
        assert all(isinstance(r, float) for r in readings)

    def test_get_probe_readings_empty(self):
        """Test getting readings when no probes exist."""
        pm = ProbeManager()
        readings = pm.get_probe_readings()
        assert readings == []

    def test_get_turbine_probe_readings(self):
        """Test getting readings from turbine-specific probes."""
        pm = ProbeManager()

        mock_probe1 = MagicMock()
        mock_probe1.read.return_value = 10.0
        mock_probe1.turbine_index = 0

        mock_probe2 = MagicMock()
        mock_probe2.read.return_value = 9.5
        mock_probe2.turbine_index = 0

        mock_probe3 = MagicMock()
        mock_probe3.read.return_value = 11.0
        mock_probe3.turbine_index = 1

        pm.probes = [mock_probe1, mock_probe2, mock_probe3]
        pm.turbine_probes[0] = [mock_probe1, mock_probe2]
        pm.turbine_probes[1] = [mock_probe3]

        # Get readings for turbine 0
        readings_t0 = pm.get_turbine_probe_readings(0)
        assert len(readings_t0) == 2
        assert readings_t0 == [10.0, 9.5]

        # Get readings for turbine 1
        readings_t1 = pm.get_turbine_probe_readings(1)
        assert len(readings_t1) == 1
        assert readings_t1 == [11.0]

    def test_get_turbine_probe_readings_nonexistent(self):
        """Test getting readings for turbine with no probes."""
        pm = ProbeManager()
        pm.turbine_probes[0] = []

        readings = pm.get_turbine_probe_readings(5)  # Turbine 5 doesn't exist
        assert readings == []


class TestHasProbes:
    """Tests for has_probes method."""

    def test_has_probes_true(self):
        """Test has_probes returns True when probes exist."""
        pm = ProbeManager()
        pm.probes = [MagicMock()]
        assert pm.has_probes() is True

    def test_has_probes_false(self):
        """Test has_probes returns False when no probes exist."""
        pm = ProbeManager()
        assert pm.has_probes() is False

    def test_has_probes_after_init(self):
        """Test has_probes after initialization."""
        pm = ProbeManager(probes_config=[{"position": [0, 0, 0]}])
        # Config exists but probes not initialized yet
        assert pm.has_probes() is False


class TestEdgeCases:
    """Edge case tests."""

    def test_relative_position_with_z(self, mock_fs):
        """Test relative positioning includes z offset."""
        config = [
            {
                "turbine_index": 0,
                "relative_position": [50, 0, 10],  # 10m above hub
                "probe_type": "WS",
            },
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            mock_probe = MagicMock()
            MockWindProbe.return_value = mock_probe

            pm.initialize_probes(mock_fs, np.array([0.0, 0.0]))

            call_kwargs = MockWindProbe.call_args[1]
            position = call_kwargs["position"]

            # Z should be hub_height (90) + offset (10) = 100
            assert position[2] == 100.0

    def test_mixed_free_and_relative_probes(self, mock_fs):
        """Test mix of free placement and turbine-relative probes."""
        config = [
            {"turbine_index": 0, "relative_position": [50, 0, 0], "name": "relative"},
            {"position": [1000, 500, 90], "name": "absolute"},
        ]
        pm = ProbeManager(probes_config=config)

        with patch("WindGym.core.probe_manager.WindProbe") as MockWindProbe:
            mock_probe = MagicMock()
            MockWindProbe.return_value = mock_probe

            probes = pm.initialize_probes(mock_fs, np.array([0.0, 0.0]))

            # Both probes should be created
            assert len(probes) == 2
            assert MockWindProbe.call_count == 2
