# conftest.py
import os
import tempfile
import time

# Use non-interactive backend for matplotlib to prevent blocking on plt.show()
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring full environment setup"
    )
    config.addinivalue_line("markers", "unit: marks fast isolated unit tests (<5s)")


@pytest.fixture
def temp_yaml_file_factory():
    """Factory for creating temporary YAML files for tests.

    This fixture handles both string content and dict content (auto-converts dicts to YAML).
    """
    created_files = []

    def _create_temp_yaml(content, name_suffix=""):
        if isinstance(content, dict):
            content_str = yaml.dump(content)
        else:
            content_str = str(content)

        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f"_{name_suffix}.yaml", encoding="utf-8"
        )
        tf.write(content_str)
        filepath = tf.name
        tf.close()
        created_files.append(filepath)
        return filepath

    yield _create_temp_yaml

    for f_path in created_files:
        if os.path.exists(f_path):
            os.remove(f_path)


@pytest.fixture
def temp_yaml_filepath_factory():
    """Factory for creating temporary YAML files from config dictionaries.

    Similar to temp_yaml_file_factory but explicitly expects a dict and converts it to YAML.
    """
    created_files = []

    def _create_temp_yaml(config_dict, name_suffix=""):
        content_str = yaml.dump(config_dict)
        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f"_{name_suffix}.yaml", encoding="utf-8"
        )
        tf.write(content_str)
        filepath = tf.name
        tf.close()
        created_files.append(filepath)
        return filepath

    yield _create_temp_yaml

    for f_path in created_files:
        if os.path.exists(f_path):
            os.remove(f_path)


@pytest.fixture
def mock_mann_methods(monkeypatch):
    """Mocks turbulence generation/loading to make tests faster."""
    from dynamiks.sites.turbulence_fields import MannTurbulenceField

    def mock_generate(*args, **kwargs):
        field_data = np.zeros((1, 1, 1, 3))
        coords = (np.array([0.0]), np.array([0.0]), np.array([90.0]))
        return MannTurbulenceField(field_data, coords)

    def mock_from_netcdf(filename):
        field_data = np.zeros((1, 1, 1, 3))
        coords = (np.array([0.0]), np.array([0.0]), np.array([90.0]))
        mocked_tf = MannTurbulenceField(field_data, coords)
        mocked_tf.mocked_filename = filename
        return mocked_tf

    monkeypatch.setattr(MannTurbulenceField, "generate", mock_generate)
    monkeypatch.setattr(MannTurbulenceField, "from_netcdf", mock_from_netcdf)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # This is a hook that is called before and after each test item
    # We use 'tryfirst=True' to ensure our hook runs before others,
    # and 'hookwrapper=True' to allow us to wrap the execution.

    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    # Skip nodeid modification when running under pytest-xdist (parallel mode)
    # xdist relies on nodeid matching exactly between workers and the main process
    if hasattr(item.config, "workerinput"):
        return

    if report.when == "call":
        # 'call' refers to the actual execution phase of the test function
        duration_ms = report.duration * 1000  # Convert to milliseconds for clarity

        # Format the duration for printing
        if duration_ms < 1000:
            duration_str = f"{duration_ms:.2f} ms"
        else:
            duration_str = f"{report.duration:.2f} s"

        # Append the duration to the test outcome string
        if report.passed:
            report.nodeid = f"{report.nodeid} PASSED [{duration_str}]"
        elif report.failed:
            report.nodeid = f"{report.nodeid} FAILED [{duration_str}]"
        elif report.skipped:
            report.nodeid = f"{report.nodeid} SKIPPED [{duration_str}]"
        elif report.error:
            report.nodeid = f"{report.nodeid} ERROR [{duration_str}]"


# =============================================================================
# Module-Scoped Shared Fixtures for Performance Optimization
# =============================================================================
# These fixtures are shared across tests in the same module to avoid expensive
# environment recreation. The DWMFlowSimulation reset() takes ~20-30s, so
# reusing environments significantly speeds up test execution.


@pytest.fixture(scope="module")
def shared_turbine():
    """Module-scoped turbine for reuse across tests."""
    from py_wake.examples.data.hornsrev1 import V80

    return V80()


@pytest.fixture(scope="module")
def shared_mann_turbulence_field():
    """
    Module-scoped Mann turbulence field to be reused across all tests in a module.
    This avoids regenerating the expensive turbulence field for each test.
    """
    from dynamiks.sites.turbulence_fields import MannTurbulenceField

    tf = MannTurbulenceField.generate(
        alphaepsilon=0.1,
        L=33.6,
        Gamma=3.9,
        Nxyz=(1024, 128, 32),  # Reduced size for testing
        dxyz=(3.0, 3.0, 3.0),
        seed=1234,  # Fixed seed for reproducibility
    )
    return tf


@pytest.fixture(scope="module")
def lightweight_env_config():
    """
    Provides a lightweight configuration dict for fast test environments.
    Uses minimal settings to reduce simulation overhead.
    """
    return {
        "n_passthrough": 0.01,  # Very short episodes
        "burn_in_passthroughs": 0.0001,  # Minimal burn-in (major speedup)
        "turbtype": "None",  # No turbulence generation
        "fill_window": 1,  # Minimal history window
    }
