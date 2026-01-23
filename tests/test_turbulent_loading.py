import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from dynamiks.dwm.added_turbulence_models import (
    AutoScalingIsotropicMannTurbulence,
    SynchronizedAutoScalingIsotropicMannTurbulence,
)
from dynamiks.sites.turbulence_fields import MannTurbulenceField, RandomTurbulence
from py_wake.examples.data.hornsrev1 import V80

from WindGym import WindFarmEnv
from WindGym.utils.generate_layouts import generate_square_grid


# Helper to create a basic YAML configuration string
def assemble_base_yaml_config_string():
    config = {
        "yaw_init": "Zeros",
        "noise": "None",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {
            "yaw_min": -30,
            "yaw_max": 30,
        },
        "wind": {
            "ws_min": 8,
            "ws_max": 10,
            "TI_min": 0.05,
            "TI_max": 0.10,
            "wd_min": 268,
            "wd_max": 272,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 5, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": True,
            "turb_power": True,
            "farm_ws": True,
            "farm_wd": True,
            "farm_TI": True,
            "farm_power": True,
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": True,
            "ws_history_N": 3,
            "ws_history_length": 10,
            "ws_window_length": 2,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": True,
            "wd_history_N": 3,
            "wd_history_length": 10,
            "wd_window_length": 2,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": True,
            "yaw_history_N": 3,
            "yaw_history_length": 10,
            "yaw_window_length": 2,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": True,
            "power_history_N": 3,
            "power_history_length": 10,
            "power_window_length": 2,
        },
    }
    return yaml.dump(config)


@pytest.fixture
def temp_yaml_filepath_factory():
    created_files = []

    def _create_temp_yaml(content_str, name_suffix=""):
        if not isinstance(content_str, str):
            content_str = yaml.dump(content_str)
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


def create_mock_mann_field_instance(monkeypatch):
    """
    Creates a minimal MannTurbulenceField instance for mocking.
    Passes only essential kwargs to hipersim.MannTurbulenceField and mocks its scale_TI.
    """
    _Nxyz_tuple = (2, 2, 2)
    _dxyz_tuple = (3.0, 3.0, 3.0)
    _uvw_data = np.zeros((3, _Nxyz_tuple[2], _Nxyz_tuple[1], _Nxyz_tuple[0]))

    # Kwargs expected by hipersim.MannTurbulenceField.__init__
    # (based on common Mann box generation parameters)
    kwargs_for_hipersim_init = {
        "Nxyz": _Nxyz_tuple,
        "dxyz": _dxyz_tuple,
        "alphaepsilon": 0.1,
        "L": 30.0,
        "Gamma": 3.9,
        "seed": 1,
        # 'HighFreqComp' and 'double_xyz' might also be relevant if hipersim's init uses them
    }

    # Mock the scale_TI method of the instance *after* it's created,
    # or of the class if it's a static/class method that gets called on the result.
    # For an instance method on the created object:
    def mock_scale_ti(
        self_field, TI, U
    ):  # Renamed 'self' to 'self_field' to avoid clash
        # print(f"Mocked scale_TI called with TI={TI}, U={U}")
        # Ensure self_field.uvw remains finite if it was modified before.
        if not np.all(np.isfinite(self_field.uvw)):
            self_field.uvw = np.zeros_like(self_field.uvw)  # Reset to zeros if NaN/inf
        pass  # Do nothing or a very simple scaling to avoid NaN issues.

    # Temporarily allow MannTurbulenceField to be instantiated without error,
    # then patch its scale_TI method.
    # This is tricky if the error happens *during* __init__ due to scale_TI being called there.
    # A safer approach is to mock scale_TI at the class level *before* instantiation.
    monkeypatch.setattr(MannTurbulenceField, "scale_TI", mock_scale_ti)

    mocked_tf = MannTurbulenceField(uvw=_uvw_data, **kwargs_for_hipersim_init)

    # Ensure Nxyz and dxyz are present as attributes for dynamiks wrapper
    mocked_tf.Nxyz = _Nxyz_tuple
    mocked_tf.dxyz = _dxyz_tuple
    mocked_tf.x, mocked_tf.y, mocked_tf.z = [
        np.arange(N) * d for N, d in zip(mocked_tf.Nxyz, mocked_tf.dxyz)
    ]

    return mocked_tf


class TestTurbulenceLoading:
    def test_turbtype_none(self, temp_yaml_filepath_factory):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "turb_none")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)
        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="None",
            reset_init=True,
            seed=42,
            Baseline_comp=False,
        )
        assert env.turbtype == "None"
        assert env.addedTurbulenceModel is None
        assert isinstance(env.site.turbulenceField, RandomTurbulence)
        env.close()

    def test_turbtype_mannload_single_file(
        self, tmp_path, temp_yaml_filepath_factory, monkeypatch
    ):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_single")
        tf_file_path = tmp_path / "TF_single_test.nc"
        tf_file_path.write_text("dummy_netcdf_content_for_single_file_test")

        from_netcdf_called_with = []

        def mock_from_netcdf(filename):
            from_netcdf_called_with.append(filename)
            mocked_tf_instance = create_mock_mann_field_instance(
                monkeypatch
            )  # Pass monkeypatch
            mocked_tf_instance.mocked_filename = filename
            return mocked_tf_instance

        monkeypatch.setattr(MannTurbulenceField, "from_netcdf", mock_from_netcdf)
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="MannLoad",
            TurbBox=str(tf_file_path),
            reset_init=True,
            seed=42,
        )
        assert env.turbtype == "MannLoad"
        assert len(from_netcdf_called_with) == 1
        assert from_netcdf_called_with[0] == str(tf_file_path)
        assert len(env.TF_files) == 1 and env.TF_files[0] == str(tf_file_path)
        assert isinstance(
            env.addedTurbulenceModel, SynchronizedAutoScalingIsotropicMannTurbulence
        )
        assert isinstance(env.site.turbulenceField, MannTurbulenceField)
        env.close()

    def test_turbtype_mannload_directory(
        self, tmp_path, temp_yaml_filepath_factory, monkeypatch
    ):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_dir")
        turb_dir = tmp_path / "turbulence_boxes"
        turb_dir.mkdir()
        tf_file_path1 = turb_dir / "TF_test1.nc"
        tf_file_path1.write_text("dummy_netcdf_content_1")
        tf_file_path2 = turb_dir / "TF_test2.nc"
        tf_file_path2.write_text("dummy_netcdf_content_2")
        (turb_dir / "not_a_tf_file.txt").write_text("ignore this")

        from_netcdf_called_with = []

        def mock_from_netcdf(filename):
            from_netcdf_called_with.append(str(Path(filename).name))
            mocked_tf_instance = create_mock_mann_field_instance(
                monkeypatch
            )  # Pass monkeypatch
            mocked_tf_instance.mocked_filename = filename
            return mocked_tf_instance

        monkeypatch.setattr(MannTurbulenceField, "from_netcdf", mock_from_netcdf)
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="MannLoad",
            TurbBox=str(turb_dir),
            reset_init=True,
            seed=42,
        )
        assert env.turbtype == "MannLoad"
        # WindFarmEnv._def_site for MannLoad directory randomly picks ONE file to load per reset.
        # The TF_files list should contain all valid files found in the directory.
        assert len(env.TF_files) == 2
        assert str(tf_file_path1) in [str(f) for f in env.TF_files]
        assert str(tf_file_path2) in [str(f) for f in env.TF_files]
        # from_netcdf is called once with the chosen file during reset's _def_site
        assert (
            len(from_netcdf_called_with) == 1
        ), "from_netcdf should be called once for the selected file"
        assert from_netcdf_called_with[0] in ["TF_test1.nc", "TF_test2.nc"]

        assert isinstance(
            env.addedTurbulenceModel, SynchronizedAutoScalingIsotropicMannTurbulence
        )
        assert isinstance(env.site.turbulenceField, MannTurbulenceField)
        env.close()

    def test_turbtype_mannload_raises_error_for_empty_directory(
        self, tmp_path, temp_yaml_filepath_factory
    ):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_dir_empty")
        empty_turb_dir = tmp_path / "empty_turbulence_boxes"
        empty_turb_dir.mkdir()

        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        # Use pytest.raises to assert that the expected exception is thrown
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(f"No TF_*.nc files found at: {empty_turb_dir}"),
        ):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                turbtype="MannLoad",
                TurbBox=str(empty_turb_dir),
                reset_init=True,
                seed=42,
            )

    def test_turbtype_manngenerate(self, temp_yaml_filepath_factory, monkeypatch):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_gen")

        generate_called_with_kwargs = None

        def mock_generate(*args, **kwargs):
            nonlocal generate_called_with_kwargs
            generate_called_with_kwargs = kwargs
            return create_mock_mann_field_instance(monkeypatch)  # Pass monkeypatch

        monkeypatch.setattr(MannTurbulenceField, "generate", mock_generate)
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="MannGenerate",
            reset_init=True,
            seed=42,
        )
        assert env.turbtype == "MannGenerate"
        assert generate_called_with_kwargs is not None
        assert "seed" in generate_called_with_kwargs
        assert isinstance(
            env.addedTurbulenceModel, SynchronizedAutoScalingIsotropicMannTurbulence
        )
        assert isinstance(env.site.turbulenceField, MannTurbulenceField)
        env.close()

    def test_turbtype_random(self, temp_yaml_filepath_factory):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "turb_random")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="Random",
            reset_init=True,
            seed=42,
            Baseline_comp=False,
        )
        assert env.turbtype == "Random"
        assert isinstance(env.addedTurbulenceModel, AutoScalingIsotropicMannTurbulence)
        assert isinstance(env.site.turbulenceField, RandomTurbulence)
        env.close()

    def test_turbtype_invalid(self, temp_yaml_filepath_factory):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "turb_invalid")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        with pytest.raises(ValueError, match="Invalid turbulence type specified"):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                turbtype="ThisIsAnInvalidTurbulenceType",
                reset_init=True,
                seed=42,
            )

    def test_turbtype_mannfixed(self, temp_yaml_filepath_factory, monkeypatch):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_fixed")

        generate_called_with_kwargs = None

        def mock_generate(*args, **kwargs):
            nonlocal generate_called_with_kwargs
            generate_called_with_kwargs = kwargs
            return create_mock_mann_field_instance(monkeypatch)  # Pass monkeypatch

        monkeypatch.setattr(MannTurbulenceField, "generate", mock_generate)
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        env = WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=yaml_filepath,
            turbtype="MannFixed",
            reset_init=True,
            seed=123,
        )
        assert env.turbtype == "MannFixed"
        assert generate_called_with_kwargs is not None
        assert generate_called_with_kwargs.get("seed") == 1234
        assert isinstance(
            env.addedTurbulenceModel, SynchronizedAutoScalingIsotropicMannTurbulence
        )
        assert isinstance(env.site.turbulenceField, MannTurbulenceField)
        env.close()

    def test_turbtype_mannload_raises_error_on_invalid_path(
        self, temp_yaml_filepath_factory
    ):
        yaml_content = assemble_base_yaml_config_string()
        yaml_filepath = temp_yaml_filepath_factory(yaml_content, "mann_invalid_path")
        x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=5, yDist=3)

        invalid_path = "./this/path/does/not/exist"

        # Test with a non-existent path
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(f"No TF_*.nc files found at: {invalid_path}"),
        ):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                turbtype="MannLoad",
                TurbBox=invalid_path,
            )

        # Test with TurbBox=None
        with pytest.raises(
            FileNotFoundError, match="Provide 'TurbBox' for turbtype='MannLoad'."
        ):
            WindFarmEnv(
                turbine=V80(),
                x_pos=x_pos,
                y_pos=y_pos,
                config=yaml_filepath,
                turbtype="MannLoad",
                TurbBox=None,
            )
