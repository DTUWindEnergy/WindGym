"""Unit tests for WindGym.core.precursor_source and TurbulenceManager.set_turbulence_files."""

import numpy as np
import pytest

from WindGym.core.precursor_source import PrecursorSource
from WindGym.core.turbulence_manager import TurbulenceManager

pytestmark = pytest.mark.unit

META = {
    "Nxyz": np.array([100, 50, 20]),
    "dxyz": np.array([10.0, 10.0, 10.0]),
    "advection_speed": 8.0,
    "n_ramp": 60,
    "t_data_s": 600.0,
}
POS = np.array([[0.0, 0.0], [400.0, 0.0]])
D = 80.0


def _src():
    return PrecursorSource.from_arrays(uvw=None, meta=META)


def _check(src, rng=None, **over):
    kw = dict(
        turbine_positions=POS,
        rotor_diameter=D,
        veer_rate=0.0,
        wd_list=[270.0, 270.0],
        episode_time_budget_s=100.0,
    )
    kw.update(over)
    return src.check_episode(rng or np.random.default_rng(0), **kw)


def test_from_arrays_exposes_meta_and_zero_offset():
    src = _src()
    assert src.meta is META and src.uvw is None
    assert src.window_offset_s == 0.0


@pytest.mark.parametrize(
    "over, match",
    [
        ({"veer_rate": 0.01}, "carries the LES shear/veer"),
        ({"wd_list": [270.0, 271.0]}, "cannot express a time-varying wind direction"),
        (
            {"turbine_positions": np.array([[0.0, 0.0], [0.0, 600.0]])},
            "exceeds the precursor box width",
        ),
        (
            {"turbine_positions": np.array([[0.0, 0.0], [500.0, 0.0]])},
            "precursor ramp only covers",
        ),
        ({"episode_time_budget_s": 1e6}, "precursor provides only"),
    ],
)
def test_check_episode_error_branches(over, match):
    with pytest.raises(ValueError, match=match):
        _check(_src(), **over)


def test_check_episode_draws_window_from_slack():
    src = _src()
    _check(src, rng=np.random.default_rng(7))
    # slack = t_data + x_min/U - budget, x_min = 0 - 2D
    slack = 600.0 + (-2 * D) / 8.0 - 100.0
    assert src.window_offset_s == float(np.random.default_rng(7).uniform(0.0, slack))


def test_manager_without_precursor_has_no_meta():
    tm = TurbulenceManager("None")
    assert tm.precursor_meta is None
    assert tm.window_offset_s == 0.0


def test_set_turbulence_files_warns_when_not_mannload():
    tm = TurbulenceManager("None")
    with pytest.warns(UserWarning, match="MannLoad"):
        tm.set_turbulence_files(["/x/TF_a.nc"])
    assert tm.turbulence_files == ["/x/TF_a.nc"]


def test_set_turbulence_files_replaces_discovered_list(tmp_path):
    (tmp_path / "TF_a.nc").write_text("x")
    (tmp_path / "TF_b.nc").write_text("x")
    tm = TurbulenceManager("MannLoad", turbulence_box_path=str(tmp_path))
    assert len(tm.turbulence_files) == 2
    tm.set_turbulence_files([str(tmp_path / "TF_b.nc")])
    assert tm.turbulence_files == [str(tmp_path / "TF_b.nc")]


def test_invalid_turbulence_type_message():
    tm = TurbulenceManager("Bogus")
    tm.np_random = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Invalid turbulence type specified"):
        tm._generate_turbulence_field(ws=8.0, ti=0.06, rotor_diameter=D)


def test_set_turbulence_files_accepts_single_path():
    from pathlib import Path

    tm = TurbulenceManager("None")
    with pytest.warns(UserWarning):
        tm.set_turbulence_files("/x/TF_a.nc")
    assert tm.turbulence_files == ["/x/TF_a.nc"]
    with pytest.warns(UserWarning):
        tm.set_turbulence_files(Path("/x/TF_b.nc"))
    assert tm.turbulence_files == [Path("/x/TF_b.nc")]
