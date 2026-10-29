"""Unit tests for WindGym.core.env_config (config loading and parsing)."""

import dataclasses

import pytest
import yaml

from WindGym.core.env_config import (
    EnvSettings,
    build_farm_mes,
    load_config_dict,
    parse_env_settings,
)
from WindGym.core.mes_class import FarmMes
from test_utils import get_fast_pywake_config

pytestmark = pytest.mark.unit

# Every flat attribute WindFarmEnv used to set in _apply_config (probe objects
# excluded: the env still builds the ProbeManager itself).
SETTINGS_ATTRS = {
    "yaw_init",
    "BaseController",
    "ActionMethod",
    "Track_power",
    "yaw_min",
    "yaw_max",
    "yaw_scaling_min",
    "yaw_scaling_max",
    "tilt",
    "turb_memmap",
    "dwm_setup",
    "ws_inflow_min",
    "ws_inflow_max",
    "TI_inflow_min",
    "TI_inflow_max",
    "wd_inflow_min",
    "wd_inflow_max",
    "veer_inflow_min",
    "veer_inflow_max",
    "act_pen",
    "power_def",
    "mes_level",
    "ws_mes",
    "wd_mes",
    "yaw_mes",
    "power_mes",
    "ti_sample_count",
    "action_penalty",
    "action_penalty_type",
    "Power_scaling",
    "power_avg",
    "power_reward",
    "tau",
    "derate_action",
    "yaw_action",
    "derate_min",
    "derate_max",
    "derate_penalty",
    "derate_penalty_type",
    "derate_method",
    "derate_step_env",
    "derate_step_sim",
    "track_reward_type",
    "track_sigma",
    "track_ref_range",
    "track_obs_setpoint",
    "track_obs_error",
    "track_obs_preview",
    "derate_reference",
    "derate_mes",
    "probes_config",
}


# --- load_config_dict --------------------------------------------------------


def test_dict_passthrough():
    cfg = {"a": 1}
    out, yaml_path = load_config_dict(cfg)
    assert out is cfg and yaml_path is None


def test_yaml_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    out, yaml_path = load_config_dict(str(p))
    assert out == {"a": 1} and yaml_path == str(p)


def test_yaml_string():
    out, yaml_path = load_config_dict("a: 1\nb: [1, 2]\n")
    assert out == {"a": 1, "b": [1, 2]} and yaml_path is None


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config_dict("/no/such/dir/config.yaml")


def test_none_raises():
    with pytest.raises(ValueError, match="configuration must be provided"):
        load_config_dict(None)


def test_bad_type_raises():
    with pytest.raises(TypeError, match="must be a dict, YAML string, or path"):
        load_config_dict(42)


# --- parse_env_settings ------------------------------------------------------


def test_settings_attribute_set_is_exact():
    s = parse_env_settings(get_fast_pywake_config())
    assert isinstance(s, EnvSettings)
    assert dataclasses.is_dataclass(s)
    assert set(vars(s)) == SETTINGS_ATTRS


def test_settings_values_and_defaults():
    s = parse_env_settings(get_fast_pywake_config())
    assert s.yaw_min == -30 and s.yaw_scaling_min == -30
    assert s.yaw_scaling_max == 30
    assert s.tilt == 0.0
    assert s.turb_memmap is False
    assert s.veer_inflow_min == 0.0 and s.veer_inflow_max == 0.0
    assert s.ti_sample_count == 30
    assert s.tau == 0.02
    assert s.derate_action is False and s.yaw_action is True
    assert s.derate_method == "absolute" and s.derate_reference == "available"
    assert s.derate_step_sim is None
    assert s.track_ref_range == [0.2, 0.8] and s.track_obs_preview == 0
    assert isinstance(s.track_obs_preview, int)
    assert s.derate_mes == {
        "derate_current": False,
        "derate_rolling_mean": False,
        "derate_history_N": 1,
        "derate_history_length": 10,
        "derate_window_length": 10,
    }
    assert s.probes_config == []


def test_derate_mes_defaults_follow_derate_action():
    s = parse_env_settings(get_fast_pywake_config(derate_action=True))
    assert s.derate_mes["derate_current"] is True


@pytest.mark.parametrize(
    "override, match",
    [
        ({"derate_method": "bogus"}, "derate_method must be 'absolute' or 'step'"),
        ({"derate_step_sim": 0}, "derate_step_sim must be positive"),
        ({"derate_reference": "x"}, "derate_reference must be 'available', 'rated' or 'setpoint'"),
        ({"track_def": {"track_ref_range": [0.9, 0.1]}}, "track_ref_range must be"),
        ({"track_def": {"track_obs_preview": 1.5}}, "track_obs_preview must be"),
    ],
)
def test_validation_errors(override, match):
    with pytest.raises(ValueError, match=match):
        parse_env_settings(get_fast_pywake_config(**override))


def test_missing_section_and_key_messages():
    cfg = get_fast_pywake_config()
    del cfg["farm"]
    with pytest.raises(ValueError, match="Config section 'farm' is required"):
        parse_env_settings(cfg)
    cfg = get_fast_pywake_config()
    del cfg["wind"]["ws_min"]
    with pytest.raises(ValueError, match="Key 'ws_min' is required in section 'wind'"):
        parse_env_settings(cfg)


def test_legacy_mann_keys_warn():
    cfg = get_fast_pywake_config(mann_nxyz=[1, 2, 3])
    with pytest.warns(UserWarning, match="config key 'mann_nxyz' is ignored"):
        parse_env_settings(cfg)


# --- build_farm_mes ----------------------------------------------------------


def test_build_farm_mes_returns_farm_mes():
    s = parse_env_settings(get_fast_pywake_config())
    scaling = dict(
        ws_min=0.0, ws_max=30.0, wd_min=0.0, wd_max=360.0, TI_min=0.0, TI_max=1.0
    )
    fm = build_farm_mes(s, n_turb=2, scaling=scaling, maxturbpower=2.0e6)
    assert isinstance(fm, FarmMes)
    assert len(fm.turb_mes) == 2
    assert fm.observed_variables() > 0


def test_yaml_roundtrip_settings_equal_dict_settings(tmp_path):
    cfg = get_fast_pywake_config()
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert parse_env_settings(load_config_dict(str(p))[0]) == parse_env_settings(cfg)
