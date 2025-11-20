# test_coliseum_basic.py

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import tempfile
import os
from unittest.mock import patch

# --- Imports from the WindGym project ---
from WindGym.utils.evaluate_PPO import Coliseum
from WindGym.farm_eval import FarmEval
from WindGym import WindFarmEnv
from WindGym.Agents import ConstantAgent
from WindGym.utils.generate_layouts import generate_square_grid
from py_wake.examples.data.hornsrev1 import V80

# --- Fixtures specific to this file ---


@pytest.fixture(scope="module")
def coliseum_agents_timeseries():
    """Provides a dictionary of simple, deterministic agents for testing."""
    # Using 2 turbines in the env_factory
    agent_a = ConstantAgent(yaw_angles=[10, 0])
    agent_b = ConstantAgent(yaw_angles=[-10, 0])
    return {"Steering_Agent": agent_a, "Non_Steering_Agent": agent_b}


@pytest.fixture
def coliseum_instance(temp_yaml_file_for_coliseum, coliseum_agents):
    """
    Provides a Coliseum instance configured with the flexible FarmEval class.
    """
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    def env_factory():
        return FarmEval(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=temp_yaml_file_for_coliseum,
            turbtype="None",
            Baseline_comp=True,
            reset_init=True,
            finite_episode=True,
            n_passthrough=0.05,  # A very short episode for fast testing
            burn_in_passthroughs=0.01,
        )

    return Coliseum(
        env_factory=env_factory,
        agents=coliseum_agents,
        n_passthrough=0.05,  # A very short episode for fast testing
        burn_in_passthroughs=0.01,
    )


@pytest.fixture
def coliseum_instance_timeseries(
    temp_yaml_file_for_coliseum, coliseum_agents_timeseries
):
    """
    This fixture provides a Coliseum instance with a fast environment factory.
    It uses WindFarmEnv so that the episode length is controlled by n_passthrough.
    """
    x_pos, y_pos = generate_square_grid(turbine=V80(), nx=2, ny=1, xDist=7, yDist=7)

    def env_factory():
        return WindFarmEnv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=temp_yaml_file_for_coliseum,
            turbtype="None",
            Baseline_comp=True,
            reset_init=False,
            n_passthrough=0.05,  # A very short episode for fast testing
            burn_in_passthroughs=0.01,
        )

    return Coliseum(
        env_factory=env_factory,
        agents=coliseum_agents_timeseries,
        n_passthrough=0.05,  # A very short episode for fast testing
        burn_in_passthroughs=0.01,
    )


# --- Grid Evaluation Tests ---


def test_run_wind_grid_evaluation_basic(coliseum_instance, coliseum_agents):
    """
    Tests the grid evaluation with a 2x2 grid, asserting the output format and content.
    """
    # Define a simple 2x2 grid (2 wind directions, 2 wind speeds)
    wd_min, wd_max, wd_step = 270, 275, 5
    ws_min, ws_max, ws_step = 8, 10, 2
    ti_points = 1

    dataset = coliseum_instance.run_wind_grid_evaluation(
        wd_min=wd_min,
        wd_max=wd_max,
        wd_step=wd_step,
        ws_min=ws_min,
        ws_max=ws_max,
        ws_step=ws_step,
        ti_points=ti_points,
    )

    # --- Assertions on the xarray.Dataset ---
    assert isinstance(dataset, xr.Dataset)

    # Check dimensions
    assert "wd" in dataset.dims and dataset.dims["wd"] == 2
    assert "ws" in dataset.dims and dataset.dims["ws"] == 2
    assert "ti" in dataset.dims and dataset.dims["ti"] == 1

    # Check coordinates
    np.testing.assert_array_equal(dataset.coords["wd"], [270, 275])
    np.testing.assert_array_equal(dataset.coords["ws"], [8, 10])

    # Check data variables
    for agent_name in coliseum_agents:
        var_name = f"{agent_name}_mean_reward"
        assert var_name in dataset.data_vars
        assert dataset[var_name].shape == (2, 2, 1)  # (wd, ws, ti)
        # Ensure the results are numbers and not empty
        assert not np.isnan(dataset[var_name].values).any()

    # Check attributes
    assert dataset.attrs["evaluation_type"] == "wind_grid"
    assert "bounds_used" in dataset.attrs


def test_run_wind_grid_evaluation_single_point(coliseum_instance):
    """
    Tests the grid evaluation for the edge case of a single condition.
    """
    dataset = coliseum_instance.run_wind_grid_evaluation(
        wd_min=270, wd_max=270, wd_step=5, ws_min=10, ws_max=10, ws_step=2, ti_points=1
    )

    assert isinstance(dataset, xr.Dataset)
    # Check that all dimensions have size 1
    assert dataset.dims["wd"] == 1
    assert dataset.dims["ws"] == 1
    assert dataset.dims["ti"] == 1
    np.testing.assert_array_equal(dataset.coords["wd"], [270])
    np.testing.assert_array_equal(dataset.coords["ws"], [10])


@patch("matplotlib.pyplot.savefig")
def test_plot_wind_grid_results_runs_without_error(mock_savefig, coliseum_instance):
    """
    Tests that the grid plotting function can be called without crashing after
    an evaluation has been run. Mocks `savefig` to avoid creating files.
    """
    # First, run a evaluation to generate data
    dataset = coliseum_instance.run_wind_grid_evaluation(
        wd_min=270, wd_max=275, wd_step=5, ws_min=8, ws_max=10, ws_step=2, ti_points=1
    )

    # Now, attempt to plot the results
    coliseum_instance.plot_wind_grid_results(dataset, save_path="test_grid_plot.png")

    # Check that savefig was called
    mock_savefig.assert_called_once_with(
        "test_grid_plot.png", dpi=150, bbox_inches="tight"
    )


# --- Time Series Evaluation Tests ---


def test_coliseum_run_time_series_evaluation(
    coliseum_instance_timeseries, coliseum_agents_timeseries
):
    """
    Tests the time series evaluation mode of the Coliseum, checking the format
    and content of the returned summary and detailed history data.
    """
    num_episodes = 2
    summary_df = coliseum_instance_timeseries.run_time_series_evaluation(
        num_episodes=num_episodes, seed=123, save_detailed_history=True
    )

    # --- 1. Assertions on the summary DataFrame ---
    assert isinstance(
        summary_df, pd.DataFrame
    ), "The returned summary should be a pandas DataFrame."
    assert (
        summary_df.shape[0] == num_episodes
    ), f"Expected {num_episodes} rows in the summary DataFrame."

    expected_columns = ["episode"] + list(coliseum_agents_timeseries.keys())
    assert all(
        col in summary_df.columns for col in expected_columns
    ), "Summary DataFrame is missing expected columns."
    assert (
        not summary_df.drop(columns=["episode"]).isnull().values.any()
    ), "Summary DataFrame should not contain NaN values."

    # --- 2. Assertions on the detailed time_series_results ---
    ts_results = coliseum_instance_timeseries.time_series_results
    assert isinstance(ts_results, dict), "time_series_results should be a dictionary."
    assert len(ts_results) == len(
        coliseum_agents_timeseries
    ), "Number of agents in time_series_results should match input."

    for agent_name, results_list in ts_results.items():
        assert (
            agent_name in coliseum_agents_timeseries
        ), f"Unexpected agent '{agent_name}' found in results."
        assert (
            len(results_list) == num_episodes
        ), f"Expected {num_episodes} episode results for agent '{agent_name}'."

        single_episode_result = results_list[0]
        assert isinstance(single_episode_result, dict)
        expected_keys = [
            "history",
            "final_mean_reward",
            "total_reward",
            "episode_length",
            "agent_name",
        ]
        assert all(
            key in single_episode_result for key in expected_keys
        ), "An episode result dict is missing keys."

        history = single_episode_result["history"]
        assert isinstance(history, dict)
        expected_history_keys = ["step", "reward", "mean_cumulative_reward", "info"]
        assert all(
            key in history for key in expected_history_keys
        ), "The history dict is missing keys."

        assert len(history["step"]) > 0, "History 'step' array should not be empty."
        assert (
            len(history["step"]) == single_episode_result["episode_length"]
        ), "History length should match episode length."
        assert (
            "Power agent" in history["info"][0]
        ), "Info dict within history is missing 'Power agent' key."


# FIX: Patch the correct plotting function, which is 'savefig', not 'show'.
@patch("matplotlib.pyplot.savefig")
def test_coliseum_plot_time_series_runs_without_error(
    mock_savefig, coliseum_instance_timeseries
):
    """
    Tests that the time series plotting function can be called without crashing
    after an evaluation has been run. It correctly mocks `plt.savefig`.
    """
    # First, run an evaluation to generate the necessary data.
    coliseum_instance_timeseries.run_time_series_evaluation(
        num_episodes=1, save_detailed_history=True
    )

    # Now, attempt to plot the results.
    coliseum_instance_timeseries.plot_time_series_comparison(save_path="test_plot.png")

    # Check that the matplotlib 'savefig' function was called, indicating the plot was generated and saved.
    mock_savefig.assert_called_once()
    # Optional: More robustly check if it was called with the correct filename.
    mock_savefig.assert_called_once_with("test_plot.png", dpi=150, bbox_inches="tight")
