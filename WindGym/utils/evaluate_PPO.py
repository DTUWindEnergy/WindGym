"""
Enhanced Coliseum Evaluation Framework
=====================================

A flexible evaluation framework for comparing multiple agents in WindFarm environments.

Wind Sampling Behavior:
- Time Series Evaluation: Uses sample_site for realistic stochastic wind sampling
- Wind Grid Evaluation: Uses fixed wind conditions across the specified grid

Supports time series evaluation with mean cumulative rewards and wind condition grid evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm.auto import tqdm
from typing import Callable, Dict, List, Optional, Union, Any
from collections import defaultdict
import copy


class Coliseum:
    """
    Enhanced evaluation framework to compare multiple agents in WindFarm environments.

    Features:
    - Time series evaluation with detailed episode history
    - Wind condition grid evaluation with NetCDF export
    - Mean cumulative reward tracking
    - Flexible agent management with custom labels
    - Comprehensive plotting capabilities
    """

    def __init__(
        self,
        env_factory: Callable,
        agents: Union[Dict[str, object], List[object]],
        agent_labels: Optional[List[str]] = None,
        n_passthrough: float = 1.0,
        burn_in_passthroughs: float = 2.0,
    ):
        """
        Initialize the Coliseum evaluation framework.

        Args:
            env_factory (Callable): Function that returns a new environment instance.
                                   Example: `lambda: WindFarmEnv(...)`
            agents (Union[Dict[str, object], List[object]]):
                   Either a dictionary {name: agent} or list of agent objects.
                   All agents must have a `.predict(obs, deterministic)` method.
            agent_labels (Optional[List[str]]): Custom labels for agents when using list input.
                                              If None, defaults to "Agent_0", "Agent_1", etc.
            n_passthrough (float, optional): Number of flow passthroughs for episode length.
                                           Defaults to 1.0.
            burn_in_passthroughs (float, optional): Number of flow passthroughs before episode
        """
        # Handle different agent input formats
        if isinstance(agents, dict):
            self.agents = agents
            self.agent_names = list(agents.keys())
        elif isinstance(agents, list):
            if agent_labels is not None:
                if len(agent_labels) != len(agents):
                    raise ValueError(
                        "Number of agent_labels must match number of agents"
                    )
                self.agent_names = agent_labels
            else:
                self.agent_names = [f"Agent_{i}" for i in range(len(agents))]
            self.agents = dict(zip(self.agent_names, agents))
        else:
            raise ValueError("agents must be a dictionary or list")

        if not self.agents:
            raise ValueError("At least one agent must be provided")

        self.env_factory = env_factory
        self.n_passthrough = n_passthrough
        self.burn_in_passthroughs = burn_in_passthroughs
        self.results: Optional[pd.DataFrame] = None
        self.time_series_results: Optional[Dict[str, Any]] = None

    def _run_single_episode_with_history(
        self,
        env: Any,
        agent: object,
        agent_name: str,
        seed: int,
        deterministic: bool = True,
        use_stochastic_wind: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single episode and return detailed history including mean cumulative reward.

        Args:
            use_stochastic_wind: If True, uses sample_site for realistic wind sampling.
                               If False, relies on any pre-set wind conditions.

        Returns:
            Dictionary with episode history and final statistics
        """
        # For stochastic evaluation, ensure sample_site is used for realistic wind sampling
        if (
            use_stochastic_wind
            and hasattr(env, "sample_site")
            and env.sample_site is None
        ):
            print(
                "Warning: No sample_site configured for realistic wind sampling. Using uniform sampling."
            )

        obs, info = env.reset(seed=seed)

        base_env = env.unwrapped
        # If the agent needs to be updated with the environment's wind conditions, do so now.
        if hasattr(agent, "update_wind") and callable(getattr(agent, "update_wind")):
            # Check the base_env for the attributes, not the wrapper
            if (
                hasattr(base_env, "ws")
                and hasattr(base_env, "wd")
                and hasattr(base_env, "ti")
            ):
                agent.update_wind(
                    wind_speed=base_env.ws, wind_direction=base_env.wd, TI=base_env.ti
                )

            else:
                print(
                    f"Warning: Agent '{agent_name}' has 'update_wind' method, but the base env is missing ws, wd, or ti attributes."
                )

        if hasattr(agent, "UseEnv"):
            agent.env = env

        # Initialize tracking
        history = {"step": [], "reward": [], "mean_cumulative_reward": [], "info": []}

        cumulative_reward = 0.0
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            # Get action
            action, _ = agent.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Update tracking
            cumulative_reward += reward
            step_count += 1
            mean_cumulative_reward = cumulative_reward / step_count

            # Record history
            history["step"].append(step_count - 1)
            history["reward"].append(reward)
            history["mean_cumulative_reward"].append(mean_cumulative_reward)
            history["info"].append(copy.deepcopy(info))

        # Convert to numpy arrays
        for key in ["step", "reward", "mean_cumulative_reward"]:
            history[key] = np.array(history[key])

        return {
            "history": history,
            "final_mean_reward": mean_cumulative_reward,
            "total_reward": cumulative_reward,
            "episode_length": step_count,
            "agent_name": agent_name,
        }

    def _run_single_episode_summary(
        self,
        env: Any,
        agent: object,
        seed: int,
        deterministic: bool = True,
        use_stochastic_wind: bool = True,
        agent_name: str = "agent",
    ) -> float:
        """
        Run a single episode and return only the final mean cumulative reward.

        Args:
            use_stochastic_wind: If True, uses sample_site for realistic wind sampling.
        """
        # For stochastic evaluation, ensure sample_site is used
        if (
            use_stochastic_wind
            and hasattr(env, "sample_site")
            and env.sample_site is None
        ):
            print(
                "Warning: No sample_site configured for realistic wind sampling. Using uniform sampling."
            )

        obs, info = env.reset(seed=seed)

        base_env = env.unwrapped
        if hasattr(agent, "update_wind") and callable(getattr(agent, "update_wind")):
            if (
                hasattr(base_env, "ws")
                and hasattr(base_env, "wd")
                and hasattr(base_env, "ti")
            ):
                agent.update_wind(
                    wind_speed=base_env.ws, wind_direction=base_env.wd, TI=base_env.ti
                )

            else:
                # This warning should no longer appear with the corrected check.
                print(
                    f"Warning: Agent '{agent_name}' has 'update_wind' method, but the base env is missing ws, wd, or ti attributes."
                )

        if hasattr(agent, "UseEnv"):
            agent.env = env

        cumulative_reward = 0.0
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            step_count += 1
            if step_count % 50 == 0:
                print(step_count)

        return cumulative_reward / step_count if step_count > 0 else 0.0

    def run_time_series_evaluation(
        self,
        num_episodes: int = 10,
        seed: int = 42,
        deterministic: bool = True,
        save_detailed_history: bool = True,
    ) -> pd.DataFrame:
        """
        Run time series evaluation with stochastic wind conditions using sample_site.

        This method relies on the environment's sample_site for realistic wind sampling.
        Each episode will have different wind conditions sampled from the site's
        wind resource distributions (Weibull for wind speed, frequency for direction).

        Args:
            num_episodes (int): Number of episodes to run
            seed (int): Master seed for reproducibility
            deterministic (bool): Whether to use deterministic agent policies
            save_detailed_history (bool): Whether to save detailed time series data

        Returns:
            pd.DataFrame: Summary results with mean cumulative rewards
        """
        rng = np.random.default_rng(seed)
        results_data = []

        if save_detailed_history:
            self.time_series_results = defaultdict(list)

        # Check if any environment has sample_site configured
        temp_env = self.env_factory()
        has_sample_site = (
            hasattr(temp_env, "sample_site") and temp_env.sample_site is not None
        )
        temp_env.close()

        if not has_sample_site:
            print(
                "Warning: No sample_site configured in environment. Time series evaluation"
            )
            print(
                "will use uniform wind sampling instead of realistic site-based sampling."
            )
            print(
                "Consider configuring sample_site for more realistic wind conditions."
            )

        print(f"Running time series evaluation for {num_episodes} episodes...")
        print(
            f"Wind sampling: {'Site-based (realistic)' if has_sample_site else 'Uniform (simplified)'}"
        )

        for episode in tqdm(range(num_episodes), desc="Time Series Evaluation"):
            episode_seed = int(rng.integers(2**31))
            episode_results = {"episode": episode}

            for agent_name in self.agent_names:
                # Create fresh environment for each agent
                env = self.env_factory()
                env.n_passthrough = self.n_passthrough
                env.burn_in_passthroughs = self.burn_in_passthroughs

                try:
                    if hasattr(env.unwrapped, "n_passthrough"):
                        env.unwrapped.n_passthrough = self.n_passthrough
                        env.unwrapped.burn_in_passthroughs = self.burn_in_passthroughs

                    agent = self.agents[agent_name]

                    if save_detailed_history:
                        # Run with full history tracking and stochastic wind
                        result = self._run_single_episode_with_history(
                            env,
                            agent,
                            agent_name,
                            episode_seed,
                            deterministic,
                            use_stochastic_wind=True,
                        )
                        self.time_series_results[agent_name].append(result)
                        episode_results[agent_name] = result["final_mean_reward"]
                    else:
                        # Run summary only with stochastic wind
                        mean_reward = self._run_single_episode_summary(
                            env,
                            agent,
                            episode_seed,
                            deterministic,
                            use_stochastic_wind=True,
                        )
                        episode_results[agent_name] = mean_reward

                finally:
                    env.close()

            results_data.append(episode_results)

        self.results = pd.DataFrame(results_data)
        return self.results

    def run_wind_grid_evaluation(
        self,
        wd_step: int = 10,
        ws_step: int = 2,
        ti_points: int = 3,
        wd_min: Optional[float] = None,
        wd_max: Optional[float] = None,
        ws_min: Optional[float] = None,
        ws_max: Optional[float] = None,
        ti_min: Optional[float] = None,
        ti_max: Optional[float] = None,
        deterministic: bool = True,
        save_netcdf: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Run evaluation over a grid of wind conditions and return as xarray Dataset.

        Args:
            wd_step (int): Wind direction step size in degrees
            ws_step (int): Wind speed step size in m/s
            ti_points (int): Number of turbulence intensity points
            wd_min (Optional[float]): Minimum wind direction. If None, uses env.wd_min
            wd_max (Optional[float]): Maximum wind direction. If None, uses env.wd_max
            ws_min (Optional[float]): Minimum wind speed. If None, uses env.ws_min
            ws_max (Optional[float]): Maximum wind speed. If None, uses env.ws_max
            ti_min (Optional[float]): Minimum turbulence intensity. If None, uses env.TI_min
            ti_max (Optional[float]): Maximum turbulence intensity. If None, uses env.TI_max
            deterministic (bool): Whether to use deterministic policies
            save_netcdf (Optional[str]): Path to save NetCDF file

        Returns:
            xr.Dataset: Results with dimensions (wd, ws, ti) and variables for each agent
        """
        # Get environment bounds (use as defaults)
        temp_env = self.env_factory()

        # Fixed-condition evaluation needs set_wind_vals (FarmEval has it;
        # plain WindFarmEnv does not) — fail fast with a clear message.
        if not hasattr(temp_env, "set_wind_vals"):
            temp_env.close()
            raise TypeError(
                "run_wind_grid_evaluation requires an environment with a "
                "set_wind_vals() method (e.g. FarmEval), got "
                f"{type(temp_env).__name__}. Build your env_factory around "
                "FarmEval instead of WindFarmEnv."
            )

        # Use user-provided bounds or fall back to environment defaults
        wd_min_actual = wd_min if wd_min is not None else temp_env.wd_inflow_min
        wd_max_actual = wd_max if wd_max is not None else temp_env.wd_inflow_max
        ws_min_actual = ws_min if ws_min is not None else temp_env.ws_inflow_min
        ws_max_actual = ws_max if ws_max is not None else temp_env.ws_inflow_max
        ti_min_actual = ti_min if ti_min is not None else temp_env.TI_inflow_min
        ti_max_actual = ti_max if ti_max is not None else temp_env.TI_inflow_max

        temp_env.close()

        # Create grids using actual bounds
        wd_grid = np.arange(wd_min_actual, wd_max_actual + 1, wd_step)
        # A full-circle sweep (e.g. 0..360) would evaluate the same direction
        # twice at both ends; drop the duplicate endpoint.
        if len(wd_grid) > 1 and (wd_grid[-1] - wd_grid[0]) % 360 == 0:
            wd_grid = wd_grid[:-1]
        ws_grid = np.arange(ws_min_actual, ws_max_actual + 1, ws_step)
        ti_grid = np.linspace(ti_min_actual, ti_max_actual, ti_points)

        total_cases = len(wd_grid) * len(ws_grid) * len(ti_grid)
        print(f"Running wind grid evaluation over {total_cases} conditions...")
        print(
            f"  Wind Direction: {wd_min_actual}° to {wd_max_actual}° (step: {wd_step}°)"
        )
        print(
            f"  Wind Speed: {ws_min_actual} to {ws_max_actual} m/s (step: {ws_step} m/s)"
        )
        print(
            f"  Turbulence Intensity: {ti_min_actual:.3f} to {ti_max_actual:.3f} ({ti_points} points)"
        )

        # Initialize result arrays
        results = {}
        for agent_name in self.agent_names:
            results[agent_name] = np.full(
                (len(wd_grid), len(ws_grid), len(ti_grid)), np.nan
            )

        # Run evaluation
        with tqdm(total=total_cases, desc="Wind Grid Evaluation") as pbar:
            for i, wd in enumerate(wd_grid):
                for j, ws in enumerate(ws_grid):
                    for k, ti in enumerate(ti_grid):
                        # Use condition-based seed for reproducibility. Mix
                        # with distinct multipliers so nearby conditions
                        # cannot collide (wd*100 + ws*10 + ti*1000 did).
                        condition_seed = (
                            int(round(wd * 1e3)) * 1_000_003
                            + int(round(ws * 1e3)) * 1_009
                            + int(round(ti * 1e4))
                        ) % (2**31)

                        for agent_name in self.agent_names:
                            env = self.env_factory()
                            env.n_passthrough = self.n_passthrough
                            env.burn_in_passthroughs = self.burn_in_passthroughs

                            try:
                                # Set fixed wind conditions (overrides sample_site)
                                env.set_wind_vals(ws=ws, ti=ti, wd=wd)
                                agent = self.agents[agent_name]

                                # Run episode with fixed conditions (no stochastic sampling)
                                mean_reward = self._run_single_episode_summary(
                                    env,
                                    agent,
                                    condition_seed,
                                    deterministic,
                                    use_stochastic_wind=False,  # Use fixed conditions
                                )
                                results[agent_name][i, j, k] = mean_reward

                            finally:
                                env.close()

                        pbar.update(1)

        # Create xarray Dataset
        coords = {"wd": wd_grid, "ws": ws_grid, "ti": ti_grid}

        data_vars = {}
        for agent_name in self.agent_names:
            data_vars[f"{agent_name}_mean_reward"] = (
                ("wd", "ws", "ti"),
                results[agent_name],
            )

        dataset = xr.Dataset(data_vars, coords=coords)

        # Add metadata including actual bounds used
        dataset.attrs.update(
            {
                "description": "Wind farm agent evaluation results",
                "agents": list(self.agent_names),
                "n_passthrough": self.n_passthrough,
                "burn_in_passthroughs": self.burn_in_passthroughs,
                "evaluation_type": "wind_grid",
                # FIX: Convert the dictionary to a string for NetCDF compatibility
                "bounds_used": str(
                    {
                        "wd_min": wd_min_actual,
                        "wd_max": wd_max_actual,
                        "ws_min": ws_min_actual,
                        "ws_max": ws_max_actual,
                        "ti_min": ti_min_actual,
                        "ti_max": ti_max_actual,
                    }
                ),
            }
        )

        # Save NetCDF if requested
        if save_netcdf:
            dataset.to_netcdf(save_netcdf)
            print(f"Results saved to {save_netcdf}")

        return dataset

    def plot_time_series_comparison(
        self,
        episodes_to_plot: Optional[List[int]] = None,
        save_path: str = "time_series_comparison.png",
    ):
        """
        Plot time series comparison of mean cumulative rewards.

        Args:
            episodes_to_plot (Optional[List[int]]): Specific episodes to plot.
                                                  If None, plots first 3 episodes.
            save_path (str): Path to save the figure
        """
        if self.time_series_results is None:
            print("No time series data available. Run time series evaluation first.")
            return

        if episodes_to_plot is None:
            episodes_to_plot = list(
                range(min(3, len(list(self.time_series_results.values())[0])))
            )

        fig, axes = plt.subplots(
            len(episodes_to_plot), 1, figsize=(12, 4 * len(episodes_to_plot))
        )
        if len(episodes_to_plot) == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.agent_names)))

        for ep_idx, episode in enumerate(episodes_to_plot):
            ax = axes[ep_idx]

            for agent_idx, agent_name in enumerate(self.agent_names):
                if episode < len(self.time_series_results[agent_name]):
                    result = self.time_series_results[agent_name][episode]
                    history = result["history"]

                    ax.plot(
                        history["step"],
                        history["mean_cumulative_reward"],
                        color=colors[agent_idx],
                        label=agent_name,
                        linewidth=2,
                    )

            ax.set_title(f"Episode {episode}: Mean Cumulative Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean Cumulative Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Time series plot saved to {save_path}")
        plt.clf()
        plt.close(fig)

    def plot_summary_comparison(self, save_path: str = "summary_comparison.png"):
        """
        Plot summary comparison showing average performance across all episodes.

        Args:
            save_path (str): Path to save the figure
        """
        if self.results is None:
            print("No results to plot. Run an evaluation first.")
            return

        mean_rewards = self.results[self.agent_names].mean()
        std_rewards = self.results[self.agent_names].std()

        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.bar(
            mean_rewards.index,
            mean_rewards.values,
            yerr=std_rewards.values,
            capsize=5,
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(self.agent_names))),
        )

        ax.bar_label(bars, fmt="{:.3f}", padding=3)

        ax.set_ylabel("Mean Cumulative Reward", fontsize=12)
        ax.set_title("Agent Performance Comparison", fontsize=14, weight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Summary plot saved to {save_path}")
        plt.clf()
        plt.close(fig)

    def plot_wind_grid_results(
        self,
        dataset: xr.Dataset,
        agent_name: Optional[str] = None,
        save_path: str = "wind_grid_results.png",
    ):
        """
        Plot wind grid evaluation results as heatmaps.

        Args:
            dataset (xr.Dataset): Results from wind grid evaluation
            agent_name (Optional[str]): Specific agent to plot. If None, plots all agents.
            save_path (str): Path to save the figure
        """
        agents_to_plot = [agent_name] if agent_name else self.agent_names

        n_agents = len(agents_to_plot)
        n_ti = len(dataset.ti)

        # Create subplots, one for each agent and TI combination
        fig, axes = plt.subplots(
            n_agents, n_ti, figsize=(7 * n_ti, 5 * n_agents), squeeze=False
        )

        for i, agent in enumerate(agents_to_plot):
            for j, ti_val in enumerate(dataset.ti.values):
                ax = axes[i, j]

                # Select the data for the current agent and TI value
                data_for_ti = dataset[f"{agent}_mean_reward"].sel(ti=ti_val)

                # Loop through each wind speed to plot it as a separate line
                for ws_val in data_for_ti.ws.values:
                    # Select the data for this specific wind speed
                    data_for_ws = data_for_ti.sel(ws=ws_val)

                    # Plot Reward vs. Wind Direction
                    ax.plot(
                        data_for_ws.wd,
                        data_for_ws.values,
                        marker="o",
                        linestyle="-",
                        label=f"WS = {ws_val} m/s",
                    )

                ax.set_title(f"{agent} (TI={ti_val:.3f})")
                ax.set_xlabel("Wind Direction (deg)")
                ax.set_ylabel("Mean Reward")
                ax.legend()
                ax.grid(True, alpha=0.4)

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for suptitle
        fig.suptitle(
            "Agent Performance Across Wind Conditions", fontsize=16, weight="bold"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Wind grid plot saved to {save_path}")
        plt.clf()
        plt.close(fig)

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all agents across all episodes."""
        if self.results is None:
            print("No results available. Run an evaluation first.")
            return pd.DataFrame()

        summary_stats = []
        for agent_name in self.agent_names:
            agent_results = self.results[agent_name]
            stats = {
                "Agent": agent_name,
                "Mean": agent_results.mean(),
                "Std": agent_results.std(),
                "Min": agent_results.min(),
                "Max": agent_results.max(),
                "Median": agent_results.median(),
            }
            summary_stats.append(stats)

        return pd.DataFrame(summary_stats)

    @staticmethod
    def create_env_factory_with_site(env_class, site, **env_kwargs):
        """
        Helper method to create an environment factory with sample_site configured.

        Args:
            env_class: Environment class (e.g., WindFarmEnv, EvaluationEnv)
            site: PyWake site object for realistic wind sampling
            **env_kwargs: Additional environment parameters

        Returns:
            Callable: Environment factory function

        Example:
            from py_wake.examples.data.hornsrev1 import Hornsrev1Site
            from WindGym import FarmEval

            site = Hornsrev1Site()
            env_factory = Coliseum.create_env_factory_with_site(
                FarmEval, site,
                turbine=V80(), x_pos=x_pos, y_pos=y_pos,
                config="config.yaml"
            )
            coliseum = Coliseum(env_factory, agents)

        Note: run_wind_grid_evaluation needs an env with set_wind_vals(),
        which FarmEval provides but plain WindFarmEnv does not.
        """

        def factory():
            return env_class(sample_site=site, **env_kwargs)

        return factory
