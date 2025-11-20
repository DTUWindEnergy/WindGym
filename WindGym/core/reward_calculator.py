"""
Reward calculation module for WindGym environments.

This module handles all reward and penalty calculations, providing a clean
interface for different reward strategies.
"""

from typing import Optional
import numpy as np
import itertools


class RewardCalculator:
    """
    Calculates rewards and penalties for wind farm control.

    Supports multiple reward strategies:
    - Baseline: Compare agent performance to baseline controller
    - Power_avg: Reward based on average power production
    - Power_diff: Reward based on power improvement over time
    - None: No power reward

    Also handles action penalties to encourage stable control.
    """

    def __init__(
        self,
        power_reward_type: str = "Baseline",
        track_power: bool = False,
        power_scaling: float = 1.0,
        action_penalty: float = 0.0,
        action_penalty_type: Optional[str] = None,
        power_window_size: Optional[int] = None,
    ):
        """
        Initialize the reward calculator.

        Args:
            power_reward_type: Type of power reward ("Baseline", "Power_avg", "Power_diff", "None")
            track_power: Whether to include power tracking reward (not yet implemented)
            power_scaling: Scaling factor for power reward
            action_penalty: Weight for action penalty (0 = no penalty)
            action_penalty_type: Type of penalty ("change" or "total")
            power_window_size: Window size for Power_diff reward type
        """
        self.power_reward_type = power_reward_type
        self.track_power = track_power
        self.power_scaling = power_scaling
        self.action_penalty = action_penalty
        self.action_penalty_type = action_penalty_type
        self._power_window_size = power_window_size

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate reward calculator configuration."""
        valid_power_rewards = {"Baseline", "Power_avg", "Power_diff", "None"}
        if self.power_reward_type not in valid_power_rewards:
            raise ValueError(
                "The Power_reward must be either Baseline, Power_avg, None or Power_diff"
            )

        if self.power_reward_type == "Power_diff":
            if self._power_window_size is None:
                raise ValueError(
                    "power_window_size must be provided for Power_diff reward type"
                )
            if self._power_window_size < 40:
                raise ValueError(
                    "The Power_avg must be larger then 40 for the Power_diff reward"
                )

        if self.track_power:
            raise NotImplementedError("Power tracking reward is not yet implemented.")

        if self.action_penalty_type is not None:
            valid_penalty_types = {"change", "total"}
            penalty_lower = self.action_penalty_type.lower()
            if penalty_lower not in valid_penalty_types:
                raise ValueError(
                    f"action_penalty_type must be one of {valid_penalty_types}, "
                    f"got '{self.action_penalty_type}'"
                )

    def calculate_power_reward(
        self,
        farm_power_deque,
        baseline_power_deque: Optional[object] = None,
        rated_power: Optional[float] = None,
        n_turbines: int = 1,
    ) -> float:
        """
        Calculate the power production reward.

        Args:
            farm_power_deque: Deque containing farm power history
            baseline_power_deque: Deque containing baseline power history (for Baseline reward)
            rated_power: Rated power of a single turbine (for Power_avg reward)
            n_turbines: Number of turbines in the farm

        Returns:
            float: The calculated power reward
        """
        if self.power_reward_type == "Baseline":
            if baseline_power_deque is None:
                raise ValueError(
                    "baseline_power_deque required for Baseline reward type"
                )
            return self._power_reward_baseline(farm_power_deque, baseline_power_deque)

        elif self.power_reward_type == "Power_avg":
            if rated_power is None:
                raise ValueError("rated_power required for Power_avg reward type")
            return self._power_reward_avg(farm_power_deque, rated_power, n_turbines)

        elif self.power_reward_type == "Power_diff":
            return self._power_reward_diff(farm_power_deque, n_turbines)

        elif self.power_reward_type == "None":
            return 0.0

        else:
            raise ValueError(f"Unknown power_reward_type: {self.power_reward_type}")

    def _power_reward_baseline(self, farm_power_deque, baseline_power_deque) -> float:
        """
        Calculate reward based on baseline farm comparison.

        Reward = (agent_power / baseline_power) - 1

        Args:
            farm_power_deque: Agent farm power history
            baseline_power_deque: Baseline farm power history

        Returns:
            float: Relative performance vs baseline
        """
        power_agent_avg = np.mean(farm_power_deque)
        power_baseline_avg = np.mean(baseline_power_deque)

        if power_baseline_avg == 0:
            raise ValueError(
                f"Baseline power is zero - invalid configuration. "
                f"Agent power deque: {list(farm_power_deque)}, "
                f"Baseline power deque: {list(baseline_power_deque)}"
            )

        reward = power_agent_avg / power_baseline_avg - 1
        return reward

    def _power_reward_avg(
        self, farm_power_deque, rated_power: float, n_turbines: int
    ) -> float:
        """
        Calculate power reward based on average production.

        Reward = avg_power / (n_turbines * rated_power)

        Args:
            farm_power_deque: Farm power history
            rated_power: Rated power of a single turbine
            n_turbines: Number of turbines

        Returns:
            float: Normalized average power production
        """
        power_agent = np.mean(farm_power_deque)
        reward = power_agent / n_turbines / rated_power
        return reward

    def _power_reward_diff(self, farm_power_deque, n_turbines: int) -> float:
        """
        Calculate reward based on power improvement over time.

        Compares recent power (latest window) to older power (oldest window).
        Encourages increasing power production over the episode.

        Args:
            farm_power_deque: Farm power history
            n_turbines: Number of turbines

        Returns:
            float: Power improvement per turbine
        """
        power_len = len(farm_power_deque)

        # Get the latest window of power values
        power_latest = np.mean(
            list(
                itertools.islice(
                    farm_power_deque,
                    power_len - (self._power_window_size // 10),
                    power_len,
                )
            )
        )

        # Get the oldest window of power values
        power_oldest = np.mean(
            list(itertools.islice(farm_power_deque, 0, (self._power_window_size // 10)))
        )

        return (power_latest - power_oldest) / n_turbines

    def calculate_action_penalty(
        self,
        old_yaws: np.ndarray,
        new_yaws: np.ndarray,
        yaw_max: float,
    ) -> float:
        """
        Calculate penalty for turbine actions.

        Supports two penalty types:
        - "change": Penalize changes in yaw angle (encourages stability)
        - "total": Penalize absolute yaw magnitude (encourages alignment)

        Args:
            old_yaws: Previous yaw angles (degrees)
            new_yaws: Current yaw angles (degrees)
            yaw_max: Maximum allowed yaw angle (degrees)

        Returns:
            float: Action penalty value
        """
        if self.action_penalty < 0.001:
            # Skip calculation if penalty is negligible
            return 0.0

        if self.action_penalty_type is None:
            return 0.0

        penalty_type = self.action_penalty_type.lower()

        if penalty_type == "change":
            # Penalize the magnitude of yaw changes
            pen_val = float(np.mean(np.abs(old_yaws - new_yaws)))

        elif penalty_type == "total":
            # Penalize the absolute yaw angles (normalized by max yaw)
            pen_val = float(np.mean(np.abs(new_yaws)) / max(1e-6, yaw_max))

        else:
            pen_val = 0.0

        return float(self.action_penalty) * pen_val

    def calculate_total_reward(
        self,
        farm_power_deque,
        old_yaws: np.ndarray,
        new_yaws: np.ndarray,
        yaw_max: float,
        baseline_power_deque: Optional[object] = None,
        rated_power: Optional[float] = None,
        n_turbines: int = 1,
    ) -> tuple[float, dict]:
        """
        Calculate total reward including power reward and action penalty.

        This is a convenience method that combines power reward and action penalty
        calculations, returning both the total reward and a breakdown.

        Args:
            farm_power_deque: Agent farm power history
            old_yaws: Previous yaw angles
            new_yaws: Current yaw angles
            yaw_max: Maximum yaw angle
            baseline_power_deque: Baseline power history (if needed)
            rated_power: Rated power per turbine (if needed)
            n_turbines: Number of turbines

        Returns:
            tuple: (total_reward, reward_breakdown_dict)
        """
        # Calculate power reward
        power_reward = self.calculate_power_reward(
            farm_power_deque=farm_power_deque,
            baseline_power_deque=baseline_power_deque,
            rated_power=rated_power,
            n_turbines=n_turbines,
        )

        # Apply power scaling
        scaled_power_reward = power_reward * self.power_scaling

        # Calculate action penalty
        action_penalty = self.calculate_action_penalty(
            old_yaws=old_yaws,
            new_yaws=new_yaws,
            yaw_max=yaw_max,
        )

        # Total reward
        total_reward = scaled_power_reward - action_penalty

        # Return breakdown for logging/debugging
        breakdown = {
            "power_reward": power_reward,
            "scaled_power_reward": scaled_power_reward,
            "action_penalty": action_penalty,
            "total_reward": total_reward,
        }

        return total_reward, breakdown
