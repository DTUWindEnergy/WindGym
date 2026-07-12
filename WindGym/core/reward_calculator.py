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
        tau: float = 0.02,
        derate_penalty: float = 0.0,
        derate_penalty_type: Optional[str] = None,
        track_reward_type: str = "abs",
        track_sigma: float = 0.1,
    ):
        """
        Initialize the reward calculator.

        Args:
            power_reward_type: Type of power reward ("Baseline", "Power_avg", "Power_diff", "None")
            track_power: Whether to use the power tracking reward instead of a
                power maximization reward (requires power_reward_type "None")
            power_scaling: Scaling factor for power reward
            action_penalty: Weight for action penalty (0 = no penalty)
            action_penalty_type: Type of penalty ("change" or "total")
            power_window_size: Window size for Power_diff reward type
            derate_penalty: Weight for derate penalty (0 = no penalty)
            derate_penalty_type: Type of derate penalty ("change" or "total")
            track_reward_type: Shape of the tracking reward ("abs" or "gaussian")
            track_sigma: Width of the gaussian tracking reward, as a fraction
                of the power normalization (rated farm power)
        """
        self.power_reward_type = power_reward_type
        self.track_power = track_power
        self.power_scaling = power_scaling
        self.action_penalty = action_penalty
        self.action_penalty_type = action_penalty_type
        self._power_window_size = power_window_size
        self.tau = tau
        self.derate_penalty = derate_penalty
        self.derate_penalty_type = derate_penalty_type
        self.track_reward_type = track_reward_type
        self.track_sigma = track_sigma

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate reward calculator configuration."""
        valid_power_rewards = {
            "Baseline",
            "Power_avg",
            "Power_diff",
            "Wake_recovery",
            "None",
        }
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
            # Tracking and maximization are mutually exclusive objectives.
            if self.power_reward_type != "None":
                raise ValueError(
                    "Track_power is mutually exclusive with a power reward: "
                    f"set Power_reward to 'None' (got '{self.power_reward_type}')."
                )
            valid_track_rewards = {"abs", "gaussian"}
            if self.track_reward_type not in valid_track_rewards:
                raise ValueError(
                    f"track_reward_type must be one of {valid_track_rewards}, "
                    f"got '{self.track_reward_type}'"
                )
            if self.track_sigma <= 0:
                raise ValueError(
                    f"track_sigma must be positive, got {self.track_sigma}"
                )

        if self.action_penalty_type is not None:
            valid_penalty_types = {"change", "total"}
            penalty_lower = self.action_penalty_type.lower()
            if penalty_lower not in valid_penalty_types:
                raise ValueError(
                    f"action_penalty_type must be one of {valid_penalty_types}, "
                    f"got '{self.action_penalty_type}'"
                )

        if self.derate_penalty_type is not None:
            valid_penalty_types = {"change", "total"}
            penalty_lower = self.derate_penalty_type.lower()
            if penalty_lower not in valid_penalty_types:
                raise ValueError(
                    f"derate_penalty_type must be one of {valid_penalty_types}, "
                    f"got '{self.derate_penalty_type}'"
                )

    def calculate_power_reward(
        self,
        farm_power_deque,
        baseline_power_deque: Optional[object] = None,
        rated_power: Optional[float] = None,
        n_turbines: int = 1,
        nowake_power_deque: Optional[object] = None,
        power_ref_deque: Optional[object] = None,
        power_norm: Optional[float] = None,
    ) -> float:
        """
        Calculate the power production reward.

        Args:
            farm_power_deque: Deque containing farm power history
            baseline_power_deque: Deque containing baseline power history (for Baseline reward)
            rated_power: Freestream power of a single turbine at the episode
                inflow wind speed (for Power_avg reward)
            n_turbines: Number of turbines in the farm
            power_ref_deque: Deque containing the power reference history
                (for the tracking reward; same window as farm_power_deque)
            power_norm: Power normalization for the tracking reward, the
                rated (nameplate) farm power in watts

        Returns:
            float: The calculated power reward
        """
        if self.track_power:
            return self._power_reward_tracking(
                farm_power_deque, power_ref_deque, power_norm
            )

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

        elif self.power_reward_type == "Wake_recovery":
            if baseline_power_deque is None:
                raise ValueError(
                    "baseline_power_deque required for Wake_recovery reward type"
                )
            if nowake_power_deque is None:
                raise ValueError(
                    "nowake_power_deque required for Wake_recovery reward type"
                )
            return self._power_reward_wake_recovery(
                farm_power_deque, baseline_power_deque, nowake_power_deque
            )

        elif self.power_reward_type == "Power_diff":
            return self._power_reward_diff(farm_power_deque, n_turbines)

        elif self.power_reward_type == "None":
            return 0.0

        else:
            raise ValueError(f"Unknown power_reward_type: {self.power_reward_type}")

    def _power_reward_tracking(
        self, farm_power_deque, power_ref_deque, power_norm
    ) -> float:
        """
        Calculate the power tracking reward.

        Compares the window mean of the farm power to the window mean of the
        reference (both deques share the same window, so a reference step
        change does not cause an unavoidable penalty spike):

        - "abs":      r = -|P_farm - P_ref| / power_norm
        - "gaussian": r = exp(-((P_farm - P_ref) / (sigma * power_norm))^2)

        Args:
            farm_power_deque: Farm power history
            power_ref_deque: Power reference history (same window)
            power_norm: Rated (nameplate) farm power in watts

        Returns:
            float: The tracking reward
        """
        if power_ref_deque is None or len(power_ref_deque) == 0:
            raise ValueError(
                "power_ref_deque required (and non-empty) for the tracking reward"
            )
        if power_norm is None or power_norm <= 0:
            raise ValueError(
                f"power_norm must be a positive number for the tracking reward, "
                f"got {power_norm}"
            )

        err = np.mean(farm_power_deque) - np.mean(power_ref_deque)

        if self.track_reward_type == "abs":
            return -abs(err) / power_norm
        else:  # "gaussian" (validated in _validate_config)
            return float(np.exp(-((err / (self.track_sigma * power_norm)) ** 2)))

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

    def _power_reward_wake_recovery(
        self, farm_power_deque, baseline_power_deque, nowake_power_deque
    ) -> float:
        P_agent = np.mean(farm_power_deque)
        P_greedy = np.mean(baseline_power_deque)
        P_freestream = np.mean(nowake_power_deque)

        gain = P_agent - P_greedy
        headroom = max(P_freestream - P_greedy, self.tau * P_freestream)

        if headroom <= 0:
            raise ValueError(
                f"Wake_recovery headroom is non-positive ({headroom}). "
                f"P_freestream={P_freestream}, P_greedy={P_greedy}, tau={self.tau}"
            )

        return gain / headroom

    def _power_reward_avg(
        self, farm_power_deque, rated_power: float, n_turbines: int
    ) -> float:
        """
        Calculate power reward based on average production.

        Reward = avg_power / (n_turbines * rated_power)

        Args:
            farm_power_deque: Farm power history
            rated_power: Freestream power of a single turbine at the episode
                inflow wind speed (the env sets this per reset; it is NOT the
                turbine's nameplate rating)
            n_turbines: Number of turbines

        Returns:
            float: Normalized average power production
        """
        if rated_power <= 0:
            # Below cut-in the freestream reference power is zero and farm
            # power is ~0 too; 0 is the principled limit (avoids inf/NaN).
            return 0.0
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
        window = self._power_window_size // 10

        # Get the latest window of power values. The deque can be shorter
        # than the window early in the episode (fill_window=False), so clamp
        # the start index to 0 — islice raises on negative indices.
        power_latest = np.mean(
            list(
                itertools.islice(
                    farm_power_deque,
                    max(0, power_len - window),
                    power_len,
                )
            )
        )

        # Get the oldest window of power values
        power_oldest = np.mean(list(itertools.islice(farm_power_deque, 0, window)))

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

    def calculate_derate_penalty(
        self,
        old_derates: Optional[np.ndarray],
        new_derates: Optional[np.ndarray],
        derate_max: float = 1.0,
    ) -> float:
        """
        Calculate penalty for derating actions (mirrors the yaw action penalty).

        Supports two penalty types:
        - "change": Penalize changes in derate level (encourages stability)
        - "total": Penalize derate magnitude (encourages full production),
          normalized by derate_max

        Args:
            old_derates: Derate levels at the previous env step
            new_derates: Current derate levels
            derate_max: Maximum allowed derate level

        Returns:
            float: Derate penalty value
        """
        if self.derate_penalty < 0.001:
            return 0.0

        if self.derate_penalty_type is None or new_derates is None:
            return 0.0

        penalty_type = self.derate_penalty_type.lower()

        if penalty_type == "change":
            if old_derates is None:
                return 0.0
            pen_val = float(np.mean(np.abs(old_derates - new_derates)))

        elif penalty_type == "total":
            pen_val = float(np.mean(np.abs(new_derates)) / max(1e-6, derate_max))

        else:
            pen_val = 0.0

        return float(self.derate_penalty) * pen_val

    def calculate_total_reward(
        self,
        farm_power_deque,
        old_yaws: np.ndarray,
        new_yaws: np.ndarray,
        yaw_max: float,
        baseline_power_deque: Optional[object] = None,
        rated_power: Optional[float] = None,
        n_turbines: int = 1,
        nowake_power_deque: Optional[object] = None,
        old_derates: Optional[np.ndarray] = None,
        new_derates: Optional[np.ndarray] = None,
        derate_max: float = 1.0,
        power_ref_deque: Optional[object] = None,
        power_norm: Optional[float] = None,
    ) -> tuple[float, dict]:
        """
        Calculate total reward including power reward and action penalties.

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
            old_derates: Derate levels at the previous env step (if derating)
            new_derates: Current derate levels (if derating)
            derate_max: Maximum allowed derate level
            power_ref_deque: Power reference history (if tracking)
            power_norm: Rated farm power in watts (if tracking)

        Returns:
            tuple: (total_reward, reward_breakdown_dict)
        """
        # Calculate power reward
        power_reward = self.calculate_power_reward(
            farm_power_deque=farm_power_deque,
            baseline_power_deque=baseline_power_deque,
            rated_power=rated_power,
            n_turbines=n_turbines,
            nowake_power_deque=nowake_power_deque,
            power_ref_deque=power_ref_deque,
            power_norm=power_norm,
        )

        # Apply power scaling
        scaled_power_reward = power_reward * self.power_scaling

        # Calculate action penalty
        action_penalty = self.calculate_action_penalty(
            old_yaws=old_yaws,
            new_yaws=new_yaws,
            yaw_max=yaw_max,
        )

        # Calculate derate penalty
        derate_penalty = self.calculate_derate_penalty(
            old_derates=old_derates,
            new_derates=new_derates,
            derate_max=derate_max,
        )

        # Total reward
        total_reward = scaled_power_reward - action_penalty - derate_penalty

        # Return breakdown for logging/debugging
        breakdown = {
            "power_reward": power_reward,
            "scaled_power_reward": scaled_power_reward,
            "action_penalty": action_penalty,
            "derate_penalty": derate_penalty,
            "total_reward": total_reward,
        }
        if self.track_power:
            breakdown["tracking_error"] = float(
                np.mean(farm_power_deque) - np.mean(power_ref_deque)
            )

        return total_reward, breakdown
