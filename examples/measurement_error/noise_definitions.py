# In examples/measurement_error/noise_definitions.py

from WindGym.Measurement_Manager import (
    HybridNoiseModel,
    WhiteNoiseModel,
    EpisodicBiasNoiseModel,
    MeasurementType,
    NoiseModel,
    MeasurementSpec,
)
import numpy as np
import torch
from typing import Dict, List

# --- Centralized Definitions for Adversarial Attacks ---


def get_adversarial_constraints() -> dict:
    """Returns the standardized dictionary of constraints for adversarial training."""
    return {
        "max_bias_ws": 4.0,  # Max wind speed bias in m/s
        "max_bias_wd": 10.0,  # Max wind direction bias in degrees
        "max_bias_power": 500000.0,  # Max power bias in Watts (+/- 500 kW)
        "max_bias_yaw": 20.0,  # Max yaw angle bias in degrees
    }


def create_adversarial_noise_model(
    antagonist_agent, device: str
) -> "AdversarialNoiseModel":
    """Factory function for creating a stateful adversarial noise model with default constraints."""
    constraints = get_adversarial_constraints()
    return AdversarialNoiseModel(
        antagonist_agent=antagonist_agent, constraints=constraints, device=device
    )


# --- Centralized Definition for Procedural Noise ---


def create_procedural_noise_model(
    ws_std: float = 0.5,
    wd_std: float = 2.0,
    ws_bias_range: tuple[float, float] = (-4.0, 4.0),
    wd_bias_range: tuple[float, float] = (-10.0, 10.0),
    power_bias_range: tuple[float, float] = (-500000.0, 500000.0),
    yaw_bias_range: tuple[float, float] = (-20.0, 20.0),
) -> HybridNoiseModel:
    """
    Factory function for the standardized "Procedural Noise" model.
    This model combines Gaussian white noise with a persistent episodic bias.
    """
    white_noise = WhiteNoiseModel(
        noise_std_devs={
            MeasurementType.WIND_SPEED: ws_std,
            MeasurementType.WIND_DIRECTION: wd_std,
        }
    )
    bias_noise = EpisodicBiasNoiseModel(
        {
            MeasurementType.WIND_SPEED: ws_bias_range,
            MeasurementType.WIND_DIRECTION: wd_bias_range,
            MeasurementType.POWER: power_bias_range,
            MeasurementType.YAW_ANGLE: yaw_bias_range,
        }
    )
    return HybridNoiseModel(models=[white_noise, bias_noise])


# --- Core Adversarial Noise Model Class ---


class AdversarialNoiseModel(NoiseModel):
    """
    Applies a dynamically evolving bias based on the actions of an antagonist agent.
    This noise model is stateful and self-contained.
    """

    def __init__(
        self, antagonist_agent=None, constraints: dict = None, device: str = "cpu"
    ):
        super().__init__()
        self.antagonist = antagonist_agent
        self.constraints = (
            constraints if constraints is not None else get_adversarial_constraints()
        )
        self.device = device
        self.current_bias_state: Dict[str, float] = {}

    def set_antagonist_agent(self, agent):
        self.antagonist = agent

    def reset_noise(self, specs: List[MeasurementSpec], rng: np.random.Generator):
        self.current_bias_state = {}

    def apply_noise(
        self,
        clean_observations: np.ndarray,
        specs: List[MeasurementSpec],
        rng: np.random.Generator,
        adv_action: np.ndarray = None,  # <-- Make the action optional
    ) -> np.ndarray:
        """
        Calculates and applies the adversarial bias.
        - If adv_action is provided (from a training loop), it uses that action.
        - If adv_action is None (from a simple evaluation loop), it generates its own action.
        """
        if self.antagonist is None:
            return clean_observations

        antagonist_action = None
        if adv_action is not None:
            # Use the action passed in by the training loop
            antagonist_action = adv_action
        else:
            # Get the action itself if none was provided
            obs_tensor = torch.Tensor(clean_observations).to(self.device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(self.antagonist, "predict"):
                    action, _ = self.antagonist.predict(obs_tensor, deterministic=True)
                    antagonist_action = action.flatten()
                else:
                    antagonist_action = (
                        self.antagonist.actor_mean(obs_tensor).squeeze(0).cpu().numpy()
                    )

        noisy_obs = clean_observations.copy()
        action_idx = 0

        # 2. Update the internal bias state based on the antagonist's action
        for spec in specs:
            is_ws = "ws_current" in spec.name and "turb" in spec.name
            is_wd = "wd_current" in spec.name and "turb" in spec.name
            is_power = "power_current" in spec.name and "turb" in spec.name
            is_yaw = "yaw_current" in spec.name and "turb" in spec.name

            if (is_ws or is_wd or is_power or is_yaw) and action_idx < len(
                antagonist_action
            ):
                if is_ws:
                    max_bias = self.constraints.get("max_bias_ws", 0.0)
                elif is_wd:
                    max_bias = self.constraints.get("max_bias_wd", 0.0)
                elif is_power:
                    max_bias = self.constraints.get("max_bias_power", 0.0)
                elif is_yaw:
                    max_bias = self.constraints.get("max_bias_yaw", 0.0)
                else:
                    continue
                max_change = max_bias * 0.1
                bias_delta = antagonist_action[action_idx] * max_change
                current_bias = self.current_bias_state.get(spec.name, 0.0)
                new_bias = np.clip(current_bias + bias_delta, -max_bias, max_bias)
                self.current_bias_state[spec.name] = new_bias
                action_idx += 1

        # 3. Apply the final, confounded biases to all relevant observation vector components
        for spec in specs:
            base_spec_name = None
            if "turb" in spec.name:
                if "ws" in spec.name:
                    base_spec_name = f"turb_{spec.turbine_id}/ws_current"
                elif "wd" in spec.name:
                    base_spec_name = f"turb_{spec.turbine_id}/wd_current"
                elif "yaw" in spec.name:
                    base_spec_name = f"turb_{spec.turbine_id}/yaw_current"

            if not base_spec_name:
                continue

            primary_bias = self.current_bias_state.get(base_spec_name, 0.0)
            final_bias = primary_bias

            if spec.measurement_type == MeasurementType.YAW_ANGLE:
                wd_spec_name = f"turb_{spec.turbine_id}/wd_current"
                wd_bias = self.current_bias_state.get(wd_spec_name, 0.0)
                final_bias = primary_bias - wd_bias

            if final_bias != 0.0:
                span = spec.max_val - spec.min_val
                if span > 0:
                    scaled_delta = (final_bias * 2.0) / span
                    # Apply to observation
                    noisy_obs[spec.index_range[0] : spec.index_range[1]] += scaled_delta

        return np.clip(noisy_obs, -1.0, 1.0)

    def get_info(self) -> Dict:
        return {
            "noise_type": "adversarial_stateful",
            "applied_bias_physical": self.current_bias_state.copy(),
        }
