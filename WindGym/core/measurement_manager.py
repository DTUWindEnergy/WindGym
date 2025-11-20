"""
Modular Measurement and Noise System for WindGym Environments.

This module provides a structured way to define, manage, and apply various
types of noise (e.g., white noise, episodic bias, adversarial) to the
observations from a WindGym environment. It is designed for modularity and extensibility.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Callable,
)  # Import Callable
from dataclasses import dataclass
from enum import Enum
import gymnasium as gym


class MeasurementType(Enum):
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    YAW_ANGLE = "yaw_angle"
    TURBULENCE_INTENSITY = "turbulence_intensity"
    POWER = "power"
    GENERIC = "generic"


@dataclass
class MeasurementSpec:
    """
    Specification for a single component of the observation vector.

    Attributes:
        name (str): A descriptive name for the measurement (e.g., 'turb_0/ws_current').
        measurement_type (MeasurementType): The physical type of the measurement.
        index_range (Tuple[int, int]): The start and end indices in the flat observation array.
        min_val (float): The minimum physical value for scaling.
        max_val (float): The maximum physical value for scaling.
        turbine_id (Optional[int]): The turbine index, if applicable.
        noise_sensitivity (float): A multiplier for the applied noise level.
    """

    name: str
    measurement_type: MeasurementType
    index_range: Tuple[int, int]
    min_val: float
    max_val: float
    turbine_id: Optional[int] = None
    noise_sensitivity: float = 1.0
    is_circular: bool = False
    circular_range: float = 360.0


# --- Noise Model Definitions ---


class NoiseModel(ABC):
    # These will be set by MeasurementManager during its initialization
    # and will be accessible as NoiseModel._unscale_value_static
    _unscale_value_static: Optional[Callable] = None
    _scale_value_static: Optional[Callable] = None

    def reset_noise(self, specs: List[MeasurementSpec], rng: np.random.Generator):
        pass

    @abstractmethod
    def apply_noise(
        self,
        observations: np.ndarray,
        specs: List[MeasurementSpec],
        rng: np.random.Generator,
    ) -> np.ndarray:
        pass  # pragma: no cover

    @abstractmethod
    def get_info(self) -> Dict:
        pass  # pragma: no cover

    # This method is now defined once in the base class and uses the static helper functions.
    def _handle_circular_noise(
        self,
        values_scaled: np.ndarray,  # Input is already scaled
        noise_unscaled_physical: np.ndarray,  # Noise is in physical units (degrees)
        spec: MeasurementSpec,
    ) -> np.ndarray:
        """Handle noise addition for circular measurements in a robust way."""
        # Ensure the static scaling methods have been set
        if (
            NoiseModel._unscale_value_static is None
            or NoiseModel._scale_value_static is None
        ):
            raise RuntimeError(
                "Scaling functions not set on NoiseModel base class. "
                "Ensure MeasurementManager initializes NoiseModel._scale_value_static and _unscale_value_static."
            )

        if spec.is_circular:
            # 1. Convert scaled observation values to physical units (0-360 degrees)
            values_physical = NoiseModel._unscale_value_static(  # Use static method
                values_scaled, spec.min_val, spec.max_val
            )

            # 2. Add noise in physical units
            noisy_physical = values_physical + noise_unscaled_physical

            # 3. Apply circular wrapping (e.g., 0-360 degrees)
            wrapped_physical = (
                noisy_physical + spec.circular_range
            ) % spec.circular_range

            # 4. Convert back to scaled representation [-1, 1] using the original min/max of the observation space
            return NoiseModel._scale_value_static(
                wrapped_physical, spec.min_val, spec.max_val
            )  # Use static method
        else:
            # For non-circular, this method expects noise_unscaled_physical to be the physical noise value.
            # However, this method is only called for circular types, so this 'else' branch is largely theoretical
            # if `apply_noise` calls it correctly. If it *were* called for non-circular, the noise_unscaled_physical
            # would need to be scaled before adding, but apply_noise handles that.
            # We keep it consistent with previous logic that expected noise_unscaled_physical.
            return (
                values_scaled + noise_unscaled_physical
            )  # This line is functionally equivalent to `observations[start:end] += scaled_noise_value` below


class WhiteNoiseModel(NoiseModel):
    """Applies Gaussian white noise defined in physical units (e.g., m/s, degrees)."""

    def __init__(self, noise_std_devs: Dict[MeasurementType, float]):
        super().__init__()  # Call parent constructor
        self.noise_std_devs = noise_std_devs

    def apply_noise(
        self,
        observations: np.ndarray,
        specs: List[MeasurementSpec],
        rng: np.random.Generator,
    ) -> np.ndarray:
        noisy_obs = observations.copy()

        # --- Pass 1: Generate and store all primary unscaled noise values ---
        physical_noise_map = {}
        for spec in specs:
            if spec.measurement_type in self.noise_std_devs:
                unscaled_std = (
                    self.noise_std_devs[spec.measurement_type] * spec.noise_sensitivity
                )
                # Generate noise in physical units
                physical_noise = rng.normal(
                    0, unscaled_std, size=spec.index_range[1] - spec.index_range[0]
                )
                physical_noise_map[spec.name] = physical_noise

        # --- Pass 2: Apply noise, with special handling for yaw ---
        for spec in specs:
            # Skip if no noise is defined for this measurement type
            if spec.name not in physical_noise_map:
                continue

            start, end = spec.index_range
            primary_noise_physical = physical_noise_map[spec.name]
            final_noise_physical = primary_noise_physical

            # --- Confounding Logic for Yaw Angle ---
            if spec.measurement_type == MeasurementType.YAW_ANGLE:
                if spec.turbine_id is not None:
                    # Find corresponding 'current' wind direction noise for this turbine
                    wd_spec_name = f"turb_{spec.turbine_id}/wd_current"
                    wd_noise = physical_noise_map.get(wd_spec_name, 0.0)
                else:
                    # Fallback for farm-level yaw
                    wd_spec_name = "farm/wd_current"
                    wd_noise = physical_noise_map.get(wd_spec_name, 0.0)

                # Sensed Yaw = True Yaw - WD_noise + Yaw_noise
                final_noise_physical = primary_noise_physical - wd_noise

            # --- Apply the final calculated noise ---
            if spec.is_circular:
                noisy_obs[start:end] = self._handle_circular_noise(
                    observations[start:end], final_noise_physical, spec
                )
            else:
                span = spec.max_val - spec.min_val
                if span > 0:
                    scaled_noise_delta = final_noise_physical * (2.0 / span)
                    noisy_obs[start:end] += scaled_noise_delta

        return noisy_obs

    def get_info(self) -> Dict:
        return {
            "noise_type": "white",
            "std_by_type (physical_units)": {
                k.value: v for k, v in self.noise_std_devs.items()
            },
        }


class EpisodicBiasNoiseModel(NoiseModel):
    """Applies a consistent bias for an entire episode, defined in physical units."""

    def __init__(self, bias_ranges: Dict[MeasurementType, Tuple[float, float]]):
        super().__init__()  # Call parent constructor
        self.bias_ranges = bias_ranges
        self.current_unscaled_biases_by_spec_name: Dict[str, float] = {}
        self.current_bias_vector: Optional[np.ndarray] = None
        self.rng: Optional[np.random.Generator] = None  # Will be set in reset_noise

    def reset_noise(self, specs: List[MeasurementSpec], rng: np.random.Generator):
        self.rng = rng  # Store the rng
        self._resample_bias(specs, rng)

    def _resample_bias(self, specs: List[MeasurementSpec], rng: np.random.Generator):
        # This method is called by reset_noise() to generate the bias for an episode.
        self.current_unscaled_biases_by_spec_name = {}

        if not specs:
            self.current_bias_vector = np.array([], dtype=np.float32)
            return

        # Pass 1: Sample all intrinsic physical biases from their defined ranges.
        for spec in specs:
            if spec.measurement_type in self.bias_ranges:
                min_bias, max_bias = self.bias_ranges[spec.measurement_type]
                unscaled_bias = rng.uniform(min_bias, max_bias) * spec.noise_sensitivity
                self.current_unscaled_biases_by_spec_name[spec.name] = unscaled_bias

        # Pass 2: Build the final scaled bias vector, applying confounding logic for yaw.
        total_obs_size = max(s.index_range[1] for s in specs) if specs else 0
        temp_scaled_bias_vector = np.zeros(total_obs_size, dtype=np.float32)

        for spec in specs:
            primary_bias_physical = self.current_unscaled_biases_by_spec_name.get(
                spec.name, 0.0
            )
            final_bias_physical = primary_bias_physical

            # --- CONFOUNDING LOGIC ---
            if spec.measurement_type == MeasurementType.YAW_ANGLE:
                wd_bias = 0.0
                if spec.turbine_id is not None:
                    # Find the corresponding wind direction bias for this turbine.
                    wd_spec_name = f"turb_{spec.turbine_id}/wd_current"
                    wd_bias = self.current_unscaled_biases_by_spec_name.get(
                        wd_spec_name, 0.0
                    )

                # Apply the formula: Sensed Yaw = True Yaw - WD_Noise + Yaw_Noise
                final_bias_physical = primary_bias_physical - wd_bias

            # --- SCALING ---
            # Scale the final physical bias and place it in the vector for apply_noise to use.
            span = spec.max_val - spec.min_val
            scaled_bias_delta = 0.0
            if span > 0:
                # This correctly scales a physical delta into the [-1, 1] observation space.
                scaled_bias_delta = final_bias_physical * (2.0 / span)

            temp_scaled_bias_vector[spec.index_range[0] : spec.index_range[1]] = (
                scaled_bias_delta
            )

        self.current_bias_vector = temp_scaled_bias_vector

    def apply_noise(
        self,
        observations: np.ndarray,
        specs: List[MeasurementSpec],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Applies the sampled episodic bias to the given observations.
        """
        if (
            self.current_bias_vector is None
            or self.current_bias_vector.shape != observations.shape
        ):
            self._resample_bias(specs, rng)
            if self.current_bias_vector is None or self.current_bias_vector.size == 0:
                return observations.copy()

        noisy_obs = observations.copy()

        for spec in specs:
            start, end = spec.index_range

            # Get the intrinsic physical bias for this measurement
            primary_bias_physical = self.current_unscaled_biases_by_spec_name.get(
                spec.name, 0.0
            )
            final_bias_physical = primary_bias_physical

            # --- Confounding Logic for Yaw Angle ---
            if spec.measurement_type == MeasurementType.YAW_ANGLE:
                # Find the corresponding wind direction bias for this turbine.
                # Assumes a naming convention like 'turb_0/wd_current'.
                if spec.turbine_id is not None:
                    # Look for the current (non-historic) wind direction measurement for this turbine
                    wd_spec_name = f"turb_{spec.turbine_id}/wd_current"
                    wd_bias = self.current_unscaled_biases_by_spec_name.get(
                        wd_spec_name, 0.0
                    )
                else:
                    # Fallback for a farm-level yaw angle (unlikely)
                    wd_spec_name = "farm/wd_current"
                    wd_bias = self.current_unscaled_biases_by_spec_name.get(
                        wd_spec_name, 0.0
                    )

                # Sensed Yaw = True Yaw - WD_noise + Yaw_noise
                final_bias_physical = primary_bias_physical - wd_bias

            # --- Apply the final calculated bias ---
            if spec.is_circular:
                # _handle_circular_noise expects an array of physical noise values
                physical_noise_array = np.full_like(
                    observations[start:end], final_bias_physical
                )
                noisy_obs[start:end] = self._handle_circular_noise(
                    observations[start:end],
                    physical_noise_array,
                    spec,
                )
            else:
                # For non-circular, convert physical bias to a scaled delta and add it
                span = spec.max_val - spec.min_val
                if span > 0:
                    scaled_bias_delta = final_bias_physical * (2.0 / span)
                    noisy_obs[start:end] += scaled_bias_delta

        return noisy_obs

    def get_info(self) -> Dict:
        return {
            "noise_type": "episodic_bias",
            "applied_bias (physical_units)": self.current_unscaled_biases_by_spec_name,
        }


class HybridNoiseModel(NoiseModel):
    def __init__(self, models: List[NoiseModel]):
        super().__init__()  # Call parent constructor
        self.models = models
        # Sub-models will access scaling functions via the NoiseModel._static_methods
        # No need to explicitly pass them here.

    def reset_noise(self, specs: List[MeasurementSpec], rng: np.random.Generator):
        for model in self.models:
            model.reset_noise(specs, rng)

    def apply_noise(
        self,
        observations: np.ndarray,
        specs: List[MeasurementSpec],
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Initial noisy_obs starts as the clean observations
        current_noisy_observations = observations.copy()

        # Apply each component model's noise sequentially.
        # This means noise from model[0] is applied to clean_observations,
        # then noise from model[1] is applied to the output of model[0], and so on.
        for model in self.models:
            current_noisy_observations = model.apply_noise(
                current_noisy_observations, specs, rng
            )

        return current_noisy_observations

    def get_info(self) -> Dict:
        return {
            "noise_type": "hybrid",
            "component_models": [model.get_info() for model in self.models],
        }


# --- MeasurementManager and Wrapper ---


class MeasurementManager:
    """Orchestrates measurement specifications and the application of noise."""

    def __init__(self, env, seed=None):
        self.env = env
        self.noise_model: Optional[NoiseModel] = None
        self.rng = np.random.default_rng(seed)
        self.specs = self._build_from_env()

        # Set the static scaling methods on the NoiseModel base class
        # This makes them accessible to all NoiseModel instances without
        # needing to pass them explicitly to every constructor or method call.
        NoiseModel._unscale_value_static = self._unscale_value
        NoiseModel._scale_value_static = self._scale_value

    def seed(self, seed: Optional[int] = None):
        """Reseeds the random number generator for the noise model."""
        self.rng = np.random.default_rng(seed)

    # Helper methods for scaling/unscaling values. These belong to MeasurementManager.
    # Make them static methods so they can be easily passed and used by NoiseModels.
    @staticmethod
    def _unscale_value(
        scaled_value: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """Helper to convert a scaled value from [-1, 1] back to its physical unit."""
        span = max_val - min_val
        if span == 0:
            return scaled_value * 0.0
        return (scaled_value + 1) / 2 * span + min_val

    @staticmethod
    def _scale_value(
        unscaled_value: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """Helper to convert a physical value to its scaled [-1, 1] unit."""
        span = max_val - min_val
        if span == 0:
            return unscaled_value * 0.0
        return 2 * (unscaled_value - min_val) / span - 1

    def _get_physical_values_from_obs_for_logging(self, obs_vector: np.ndarray) -> dict:
        """Extracts physical values for logging from a given observation vector."""
        physical_values = {}
        for spec in self.specs:
            val_scaled = obs_vector[spec.index_range[0] : spec.index_range[1]]
            if val_scaled.size == 1:
                physical_values[spec.name] = (
                    self._unscale_value(val_scaled, spec.min_val, spec.max_val)
                ).item()
            else:
                physical_values[spec.name] = self._unscale_value(
                    val_scaled, spec.min_val, spec.max_val
                )

        return physical_values

    def set_noise_model(self, noise_model: NoiseModel):
        self.noise_model = noise_model

    def reset_noise(self):
        if self.noise_model:
            self.noise_model.reset_noise(self.specs, self.rng)

    def apply_noise(self, clean_observations: np.ndarray) -> Tuple[np.ndarray, Dict]:
        info = {}

        # --- 1. Capture TRUE (clean) physical observations ---
        true_physical_obs_values = self._get_physical_values_from_obs_for_logging(
            clean_observations
        )
        for key, val in true_physical_obs_values.items():
            info[f"obs_true/{key}"] = val

        if self.noise_model is None:
            info.update({"noise_info": {"type": "none"}})
            for key, val in true_physical_obs_values.items():
                info[f"obs_sensed/{key}"] = val
            return clean_observations.astype(np.float32), info

        # --- 2. Apply noise using the assigned noise model ---
        # The noise model now handles its own scaling/unscaling internally
        noisy_obs = self.noise_model.apply_noise(
            clean_observations, self.specs, self.rng
        )

        # --- 3. (Optional) Global clipping to [-1, 1] range after all noise and wrapping ---
        # It's good practice to ensure final observations are within the expected range.
        clipped_obs = np.clip(noisy_obs, -1.0, 1.0)

        # --- 4. Capture SENSED (noisy) physical observations from the final clipped_obs ---
        sensed_physical_obs_values = self._get_physical_values_from_obs_for_logging(
            clipped_obs
        )
        for key, val in sensed_physical_obs_values.items():
            info[f"obs_sensed/{key}"] = val

        # --- 5. Update info with noise model details ---
        info.update({"noise_info": self.noise_model.get_info()})

        return clipped_obs.astype(np.float32), info

    def _build_from_env(self) -> List[MeasurementSpec]:
        """
        Builds a list of MeasurementSpec by inspecting the environment's internal
        MesClass structure, now including scaling parameters.
        """
        specs: List[MeasurementSpec] = []
        current_idx = 0
        fm = self.env.farm_measurements

        def get_mes_names(mes_obj, prefix=""):
            names = []
            if mes_obj.current:
                names.append(f"{prefix}_current")
            if mes_obj.rolling_mean:
                for i in range(mes_obj.history_N):
                    names.append(f"{prefix}_hist_{i}")
            return names

        for i in range(fm.n_turbines):
            turb_mes_obj = fm.turb_mes[i]

            if fm.turb_ws:
                for name in get_mes_names(turb_mes_obj.ws, "ws"):
                    specs.append(
                        MeasurementSpec(
                            name=f"turb_{i}/{name}",
                            measurement_type=MeasurementType.WIND_SPEED,
                            index_range=(current_idx, current_idx + 1),
                            min_val=turb_mes_obj.ws_min,
                            max_val=turb_mes_obj.ws_max,
                            turbine_id=i,
                        )
                    )
                    current_idx += 1
            if fm.turb_wd:
                for name in get_mes_names(turb_mes_obj.wd, "wd"):
                    specs.append(
                        MeasurementSpec(
                            name=f"turb_{i}/{name}",
                            measurement_type=MeasurementType.WIND_DIRECTION,
                            index_range=(current_idx, current_idx + 1),
                            min_val=turb_mes_obj.wd_min,
                            max_val=turb_mes_obj.wd_max,
                            turbine_id=i,
                            is_circular=True,  # Add this line
                            circular_range=360.0,  # Add this line
                        )
                    )
                    current_idx += 1

            # Yaw angles
            for name in get_mes_names(turb_mes_obj.yaw, "yaw"):
                specs.append(
                    MeasurementSpec(
                        name=f"turb_{i}/{name}",
                        measurement_type=MeasurementType.YAW_ANGLE,
                        index_range=(current_idx, current_idx + 1),
                        min_val=turb_mes_obj.yaw_min,
                        max_val=turb_mes_obj.yaw_max,
                        turbine_id=i,
                    )
                )
                current_idx += 1

            # Turbulence intensity
            if fm.turb_TI:
                specs.append(
                    MeasurementSpec(
                        name=f"turb_{i}/TI",
                        measurement_type=MeasurementType.TURBULENCE_INTENSITY,
                        index_range=(current_idx, current_idx + 1),
                        min_val=turb_mes_obj.TI_min,
                        max_val=turb_mes_obj.TI_max,
                        turbine_id=i,
                    )
                )
                current_idx += 1

            # Power
            if fm.turb_power:
                for name in get_mes_names(turb_mes_obj.power, "power"):
                    specs.append(
                        MeasurementSpec(
                            name=f"turb_{i}/{name}",
                            measurement_type=MeasurementType.POWER,
                            index_range=(current_idx, current_idx + 1),
                            min_val=0,  # Min power is 0
                            max_val=turb_mes_obj.power_max,
                            turbine_id=i,
                        )
                    )
                    current_idx += 1

        # Farm-level measurements are last (if any)
        farm_mes_obj = fm.farm_mes
        if fm.farm_ws:
            for name in get_mes_names(farm_mes_obj.ws, "ws"):
                specs.append(
                    MeasurementSpec(
                        name=f"farm/{name}",
                        measurement_type=MeasurementType.WIND_SPEED,
                        index_range=(current_idx, current_idx + 1),
                        min_val=farm_mes_obj.ws_min,
                        max_val=farm_mes_obj.ws_max,
                    )
                )
                current_idx += 1

        if fm.farm_wd:  # Check if farm-level wind direction is enabled
            for name in get_mes_names(farm_mes_obj.wd, "wd"):
                specs.append(
                    MeasurementSpec(
                        name=f"farm/{name}",
                        measurement_type=MeasurementType.WIND_DIRECTION,
                        index_range=(current_idx, current_idx + 1),
                        min_val=farm_mes_obj.wd_min,
                        max_val=farm_mes_obj.wd_max,
                        is_circular=True,
                        circular_range=360.0,
                    )
                )
                current_idx += 1

        if fm.farm_power:
            for name in get_mes_names(farm_mes_obj.power, "power"):
                specs.append(
                    MeasurementSpec(
                        name=f"farm/{name}",
                        measurement_type=MeasurementType.POWER,
                        index_range=(current_idx, current_idx + 1),
                        min_val=0,  # Min power is 0
                        max_val=farm_mes_obj.power_max,
                    )
                )
                current_idx += 1

        if fm.farm_TI:
            # Note: farm_TI doesn't use get_mes_names as it's a single value, not a history object
            specs.append(
                MeasurementSpec(
                    name="farm/TI",
                    measurement_type=MeasurementType.TURBULENCE_INTENSITY,
                    index_range=(current_idx, current_idx + 1),
                    min_val=farm_mes_obj.TI_min,
                    max_val=farm_mes_obj.TI_max,
                )
            )
            current_idx += 1

        return specs


class NoisyWindFarmEnv(gym.Wrapper):
    """A Gym wrapper that applies measurement errors to a base WindFarm environment."""

    def __init__(
        self, base_env_class, measurement_manager: MeasurementManager, **env_kwargs
    ):
        self.base_env = base_env_class(**env_kwargs)
        super().__init__(self.base_env)
        self.measurement_manager = measurement_manager
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.measurement_manager.env = self.base_env

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        # Seed the measurement manager first to ensure noise is reproducible for this episode
        if seed is not None:
            self.measurement_manager.seed(seed)

        # Now, call the base environment's reset, which will also be seeded
        clean_obs, info = self.base_env.reset(seed=seed, options=options)

        # Reset the noise model (e.g., sample a new bias for the episode) using the now-seeded RNG
        self.measurement_manager.reset_noise()

        # Apply the noise to the initial observation
        noisy_obs, noise_info = self.measurement_manager.apply_noise(clean_obs)
        info.update(noise_info)
        info["clean_obs"] = clean_obs
        return noisy_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        clean_obs, reward, terminated, truncated, info = self.base_env.step(action)
        noisy_obs, noise_info = self.measurement_manager.apply_noise(clean_obs)
        info.update(noise_info)
        info["clean_obs"] = clean_obs
        return noisy_obs, reward, terminated, truncated, info

    def close(self):
        self.base_env.close()


class AdversarialNoiseModel(NoiseModel):
    def __init__(self, antagonist_agent, constraints, device):
        super().__init__()
        self.antagonist = antagonist_agent
        self.constraints = constraints
        self.device = device
        self.current_bias_state = {}

    def reset_noise(self, specs: list, rng: np.random.Generator):
        self.current_bias_state = {}

    def apply_noise(self, clean_observations, specs, rng):
        obs_tensor = torch.Tensor(clean_observations).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.antagonist.actor_mean(obs_tensor)
        antagonist_action = action.squeeze(0).cpu().numpy()
        noisy_obs = clean_observations.copy()
        action_idx = 0
        for spec in specs:
            is_ws = "ws_current" in spec.name and "turb" in spec.name
            is_wd = "wd_current" in spec.name and "turb" in spec.name
            if (is_ws or is_wd) and action_idx < len(antagonist_action):
                max_bias = (
                    self.constraints["max_bias_ws"]
                    if is_ws
                    else self.constraints["max_bias_wd"]
                )
                max_physical_change_per_step = max_bias * 0.1
                bias_change_physical_delta = (
                    antagonist_action[action_idx] * max_physical_change_per_step
                )
                current_physical_bias = self.current_bias_state.get(spec.name, 0.0)
                new_physical_bias = current_physical_bias + bias_change_physical_delta
                new_physical_bias = np.clip(new_physical_bias, -max_bias, max_bias)
                self.current_bias_state[spec.name] = new_physical_bias
                action_idx += 1
                span = spec.max_val - spec.min_val
                if span > 0:
                    scaled_bias = (new_physical_bias * 2.0) / span
                    noisy_obs[spec.index_range[0] : spec.index_range[1]] += scaled_bias
        return np.clip(noisy_obs, -1.0, 1.0)

    def get_info(self):
        return {
            "noise_type": "adversarial (stateful)",
            "applied_bias (physical_units)": self.current_bias_state.copy(),
        }
