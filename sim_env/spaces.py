"""
Action and Observation Space definitions and schemas.
Supports continuous, discrete, and dictionary/vector observation configurations.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np


class ActionSpaceType:
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class ActionSpaceConfig:
    type: str = ActionSpaceType.CONTINUOUS
    # Continuous limits: [steer, throttle, brake]
    continuous_low: List[float] = field(default_factory=lambda: [-1.0, 0.0, 0.0])
    continuous_high: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Discrete options: 0=Coast, 1=Forward, 2=Brake, 3=Left, 4=Right
    discrete_actions: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],   # 0: Coast
        [0.0, 0.7, 0.0],   # 1: Accelerate
        [0.0, 0.0, 0.8],   # 2: Brake
        [-0.6, 0.4, 0.0],  # 3: Steer Left + Throttle
        [0.6, 0.4, 0.0],   # 4: Steer Right + Throttle
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionSpaceConfig:
        return cls(
            type=str(data.get('type', ActionSpaceType.CONTINUOUS)),
            continuous_low=list(data.get('continuous_low', [-1.0, 0.0, 0.0])),
            continuous_high=list(data.get('continuous_high', [1.0, 1.0, 1.0])),
            discrete_actions=list(data.get('discrete_actions', [
                [0.0, 0.0, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.0, 0.8],
                [-0.6, 0.4, 0.0],
                [0.6, 0.4, 0.0],
            ])),
        )

    def validate_action(self, raw_action: Any) -> Tuple[bool, str]:
        """
        Validates whether raw_action is structurally and numerically valid.
        Returns: (is_valid: bool, error_message: str)
        """
        if raw_action is None:
            return False, "Action cannot be None"

        if self.type == ActionSpaceType.DISCRETE:
            try:
                if isinstance(raw_action, (np.ndarray, list, tuple)):
                    if len(raw_action) == 0:
                        return False, "Empty action container"
                    val = raw_action[0]
                else:
                    val = raw_action
                
                # Check for NaN / Inf
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return False, f"Discrete action cannot be NaN or Infinity: {val}"
                
                idx = int(val)
                if idx < 0 or idx >= len(self.discrete_actions):
                    return False, f"Discrete action index {idx} out of range [0, {len(self.discrete_actions)-1}]"
                return True, ""
            except (ValueError, TypeError) as e:
                return False, f"Invalid discrete action value: {e}"
        else:
            try:
                act = np.asarray(raw_action, dtype=np.float32).flatten()
                if len(act) == 0:
                    return False, "Action array is empty"
                if np.any(np.isnan(act)):
                    return False, "Action contains NaN values"
                if np.any(np.isinf(act)):
                    return False, "Action contains Infinity values"
                if len(act) < 3:
                    return False, f"Continuous action expected at least 3 elements [steer, throttle, brake], got {len(act)}"
                return True, ""
            except Exception as e:
                return False, f"Failed to parse continuous action: {e}"

    def decode_action(self, raw_action: Union[np.ndarray, List[float], int, float, Any]) -> Tuple[float, float, float]:
        """
        Translates raw agent action into (steer, throttle, brake) floats.
        Guarantees finite, clamped values within configured bounds, never NaN or Inf.
        """
        if self.type == ActionSpaceType.DISCRETE:
            idx = 0
            try:
                if isinstance(raw_action, (np.ndarray, list, tuple)):
                    if len(raw_action) > 0:
                        val = raw_action[0]
                    else:
                        val = 0
                else:
                    val = raw_action
                
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        val = 0
                idx = int(val)
            except (ValueError, TypeError, OverflowError):
                idx = 0

            idx = max(0, min(idx, len(self.discrete_actions) - 1))
            act = self.discrete_actions[idx]
            return float(act[0]), float(act[1]), float(act[2])
        else:
            # Continuous: [steer, throttle, brake]
            try:
                act = np.asarray(raw_action, dtype=np.float32).flatten()
            except Exception:
                act = np.zeros(3, dtype=np.float32)

            # Sanitize NaNs and Infs: NaN -> 0.0, Inf -> clamped
            act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)

            steer = float(np.clip(act[0], self.continuous_low[0], self.continuous_high[0])) if len(act) > 0 else 0.0
            throttle = float(np.clip(act[1], self.continuous_low[1], self.continuous_high[1])) if len(act) > 1 else 0.0
            brake = float(np.clip(act[2], self.continuous_low[2], self.continuous_high[2])) if len(act) > 2 else 0.0
            return steer, throttle, brake


@dataclass
class ObservationSchema:
    """
    User-configurable schema specifying which telemetry and sensor features
    are exposed to the AI model.
    """
    include_speed: bool = True
    include_velocity: bool = True
    include_yaw_rate: bool = True
    include_steering_angle: bool = True
    include_distance_from_center: bool = True
    include_heading_error: bool = True
    include_distance_to_checkpoint: bool = True
    include_lidar_rays: bool = True
    include_camera_rgb: bool = False  # Set to True for vision-based RL models
    flatten_vector: bool = True       # If True, vector features are flattened into 1D Box

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationSchema:
        schema = cls()
        for k, v in data.items():
            if hasattr(schema, k):
                setattr(schema, k, bool(v))
        return schema

    def compute_vector_dim(self, num_lidar_rays: int = 15) -> int:
        """Computes total dimension of flattened vector observation."""
        dim = 0
        if self.include_speed:
            dim += 1
        if self.include_velocity:
            dim += 2  # vx, vy
        if self.include_yaw_rate:
            dim += 1
        if self.include_steering_angle:
            dim += 1
        if self.include_distance_from_center:
            dim += 1
        if self.include_heading_error:
            dim += 1
        if self.include_distance_to_checkpoint:
            dim += 1
        if self.include_lidar_rays:
            dim += num_lidar_rays
        return dim
