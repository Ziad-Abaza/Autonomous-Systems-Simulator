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

    def decode_action(self, raw_action: Union[np.ndarray, List[float], int, float]) -> Tuple[float, float, float]:
        """
        Translates raw agent action into (steer, throttle, brake) floats.
        """
        if self.type == ActionSpaceType.DISCRETE:
            idx = int(raw_action)
            idx = max(0, min(idx, len(self.discrete_actions) - 1))
            act = self.discrete_actions[idx]
            return float(act[0]), float(act[1]), float(act[2])
        else:
            # Continuous: [steer, throttle, brake]
            act = np.asarray(raw_action, dtype=np.float32).flatten()
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
