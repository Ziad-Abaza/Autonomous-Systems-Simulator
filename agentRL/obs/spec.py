"""ObservationSpec — declarative policy-input layout.

The spec names a subset/order of the environment's AGENT_OBSERVATION
channels (the env is built so its flat vector is already exactly this
layout) plus client-side temporal features (frame stacking) and
prev_action. Track identity never enters the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# name -> flat width in the default env observation vector layout
DEFAULT_CHANNELS: dict[str, int] = {
    "speed": 1,
    "velocity_body": 2,
    "yaw_rate": 1,
    "steering_angle": 1,
    "distance_from_center": 1,
    "heading_error": 1,
    "distance_to_checkpoint": 1,
    "lidar_ranges": 15,
}

_STATE_CHANNELS = (
    "speed", "velocity_body", "yaw_rate", "steering_angle",
    "distance_from_center", "heading_error", "distance_to_checkpoint",
)

PRESETS: dict[str, tuple[str, ...]] = {
    "state8": _STATE_CHANNELS,
    "full23": _STATE_CHANNELS + ("lidar_ranges",),
    "lidar15": ("lidar_ranges",),
}

ACTION_DIM = 3  # [steer, throttle, brake]


@dataclass(frozen=True)
class ObservationSpec:
    channel_names: tuple[str, ...]
    frame_stack: int = 1
    prev_action: bool = False
    image: bool = False  # reserved — encoder raises until CNN support lands

    def __post_init__(self) -> None:
        if not self.channel_names:
            raise ValueError("channel_names must be non-empty")
        unknown = [c for c in self.channel_names if c not in DEFAULT_CHANNELS]
        if unknown:
            raise ValueError(
                f"unknown observation channels {unknown}; "
                f"known: {sorted(DEFAULT_CHANNELS)}"
            )
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")

    @property
    def vector_dim(self) -> int:
        return sum(DEFAULT_CHANNELS[c] for c in self.channel_names)

    @property
    def input_dim(self) -> int:
        return self.vector_dim * self.frame_stack + (
            ACTION_DIM if self.prev_action else 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "frame_stack": self.frame_stack,
            "prev_action": self.prev_action,
            "image": self.image,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSpec":
        return cls(
            channel_names=tuple(data["channel_names"]),
            frame_stack=int(data.get("frame_stack", 1)),
            prev_action=bool(data.get("prev_action", False)),
            image=bool(data.get("image", False)),
        )
