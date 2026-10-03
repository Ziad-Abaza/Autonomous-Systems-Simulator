"""
Episode Lifecycle Configuration.
Encapsulates episode limits, deterministic spawn parameters, initial speeds,
random seeds, and reset behaviors.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple


class SpawnMode:
    TRACK_SPAWN = "track_spawn"          # Use track's defined spawn point
    RANDOM_CHECKPOINT = "random_cp"      # Spawn randomly at any checkpoint along track
    CUSTOM_POSE = "custom_pose"          # Use custom fixed x, y, z, yaw


class ResetBehavior:
    HARD_RESET = "hard"                  # Reinitialize full clock, physics, and state
    SOFT_RESET = "soft"                  # Keep accumulated clock, reposition entity only


@dataclass
class EpisodeConfiguration:
    """
    User-configurable episode lifecycle settings.
    """
    max_duration_seconds: float = 60.0
    max_steps: int = 3600
    spawn_mode: str = SpawnMode.TRACK_SPAWN
    initial_speed: float = 0.0           # m/s
    custom_spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    custom_spawn_yaw_deg: float = 0.0
    spawn_lateral_jitter_m: float = 0.0
    spawn_heading_jitter_deg: float = 0.0
    random_seed: int = 42
    reset_behavior: str = ResetBehavior.HARD_RESET
    auto_reset_on_done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_duration_seconds": self.max_duration_seconds,
            "max_steps": self.max_steps,
            "spawn_mode": self.spawn_mode,
            "initial_speed": self.initial_speed,
            "custom_spawn_pos": list(self.custom_spawn_pos),
            "custom_spawn_yaw_deg": self.custom_spawn_yaw_deg,
            "spawn_lateral_jitter_m": self.spawn_lateral_jitter_m,
            "spawn_heading_jitter_deg": self.spawn_heading_jitter_deg,
            "random_seed": self.random_seed,
            "reset_behavior": self.reset_behavior,
            "auto_reset_on_done": self.auto_reset_on_done,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeConfiguration:
        cfg = cls()
        cfg.max_duration_seconds = float(data.get("max_duration_seconds", 60.0))
        cfg.max_steps = int(data.get("max_steps", 3600))
        cfg.spawn_mode = str(data.get("spawn_mode", SpawnMode.TRACK_SPAWN))
        cfg.initial_speed = float(data.get("initial_speed", 0.0))
        if "custom_spawn_pos" in data:
            cfg.custom_spawn_pos = tuple(data["custom_spawn_pos"])
        cfg.custom_spawn_yaw_deg = float(data.get("custom_spawn_yaw_deg", 0.0))
        cfg.spawn_lateral_jitter_m = float(data.get("spawn_lateral_jitter_m", 0.0))
        cfg.spawn_heading_jitter_deg = float(data.get("spawn_heading_jitter_deg", 0.0))
        cfg.random_seed = int(data.get("random_seed", 42))
        cfg.reset_behavior = str(data.get("reset_behavior", ResetBehavior.HARD_RESET))
        cfg.auto_reset_on_done = bool(data.get("auto_reset_on_done", False))
        return cfg
