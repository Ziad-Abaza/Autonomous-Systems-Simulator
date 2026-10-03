"""
Termination and truncation evaluator with explicit cause attribution.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple


@dataclass
class TerminationConfig:
    terminate_on_collision: bool = True
    terminate_on_off_road: bool = True
    terminate_on_wrong_direction: bool = True
    wrong_direction_max_angle_deg: float = 120.0
    terminate_on_lap_completion: bool = False
    laps_to_complete: int = 1

    # Truncation limits
    max_episode_steps: int = 1500
    max_seconds_without_checkpoint: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TerminationConfig:
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                val_type = type(getattr(cfg, k))
                setattr(cfg, k, val_type(v))
        return cfg


class TerminationEngine:
    """
    Evaluates termination and truncation conditions per environment step.
    """
    def __init__(self, config: TerminationConfig | None = None):
        self.config = config or TerminationConfig()
        self.episode_steps: int = 0
        self.time_since_last_checkpoint: float = 0.0

    def reset(self) -> None:
        self.episode_steps = 0
        self.time_since_last_checkpoint = 0.0

    def evaluate(
        self,
        dt: float,
        is_colliding: bool,
        is_on_road: bool,
        heading_error: float,
        checkpoint_passed: bool,
        laps_completed: int
    ) -> Tuple[bool, bool, str]:
        """
        Evaluates conditions.
        Returns: (terminated, truncated, reason)
        """
        cfg = self.config
        self.episode_steps += 1

        if checkpoint_passed:
            self.time_since_last_checkpoint = 0.0
        else:
            self.time_since_last_checkpoint += dt

        # 1. Collision termination
        if cfg.terminate_on_collision and is_colliding:
            return True, False, "collision"

        # 2. Off-road termination
        if cfg.terminate_on_off_road and not is_on_road:
            return True, False, "off_road"

        # 3. Wrong direction termination
        max_angle_rad = math.radians(cfg.wrong_direction_max_angle_deg)
        if cfg.terminate_on_wrong_direction and abs(heading_error) > max_angle_rad:
            return True, False, "wrong_direction"

        # 4. Lap completion termination
        if cfg.terminate_on_lap_completion and laps_completed >= cfg.laps_to_complete:
            return True, False, "lap_completed"

        # 5. Checkpoint timeout truncation (stuck vehicle)
        if self.time_since_last_checkpoint > cfg.max_seconds_without_checkpoint:
            return False, True, "checkpoint_timeout"

        # 6. Max steps truncation
        if self.episode_steps >= cfg.max_episode_steps:
            return False, True, "max_steps_exceeded"

        return False, False, "running"
