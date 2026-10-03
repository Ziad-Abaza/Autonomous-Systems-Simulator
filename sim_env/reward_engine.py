"""
Modular composable Reward Engine with full component decomposition.
Supports positive progress bonuses, centering penalties, speed rewards,
heading alignment, checkpoint rewards, and collision penalties.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple


@dataclass
class RewardConfig:
    # Component weights
    weight_progress: float = 1.0           # Reward per meter of forward progress along centerline
    weight_centering: float = 0.5          # Reward for staying near track centerline [0, 1]
    weight_speed: float = 0.2              # Reward for maintaining target speed
    target_speed: float = 20.0             # m/s target
    weight_heading: float = 0.3            # Reward for aligning with track direction (cos(heading_error))
    weight_action_smoothness: float = 0.05 # Penalty for rapid steering changes

    # Event rewards & penalties
    checkpoint_bonus: float = 10.0         # Bonus on passing each checkpoint
    lap_completion_bonus: float = 100.0    # Bonus on completing a full lap
    collision_penalty: float = 50.0        # Penalty on hitting barrier/obstacle
    off_road_penalty: float = 25.0         # Penalty on leaving track surface
    backward_penalty: float = 1.0          # Penalty per step for driving backward

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RewardConfig:
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, float(v))
        return cfg


class RewardEngine:
    """
    Computes composable step rewards and decomposes the total into individual terms
    for transparent inspection and debugging.
    """
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.prev_s: float = 0.0
        self.prev_steer: float = 0.0
        self.total_accumulated_reward: float = 0.0
        self.last_breakdown: Dict[str, float] = {}

    def reset(self, initial_s: float = 0.0) -> None:
        self.prev_s = initial_s
        self.prev_steer = 0.0
        self.total_accumulated_reward = 0.0
        self.last_breakdown = {
            'progress': 0.0,
            'centering': 0.0,
            'speed': 0.0,
            'heading': 0.0,
            'smoothness': 0.0,
            'checkpoint': 0.0,
            'lap': 0.0,
            'collision': 0.0,
            'off_road': 0.0,
            'backward': 0.0,
            'total': 0.0,
        }

    def compute_step_reward(
        self,
        current_s: float,
        track_length: float,
        is_closed: bool,
        lateral_offset: float,
        road_width: float,
        speed: float,
        heading_error: float,
        current_steer: float,
        is_colliding: bool,
        is_on_road: bool,
        checkpoint_passed: bool,
        lap_completed: bool
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config

        # 1. Progress along track
        delta_s = current_s - self.prev_s
        # Handle wrap-around on closed tracks
        if is_closed and track_length > 0:
            if delta_s < -track_length * 0.5:
                delta_s += track_length
            elif delta_s > track_length * 0.5:
                delta_s -= track_length

        self.prev_s = current_s

        # Forward progress reward
        r_progress = delta_s * cfg.weight_progress

        # Backward driving penalty
        r_backward = 0.0
        if delta_s < -0.05 or abs(heading_error) > (math.pi * 0.6):
            r_backward = -cfg.backward_penalty

        # 2. Centering reward: 1.0 at center, falling to 0.0 at edge
        half_w = max(1.0, road_width * 0.5)
        norm_lat = min(1.0, abs(lateral_offset) / half_w)
        r_centering = (1.0 - norm_lat) * cfg.weight_centering

        # 3. Speed reward: reward matching target speed smoothly
        norm_speed = min(1.0, max(0.0, speed / max(1.0, cfg.target_speed)))
        r_speed = norm_speed * cfg.weight_speed

        # 4. Heading alignment: cos(heading_error) is +1 when aligned, -1 when opposite
        r_heading = math.cos(heading_error) * cfg.weight_heading

        # 5. Action smoothness penalty: penalizes twitchy steering
        delta_steer = current_steer - self.prev_steer
        self.prev_steer = current_steer
        r_smoothness = - (delta_steer * delta_steer) * cfg.weight_action_smoothness

        # 6. Event rewards
        r_checkpoint = cfg.checkpoint_bonus if checkpoint_passed else 0.0
        r_lap = cfg.lap_completion_bonus if lap_completed else 0.0
        r_collision = -cfg.collision_penalty if is_colliding else 0.0
        r_off_road = -cfg.off_road_penalty if not is_on_road else 0.0

        total = (
            r_progress +
            r_centering +
            r_speed +
            r_heading +
            r_smoothness +
            r_checkpoint +
            r_lap +
            r_collision +
            r_off_road +
            r_backward
        )

        breakdown = {
            'progress': float(r_progress),
            'centering': float(r_centering),
            'speed': float(r_speed),
            'heading': float(r_heading),
            'smoothness': float(r_smoothness),
            'checkpoint': float(r_checkpoint),
            'lap': float(r_lap),
            'collision': float(r_collision),
            'off_road': float(r_off_road),
            'backward': float(r_backward),
            'total': float(total),
        }

        self.last_breakdown = breakdown
        self.total_accumulated_reward += total
        return total, breakdown
