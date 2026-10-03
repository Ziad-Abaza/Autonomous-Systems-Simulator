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
    motion_gate_ms: float = 5.0            # Speed (m/s) at which alignment shaping reaches full credit; 0 disables the gate (legacy semantics)

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

    def validate(self) -> bool:
        """
        Validates reward parameters to prevent silent contract breaks.
        Returns True if valid, raises ValueError with explanation otherwise.
        """
        if self.weight_progress < 0.0:
            raise ValueError(f"weight_progress cannot be negative: {self.weight_progress}")
        if self.weight_centering < 0.0:
            raise ValueError(f"weight_centering cannot be negative: {self.weight_centering}")
        if self.weight_speed < 0.0:
            raise ValueError(f"weight_speed cannot be negative: {self.weight_speed}")
        if self.target_speed <= 0.0:
            raise ValueError(f"target_speed must be positive: {self.target_speed}")
        if self.weight_heading < 0.0:
            raise ValueError(f"weight_heading cannot be negative: {self.weight_heading}")
        if self.weight_action_smoothness < 0.0:
            raise ValueError(f"weight_action_smoothness cannot be negative: {self.weight_action_smoothness}")
        if self.checkpoint_bonus < 0.0:
            raise ValueError(f"checkpoint_bonus cannot be negative: {self.checkpoint_bonus}")
        if self.lap_completion_bonus < 0.0:
            raise ValueError(f"lap_completion_bonus cannot be negative: {self.lap_completion_bonus}")
        if self.collision_penalty < 0.0:
            raise ValueError(f"collision_penalty cannot be negative: {self.collision_penalty}")
        if self.off_road_penalty < 0.0:
            raise ValueError(f"off_road_penalty cannot be negative: {self.off_road_penalty}")
        if self.backward_penalty < 0.0:
            raise ValueError(f"backward_penalty cannot be negative: {self.backward_penalty}")
        return True


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

        # Sanity check: Discard discontinuous jumps / teleportation (max realistic 1-step delta at 60Hz is ~1.5m)
        if abs(delta_s) > 5.0:
            delta_s = 0.0

        self.prev_s = current_s

        # Forward progress reward
        r_progress = delta_s * cfg.weight_progress

        # Backward driving penalty
        r_backward = 0.0
        if delta_s < -0.05 or abs(heading_error) > (math.pi * 0.6):
            r_backward = -cfg.backward_penalty

        # Motion gate: alignment shaping only applies while actually moving —
        # a parked vehicle is not "keeping" the lane (stationary-policy exploit).
        motion = 1.0 if cfg.motion_gate_ms <= 0.0 else min(1.0, max(0.0, speed / cfg.motion_gate_ms))

        # 2. Centering reward: 1.0 at center, falling to 0.0 at edge
        half_w = max(1.0, road_width * 0.5)
        norm_lat = min(1.0, abs(lateral_offset) / half_w)
        r_centering = (1.0 - norm_lat) * cfg.weight_centering * motion

        # 3. Speed reward: reward matching target speed smoothly
        norm_speed = min(1.0, max(0.0, speed / max(1.0, cfg.target_speed)))
        r_speed = norm_speed * cfg.weight_speed

        # 4. Heading alignment: cos(heading_error) is +1 when aligned, -1 when opposite
        r_heading = math.cos(heading_error) * cfg.weight_heading * motion

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
