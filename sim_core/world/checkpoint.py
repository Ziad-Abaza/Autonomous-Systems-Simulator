"""
Checkpoint tracking and lap progress manager.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from sim_core.math_utils import Vec2, Vec3
from sim_core.track.mesh_generator import GeneratedTrack


class CheckpointTracker:
    """
    Tracks vehicle progress across sequential checkpoints along the track.
    """
    def __init__(self, track: GeneratedTrack):
        self.track = track
        self.current_index: int = 0
        self.laps_completed: int = 0
        self.total_checkpoints_passed: int = 0
        self.last_cross_time: float = 0.0
        self.lap_start_time: float = 0.0
        self.last_lap_time: float = 0.0
        self.best_lap_time: float = float('inf')

    def reset(self, start_time: float = 0.0) -> None:
        self.current_index = 0
        self.laps_completed = 0
        self.total_checkpoints_passed = 0
        self.last_cross_time = start_time
        self.lap_start_time = start_time
        self.last_lap_time = 0.0

    def update(self, prev_pos: Vec2, curr_pos: Vec2, current_time: float) -> Tuple[bool, bool]:
        """
        Updates checkpoint progression.
        Returns: (checkpoint_passed: bool, lap_completed: bool)
        """
        if not self.track.checkpoints:
            return False, False

        num_cp = len(self.track.checkpoints)
        target_cp = self.track.checkpoints[self.current_index % num_cp]

        gate_left = target_cp['gate_left']
        gate_right = target_cp['gate_right']

        from sim_core.math_utils import segments_intersect
        if segments_intersect(prev_pos, curr_pos, gate_left, gate_right):
            # Passed target checkpoint!
            self.total_checkpoints_passed += 1
            self.last_cross_time = current_time
            self.current_index += 1

            lap_completed = False
            if self.current_index >= num_cp:
                self.current_index = 0
                self.laps_completed += 1
                lap_completed = True
                self.last_lap_time = current_time - self.lap_start_time
                self.lap_start_time = current_time
                if self.last_lap_time < self.best_lap_time:
                    self.best_lap_time = self.last_lap_time

            return True, lap_completed

        return False, False

    def get_progress_fraction(self) -> float:
        """Fraction of current lap completed [0.0, 1.0]."""
        num_cp = len(self.track.checkpoints)
        if num_cp == 0:
            return 0.0
        return (self.current_index % num_cp) / float(num_cp)
