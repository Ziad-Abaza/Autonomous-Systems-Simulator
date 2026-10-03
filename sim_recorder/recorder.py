"""
Episode recording system capturing high-fidelity trajectories, actions,
reward breakdowns, and collision events for research reproducibility and replay.
"""

from __future__ import annotations
import json
import gzip
import time
from typing import Dict, Any, List, Optional
import numpy as np


class EpisodeRecorder:
    """
    Captures step-by-step episode telemetry with memory limits and compressed export.
    """
    def __init__(self, max_steps: int = 5000):
        self.max_steps = max_steps
        self.is_recording = False
        self.metadata: Dict[str, Any] = {}
        self.frames: List[Dict[str, Any]] = []

    def start_recording(
        self,
        track_name: str,
        seed: int,
        env_config: Optional[Dict[str, Any]] = None,
        env_version: str = "1.0.0",
        scenario_name: str = "Standard",
        observation_schema: Optional[Dict[str, Any]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        fingerprint: str = ""
    ) -> None:
        self.is_recording = True
        self.frames = []
        self.metadata = {
            'timestamp': time.time(),
            'track_name': track_name,
            'seed': seed,
            'env_version': env_version,
            'scenario_name': scenario_name,
            'env_fingerprint': fingerprint,
            'env_config': env_config or {},
            'observation_schema': observation_schema or {},
            'action_schema': action_schema or {},
            'reward_config': reward_config or {},
            'total_steps': 0,
            'termination_reason': 'in_progress',
            'episode_result': {}
        }

    def record_step(
        self,
        step: int,
        sim_time: float,
        vehicle_pos: tuple[float, float, float],
        vehicle_yaw: float,
        speed: float,
        action: list[float],
        reward: float,
        reward_breakdown: Dict[str, float],
        telemetry: Dict[str, Any],
        is_colliding: bool = False
    ) -> None:
        if not self.is_recording:
            return

        if len(self.frames) >= self.max_steps:
            # Memory safeguard
            self.frames.pop(0)

        frame = {
            'step': step,
            't': round(sim_time, 4),
            'pos': [round(p, 3) for p in vehicle_pos],
            'yaw': round(vehicle_yaw, 4),
            'speed': round(speed, 2),
            'action': [round(a, 3) for a in action],
            'reward': round(reward, 4),
            'breakdown': {k: round(v, 4) for k, v in reward_breakdown.items()},
            'lat_offset': round(telemetry.get('lateral_offset', 0.0), 3),
            'heading_err': round(telemetry.get('heading_error', 0.0), 4),
            'collision': is_colliding,
        }
        self.frames.append(frame)

    def stop_recording(self, termination_reason: str = "completed", episode_result: Optional[Dict[str, Any]] = None) -> None:
        self.is_recording = False
        self.metadata['total_steps'] = len(self.frames)
        self.metadata['termination_reason'] = termination_reason
        if episode_result:
            self.metadata['episode_result'] = episode_result

    def save_to_file(self, filepath: str) -> None:
        """Saves recorded episode to JSON or gzipped JSON file."""
        data = {
            'metadata': self.metadata,
            'frames': self.frames
        }
        json_str = json.dumps(data, indent=2)
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)

    @classmethod
    def load_from_file(cls, filepath: str) -> Dict[str, Any]:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
