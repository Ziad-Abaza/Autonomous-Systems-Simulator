"""
Vehicle kinematics and telemetry state sensor.
Provides ground-truth or noisy measurements of vehicle velocities,
heading error, lateral distance from center, and track progress.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from sim_core.math_utils import Vec2, Vec3
from sim_core.sensors.base_sensor import BaseSensor


class VehicleStateSensor(BaseSensor):
    """
    Measures vehicle kinematics and track relationship.
    """
    def __init__(
        self,
        name: str = "vehicle_state",
        update_frequency_hz: float = 60.0,
        noise_std: float = 0.0,
        latency_seconds: float = 0.0
    ):
        super().__init__(
            name=name,
            sensor_type="vehicle_state",
            update_frequency_hz=update_frequency_hz,
            noise_std=noise_std,
            latency_seconds=latency_seconds
        )

    def _generate_raw_sample(self, context: Any, rng: np.random.Generator) -> Dict[str, float]:
        """
        Extracts telemetry from context (vehicle, track_queries, track).
        """
        vehicle = context['vehicle']
        track_queries = context['track_queries']
        st = vehicle.state

        pos_2d = Vec2(st.pos.x, st.pos.y)
        track_info = track_queries.query_vehicle_pose(pos_2d, st.yaw)

        # Distance to next checkpoint
        checkpoints = track_queries.track.checkpoints
        curr_cp_idx = context.get('checkpoint_idx', 0)
        dist_to_cp = 0.0
        if checkpoints:
            target_cp = checkpoints[curr_cp_idx % len(checkpoints)]
            cp_pos = Vec2(target_cp['pos'].x, target_cp['pos'].y)
            dist_to_cp = (cp_pos - pos_2d).length()

        sample = {
            'speed': float(st.speed),
            'vel_x': float(st.vel_body.x),
            'vel_y': float(st.vel_body.y),
            'yaw': float(st.yaw),
            'yaw_rate': float(st.yaw_rate),
            'accel_x': float(st.accel_body.x),
            'accel_y': float(st.accel_body.y),
            'steering_angle': float(st.steering_angle),
            'throttle': float(st.throttle),
            'brake': float(st.brake),
            'distance_from_center': float(track_info['lateral_offset']),
            'heading_error': float(track_info['heading_error']),
            'is_on_road': 1.0 if track_info['is_on_road'] else 0.0,
            'distance_to_checkpoint': float(dist_to_cp),
            'track_progress_s': float(track_info['s']),
            'road_width': float(track_info['road_width']),
        }
        return sample

    def _apply_noise(self, data: Any, rng: np.random.Generator) -> Any:
        if self.noise_std <= 0.0 or not isinstance(data, dict):
            return data
        noisy = {}
        for k, v in data.items():
            if k in ('is_on_road',):
                noisy[k] = v
            else:
                noisy[k] = v + float(rng.normal(0.0, self.noise_std))
        return noisy
