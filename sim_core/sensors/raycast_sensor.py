"""
Multi-beam Raycast LiDAR / Rangefinder sensor.
Casts radial distance rays against track barriers, walls, and obstacles.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import numpy as np
from sim_core.math_utils import Vec2, Vec3
from sim_core.sensors.base_sensor import BaseSensor


class RaycastSensor(BaseSensor):
    """
    Simulates a multi-beam forward or 360-degree LiDAR rangefinder.
    """
    def __init__(
        self,
        name: str = "lidar_rays",
        num_rays: int = 15,
        fov_degrees: float = 180.0,
        max_range: float = 40.0,
        update_frequency_hz: float = 30.0,
        local_pos: Optional[Vec3] = None,
        local_yaw: float = 0.0,
        noise_std: float = 0.0,
        latency_seconds: float = 0.0
    ):
        super().__init__(
            name=name,
            sensor_type="raycast_lidar",
            update_frequency_hz=update_frequency_hz,
            local_pos=local_pos or Vec3(1.5, 0.0, 0.5),  # Front bumper
            local_yaw=local_yaw,
            noise_std=noise_std,
            latency_seconds=latency_seconds
        )
        self.num_rays = max(1, int(num_rays))
        self.fov_degrees = float(fov_degrees)
        self.max_range = float(max_range)
        self.ray_angles = self._compute_angles()

        # Cached visualization data: list of (ray_start_2d, ray_end_or_hit_2d, fraction)
        self.ray_visuals: List[Dict[str, Any]] = []

    def _compute_angles(self) -> List[float]:
        """Angles in radians relative to sensor forward direction."""
        fov_rad = math.radians(self.fov_degrees)
        if self.num_rays == 1:
            return [0.0]
        half_fov = fov_rad * 0.5
        angles = []
        for i in range(self.num_rays):
            a = -half_fov + (i / (self.num_rays - 1)) * fov_rad
            angles.append(a)
        return angles

    def _generate_raw_sample(self, context: Any, rng: np.random.Generator) -> Dict[str, Any]:
        vehicle = context['vehicle']
        track_queries = context['track_queries']
        st = vehicle.state

        # Compute sensor world position and base yaw
        cos_y = math.cos(st.yaw)
        sin_y = math.sin(st.yaw)
        s_world_x = st.pos.x + cos_y * self.local_pos.x - sin_y * self.local_pos.y
        s_world_y = st.pos.y + sin_y * self.local_pos.x + cos_y * self.local_pos.y
        sensor_origin = Vec2(s_world_x, s_world_y)
        sensor_yaw = st.yaw + self.local_yaw

        # Get additional obstacle segments if any
        obstacle_segs = context.get('obstacle_segments', None)

        distances = []
        normalized_distances = []
        visuals = []

        for angle in self.ray_angles:
            ray_yaw = sensor_yaw + angle
            ray_dir = Vec2(math.cos(ray_yaw), math.sin(ray_yaw))

            dist, hit_pt = track_queries.cast_ray(
                ray_origin=sensor_origin,
                ray_dir=ray_dir,
                max_range=self.max_range,
                additional_segments=obstacle_segs
            )

            distances.append(dist)
            normalized_distances.append(dist / self.max_range)

            end_pt = hit_pt if hit_pt is not None else (sensor_origin + ray_dir * self.max_range)
            visuals.append({
                'origin': sensor_origin,
                'end': end_pt,
                'hit': hit_pt is not None,
                'distance': dist,
                'fraction': dist / self.max_range,
                'angle_rad': angle
            })

        self.ray_visuals = visuals

        return {
            'distances': np.array(distances, dtype=np.float32),
            'ranges_norm': np.array(normalized_distances, dtype=np.float32),
            'num_rays': self.num_rays,
            'max_range': self.max_range,
        }

    def _apply_noise(self, data: Any, rng: np.random.Generator) -> Any:
        if self.noise_std <= 0.0 or not isinstance(data, dict):
            return data
        noisy = dict(data)
        dists = data['distances']
        noise = rng.normal(0.0, self.noise_std, size=dists.shape)
        noisy_dists = np.clip(dists + noise, 0.0, self.max_range).astype(np.float32)
        noisy['distances'] = noisy_dists
        noisy['ranges_norm'] = (noisy_dists / self.max_range).astype(np.float32)
        return noisy

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'num_rays': self.num_rays,
            'fov_degrees': self.fov_degrees,
            'max_range': self.max_range,
        })
        return d
