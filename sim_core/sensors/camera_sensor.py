"""
Synthetic RGB Camera sensor simulating front/hood/roof-mounted visual perception.
Produces RGB pixel arrays for visual RL or computer vision perception tasks.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from sim_core.math_utils import Vec2, Vec3
from sim_core.sensors.base_sensor import BaseSensor


class CameraSensor(BaseSensor):
    """
    Simulates an on-board synthetic RGB camera.
    """
    def __init__(
        self,
        name: str = "rgb_camera",
        width: int = 84,
        height: int = 84,
        fov_degrees: float = 75.0,
        update_frequency_hz: float = 30.0,
        local_pos: Optional[Vec3] = None,
        local_yaw: float = 0.0,
        local_pitch: float = -0.05,  # Slightly angled down towards road
        noise_std: float = 0.0,
        latency_seconds: float = 0.0
    ):
        super().__init__(
            name=name,
            sensor_type="camera_rgb",
            update_frequency_hz=update_frequency_hz,
            local_pos=local_pos or Vec3(1.0, 0.0, 1.1),  # Vehicle hood/roof
            local_yaw=local_yaw,
            noise_std=noise_std,
            latency_seconds=latency_seconds
        )
        self.width = int(width)
        self.height = int(height)
        self.fov_degrees = float(fov_degrees)
        self.local_pitch = float(local_pitch)

        # Offscreen renderer reference if set by rendering engine
        self._offscreen_renderer = None
        self._last_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def set_offscreen_renderer(self, renderer_callable: Any) -> None:
        self._offscreen_renderer = renderer_callable

    def _generate_raw_sample(self, context: Any, rng: np.random.Generator) -> np.ndarray:
        """
        Renders or synthesizes the forward camera view.
        If an OpenGL offscreen renderer is provided, captures the real 3D frame.
        Otherwise, uses procedural perspective road rasterizer (works in headless tests).
        """
        if self._offscreen_renderer is not None:
            try:
                img = self._offscreen_renderer(self)
                if img is not None and img.shape == (self.height, self.width, 3):
                    self._last_image = img
                    return img
            except Exception:
                pass

        # Fast procedural perspective synthesis of road and boundaries
        img = self._synthesize_perspective_road(context)
        self._last_image = img
        return img

    def _synthesize_perspective_road(self, context: Any) -> np.ndarray:
        """
        Synthesizes a clean 3D perspective projection of the road, lane markings,
        and sky/grass into the image buffer.
        """
        w, h = self.width, self.height
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky background (gradient blue)
        horizon_y = int(h * 0.45)
        img[:horizon_y, :] = [135, 206, 235]  # Sky blue

        # Ground grass (green)
        img[horizon_y:, :] = [34, 139, 34]    # Forest green

        vehicle = context['vehicle']
        track_queries = context['track_queries']
        st = vehicle.state

        # Query vehicle pose on track
        pos_2d = Vec2(st.pos.x, st.pos.y)
        track_info = track_queries.query_vehicle_pose(pos_2d, st.yaw)

        lat_offset = track_info['lateral_offset']  # positive left, negative right
        heading_err = track_info['heading_error']  # vehicle yaw - road tangent

        # Perspective road trapezoid
        center_x = w * 0.5 - (lat_offset * (w / 18.0)) - (heading_err * (w / 2.5))
        road_half_w_bottom = w * 0.42
        road_half_w_top = w * 0.08

        pts = np.array([
            [center_x - road_half_w_bottom, h],
            [center_x + road_half_w_bottom, h],
            [center_x + road_half_w_top, horizon_y],
            [center_x - road_half_w_top, horizon_y],
        ], dtype=np.int32)

        # Draw road asphalt
        cv2.fillPoly(img, [pts], (50, 50, 50))

        # Draw curbs (red and white)
        curb_w_bottom = 8
        curb_w_top = 2
        left_curb = np.array([
            [center_x - road_half_w_bottom - curb_w_bottom, h],
            [center_x - road_half_w_bottom, h],
            [center_x - road_half_w_top, horizon_y],
            [center_x - road_half_w_top - curb_w_top, horizon_y],
        ], dtype=np.int32)
        cv2.fillPoly(img, [left_curb], (0, 0, 200))

        right_curb = np.array([
            [center_x + road_half_w_bottom, h],
            [center_x + road_half_w_bottom + curb_w_bottom, h],
            [center_x + road_half_w_top + curb_w_top, horizon_y],
            [center_x + road_half_w_top, horizon_y],
        ], dtype=np.int32)
        cv2.fillPoly(img, [right_curb], (0, 0, 200))

        # Dashed center line
        dash_steps = 6
        for step in range(dash_steps):
            t0 = step / dash_steps
            t1 = (step + 0.5) / dash_steps
            y0 = int(h - (h - horizon_y) * t0)
            y1 = int(h - (h - horizon_y) * t1)
            cx0 = int(center_x + (w * 0.5 - center_x) * (1.0 - (y0 - horizon_y) / (h - horizon_y)))
            cx1 = int(center_x + (w * 0.5 - center_x) * (1.0 - (y1 - horizon_y) / (h - horizon_y)))
            cv2.line(img, (cx0, y0), (cx1, y1), (255, 255, 255), max(1, int(3 * (y0 / h))))

        return img

    def _apply_noise(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_std <= 0.0:
            return data
        noise = rng.normal(0.0, self.noise_std * 255.0, size=data.shape).astype(np.float32)
        noisy = np.clip(data.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return noisy

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'width': self.width,
            'height': self.height,
            'fov_degrees': self.fov_degrees,
            'local_pitch': self.local_pitch,
        })
        return d
