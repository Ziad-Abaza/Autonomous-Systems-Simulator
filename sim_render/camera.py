"""
Camera system supporting Chase, Cockpit/Hood, Top-Down Ortho/Perspective,
and Orbit trackside inspection modes.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from sim_core.math_utils import Vec3, clamp


class CameraMode:
    CHASE = "chase"
    HOOD = "hood"
    TOP_DOWN = "top_down"
    ORBIT = "orbit"


class SimulationCamera:
    """
    Computes View and Projection matrices for 3D rendering.
    Coordinates: X East/Right, Y North/Forward, Z Up.
    """
    def __init__(self, aspect_ratio: float = 16.0 / 9.0):
        self.mode = CameraMode.CHASE
        self.aspect_ratio = aspect_ratio
        self.fov_degrees = 60.0
        self.near_clip = 0.5
        self.far_clip = 1000.0

        # Camera pose in 3D
        self.pos = Vec3(0.0, -10.0, 5.0)
        self.target = Vec3(0.0, 0.0, 0.0)
        self.up = Vec3(0.0, 0.0, 1.0)

        # Chase parameters
        self.chase_distance = 7.5
        self.chase_height = 3.2
        self.chase_damping = 0.15
        self._current_yaw = 0.0

        # Top-down parameters
        self.top_down_height = 90.0
        self.pan_offset = Vec3(0.0, 0.0, 0.0)

        # Orbit parameters
        self.orbit_distance = 25.0
        self.orbit_yaw = 0.0
        self.orbit_pitch = 0.45

    def set_aspect_ratio(self, width: int, height: int) -> None:
        self.aspect_ratio = max(0.1, float(width) / max(1, height))

    def update(self, vehicle_pos: Vec3, vehicle_yaw: float, dt: float) -> None:
        """Updates camera pose based on active mode."""
        if self.mode == CameraMode.CHASE:
            # Smooth yaw following
            yaw_diff = vehicle_yaw - self._current_yaw
            while yaw_diff > math.pi:
                yaw_diff -= 2.0 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2.0 * math.pi
            self._current_yaw += yaw_diff * min(1.0, 8.0 * dt)

            cos_y = math.cos(self._current_yaw)
            sin_y = math.sin(self._current_yaw)

            # Position behind and above car
            cam_x = vehicle_pos.x - cos_y * self.chase_distance
            cam_y = vehicle_pos.y - sin_y * self.chase_distance
            cam_z = vehicle_pos.z + self.chase_height

            self.pos = Vec3(cam_x, cam_y, cam_z)
            self.target = Vec3(vehicle_pos.x, vehicle_pos.y, vehicle_pos.z + 1.0)

        elif self.mode == CameraMode.HOOD:
            cos_y = math.cos(vehicle_yaw)
            sin_y = math.sin(vehicle_yaw)
            # Front hood
            cam_x = vehicle_pos.x + cos_y * 1.2
            cam_y = vehicle_pos.y + sin_y * 1.2
            cam_z = vehicle_pos.z + 1.1
            self.pos = Vec3(cam_x, cam_y, cam_z)
            self.target = Vec3(cam_x + cos_y * 10.0, cam_y + sin_y * 10.0, cam_z - 0.4)

        elif self.mode == CameraMode.TOP_DOWN:
            self.pos = Vec3(vehicle_pos.x + self.pan_offset.x, vehicle_pos.y + self.pan_offset.y, self.top_down_height)
            self.target = Vec3(vehicle_pos.x + self.pan_offset.x, vehicle_pos.y + self.pan_offset.y + 0.01, 0.0)

        elif self.mode == CameraMode.ORBIT:
            cos_p = math.cos(self.orbit_pitch)
            sin_p = math.sin(self.orbit_pitch)
            cos_y = math.cos(self.orbit_yaw)
            sin_y = math.sin(self.orbit_yaw)

            cam_x = vehicle_pos.x + self.orbit_distance * cos_p * sin_y
            cam_y = vehicle_pos.y - self.orbit_distance * cos_p * cos_y
            cam_z = vehicle_pos.z + self.orbit_distance * sin_p

            self.pos = Vec3(cam_x, cam_y, cam_z)
            self.target = Vec3(vehicle_pos.x, vehicle_pos.y, vehicle_pos.z + 0.8)

    def get_view_matrix(self) -> np.ndarray:
        """Computes 4x4 View Matrix (LookAt)."""
        eye = np.array([self.pos.x, self.pos.y, self.pos.z], dtype=np.float32)
        target = np.array([self.target.x, self.target.y, self.target.z], dtype=np.float32)
        up = np.array([self.up.x, self.up.y, self.up.z], dtype=np.float32)

        # Forward vector (from target to eye for OpenGL camera look)
        f = target - eye
        f_norm = np.linalg.norm(f)
        f = f / max(1e-6, f_norm)

        # Right vector = f cross up
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-6:
            s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            s = s / s_norm

        # Recompute up = s cross f
        u = np.cross(s, f)

        view = np.identity(4, dtype=np.float32)
        view[0, 0:3] = s
        view[1, 0:3] = u
        view[2, 0:3] = -f
        view[0, 3] = -np.dot(s, eye)
        view[1, 3] = -np.dot(u, eye)
        view[2, 3] = np.dot(f, eye)

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Computes 4x4 Perspective Projection Matrix."""
        fov_rad = math.radians(self.fov_degrees)
        tan_half = math.tan(fov_rad / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)

        proj[0, 0] = 1.0 / (self.aspect_ratio * tan_half)
        proj[1, 1] = 1.0 / tan_half
        proj[2, 2] = -(self.far_clip + self.near_clip) / (self.far_clip - self.near_clip)
        proj[2, 3] = -(2.0 * self.far_clip * self.near_clip) / (self.far_clip - self.near_clip)
        proj[3, 2] = -1.0
        return proj
