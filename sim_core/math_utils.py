"""
High-performance math utilities for 3D simulation and geometry calculations.
Includes vector operations, oriented bounding boxes (OBB), ray-segment intersections,
and spatial queries for road geometry.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np


class Vec2:
    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:.3f}, {self.y:.3f})"

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """2D cross product: self.x * other.y - self.y * other.x"""
        return self.x * other.y - self.y * other.x

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        l = self.length()
        if l < 1e-8:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)

    def perpendicular(self) -> Vec2:
        """Rotate 90 degrees counter-clockwise (left normal): (-y, x)"""
        return Vec2(-self.y, self.x)

    def rotated(self, angle_rad: float) -> Vec2:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class Vec3:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> Vec3:
        l = self.length()
        if l < 1e-8:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_vec2(self) -> Vec2:
        return Vec2(self.x, self.y)


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def point_segment_distance(pt: Vec2, a: Vec2, b: Vec2) -> Tuple[float, Vec2, float]:
    """
    Computes shortest distance from point pt to segment a-b.
    Returns: (distance, closest_point, parameter_t_in_0_1)
    """
    ab = b - a
    ab_len_sq = ab.length_sq()
    if ab_len_sq < 1e-8:
        return (pt - a).length(), a, 0.0

    ap = pt - a
    t = ap.dot(ab) / ab_len_sq
    t_clamped = clamp(t, 0.0, 1.0)
    closest = a + ab * t_clamped
    dist = (pt - closest).length()
    return dist, closest, t_clamped


def ray_segment_intersection(
    ray_origin: Vec2,
    ray_dir: Vec2,
    seg_a: Vec2,
    seg_b: Vec2,
    max_range: float
) -> Optional[float]:
    """
    2D Ray vs Line Segment intersection test.
    Ray: origin + t * dir, t >= 0
    Segment: seg_a + u * (seg_b - seg_a), u in [0, 1]
    Returns t (distance along ray) if intersection occurs within max_range, else None.
    """
    v1 = ray_origin - seg_a
    v2 = seg_b - seg_a
    v3 = Vec2(-ray_dir.y, ray_dir.x)

    dot = v2.dot(v3)
    if abs(dot) < 1e-8:
        return None  # Parallel

    t1 = v2.cross(v1) / dot
    t2 = v1.dot(v3) / dot

    if t1 >= 0.0 and (t1 <= max_range) and (0.0 <= t2 <= 1.0):
        return t1
    return None


class OBB2D:
    """
    Oriented Bounding Box in 2D (chassis collision, obstacles).
    Defined by center, half_extents (half_length, half_width), and yaw angle.
    """
    def __init__(self, center: Vec2, half_length: float, half_width: float, yaw: float):
        self.center = center
        self.half_length = half_length
        self.half_width = half_width
        self.yaw = yaw

    def get_corners(self) -> List[Vec2]:
        """Returns 4 corners in clockwise order: Front-Right, Rear-Right, Rear-Left, Front-Left."""
        fwd = Vec2(math.cos(self.yaw), math.sin(self.yaw))
        right = Vec2(math.sin(self.yaw), -math.cos(self.yaw))

        c = self.center
        hl = self.half_length
        hw = self.half_width

        return [
            c + fwd * hl + right * hw,  # Front-Right
            c - fwd * hl + right * hw,  # Rear-Right
            c - fwd * hl - right * hw,  # Rear-Left
            c + fwd * hl - right * hw,  # Front-Left
        ]

    def get_axes(self) -> List[Vec2]:
        """Returns the two orthogonal axes for Separating Axis Theorem (SAT)."""
        fwd = Vec2(math.cos(self.yaw), math.sin(self.yaw))
        right = Vec2(math.sin(self.yaw), -math.cos(self.yaw))
        return [fwd, right]

    def intersects_segment(self, seg_a: Vec2, seg_b: Vec2) -> bool:
        """Checks if line segment intersects this OBB using SAT."""
        corners = self.get_corners()
        # Check segment vs each of the 4 edges of the box
        for i in range(4):
            c1 = corners[i]
            c2 = corners[(i + 1) % 4]
            if segments_intersect(seg_a, seg_b, c1, c2):
                return True
        # Check if segment endpoint is inside box
        if self.contains_point(seg_a) or self.contains_point(seg_b):
            return True
        return False

    def contains_point(self, pt: Vec2) -> bool:
        """Tests if a point is inside the OBB."""
        d = pt - self.center
        fwd = Vec2(math.cos(self.yaw), math.sin(self.yaw))
        right = Vec2(math.sin(self.yaw), -math.cos(self.yaw))

        proj_fwd = abs(d.dot(fwd))
        proj_right = abs(d.dot(right))

        return proj_fwd <= self.half_length and proj_right <= self.half_width


def segments_intersect(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> bool:
    """Checks whether segment p1-p2 intersects segment p3-p4."""
    d1 = (p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)
    d2 = (p4.x - p3.x) * (p2.y - p3.y) - (p4.y - p3.y) * (p2.x - p3.x)
    d3 = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    d4 = (p2.x - p1.x) * (p4.y - p1.y) - (p2.y - p1.y) * (p4.x - p1.x)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False
