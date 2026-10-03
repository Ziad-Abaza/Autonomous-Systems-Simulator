"""
Spline geometry and arc-length parameterization for smooth track centerlines.
Supports closed and open tracks, uniform arc-length sampling, tangent and normal vectors,
curvature computation, and spatial projection queries.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sim_core.math_utils import Vec2, Vec3, clamp, normalize_angle


class SplinePoint:
    """A sampled point along the parameterized spline."""
    __slots__ = ('pos', 'tangent', 'normal', 's', 'width', 'elevation', 'banking')

    def __init__(
        self,
        pos: Vec3,
        tangent: Vec3,
        normal: Vec3,
        s: float,
        width: float,
        elevation: float,
        banking: float
    ):
        self.pos = pos
        self.tangent = tangent
        self.normal = normal  # 2D/3D perpendicular pointing to the left
        self.s = s            # Arc-length distance from start
        self.width = width    # Road width at this point
        self.elevation = elevation
        self.banking = banking


class TrackSpline:
    """
    Parametric Catmull-Rom spline with uniform arc-length re-parameterization.
    """
    def __init__(self, is_closed: bool = True):
        self.is_closed = is_closed
        self.samples: List[SplinePoint] = []
        self.total_length: float = 0.0

    def build_from_control_points(
        self,
        control_points: List[Dict[str, Any]],
        sample_step: float = 1.0
    ) -> None:
        """
        Builds high-resolution spline samples from a list of control point dictionaries:
        Each dict has: {'x': float, 'y': float, 'z': float, 'width': float, 'banking': float}
        """
        if len(control_points) < 3:
            self.samples = []
            self.total_length = 0.0
            return

        n = len(control_points)
        raw_pts: List[Vec3] = []
        widths: List[float] = []
        bankings: List[float] = []

        for cp in control_points:
            x = float(cp.get('x', 0.0))
            y = float(cp.get('y', 0.0))
            z = float(cp.get('z', 0.0))
            w = float(cp.get('width', 10.0))
            b = float(cp.get('banking', 0.0))
            raw_pts.append(Vec3(x, y, z))
            widths.append(w)
            bankings.append(b)

        # Generate dense Catmull-Rom interpolation
        dense_positions: List[Vec3] = []
        dense_widths: List[float] = []
        dense_bankings: List[float] = []

        subdivisions = 40
        num_segments = n if self.is_closed else (n - 1)

        for i in range(num_segments):
            if self.is_closed:
                p0 = raw_pts[(i - 1) % n]
                p1 = raw_pts[i % n]
                p2 = raw_pts[(i + 1) % n]
                p3 = raw_pts[(i + 2) % n]
                w1 = widths[i % n]
                w2 = widths[(i + 1) % n]
                b1 = bankings[i % n]
                b2 = bankings[(i + 1) % n]
            else:
                p0 = raw_pts[max(0, i - 1)]
                p1 = raw_pts[i]
                p2 = raw_pts[min(n - 1, i + 1)]
                p3 = raw_pts[min(n - 1, i + 2)]
                w1 = widths[i]
                w2 = widths[min(n - 1, i + 1)]
                b1 = bankings[i]
                b2 = bankings[min(n - 1, i + 1)]

            for step in range(subdivisions):
                if not self.is_closed and i == num_segments - 1 and step == subdivisions - 1:
                    t = 1.0
                else:
                    t = step / subdivisions
                pos = self._catmull_rom(p0, p1, p2, p3, t)
                w = w1 + (w2 - w1) * t
                b = b1 + (b2 - b1) * t
                dense_positions.append(pos)
                dense_widths.append(w)
                dense_bankings.append(b)

        if self.is_closed:
            dense_positions.append(dense_positions[0])
            dense_widths.append(dense_widths[0])
            dense_bankings.append(dense_bankings[0])
        else:
            dense_positions.append(raw_pts[-1])
            dense_widths.append(widths[-1])
            dense_bankings.append(bankings[-1])

        # Arc-length parameterization
        arc_lengths: List[float] = [0.0]
        for i in range(1, len(dense_positions)):
            d = (dense_positions[i] - dense_positions[i - 1]).length()
            arc_lengths.append(arc_lengths[-1] + d)

        total_len = arc_lengths[-1]
        self.total_length = total_len

        if total_len < 1e-4:
            self.samples = []
            return

        # Resample at uniform intervals of sample_step
        num_uniform = max(10, int(math.ceil(total_len / sample_step)))
        uniform_samples: List[SplinePoint] = []

        for k in range(num_uniform):
            target_s = (k / num_uniform) * total_len if self.is_closed else min((k * sample_step), total_len)

            # Binary search / search in arc_lengths
            idx = np.searchsorted(arc_lengths, target_s)
            if idx <= 0:
                pos = dense_positions[0]
                width = dense_widths[0]
                banking = dense_bankings[0]
            elif idx >= len(arc_lengths):
                pos = dense_positions[-1]
                width = dense_widths[-1]
                banking = dense_bankings[-1]
            else:
                s0 = arc_lengths[idx - 1]
                s1 = arc_lengths[idx]
                frac = (target_s - s0) / max(1e-6, (s1 - s0))
                pos = dense_positions[idx - 1] + (dense_positions[idx] - dense_positions[idx - 1]) * frac
                width = dense_widths[idx - 1] + (dense_widths[idx] - dense_widths[idx - 1]) * frac
                banking = dense_bankings[idx - 1] + (dense_bankings[idx] - dense_bankings[idx - 1]) * frac

            uniform_samples.append(SplinePoint(
                pos=pos,
                tangent=Vec3(1.0, 0.0, 0.0),
                normal=Vec3(0.0, 1.0, 0.0),
                s=target_s,
                width=width,
                elevation=pos.z,
                banking=banking
            ))

        # Compute accurate tangent and normal vectors
        m = len(uniform_samples)
        for i in range(m):
            if self.is_closed:
                prev_p = uniform_samples[(i - 1) % m].pos
                next_p = uniform_samples[(i + 1) % m].pos
            else:
                prev_p = uniform_samples[max(0, i - 1)].pos
                next_p = uniform_samples[min(m - 1, i + 1)].pos

            tangent_3d = (next_p - prev_p).normalized()
            # 2D horizontal tangent and left-hand normal
            tangent_2d = Vec2(tangent_3d.x, tangent_3d.y).normalized()
            normal_2d = tangent_2d.perpendicular()  # (-y, x) pointing left

            uniform_samples[i].tangent = tangent_3d
            uniform_samples[i].normal = Vec3(normal_2d.x, normal_2d.y, 0.0)

        self.samples = uniform_samples

    def _catmull_rom(self, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
        """Standard Catmull-Rom cubic spline formulation."""
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    def get_closest_point(self, pos_2d: Vec2) -> Tuple[float, float, float, SplinePoint]:
        """
        Projects a 2D position onto the track centerline.
        Returns:
            s: Arc-length along track [0, total_length]
            lateral_offset: Signed distance from centerline (positive = left, negative = right)
            heading_error: Heading error angle [-pi, pi] between tangent and vehicle heading if needed
            sample: The nearest SplinePoint
        """
        if not self.samples:
            dummy = SplinePoint(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), 0.0, 10.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0, dummy

        best_dist_sq = float('inf')
        best_idx = 0

        # Find closest sample point
        for i, sample in enumerate(self.samples):
            dx = sample.pos.x - pos_2d.x
            dy = sample.pos.y - pos_2d.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i

        sample = self.samples[best_idx]
        to_pt = pos_2d - Vec2(sample.pos.x, sample.pos.y)

        # Signed lateral offset: dot product with left-hand normal
        # normal points left: positive = vehicle is left of center, negative = right of center
        normal_2d = Vec2(sample.normal.x, sample.normal.y)
        lateral_offset = to_pt.dot(normal_2d)

        # Tangent angle
        tangent_angle = math.atan2(sample.tangent.y, sample.tangent.x)

        return sample.s, lateral_offset, tangent_angle, sample

    def get_heading_error(self, vehicle_yaw: float, tangent_angle: float) -> float:
        """Computes shortest signed heading error angle between vehicle and track forward direction."""
        return normalize_angle(vehicle_yaw - tangent_angle)

    def sample_at_distance(self, s: float) -> SplinePoint:
        """Samples spline point at a given arc-length distance s."""
        if not self.samples:
            return SplinePoint(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), 0.0, 10.0, 0.0, 0.0)

        if self.is_closed and self.total_length > 0.0:
            s = s % self.total_length
        else:
            s = clamp(s, 0.0, self.total_length)

        # Linear search or interpolation
        idx = int((s / max(1e-4, self.total_length)) * len(self.samples))
        idx = clamp(idx, 0, len(self.samples) - 1)
        return self.samples[idx]
