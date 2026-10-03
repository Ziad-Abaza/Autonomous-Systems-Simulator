"""
Spatial queries and collision tests against track geometry and checkpoints.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional, Dict, Any
from sim_core.math_utils import Vec2, Vec3, ray_segment_intersection, point_segment_distance, clamp, normalize_angle
from sim_core.track.mesh_generator import GeneratedTrack


class TrackSpatialQueries:
    """
    High-performance spatial query engine for road queries, boundary raycasts, and checkpoint crossing.
    """
    def __init__(self, track: GeneratedTrack):
        self.track = track

    def query_vehicle_pose(self, pos_2d: Vec2, yaw: float) -> Dict[str, Any]:
        """
        Projects vehicle position onto the track centerline and returns:
        - s: arc length along track
        - lateral_offset: distance from center (positive left, negative right)
        - heading_error: angle between vehicle heading and road tangent [-pi, pi]
        - is_on_road: boolean whether vehicle center is inside road width
        - road_width: width at this location
        """
        s, lateral_offset, tangent_angle, sp = self.track.spline.get_closest_point(pos_2d)
        heading_error = self.track.spline.get_heading_error(yaw, tangent_angle)
        half_width = sp.width * 0.5
        is_on_road = abs(lateral_offset) <= half_width

        return {
            's': s,
            'lateral_offset': lateral_offset,
            'tangent_angle': tangent_angle,
            'heading_error': heading_error,
            'is_on_road': is_on_road,
            'road_width': sp.width,
            'elevation': sp.elevation,
        }

    def cast_ray(
        self,
        ray_origin: Vec2,
        ray_dir: Vec2,
        max_range: float = 50.0,
        additional_segments: Optional[List[Tuple[Vec2, Vec2]]] = None
    ) -> Tuple[float, Optional[Vec2]]:
        """
        Casts a 2D ray against all track boundaries and optional obstacle segments.
        Returns: (distance, hit_point)
        """
        closest_dist = max_range
        hit_pt = None

        # Ray bounding box
        rx2 = ray_origin.x + ray_dir.x * max_range
        ry2 = ray_origin.y + ray_dir.y * max_range
        ray_min_x = (ray_origin.x if ray_origin.x < rx2 else rx2) - 0.5
        ray_max_x = (ray_origin.x if ray_origin.x > rx2 else rx2) + 0.5
        ray_min_y = (ray_origin.y if ray_origin.y < ry2 else ry2) - 0.5
        ray_max_y = (ray_origin.y if ray_origin.y > ry2 else ry2) + 0.5

        # Check track boundary segments
        for seg_a, seg_b in self.track.all_boundary_segments:
            seg_min_x = seg_a.x if seg_a.x < seg_b.x else seg_b.x
            seg_max_x = seg_a.x if seg_a.x > seg_b.x else seg_b.x
            if seg_max_x < ray_min_x or seg_min_x > ray_max_x:
                continue

            seg_min_y = seg_a.y if seg_a.y < seg_b.y else seg_b.y
            seg_max_y = seg_a.y if seg_a.y > seg_b.y else seg_b.y
            if seg_max_y < ray_min_y or seg_min_y > ray_max_y:
                continue

            d = ray_segment_intersection(ray_origin, ray_dir, seg_a, seg_b, closest_dist)
            if d is not None and d < closest_dist:
                closest_dist = d
                hit_pt = ray_origin + ray_dir * d

        # Check additional obstacle segments if any
        if additional_segments:
            for seg_a, seg_b in additional_segments:
                seg_min_x = seg_a.x if seg_a.x < seg_b.x else seg_b.x
                seg_max_x = seg_a.x if seg_a.x > seg_b.x else seg_b.x
                if seg_max_x < ray_min_x or seg_min_x > ray_max_x:
                    continue

                seg_min_y = seg_a.y if seg_a.y < seg_b.y else seg_b.y
                seg_max_y = seg_a.y if seg_a.y > seg_b.y else seg_b.y
                if seg_max_y < ray_min_y or seg_min_y > ray_max_y:
                    continue

                d = ray_segment_intersection(ray_origin, ray_dir, seg_a, seg_b, closest_dist)
                if d is not None and d < closest_dist:
                    closest_dist = d
                    hit_pt = ray_origin + ray_dir * d

        return closest_dist, hit_pt

    def check_checkpoint_crossing(
        self,
        prev_pos: Vec2,
        curr_pos: Vec2,
        current_checkpoint_idx: int
    ) -> Tuple[bool, int]:
        """
        Checks if the segment from prev_pos to curr_pos crossed the next checkpoint gate.
        Returns: (crossed_boolean, new_checkpoint_index)
        """
        if not self.track.checkpoints:
            return False, current_checkpoint_idx

        num_cp = len(self.track.checkpoints)
        target_idx = (current_checkpoint_idx) % num_cp
        cp = self.track.checkpoints[target_idx]

        gate_left = cp['gate_left']
        gate_right = cp['gate_right']

        # Check if line from prev_pos to curr_pos intersects gate_left to gate_right
        from sim_core.math_utils import segments_intersect
        if segments_intersect(prev_pos, curr_pos, gate_left, gate_right):
            next_idx = (target_idx + 1) % num_cp
            return True, next_idx

        return False, current_checkpoint_idx
