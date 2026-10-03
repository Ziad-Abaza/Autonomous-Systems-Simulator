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
        past_course_end = False

        # Open-route end semantics: the projection clamps to the nearest
        # sample, so positions beyond either end were previously classified
        # as "on road" with ~0 lateral offset. The road physically ends at
        # the first/last sample — anything beyond along the tangent is
        # off the drivable surface. A 0.5 m tolerance keeps a spawn sitting
        # exactly on the start/end face (fwd ~= 0 within float noise) from
        # being classified off-road.
        if is_on_road and not self.track.spline.is_closed and self.track.spline.samples:
            tangent_2d = Vec2(sp.tangent.x, sp.tangent.y)
            to_pt = pos_2d - Vec2(sp.pos.x, sp.pos.y)
            fwd = to_pt.dot(tangent_2d)
            first = self.track.spline.samples[0]
            last = self.track.spline.samples[-1]
            end_tol = 0.5
            if sp is last and fwd > 0.0:
                past_course_end = True
            elif sp is first and fwd < 0.0:
                past_course_end = True
            if (sp is last and fwd > end_tol) or (sp is first and fwd < -end_tol):
                is_on_road = False

        return {
            's': s,
            'lateral_offset': lateral_offset,
            'tangent_angle': tangent_angle,
            'heading_error': heading_error,
            'is_on_road': is_on_road,
            'past_course_end': past_course_end,
            'road_width': sp.width,
            'elevation': sp.elevation,
            'friction': sp.friction,
            'banking': sp.banking,
        }

    def query_surface(self, pos_2d: Vec2) -> Dict[str, Any]:
        """
        Physical surface properties under a world position.

        Returns:
            region: 'road' | 'curb' | 'off' — road is |lateral| <= width/2,
                curb is the authored curb band just outside it, off otherwise
                (incl. past an open route's end).
            friction_mult: local surface mu multiplier (control-point
                friction on road; road_def.curb_friction / off_road_friction
                beyond it).
            banking_rad: road banking in radians (authored degrees; >0 means
                the left edge is raised so gravity pulls toward -normal).
            grade: road grade dz/ds (tangent z) — positive uphill.
            normal_2d, tangent_2d: surface frame directions in world XY.
            elevation: surface height at the closest sample.
        """
        s, lateral_offset, tangent_angle, sp = self.track.spline.get_closest_point(pos_2d)
        road_def = self.track.road_def
        curb_w = (road_def.boundary_config.curb_width
                  if road_def and road_def.boundary_config.has_curbs else 0.0)
        half_w = sp.width * 0.5

        abs_lat = abs(lateral_offset)
        if abs_lat <= half_w:
            region = "road"
            friction_mult = sp.friction
        elif abs_lat <= half_w + curb_w:
            region = "curb"
            friction_mult = road_def.curb_friction if road_def else 0.85
        else:
            region = "off"
            friction_mult = road_def.off_road_friction if road_def else 0.55

        # Beyond an open route's ends the drivable surface simply ends.
        if region != "off" and not self.track.spline.is_closed and self.track.spline.samples:
            tangent_2d = Vec2(sp.tangent.x, sp.tangent.y)
            to_pt = pos_2d - Vec2(sp.pos.x, sp.pos.y)
            fwd = to_pt.dot(tangent_2d)
            first = self.track.spline.samples[0]
            last = self.track.spline.samples[-1]
            # Same 0.5 m end tolerance as query_vehicle_pose.
            if (sp is last and fwd > 0.5) or (sp is first and fwd < -0.5):
                region = "off"
                friction_mult = road_def.off_road_friction if road_def else 0.55

        return {
            'region': region,
            'friction_mult': friction_mult,
            'banking_rad': math.radians(sp.banking),
            'grade': sp.tangent.z,
            'normal_2d': Vec2(sp.normal.x, sp.normal.y),
            'tangent_2d': Vec2(sp.tangent.x, sp.tangent.y).normalized(),
            'elevation': sp.elevation,
            'is_on_road': region != "off" and abs_lat <= half_w,
            's': s,
            'lateral_offset': lateral_offset,
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
