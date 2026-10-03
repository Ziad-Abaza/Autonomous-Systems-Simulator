"""
Collision detection for vehicles against track boundaries and world obstacles.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from sim_core.math_utils import Vec2, OBB2D
from sim_core.vehicle.vehicle_model import VehicleModel


class CollisionResult:
    __slots__ = ('collided', 'impact_point', 'collider_type', 'details')

    def __init__(
        self,
        collided: bool = False,
        impact_point: Optional[Vec2] = None,
        collider_type: str = "none",
        details: str = ""
    ):
        self.collided = collided
        self.impact_point = impact_point
        self.collider_type = collider_type
        self.details = details


class VehicleCollisionChecker:
    """
    Tests vehicle OBB against track boundary segments and obstacle entities.
    """
    @staticmethod
    def check_track_boundary_collision(
        vehicle: VehicleModel,
        boundary_segments: List[Tuple[Vec2, Vec2]]
    ) -> CollisionResult:
        obb = vehicle.get_obb()
        c = obb.center
        # Bounding radius + safety margin
        r = max(obb.half_length, obb.half_width) + 0.5
        min_x = c.x - r
        max_x = c.x + r
        min_y = c.y - r
        max_y = c.y + r

        # Check only boundary segments whose AABB overlaps vehicle AABB
        for seg_a, seg_b in boundary_segments:
            seg_min_x = seg_a.x if seg_a.x < seg_b.x else seg_b.x
            seg_max_x = seg_a.x if seg_a.x > seg_b.x else seg_b.x
            if seg_max_x < min_x or seg_min_x > max_x:
                continue

            seg_min_y = seg_a.y if seg_a.y < seg_b.y else seg_b.y
            seg_max_y = seg_a.y if seg_a.y > seg_b.y else seg_b.y
            if seg_max_y < min_y or seg_min_y > max_y:
                continue

            if obb.intersects_segment(seg_a, seg_b):
                vehicle.state.is_colliding = True
                return CollisionResult(
                    collided=True,
                    impact_point=obb.center,
                    collider_type="boundary_barrier",
                    details="Vehicle collided with track boundary barrier"
                )

        vehicle.state.is_colliding = False
        return CollisionResult(collided=False)

    @staticmethod
    def check_obstacle_collision(
        vehicle: VehicleModel,
        obstacles: List[Any]
    ) -> CollisionResult:
        obb = vehicle.get_obb()
        for obs in obstacles:
            # If obstacle has get_obb or get_segments
            if hasattr(obs, 'get_obb'):
                obs_obb = obs.get_obb()
                # Check if OBBs intersect (corners test or SAT)
                for corner in obs_obb.get_corners():
                    if obb.contains_point(corner):
                        vehicle.state.is_colliding = True
                        return CollisionResult(
                            collided=True,
                            impact_point=corner,
                            collider_type="obstacle",
                            details=f"Collision with {getattr(obs, 'name', 'obstacle')}"
                        )
                for corner in obb.get_corners():
                    if obs_obb.contains_point(corner):
                        vehicle.state.is_colliding = True
                        return CollisionResult(
                            collided=True,
                            impact_point=corner,
                            collider_type="obstacle",
                            details=f"Collision with {getattr(obs, 'name', 'obstacle')}"
                        )
        return CollisionResult(collided=False)
