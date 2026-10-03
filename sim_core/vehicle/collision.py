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
        corners = obb.get_corners()

        # Check vehicle OBB edges against all boundary segments
        for seg_a, seg_b in boundary_segments:
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
