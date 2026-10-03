"""
Collision detection for vehicles against track boundaries and world obstacles.
Supports spatial hash broadphase acceleration with exact SAT narrowphase
verification, plus impulse-based contact response.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any
from sim_core.math_utils import Vec2, OBB2D, clamp, point_segment_distance
from sim_core.vehicle.vehicle_model import VehicleModel


class CollisionResult:
    __slots__ = ('collided', 'impact_point', 'collider_type', 'details',
                 'contact_seg', 'contact_obb')

    def __init__(
        self,
        collided: bool = False,
        impact_point: Optional[Vec2] = None,
        collider_type: str = "none",
        details: str = "",
        contact_seg: Optional[Tuple[Vec2, Vec2]] = None,
        contact_obb: Optional[OBB2D] = None
    ):
        self.collided = collided
        self.impact_point = impact_point
        self.collider_type = collider_type
        self.details = details
        self.contact_seg = contact_seg
        self.contact_obb = contact_obb


class VehicleCollisionChecker:
    """
    Tests vehicle OBB against track boundary segments and obstacle entities.
    Supports optional spatial broadphase acceleration.
    """
    @staticmethod
    def check_track_boundary_collision(
        vehicle: VehicleModel,
        boundary_segments: List[Tuple[Vec2, Vec2]],
        broadphase: Optional[Any] = None
    ) -> CollisionResult:
        obb = vehicle.get_obb()
        c = obb.center
        # Bounding radius + safety margin
        r = max(obb.half_length, obb.half_width) + 0.5

        # Query candidate segments via broadphase if available
        if broadphase is not None and hasattr(broadphase, 'query_candidate_segments'):
            candidates = broadphase.query_candidate_segments(c, r)
        else:
            candidates = boundary_segments

        min_x = c.x - r
        max_x = c.x + r
        min_y = c.y - r
        max_y = c.y + r

        # Narrowphase SAT intersection against candidate boundary segments
        for seg_a, seg_b in candidates:
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
                    details="Vehicle collided with track boundary barrier",
                    contact_seg=(seg_a, seg_b)
                )

        vehicle.state.is_colliding = False
        return CollisionResult(collided=False)

    @staticmethod
    def check_obstacle_collision(
        vehicle: VehicleModel,
        obstacles: List[Any],
        broadphase: Optional[Any] = None
    ) -> CollisionResult:
        obb = vehicle.get_obb()
        c = obb.center
        r = max(obb.half_length, obb.half_width) + 1.0

        if broadphase is not None and hasattr(broadphase, 'query_candidate_entities'):
            candidate_obs = broadphase.query_candidate_entities(c, r)
        else:
            candidate_obs = obstacles

        for obs in candidate_obs:
            # Check collidability
            if hasattr(obs, 'is_collidable') and not obs.is_collidable:
                continue

            # Check OBB intersection
            if hasattr(obs, 'get_obb'):
                obs_obb = obs.get_obb()
                if obs_obb is None:
                    continue

                # Quick bounding radius distance check before corner SAT
                dx = obs_obb.center.x - obb.center.x
                dy = obs_obb.center.y - obb.center.y
                obs_r = max(obs_obb.half_length, obs_obb.half_width)
                veh_r = max(obb.half_length, obb.half_width)
                if (dx * dx + dy * dy) > ((obs_r + veh_r) * (obs_r + veh_r)):
                    continue

                # Corner inclusion SAT
                for corner in obs_obb.get_corners():
                    if obb.contains_point(corner):
                        vehicle.state.is_colliding = True
                        return CollisionResult(
                            collided=True,
                            impact_point=corner,
                            collider_type="obstacle",
                            details=f"Collision with {getattr(obs, 'name', 'obstacle')}",
                            contact_obb=obs_obb
                        )
                for corner in obb.get_corners():
                    if obs_obb.contains_point(corner):
                        vehicle.state.is_colliding = True
                        return CollisionResult(
                            collided=True,
                            impact_point=corner,
                            collider_type="obstacle",
                            details=f"Collision with {getattr(obs, 'name', 'obstacle')}",
                            contact_obb=obs_obb
                        )

                # Segment-OBB intersection in case thin barrier penetrates
                obs_segs = obs.get_boundary_segments() if hasattr(obs, 'get_boundary_segments') else []
                for s1, s2 in obs_segs:
                    if obb.intersects_segment(s1, s2):
                        vehicle.state.is_colliding = True
                        return CollisionResult(
                            collided=True,
                            impact_point=s1,
                            collider_type="obstacle",
                            details=f"Collision with {getattr(obs, 'name', 'obstacle')}",
                            contact_obb=obs_obb
                        )

        return CollisionResult(collided=False)

    # ------------------------------------------------------------------
    # Contact response (impulse-based, energy non-increasing)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_impulse(vehicle: VehicleModel, contact_pt: Vec2, normal: Vec2) -> float:
        """
        Applies a rigid-body contact impulse at `contact_pt` along `normal`
        (pointing INTO the vehicle, away from the collider), with Coulomb
        tangential friction. Returns the applied normal impulse magnitude.

        Jn = -(1+e) * v_n / (1/m + (r x n)^2 / Iz)
        Jt = clamp(-v_t / (1/m + (r x t)^2 / Iz), -mu*Jn, +mu*Jn)

        With e <= 1 and friction opposing relative motion, this only removes
        kinetic energy — it can never inject it.
        """
        st = vehicle.state
        cfg = vehicle.config
        center = Vec2(st.pos.x, st.pos.y)
        r = contact_pt - center

        vxw, vyw = st.vel_world.x, st.vel_world.y
        wz = st.yaw_rate
        # contact-point velocity: v + wz x r  (2D: w x r = (-wz*ry, wz*rx))
        vcx = vxw - wz * r.y
        vcy = vyw + wz * r.x
        vn = vcx * normal.x + vcy * normal.y
        if vn >= 0.0:
            return 0.0  # separating already — no impulse

        e = cfg.contact_restitution
        mu_c = cfg.contact_friction
        inv_m = 1.0 / cfg.mass
        rn = r.x * normal.y - r.y * normal.x
        jn = -(1.0 + e) * vn / (inv_m + rn * rn / vehicle.Iz)

        t_dir = normal.perpendicular()  # (-ny, nx) — tangential direction
        vt = vcx * t_dir.x + vcy * t_dir.y
        rt = r.x * t_dir.y - r.y * t_dir.x
        jt = -vt / (inv_m + rt * rt / vehicle.Iz)
        jt = clamp(jt, -mu_c * jn, mu_c * jn)

        jx = normal.x * jn + t_dir.x * jt
        jy = normal.y * jn + t_dir.y * jt

        st.vel_world.x += jx * inv_m
        st.vel_world.y += jy * inv_m
        st.yaw_rate += (r.x * jy - r.y * jx) / vehicle.Iz

        # Resync body-frame velocity to the post-impulse world velocity
        cyaw = math.cos(st.yaw)
        syaw = math.sin(st.yaw)
        st.vel_body = Vec2(
            cyaw * st.vel_world.x + syaw * st.vel_world.y,
            -syaw * st.vel_world.x + cyaw * st.vel_world.y
        )
        return jn

    @staticmethod
    def apply_boundary_contact(
        vehicle: VehicleModel,
        seg_a: Vec2,
        seg_b: Vec2,
        interior_hint: Optional[Vec2] = None
    ) -> bool:
        """
        Resolves vehicle OBB contact against a wall segment: positional
        depenetration along the wall normal plus a normal+tangential impulse.
        Returns True if a response was applied.

        interior_hint: direction pointing from the wall toward the road
        interior (e.g. -sign(lateral_offset)*road_normal). When provided the
        contact normal is oriented toward the interior, so a vehicle whose
        centre has just crossed the wall is pushed BACK onto the road instead
        of being ejected outward.
        """
        st = vehicle.state
        obb = vehicle.get_obb()
        seg_dir = seg_b - seg_a
        seg_len = seg_dir.length()
        if seg_len < 1e-9:
            return False
        seg_dir = seg_dir / seg_len

        # Contact normal: oriented toward the road interior when a hint is
        # given; otherwise toward the side containing the vehicle centre.
        n = seg_dir.perpendicular()
        if interior_hint is not None and interior_hint.length() > 1e-9:
            if n.dot(interior_hint) < 0.0:
                n = -n
        elif n.dot(Vec2(obb.center.x - seg_a.x, obb.center.y - seg_a.y)) < 0.0:
            n = -n

        # Deepest penetrating corner (signed distance along n < 0)
        deepest = 0.0
        contact_pt = None
        for c in obb.get_corners():
            d = (c - seg_a).dot(n)
            if d < deepest:
                deepest = d
                contact_pt = c
        if contact_pt is None:
            # Segment crosses the box without a corner beyond the wall line
            # (e.g. box straddles the segment end) — use the vehicle centre
            # projected to the wall as the nominal contact.
            contact_pt = obb.center
            deepest = 0.0

        # Positional depenetration (full — barrier is infinitely stiff)
        if deepest < 0.0:
            st.pos.x += n.x * (-deepest)
            st.pos.y += n.y * (-deepest)

        t = clamp((contact_pt - seg_a).dot(seg_dir), 0.0, seg_len)
        wall_pt = seg_a + seg_dir * t
        VehicleCollisionChecker._apply_impulse(vehicle, wall_pt, n)
        return True

    @staticmethod
    def apply_obstacle_contact(vehicle: VehicleModel, obs_obb: OBB2D) -> bool:
        """
        Resolves vehicle contact against an obstacle OBB using the normal
        from the obstacle's closest boundary point to the vehicle centre.
        """
        st = vehicle.state
        center = Vec2(st.pos.x, st.pos.y)

        best_d = float('inf')
        best_pt = None
        corners = obs_obb.get_corners()
        for i in range(4):
            d, pt, _t = point_segment_distance(center, corners[i], corners[(i + 1) % 4])
            if d < best_d:
                best_d = d
                best_pt = pt
        if best_pt is None:
            return False

        n_vec = center - best_pt
        if n_vec.length() < 1e-6:
            # Centre inside the obstacle: push out along its shallowest axis
            axes = obs_obb.get_axes()
            d = center - obs_obb.center
            p0 = d.dot(axes[0])
            p1 = d.dot(axes[1])
            n_vec = axes[0] * (1.0 if p0 >= 0 else -1.0) if abs(p0) < abs(p1) \
                else axes[1] * (1.0 if p1 >= 0 else -1.0)
        n = n_vec.normalized()

        # Depenetration: push out by the overlap of the vehicle OBB along n.
        veh_obb = vehicle.get_obb()
        fwd = Vec2(math.cos(veh_obb.yaw), math.sin(veh_obb.yaw))
        right = Vec2(math.sin(veh_obb.yaw), -math.cos(veh_obb.yaw))
        ext = (veh_obb.half_length * abs(fwd.dot(n))
               + veh_obb.half_width * abs(right.dot(n)))
        pen = ext - best_d
        if pen > 0.0:
            st.pos.x += n.x * pen
            st.pos.y += n.y * pen

        VehicleCollisionChecker._apply_impulse(vehicle, best_pt, n)
        return True
