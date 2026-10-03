"""
Procedural 3D mesh generator and spatial collision boundary extraction from logical road definitions.
Generates road surface geometry, curbs, guardrails/walls, lane markings, and checkpoint triggers.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sim_core.math_utils import Vec2, Vec3
from sim_core.track.spline import TrackSpline, SplinePoint
from sim_core.track.road_definition import RoadDefinition, RoadBoundaryConfig
from sim_core.collision.spatial_hash import SpatialHashGrid2D



class GeneratedTrack:
    """
    Holds the generated 3D meshes, collision boundaries, and checkpoints
    derived deterministically from a RoadDefinition.
    """
    def __init__(self):
        self.spline = TrackSpline()
        # Mesh vertex arrays: (N, 3) positions, (N, 3) normals, (N, 2) uvs, (M, 3) triangle indices
        self.road_vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.road_normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.road_uvs: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.road_indices: np.ndarray = np.empty((0, 3), dtype=np.int32)

        # Curbs
        self.curb_vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.curb_colors: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.curb_indices: np.ndarray = np.empty((0, 3), dtype=np.int32)

        # Guardrails / Barriers
        self.barrier_vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.barrier_indices: np.ndarray = np.empty((0, 3), dtype=np.int32)

        # Collision boundary segments: list of (Vec2(start), Vec2(end))
        self.left_boundary_segments: List[Tuple[Vec2, Vec2]] = []
        self.right_boundary_segments: List[Tuple[Vec2, Vec2]] = []
        self.all_boundary_segments: List[Tuple[Vec2, Vec2]] = []

        # Checkpoints for progress tracking
        self.checkpoints: List[Dict[str, Any]] = []

        # Track bounds
        self.min_bounds = Vec3(0, 0, 0)
        self.max_bounds = Vec3(0, 0, 0)

        # Spatial Hash Broadphase
        self.broadphase: Optional[SpatialHashGrid2D] = None

        # Source road definition (surface parameters for physics queries)
        self.road_def: Optional[RoadDefinition] = None



class TrackMeshGenerator:
    """
    Transforms logical RoadDefinition into 3D renderable geometry and 2D/3D physical boundaries.
    """
    @staticmethod
    def generate(road_def: RoadDefinition, sample_step: float = 1.0) -> GeneratedTrack:
        track = GeneratedTrack()
        track.road_def = road_def
        track.spline = TrackSpline(is_closed=road_def.is_closed)
        track.spline.build_from_control_points(
            [cp.to_dict() for cp in road_def.control_points],
            sample_step=sample_step
        )

        samples = track.spline.samples
        if len(samples) < 2:
            return track

        n = len(samples)
        num_rings = n if road_def.is_closed else (n - 1)

        # 1. Road Surface Mesh Generation
        # At each sample ring: 2 vertices: Left and Right edge of road
        road_pos = []
        road_norm = []
        road_uv = []
        road_ind = []

        curb_pos = []
        curb_col = []
        curb_ind = []

        barrier_pos = []
        barrier_ind = []

        left_bounds = []
        right_bounds = []

        cfg = road_def.boundary_config
        curb_w = cfg.curb_width if cfg.has_curbs else 0.0
        curb_h = cfg.curb_height if cfg.has_curbs else 0.0
        wall_h = cfg.wall_height

        for i, s in enumerate(samples):
            half_w = s.width * 0.5
            norm_2d = Vec2(s.normal.x, s.normal.y)
            center = Vec2(s.pos.x, s.pos.y)

            # Left and right road surface points
            left_pt = center + norm_2d * half_w
            right_pt = center - norm_2d * half_w

            # Road surface vertices
            road_pos.append([left_pt.x, left_pt.y, s.pos.z])
            road_pos.append([right_pt.x, right_pt.y, s.pos.z])

            road_norm.append([0.0, 0.0, 1.0])
            road_norm.append([0.0, 0.0, 1.0])

            v_coord = s.s * 0.1  # UV along road
            road_uv.append([0.0, v_coord])
            road_uv.append([1.0, v_coord])

            # Curbs (red and white alternating strips)
            if cfg.has_curbs:
                outer_left = center + norm_2d * (half_w + curb_w)
                outer_right = center - norm_2d * (half_w + curb_w)

                # Alternating color based on arc length
                is_red = int(s.s / 2.0) % 2 == 0
                color = [0.9, 0.1, 0.1] if is_red else [0.95, 0.95, 0.95]

                # Left curb ring (2 vertices: inner on road, outer raised)
                curb_pos.append([left_pt.x, left_pt.y, s.pos.z])
                curb_pos.append([outer_left.x, outer_left.y, s.pos.z + curb_h])
                curb_col.append(color)
                curb_col.append(color)

                # Right curb ring (2 vertices: inner on road, outer raised)
                curb_pos.append([right_pt.x, right_pt.y, s.pos.z])
                curb_pos.append([outer_right.x, outer_right.y, s.pos.z + curb_h])
                curb_col.append(color)
                curb_col.append(color)

            # Barriers (vertical walls for guardrail/wall)
            if cfg.left_type in ("guardrail", "wall"):
                b_left_base = center + norm_2d * (half_w + curb_w)
                barrier_pos.append([b_left_base.x, b_left_base.y, s.pos.z])
                barrier_pos.append([b_left_base.x, b_left_base.y, s.pos.z + wall_h])

            if cfg.right_type in ("guardrail", "wall"):
                b_right_base = center - norm_2d * (half_w + curb_w)
                barrier_pos.append([b_right_base.x, b_right_base.y, s.pos.z])
                barrier_pos.append([b_right_base.x, b_right_base.y, s.pos.z + wall_h])

        # Generate indices for road surface
        for i in range(num_rings):
            i0 = (i * 2)
            i1 = (i * 2 + 1)
            i2 = (((i + 1) % n) * 2)
            i3 = (((i + 1) % n) * 2 + 1)

            # Two triangles forming quad: i0-i2-i1, i1-i2-i3
            road_ind.append([i0, i2, i1])
            road_ind.append([i1, i2, i3])

            # Extract 2D boundary collision segments
            # Left boundary segment
            p_left_start = Vec2(road_pos[i0][0], road_pos[i0][1])
            p_left_end = Vec2(road_pos[i2][0], road_pos[i2][1])
            # If there's a curb or wall, push boundary slightly outward to curb edge
            if cfg.has_curbs:
                norm_2d = Vec2(samples[i].normal.x, samples[i].normal.y)
                p_left_start = p_left_start + norm_2d * curb_w
                norm_next = Vec2(samples[(i + 1) % n].normal.x, samples[(i + 1) % n].normal.y)
                p_left_end = p_left_end + norm_next * curb_w

            left_bounds.append((p_left_start, p_left_end))

            # Right boundary segment
            p_right_start = Vec2(road_pos[i1][0], road_pos[i1][1])
            p_right_end = Vec2(road_pos[i3][0], road_pos[i3][1])
            if cfg.has_curbs:
                norm_2d = Vec2(samples[i].normal.x, samples[i].normal.y)
                p_right_start = p_right_start - norm_2d * curb_w
                norm_next = Vec2(samples[(i + 1) % n].normal.x, samples[(i + 1) % n].normal.y)
                p_right_end = p_right_end - norm_next * curb_w

            right_bounds.append((p_right_start, p_right_end))

        # Generate indices for curbs
        if cfg.has_curbs:
            for i in range(num_rings):
                next_i = (i + 1) % n
                # Left curb quad
                l_base_cur = i * 4
                l_base_next = next_i * 4
                curb_ind.append([l_base_cur, l_base_next, l_base_cur + 1])
                curb_ind.append([l_base_cur + 1, l_base_next, l_base_next + 1])

                # Right curb quad
                r_base_cur = i * 4 + 2
                r_base_next = next_i * 4 + 2
                curb_ind.append([r_base_cur, r_base_next, r_base_cur + 1])
                curb_ind.append([r_base_cur + 1, r_base_next, r_base_next + 1])

        # Generate indices for barriers
        b_ring_size = (2 if cfg.left_type in ("guardrail", "wall") else 0) + \
                      (2 if cfg.right_type in ("guardrail", "wall") else 0)

        if b_ring_size > 0:
            for i in range(num_rings):
                next_i = (i + 1) % n
                offset_cur = i * b_ring_size
                offset_next = next_i * b_ring_size

                idx = 0
                if cfg.left_type in ("guardrail", "wall"):
                    c0 = offset_cur + idx
                    c1 = offset_cur + idx + 1
                    n0 = offset_next + idx
                    n1 = offset_next + idx + 1
                    barrier_ind.append([c0, n0, c1])
                    barrier_ind.append([c1, n0, n1])
                    idx += 2

                if cfg.right_type in ("guardrail", "wall"):
                    c0 = offset_cur + idx
                    c1 = offset_cur + idx + 1
                    n0 = offset_next + idx
                    n1 = offset_next + idx + 1
                    barrier_ind.append([c0, n0, c1])
                    barrier_ind.append([c1, n0, n1])

        # Checkpoints generation along the track
        checkpoints = []
        num_cp = max(4, road_def.num_checkpoints)
        for k in range(num_cp):
            s_target = (k / num_cp) * track.spline.total_length
            sp = track.spline.sample_at_distance(s_target)
            half_w = sp.width * 0.5 + curb_w
            norm_2d = Vec2(sp.normal.x, sp.normal.y)
            pos_2d = Vec2(sp.pos.x, sp.pos.y)

            cp_left = pos_2d + norm_2d * half_w
            cp_right = pos_2d - norm_2d * half_w

            checkpoints.append({
                'index': k,
                's': s_target,
                'pos': sp.pos,
                'tangent': sp.tangent,
                'normal': sp.normal,
                'width': sp.width,
                'gate_left': cp_left,
                'gate_right': cp_right,
            })

        # Finalize numpy arrays
        track.road_vertices = np.array(road_pos, dtype=np.float32)
        track.road_normals = np.array(road_norm, dtype=np.float32)
        track.road_uvs = np.array(road_uv, dtype=np.float32)
        track.road_indices = np.array(road_ind, dtype=np.int32)

        if curb_pos:
            track.curb_vertices = np.array(curb_pos, dtype=np.float32)
            track.curb_colors = np.array(curb_col, dtype=np.float32)
            track.curb_indices = np.array(curb_ind, dtype=np.int32)

        if barrier_pos:
            track.barrier_vertices = np.array(barrier_pos, dtype=np.float32)
            track.barrier_indices = np.array(barrier_ind, dtype=np.int32)

        track.left_boundary_segments = left_bounds
        track.right_boundary_segments = right_bounds
        track.all_boundary_segments = left_bounds + right_bounds
        track.checkpoints = checkpoints

        # Compute bounding box
        if len(road_pos) > 0:
            all_pts = np.array(road_pos)
            track.min_bounds = Vec3(float(np.min(all_pts[:, 0])), float(np.min(all_pts[:, 1])), float(np.min(all_pts[:, 2])))
            track.max_bounds = Vec3(float(np.max(all_pts[:, 0])), float(np.max(all_pts[:, 1])), float(np.max(all_pts[:, 2])))

        # Build 2D spatial acceleration broadphase
        track.broadphase = SpatialHashGrid2D(cell_size=12.0)
        track.broadphase.build_from_segments(track.all_boundary_segments)

        return track
