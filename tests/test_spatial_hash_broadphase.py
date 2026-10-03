"""
Unit and integration tests for Phase 2 Spatial Hash Grid 2D Broadphase.
Validates spatial grid insertion, candidate querying, bounding-box tests,
dynamic entity indexing, and zero-false-negative correctness.
"""

import math
import pytest
from sim_core.math_utils import Vec2, Vec3
from sim_core.collision.spatial_hash import SpatialHashGrid2D
from sim_core.world.entity import TrafficCone, Barrier, StaticObstacle


def test_spatial_hash_grid_initialization():
    grid = SpatialHashGrid2D(cell_size=15.0)
    assert grid.cell_size == 15.0
    assert len(grid.segments) == 0
    assert len(grid.grid) == 0
    assert len(grid.entities) == 0
    assert len(grid.entity_grid) == 0


def test_spatial_hash_segment_insertion_and_query():
    grid = SpatialHashGrid2D(cell_size=10.0)

    # Insert two segments via build_from_segments:
    # Seg 0: (0, 0) to (5, 5) -> cell (0, 0)
    # Seg 1: (50, 50) to (55, 55) -> cell (5, 5)
    segs = [
        (Vec2(0.0, 0.0), Vec2(5.0, 5.0)),
        (Vec2(50.0, 50.0), Vec2(55.0, 55.0))
    ]
    grid.build_from_segments(segs)

    assert len(grid.segments) == 2

    # Query near seg 0
    candidates_0 = grid.query_candidate_segment_indices(Vec2(2.5, 2.5), 3.0)
    assert 0 in candidates_0
    assert 1 not in candidates_0

    # Query candidate segments directly
    cand_segs_0 = grid.query_candidate_segments(Vec2(2.5, 2.5), 3.0)
    assert len(cand_segs_0) == 1
    assert cand_segs_0[0] == segs[0]

    # Query near seg 1
    candidates_1 = grid.query_candidate_segment_indices(Vec2(52.0, 52.0), 3.0)
    assert 1 in candidates_1
    assert 0 not in candidates_1

    # Query far empty region
    empty = grid.query_candidate_segment_indices(Vec2(100.0, 100.0), 5.0)
    assert len(empty) == 0


def test_spatial_hash_entity_indexing():
    grid = SpatialHashGrid2D(cell_size=10.0)

    cone1 = TrafficCone(name="Cone 1", pos=Vec3(5.0, 5.0, 0.0), radius=0.5)
    cone2 = TrafficCone(name="Cone 2", pos=Vec3(85.0, 85.0, 0.0), radius=0.5)

    grid.insert_entity(cone1)
    grid.insert_entity(cone2)

    assert len(grid.entities) == 2

    # Query entities around (5, 5)
    res = grid.query_candidate_entities(Vec2(5.0, 5.0), 2.0)
    assert len(res) == 1
    assert res[0].entity_id == cone1.entity_id

    # Query entities around (85, 85)
    res2 = grid.query_candidate_entities(Vec2(85.0, 85.0), 2.0)
    assert len(res2) == 1
    assert res2[0].entity_id == cone2.entity_id

    # Query empty region
    res3 = grid.query_candidate_entities(Vec2(40.0, 40.0), 5.0)
    assert len(res3) == 0


def test_spatial_hash_zero_false_negatives():
    grid = SpatialHashGrid2D(cell_size=8.0)

    # Place 100 segments along a circle of radius 50
    segments = []
    num_pts = 100
    for i in range(num_pts):
        angle1 = 2 * math.pi * i / num_pts
        angle2 = 2 * math.pi * (i + 1) / num_pts
        p1 = Vec2(50.0 * math.cos(angle1), 50.0 * math.sin(angle1))
        p2 = Vec2(50.0 * math.cos(angle2), 50.0 * math.sin(angle2))
        segments.append((p1, p2))

    grid.build_from_segments(segments)

    # Test queries at various points along the circle
    for i in range(0, num_pts, 5):
        qx = 50.0 * math.cos(2 * math.pi * i / num_pts)
        qy = 50.0 * math.sin(2 * math.pi * i / num_pts)
        query_center = Vec2(qx, qy)
        radius = 5.0

        broad_candidates = set(grid.query_candidate_segment_indices(query_center, radius))

        # Brute force: find all segments whose AABB intersects query circle AABB
        min_x, max_x = qx - radius, qx + radius
        min_y, max_y = qy - radius, qy + radius
        brute_candidates = set()
        for idx, (p1, p2) in enumerate(segments):
            s_min_x = min(p1.x, p2.x)
            s_max_x = max(p1.x, p2.x)
            s_min_y = min(p1.y, p2.y)
            s_max_y = max(p1.y, p2.y)
            if not (max_x < s_min_x or min_x > s_max_x or max_y < s_min_y or min_y > s_max_y):
                brute_candidates.add(idx)

        # Broadphase candidates MUST be a superset of brute_candidates (NO false negatives)
        assert brute_candidates.issubset(broad_candidates), (
            f"False negative detected! Missing: {brute_candidates - broad_candidates}"
        )


def test_spatial_hash_memory_footprint():
    grid = SpatialHashGrid2D(cell_size=12.0)
    segs = [(Vec2(float(i), float(i)), Vec2(float(i + 1), float(i + 1))) for i in range(50)]
    grid.build_from_segments(segs)

    footprint = grid.get_memory_footprint()
    assert footprint['cell_count'] > 0
    assert footprint['indexed_segments'] == 50
    assert footprint['cell_size'] == 12.0
    assert footprint['approx_memory_bytes'] > 0
