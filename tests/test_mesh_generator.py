"""
Unit tests for procedural 3D mesh generation and collision boundary extraction.
"""

import pytest
import numpy as np
from sim_core.track.road_definition import RoadDefinition
from sim_core.track.mesh_generator import TrackMeshGenerator


def test_track_mesh_generation():
    road = RoadDefinition.create_default_oval(radius_x=50.0, radius_y=30.0, width=12.0)
    track = TrackMeshGenerator.generate(road, sample_step=2.0)

    # Road mesh
    assert len(track.road_vertices) > 0
    assert len(track.road_normals) == len(track.road_vertices)
    assert len(track.road_uvs) == len(track.road_vertices)
    assert len(track.road_indices) > 0
    assert track.road_indices.shape[1] == 3  # Triangles

    # Curbs
    assert len(track.curb_vertices) > 0
    assert len(track.curb_colors) == len(track.curb_vertices)

    # Barriers
    assert len(track.barrier_vertices) > 0
    assert len(track.barrier_indices) > 0

    # Collision boundaries
    assert len(track.left_boundary_segments) > 0
    assert len(track.right_boundary_segments) > 0
    assert len(track.all_boundary_segments) == len(track.left_boundary_segments) + len(track.right_boundary_segments)

    # Checkpoints
    assert len(track.checkpoints) >= 4
    for cp in track.checkpoints:
        assert 'gate_left' in cp
        assert 'gate_right' in cp
        assert 's' in cp
