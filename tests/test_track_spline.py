"""
Unit tests for TrackSpline, arc-length parameterization, and projection queries.
"""

import math
import pytest
from sim_core.math_utils import Vec2, Vec3
from sim_core.track.spline import TrackSpline
from sim_core.track.road_definition import RoadDefinition


def test_spline_generation_and_arc_length():
    road = RoadDefinition.create_default_oval(radius_x=50.0, radius_y=30.0, width=10.0)
    spline = TrackSpline(is_closed=road.is_closed)
    spline.build_from_control_points([cp.to_dict() for cp in road.control_points], sample_step=1.0)

    # Arc length should approximate perimeter of ellipse ~ 2 * pi * sqrt((a^2 + b^2) / 2)
    # 2 * pi * sqrt((2500 + 900)/2) ~ 258 m
    assert spline.total_length > 200.0
    assert len(spline.samples) > 150

    # Ensure samples are monotonically increasing in s
    for i in range(1, len(spline.samples)):
        assert spline.samples[i].s >= spline.samples[i - 1].s


def test_spline_tangent_and_normal_orthogonality():
    road = RoadDefinition.create_default_oval()
    spline = TrackSpline(is_closed=True)
    spline.build_from_control_points([cp.to_dict() for cp in road.control_points])

    for s in spline.samples:
        t_2d = Vec2(s.tangent.x, s.tangent.y)
        n_2d = Vec2(s.normal.x, s.normal.y)
        # Dot product of tangent and left normal should be zero (orthogonal)
        assert abs(t_2d.dot(n_2d)) < 1e-4
        assert abs(t_2d.length() - 1.0) < 1e-3
        assert abs(n_2d.length() - 1.0) < 1e-3


def test_closest_point_projection():
    road = RoadDefinition.create_default_oval(radius_x=60.0, radius_y=35.0)
    spline = TrackSpline(is_closed=True)
    spline.build_from_control_points([cp.to_dict() for cp in road.control_points])

    # Test point on the track centerline at (60.0, 0.0)
    test_pt = Vec2(60.0, 0.0)
    s, lat_offset, tangent_angle, sp = spline.get_closest_point(test_pt)
    assert abs(lat_offset) < 1.0

    # Test point 3 meters to the left of the road
    norm_2d = Vec2(sp.normal.x, sp.normal.y)
    left_pt = test_pt + norm_2d * 3.0
    s_l, lat_l, _, _ = spline.get_closest_point(left_pt)
    assert abs(lat_l - 3.0) < 0.5
