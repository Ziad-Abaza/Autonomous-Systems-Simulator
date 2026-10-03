"""
Unit and integration tests for Phase 2 Authoring Entities and World Model.
Tests entity creation, serialization/deserialization, OBBs, boundary segments,
and backward compatibility with legacy obstacles.
"""

import math
import pytest
from sim_core.math_utils import Vec2, Vec3, OBB2D
from sim_core.world.entity import (
    WorldEntity,
    StaticObstacle,
    Barrier,
    TrafficCone,
    TrafficSign,
    TrafficLight,
    CheckpointEntity,
    SpawnEntity,
    create_entity,
    entity_from_dict,
)
from sim_core.world.obstacle import Obstacle


def test_entity_creation_and_defaults():
    cone = create_entity("cone", pos=Vec3(10.0, 20.0, 0.0), yaw=0.5, radius=0.35)
    assert isinstance(cone, TrafficCone)
    assert cone.pos.x == 10.0
    assert cone.pos.y == 20.0
    assert cone.yaw == 0.5
    assert cone.radius == 0.35
    assert cone.is_collidable is True

    barrier = create_entity("barrier", pos=Vec3(5.0, 5.0, 0.0), length=4.0, barrier_type="concrete")
    assert isinstance(barrier, Barrier)
    assert barrier.barrier_type == "concrete"
    assert barrier.length == 4.0
    assert barrier.is_collidable is True

    sign = create_entity("traffic_sign", pos=Vec3(0.0, 0.0, 0.0), sign_type="speed_limit_50")
    assert isinstance(sign, TrafficSign)
    assert sign.sign_type == "speed_limit_50"

    tl = create_entity("traffic_light", pos=Vec3(15.0, 30.0, 0.0), state="green")
    assert isinstance(tl, TrafficLight)
    assert tl.state == "green"
    assert tl.green_duration > 0.0

    cp = create_entity("checkpoint", pos=Vec3(50.0, 0.0, 0.0), gate_width=14.0, index=3)
    assert isinstance(cp, CheckpointEntity)
    assert cp.gate_width == 14.0
    assert cp.index == 3
    assert cp.is_collidable is False

    sp = create_entity("spawn", pos=Vec3(0.0, 0.0, 0.2), initial_speed=15.0)
    assert isinstance(sp, SpawnEntity)
    assert sp.initial_speed == 15.0
    assert sp.is_collidable is False


def test_obb_and_boundary_computation():
    # Box obstacle
    obs = StaticObstacle(pos=Vec3(10.0, 20.0, 0.0), length=4.0, width=2.0, yaw=0.0)
    obb = obs.get_obb()
    assert obb is not None
    assert obb.center.x == 10.0
    assert obb.center.y == 20.0
    assert obb.half_length == 2.0
    assert obb.half_width == 1.0

    segs = obs.get_boundary_segments()
    assert len(segs) == 4
    # All corner coordinates should be within bounding box [8, 12] x [19, 21]
    for p1, p2 in segs:
        assert 8.0 <= p1.x <= 12.0
        assert 19.0 <= p1.y <= 21.0

    # Traffic light state machine update
    tl = TrafficLight(state="green", green_duration=5.0, yellow_duration=2.0, red_duration=5.0)
    tl.update(3.0)
    assert tl.state == "green"
    tl.update(2.5)  # total 5.5s -> transitions to yellow
    assert tl.state == "yellow"


def test_entity_serialization_roundtrip():
    entities = [
        StaticObstacle(name="Crate", obstacle_type="crate", pos=Vec3(1.0, 2.0, 0.0), length=1.5, width=1.5),
        Barrier(name="Wall", barrier_type="concrete", pos=Vec3(10.0, 20.0, 0.0), length=6.0, yaw=0.25),
        TrafficCone(name="Cone", pos=Vec3(3.0, 4.0, 0.0), radius=0.3),
        TrafficSign(name="Stop", sign_type="stop", pos=Vec3(5.0, 6.0, 0.0)),
        TrafficLight(name="Signal", state="red", pos=Vec3(7.0, 8.0, 0.0)),
        CheckpointEntity(name="CP", index=5, pos=Vec3(100.0, 0.0, 0.0), gate_width=16.0),
        SpawnEntity(name="Spawn", pos=Vec3(0.0, 0.0, 0.2), initial_speed=12.0),
    ]

    for entity in entities:
        d = entity.to_dict()
        assert "entity_id" in d
        assert "entity_type" in d
        assert "pos" in d

        restored = entity_from_dict(d)
        assert restored.__class__ == entity.__class__
        assert restored.entity_id == entity.entity_id
        assert restored.pos.x == entity.pos.x
        assert restored.pos.y == entity.pos.y
        assert restored.yaw == entity.yaw
        assert restored.is_collidable == entity.is_collidable


def test_legacy_obstacle_backward_compatibility():
    legacy = Obstacle(x=12.0, y=24.0, radius=1.8, height=1.2, color=(200, 50, 50), is_sensor_visible=True)
    assert isinstance(legacy, StaticObstacle)
    assert legacy.pos.x == 12.0
    assert legacy.pos.y == 24.0
    assert legacy.x == 12.0
    assert legacy.y == 24.0
    assert legacy.radius == 1.8
    assert legacy.height == 1.2
    assert legacy.color == (200, 50, 50)
    assert legacy.is_sensor_visible is True

    # Check to_dict maintains properties
    d = legacy.to_dict()
    assert d["radius"] == 1.8
    assert d["height"] == 1.2
