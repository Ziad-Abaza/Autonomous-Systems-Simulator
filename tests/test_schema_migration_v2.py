"""
Unit tests for Project Schema Migration and Persistence.
Tests loading legacy Schema 1.0.0 project files, verifying automatic migration
to Schema 2.0.0, and verifying Schema 2.0.0 round-trip serialization.
"""

import os
import json
import tempfile
import pytest
from sim_project.serializer import EnvironmentProject, SCHEMA_VERSION
from sim_core.math_utils import Vec3
from sim_core.world.entity import (
    StaticObstacle,
    Barrier,
    TrafficCone,
    TrafficSign,
    TrafficLight,
    create_entity,
)
from sim_project.presets import create_oval_circuit


def test_schema_1_to_2_migration():
    # Construct synthetic Schema 1.0.0 payload
    legacy_data = {
        "schema_version": "1.0.0",
        "name": "Legacy Project",
        "author": "Phase 1",
        "road_definition": {
            "control_points": [
                {"x": 0.0, "y": 0.0, "width": 10.0, "elevation": 0.0, "banking": 0.0},
                {"x": 50.0, "y": 0.0, "width": 10.0, "elevation": 0.0, "banking": 0.0},
                {"x": 50.0, "y": 50.0, "width": 10.0, "elevation": 0.0, "banking": 0.0},
                {"x": 0.0, "y": 50.0, "width": 10.0, "elevation": 0.0, "banking": 0.0},
            ],
            "is_closed": True,
            "resolution": 200,
        },
        "obstacles": [
            {
                "x": 25.0,
                "y": 10.0,
                "radius": 2.0,
                "height": 1.5,
                "color": [220, 40, 40],
                "is_sensor_visible": True,
            }
        ],
        "vehicle_config": {"mass": 1200.0},
        "reward_config": {"weight_progress": 1.0},
        "termination_config": {"max_episode_steps": 500},
        "observation_schema": {"include_speed": True},
        "action_space_config": {"steering_limit": 0.5},
    }

    # Load from dict
    project = EnvironmentProject.from_dict(legacy_data)

    # Verify migration
    assert project.schema_version == SCHEMA_VERSION  # 2.0.0
    assert len(project.entities) == 1
    entity = project.entities[0]
    assert isinstance(entity, StaticObstacle)
    assert entity.pos.x == 25.0
    assert entity.pos.y == 10.0

    # Verify backward-compatible obstacles property
    assert len(project.obstacles) == 1
    assert project.obstacles[0].pos.x == 25.0


def test_schema_2_roundtrip_persistence():
    project = create_oval_circuit()

    # Add diverse Phase 2 entities
    cone = create_entity("cone", pos=Vec3(10.0, 15.0, 0.0))
    barrier = create_entity("barrier", pos=Vec3(20.0, 25.0, 0.0), length=5.0)
    sign = create_entity("traffic_sign", pos=Vec3(30.0, 35.0, 0.0), sign_type="turn_left")
    tl = create_entity("traffic_light", pos=Vec3(40.0, 45.0, 0.0), state="red")

    project.entities.extend([cone, barrier, sign, tl])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_v2_project.sim.json")
        project.save(filepath)

        # Inspect raw JSON file to confirm version and entities
        with open(filepath, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        assert raw_json["schema_version"] == "2.0.0"
        assert "entities" in raw_json
        assert len(raw_json["entities"]) == len(project.entities)

        # Reload project
        loaded = EnvironmentProject.load(filepath)
        assert loaded.schema_version == "2.0.0"
        assert len(loaded.entities) == len(project.entities)

        types_found = {e.entity_type for e in loaded.entities}
        assert "cone" in types_found
        assert "barrier" in types_found
        assert "traffic_sign" in types_found
        assert "traffic_light" in types_found
