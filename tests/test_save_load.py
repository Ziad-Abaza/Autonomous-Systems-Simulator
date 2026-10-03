"""
Test save and load persistence of environments and presets.
"""

import os
import tempfile
import pytest
from sim_project import (
    EnvironmentProject,
    create_serpentine_track,
    create_oval_circuit,
    create_obstacle_challenge,
)


def test_save_and_load_roundtrip():
    original = create_serpentine_track()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_track.sim.json")
        original.save(filepath)
        assert os.path.exists(filepath)

        loaded = EnvironmentProject.load(filepath)
        assert loaded.name == original.name
        assert len(loaded.road_def.control_points) == len(original.road_def.control_points)
        assert loaded.vehicle_config.mass == original.vehicle_config.mass
        assert loaded.reward_config.collision_penalty == original.reward_config.collision_penalty


def test_auto_version_increment_on_change():
    proj = create_oval_circuit()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "versioned.sim.json")
        proj.save(filepath)
        assert proj.environment_version == "1.0.0"

        # Load back
        loaded = EnvironmentProject.load(filepath)
        assert loaded.environment_version == "1.0.0"

        # Save without changes -> version stays 1.0.0
        loaded.save(filepath)
        assert loaded.environment_version == "1.0.0"

        # Modify configuration -> version increments to 1.0.1
        loaded.vehicle_config.mass = 1600.0
        loaded.save(filepath)
        assert loaded.environment_version == "1.0.1"

