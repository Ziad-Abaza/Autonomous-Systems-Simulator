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
