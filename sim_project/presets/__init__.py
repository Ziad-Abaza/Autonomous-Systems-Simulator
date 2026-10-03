"""
Built-in preset environments for training and testing.
"""

from __future__ import annotations
import math
from typing import Dict
from sim_project.serializer import EnvironmentProject
from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint, RoadBoundaryConfig
from sim_core.math_utils import Vec3


def create_oval_circuit() -> EnvironmentProject:
    proj = EnvironmentProject(name="Proving Ground Oval")
    proj.road_def = RoadDefinition.create_default_oval(radius_x=65.0, radius_y=40.0, width=12.0)
    return proj


def create_serpentine_track() -> EnvironmentProject:
    """Technical circuit with elevation and variable road width."""
    road = RoadDefinition(name="Alpine Serpentine Circuit", is_closed=True)
    # Technical winding track with S-curves and variable widths
    pts = [
        (0.0, -50.0, 0.0, 14.0),
        (50.0, -45.0, 2.0, 12.0),
        (80.0, -10.0, 4.0, 10.0),    # Narrowing crest
        (65.0, 30.0, 3.0, 11.0),
        (25.0, 15.0, 1.5, 12.0),     # Chicane
        (0.0, 50.0, 0.0, 13.0),
        (-35.0, 60.0, -1.0, 14.0),
        (-70.0, 35.0, -2.0, 12.0),   # Dip
        (-75.0, -15.0, -1.0, 11.0),
        (-40.0, -40.0, 0.0, 13.0),
    ]
    for x, y, z, w in pts:
        road.add_control_point(x=x, y=y, z=z, width=w, banking=0.0)

    road.spawn_point = SpawnPoint(x=0.0, y=-50.0, z=0.2, yaw=0.0, initial_speed=0.0)
    road.num_checkpoints = 20

    proj = EnvironmentProject(name="Alpine Serpentine Circuit", road_def=road)
    return proj


def create_obstacle_challenge() -> EnvironmentProject:
    """Track with obstacles configured in scenario."""
    proj = create_oval_circuit()
    proj.name = "Obstacle Evasion Proving Ground"
    # Place obstacles in scenario
    proj.scenario_config.obstacles = [
        {'name': 'Cone Cluster 1', 'obstacle_type': 'cone', 'pos': [45.0, 15.0, 0.0], 'yaw': 0.0, 'length': 0.8, 'width': 0.8, 'height': 0.8},
        {'name': 'Concrete Barrier', 'obstacle_type': 'barricade', 'pos': [-40.0, 10.0, 0.0], 'yaw': 0.3, 'length': 2.5, 'width': 0.8, 'height': 1.0},
        {'name': 'Barrel Stack', 'obstacle_type': 'barrel', 'pos': [0.0, 40.0, 0.0], 'yaw': 0.0, 'length': 1.2, 'width': 1.2, 'height': 1.0},
    ]
    return proj


def save_default_presets(output_dir: str) -> None:
    presets = {
        'oval_circuit.sim.json': create_oval_circuit(),
        'serpentine_track.sim.json': create_serpentine_track(),
        'obstacle_challenge.sim.json': create_obstacle_challenge(),
    }
    import os
    os.makedirs(output_dir, exist_ok=True)
    for filename, p in presets.items():
        path = os.path.join(output_dir, filename)
        p.save(path)
