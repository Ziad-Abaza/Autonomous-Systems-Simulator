"""
Project persistence and presets package.
"""

from sim_project.serializer import EnvironmentProject, SCHEMA_VERSION
from sim_project.presets import (
    create_oval_circuit,
    create_serpentine_track,
    create_obstacle_challenge,
    save_default_presets,
)

__all__ = [
    'EnvironmentProject',
    'SCHEMA_VERSION',
    'create_oval_circuit',
    'create_serpentine_track',
    'create_obstacle_challenge',
    'save_default_presets',
]
