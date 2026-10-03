"""
Track and road geometry package.
"""

from sim_core.track.spline import TrackSpline, SplinePoint
from sim_core.track.road_definition import RoadDefinition, ControlPoint, RoadBoundaryConfig, SpawnPoint
from sim_core.track.mesh_generator import TrackMeshGenerator, GeneratedTrack
from sim_core.track.track_queries import TrackSpatialQueries

__all__ = [
    'TrackSpline',
    'SplinePoint',
    'RoadDefinition',
    'ControlPoint',
    'RoadBoundaryConfig',
    'SpawnPoint',
    'TrackMeshGenerator',
    'GeneratedTrack',
    'TrackSpatialQueries',
]
