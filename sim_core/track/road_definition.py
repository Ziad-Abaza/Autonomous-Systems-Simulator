"""
Logical road definition containing semantic control points, boundary configurations,
checkpoints, spawn locations, and surface parameters.
Decoupled from renderable mesh geometry.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from sim_core.math_utils import Vec2, Vec3


@dataclass
class ControlPoint:
    x: float
    y: float
    z: float = 0.0
    width: float = 12.0
    banking: float = 0.0  # degrees; >0 = left edge raised (downhill toward -normal)
    friction: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ControlPoint:
        return cls(
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.0)),
            width=float(data.get('width', 12.0)),
            banking=float(data.get('banking', 0.0)),
            friction=float(data.get('friction', 1.0)),
        )


@dataclass
class RoadBoundaryConfig:
    left_type: str = "guardrail"   # "guardrail", "wall", "curb", "open", "invisible"
    right_type: str = "guardrail"
    wall_height: float = 0.8       # meters
    curb_width: float = 0.5        # meters
    curb_height: float = 0.15      # meters
    has_curbs: bool = True
    has_lane_markings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadBoundaryConfig:
        return cls(
            left_type=str(data.get('left_type', 'guardrail')),
            right_type=str(data.get('right_type', 'guardrail')),
            wall_height=float(data.get('wall_height', 0.8)),
            curb_width=float(data.get('curb_width', 0.5)),
            curb_height=float(data.get('curb_height', 0.15)),
            has_curbs=bool(data.get('has_curbs', True)),
            has_lane_markings=bool(data.get('has_lane_markings', True)),
        )


@dataclass
class SpawnPoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.2
    yaw: float = 0.0  # radians
    initial_speed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpawnPoint:
        return cls(
            x=float(data.get('x', 0.0)),
            y=float(data.get('y', 0.0)),
            z=float(data.get('z', 0.2)),
            yaw=float(data.get('yaw', 0.0)),
            initial_speed=float(data.get('initial_speed', 0.0)),
        )


class RoadDefinition:
    """
    Complete semantic road and track specification.
    """
    def __init__(self, name: str = "Default Track", is_closed: bool = True):
        self.name = name
        self.is_closed = is_closed
        self.control_points: List[ControlPoint] = []
        self.boundary_config = RoadBoundaryConfig()
        self.spawn_point = SpawnPoint()
        self.num_checkpoints: int = 16
        self.default_friction: float = 1.0
        # Surface friction multipliers applied off the asphalt region.
        # The drivable "road" region is |lateral| <= width/2; the curb band
        # extends curb_width further out; anything beyond is off-road.
        self.curb_friction: float = 0.85
        self.off_road_friction: float = 0.55

    def add_control_point(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        width: float = 12.0,
        banking: float = 0.0,
        friction: float = 1.0
    ) -> None:
        self.control_points.append(ControlPoint(x, y, z, width, banking, friction))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'is_closed': self.is_closed,
            'control_points': [cp.to_dict() for cp in self.control_points],
            'boundary_config': self.boundary_config.to_dict(),
            'spawn_point': self.spawn_point.to_dict(),
            'num_checkpoints': self.num_checkpoints,
            'default_friction': self.default_friction,
            'curb_friction': self.curb_friction,
            'off_road_friction': self.off_road_friction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadDefinition:
        road = cls(
            name=str(data.get('name', 'Untitled Track')),
            is_closed=bool(data.get('is_closed', True))
        )
        road.control_points = [
            ControlPoint.from_dict(cp) for cp in data.get('control_points', [])
        ]
        if 'boundary_config' in data:
            road.boundary_config = RoadBoundaryConfig.from_dict(data['boundary_config'])
        if 'spawn_point' in data:
            road.spawn_point = SpawnPoint.from_dict(data['spawn_point'])
        road.num_checkpoints = int(data.get('num_checkpoints', 16))
        road.default_friction = float(data.get('default_friction', 1.0))
        road.curb_friction = float(data.get('curb_friction', 0.85))
        road.off_road_friction = float(data.get('off_road_friction', 0.55))
        return road

    @classmethod
    def create_default_oval(cls, radius_x: float = 60.0, radius_y: float = 35.0, width: float = 12.0) -> RoadDefinition:
        """Helper to create a standard smooth test circuit."""
        road = cls(name="Proving Ground Circuit", is_closed=True)
        # 8 control points forming an oval
        import math
        points = [
            (radius_x, 0.0),
            (radius_x * 0.7, radius_y * 0.8),
            (0.0, radius_y),
            (-radius_x * 0.7, radius_y * 0.8),
            (-radius_x, 0.0),
            (-radius_x * 0.7, -radius_y * 0.8),
            (0.0, -radius_y),
            (radius_x * 0.7, -radius_y * 0.8),
        ]
        for x, y in points:
            road.add_control_point(x=x, y=y, z=0.0, width=width, banking=0.0)

        # Set spawn point at start line facing forward (+Y direction)
        road.spawn_point = SpawnPoint(x=radius_x, y=0.0, z=0.1, yaw=math.pi / 2.0)
        return road
