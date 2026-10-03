"""
Static obstacles (traffic cones, concrete barriers, barrels) placed in the environment.
Preserved for backward compatibility, inheriting from StaticObstacle.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from sim_core.math_utils import Vec2, Vec3, OBB2D
from sim_core.world.entity import StaticObstacle, WorldEntity


class Obstacle(StaticObstacle):
    """
    Physical obstacle with collision geometry and semantic category.
    Fully backwards-compatible with Phase 1 obstacle API.
    """
    def __init__(
        self,
        name: str = "Barricade",
        obstacle_type: str = "barricade",  # "barricade", "cone", "barrel", "box"
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        length: float = 2.0,
        width: float = 0.8,
        height: float = 1.0,
        entity_id: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        radius: Optional[float] = None,
        color: Optional[Tuple[int, int, int]] = None,
        is_sensor_visible: bool = True
    ):
        # Support legacy x, y, radius parameters
        if pos is None and (x is not None or y is not None):
            pos = Vec3(float(x or 0.0), float(y or 0.0), 0.0)
        if radius is not None:
            length = radius * 2.0
            width = radius * 2.0

        super().__init__(
            name=name,
            obstacle_type=obstacle_type,
            pos=pos or Vec3(0, 0, 0),
            yaw=yaw,
            length=length,
            width=width,
            height=height,
            is_collidable=True,
            entity_id=entity_id
        )
        self.color = color or (220, 50, 50)
        self.is_sensor_visible = is_sensor_visible

    @property
    def x(self) -> float:
        return self.pos.x

    @x.setter
    def x(self, val: float) -> None:
        self.pos.x = float(val)

    @property
    def y(self) -> float:
        return self.pos.y

    @y.setter
    def y(self, val: float) -> None:
        self.pos.y = float(val)

    @property
    def radius(self) -> float:
        return self.length * 0.5

    @radius.setter
    def radius(self, val: float) -> None:
        self.length = float(val) * 2.0
        self.width = float(val) * 2.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'radius': self.radius,
            'color': list(self.color),
            'is_sensor_visible': self.is_sensor_visible,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Obstacle:
        pos_tuple = data.get('pos')
        if pos_tuple is not None:
            pos = Vec3(pos_tuple[0], pos_tuple[1], pos_tuple[2])
        else:
            pos = Vec3(float(data.get('x', 0.0)), float(data.get('y', 0.0)), 0.0)

        rad = float(data.get('radius', 1.0))
        return cls(
            name=str(data.get('name', 'Barricade')),
            obstacle_type=str(data.get('obstacle_type', 'barricade')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            length=float(data.get('length', rad * 2.0)),
            width=float(data.get('width', rad * 2.0)),
            height=float(data.get('height', 1.0)),
            entity_id=data.get('entity_id'),
            color=tuple(data.get('color', [220, 50, 50])),
            is_sensor_visible=bool(data.get('is_sensor_visible', True))
        )
