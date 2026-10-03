"""
Static obstacles (traffic cones, concrete barriers, barrels) placed in the environment.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from sim_core.math_utils import Vec2, Vec3, OBB2D
from sim_core.world.entity import WorldEntity


class Obstacle(WorldEntity):
    """
    Physical obstacle with collision geometry and semantic category.
    """
    def __init__(
        self,
        name: str = "Barricade",
        obstacle_type: str = "barricade",  # "barricade", "cone", "barrel", "box"
        pos: Vec3 | None = None,
        yaw: float = 0.0,
        length: float = 2.0,
        width: float = 0.8,
        height: float = 1.0,
        entity_id: str | None = None
    ):
        super().__init__(
            name=name,
            entity_type="obstacle",
            semantic_label="hazard",
            pos=pos or Vec3(0, 0, 0),
            yaw=yaw,
            entity_id=entity_id
        )
        self.obstacle_type = obstacle_type
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)

    def get_obb(self) -> OBB2D:
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=self.length * 0.5,
            half_width=self.width * 0.5,
            yaw=self.yaw
        )

    def get_boundary_segments(self) -> List[Tuple[Vec2, Vec2]]:
        """Returns 4 boundary edges of the obstacle for raycasting and collision."""
        corners = self.get_obb().get_corners()
        return [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'obstacle_type': self.obstacle_type,
            'length': self.length,
            'width': self.width,
            'height': self.height,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Obstacle:
        pos_tuple = data.get('pos', [0.0, 0.0, 0.0])
        return cls(
            name=str(data.get('name', 'Barricade')),
            obstacle_type=str(data.get('obstacle_type', 'barricade')),
            pos=Vec3(pos_tuple[0], pos_tuple[1], pos_tuple[2]),
            yaw=float(data.get('yaw', 0.0)),
            length=float(data.get('length', 2.0)),
            width=float(data.get('width', 0.8)),
            height=float(data.get('height', 1.0)),
            entity_id=data.get('entity_id')
        )
