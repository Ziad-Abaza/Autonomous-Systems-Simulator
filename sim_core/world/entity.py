"""
Semantic world entities model for the AI Simulation Studio.
Provides identity, type classification, collision geometry, and editable properties for:
- StaticObstacle (box, barrel, crate, rock)
- Barrier (concrete jersey barrier, guardrail, tire wall)
- TrafficCone (delineator cone)
- TrafficSign (speed limit, stop, yield, turn signs)
- TrafficLight (tri-color state machine)
- CheckpointEntity (racing gate)
- SpawnEntity (vehicle start location)
"""

from __future__ import annotations
import math
import uuid
from typing import Dict, Any, Optional, List, Tuple
from sim_core.math_utils import Vec2, Vec3, OBB2D


class WorldEntity:
    """
    Base world entity with semantic identity for AI perception, segmentation, and tracking.
    """
    def __init__(
        self,
        name: str,
        entity_type: str,
        semantic_label: str,
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        is_collidable: bool = True,
        entity_id: Optional[str] = None
    ):
        self.entity_id = entity_id or str(uuid.uuid4())[:8]
        self.name = name
        self.entity_type = entity_type        # e.g., "obstacle", "barrier", "cone", "traffic_sign", "traffic_light"
        self.semantic_label = semantic_label  # e.g., "hazard", "infrastructure", "traffic_control"
        self.pos = pos if pos is not None else Vec3(0.0, 0.0, 0.0)
        self.yaw = float(yaw)
        self.is_collidable = is_collidable

    def get_obb(self) -> Optional[OBB2D]:
        """Returns 2D Oriented Bounding Box for collision detection if collidable."""
        return None

    def get_boundary_segments(self) -> List[Tuple[Vec2, Vec2]]:
        """Returns boundary segments for collision & raycasting."""
        obb = self.get_obb()
        if obb is None:
            return []
        corners = obb.get_corners()
        return [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]

    def update(self, dt: float) -> None:
        """Optional per-frame update for dynamic entities (e.g. traffic lights)."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type,
            'semantic_label': self.semantic_label,
            'pos': [self.pos.x, self.pos.y, self.pos.z],
            'yaw': self.yaw,
            'is_collidable': self.is_collidable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldEntity:
        return entity_from_dict(data)


class StaticObstacle(WorldEntity):
    """
    Physical obstacle (box, barrel, crate, rock) with configurable 3D dimensions.
    """
    def __init__(
        self,
        name: str = "Obstacle",
        obstacle_type: str = "box",  # "box", "barrel", "crate", "rock"
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        length: float = 2.0,
        width: float = 1.0,
        height: float = 1.0,
        is_collidable: bool = True,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="obstacle",
            semantic_label="hazard",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=is_collidable,
            entity_id=entity_id
        )
        self.obstacle_type = obstacle_type
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)

    def get_obb(self) -> OBB2D:
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=max(0.1, self.length * 0.5),
            half_width=max(0.1, self.width * 0.5),
            yaw=self.yaw
        )

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
    def from_dict(cls, data: Dict[str, Any]) -> StaticObstacle:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Obstacle')),
            obstacle_type=str(data.get('obstacle_type', 'box')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            length=float(data.get('length', 2.0)),
            width=float(data.get('width', 1.0)),
            height=float(data.get('height', 1.0)),
            is_collidable=bool(data.get('is_collidable', True)),
            entity_id=data.get('entity_id')
        )


class Barrier(WorldEntity):
    """
    Modular barrier entity (concrete jersey barrier, guardrail, tire wall).
    """
    def __init__(
        self,
        name: str = "Concrete Barrier",
        barrier_type: str = "concrete",  # "concrete", "guardrail", "tire_wall"
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        length: float = 3.0,
        width: float = 0.6,
        height: float = 0.9,
        is_collidable: bool = True,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="barrier",
            semantic_label="infrastructure",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=is_collidable,
            entity_id=entity_id
        )
        self.barrier_type = barrier_type
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)

    def get_obb(self) -> OBB2D:
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=max(0.1, self.length * 0.5),
            half_width=max(0.1, self.width * 0.5),
            yaw=self.yaw
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'barrier_type': self.barrier_type,
            'length': self.length,
            'width': self.width,
            'height': self.height,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Barrier:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Concrete Barrier')),
            barrier_type=str(data.get('barrier_type', 'concrete')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            length=float(data.get('length', 3.0)),
            width=float(data.get('width', 0.6)),
            height=float(data.get('height', 0.9)),
            is_collidable=bool(data.get('is_collidable', True)),
            entity_id=data.get('entity_id')
        )


class TrafficCone(WorldEntity):
    """
    Traffic cone delineator entity.
    """
    def __init__(
        self,
        name: str = "Traffic Cone",
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        radius: float = 0.3,
        height: float = 0.75,
        is_collidable: bool = True,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="cone",
            semantic_label="hazard",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=is_collidable,
            entity_id=entity_id
        )
        self.radius = float(radius)
        self.height = float(height)

    @property
    def length(self) -> float:
        return self.radius * 2.0

    @property
    def width(self) -> float:
        return self.radius * 2.0

    def get_obb(self) -> OBB2D:
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=max(0.05, self.radius),
            half_width=max(0.05, self.radius),
            yaw=self.yaw
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'radius': self.radius,
            'height': self.height,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficCone:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Traffic Cone')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            radius=float(data.get('radius', 0.3)),
            height=float(data.get('height', 0.75)),
            is_collidable=bool(data.get('is_collidable', True)),
            entity_id=data.get('entity_id')
        )


class TrafficSign(WorldEntity):
    """
    Traffic sign entity (Stop, Speed Limit, Yield, Curve Warning).
    """
    SIGN_TYPES = [
        "stop",
        "yield",
        "speed_30",
        "speed_50",
        "speed_80",
        "turn_left",
        "turn_right",
        "hazard_ahead"
    ]

    def __init__(
        self,
        name: str = "Traffic Sign",
        sign_type: str = "stop",
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        width: float = 0.8,
        height: float = 2.2,
        is_collidable: bool = False,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="traffic_sign",
            semantic_label="traffic_control",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=is_collidable,
            entity_id=entity_id
        )
        self.sign_type = sign_type
        self.width = float(width)
        self.height = float(height)

    @property
    def length(self) -> float:
        return 0.2

    def get_obb(self) -> Optional[OBB2D]:
        if not self.is_collidable:
            return None
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=0.1,
            half_width=max(0.1, self.width * 0.5),
            yaw=self.yaw
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'sign_type': self.sign_type,
            'width': self.width,
            'height': self.height,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficSign:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Traffic Sign')),
            sign_type=str(data.get('sign_type', 'stop')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            width=float(data.get('width', 0.8)),
            height=float(data.get('height', 2.2)),
            is_collidable=bool(data.get('is_collidable', False)),
            entity_id=data.get('entity_id')
        )


class TrafficLight(WorldEntity):
    """
    Traffic signal entity with cycling state machine (Green -> Yellow -> Red -> Green).
    """
    STATES = ["green", "yellow", "red"]

    def __init__(
        self,
        name: str = "Traffic Light",
        state: str = "green",
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        green_duration: float = 12.0,
        yellow_duration: float = 3.0,
        red_duration: float = 10.0,
        width: float = 0.6,
        height: float = 3.2,
        is_collidable: bool = False,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="traffic_light",
            semantic_label="traffic_control",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=is_collidable,
            entity_id=entity_id
        )
        self.state = state if state in self.STATES else "green"
        self.green_duration = float(green_duration)
        self.yellow_duration = float(yellow_duration)
        self.red_duration = float(red_duration)
        self.timer = 0.0
        self.width = float(width)
        self.height = float(height)

    @property
    def length(self) -> float:
        return 0.4

    def update(self, dt: float) -> None:
        """Advances traffic light state machine."""
        self.timer += dt
        if self.state == "green" and self.timer >= self.green_duration:
            self.state = "yellow"
            self.timer = 0.0
        elif self.state == "yellow" and self.timer >= self.yellow_duration:
            self.state = "red"
            self.timer = 0.0
        elif self.state == "red" and self.timer >= self.red_duration:
            self.state = "green"
            self.timer = 0.0

    def get_obb(self) -> Optional[OBB2D]:
        if not self.is_collidable:
            return None
        return OBB2D(
            center=Vec2(self.pos.x, self.pos.y),
            half_length=0.2,
            half_width=max(0.1, self.width * 0.5),
            yaw=self.yaw
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'state': self.state,
            'green_duration': self.green_duration,
            'yellow_duration': self.yellow_duration,
            'red_duration': self.red_duration,
            'timer': self.timer,
            'width': self.width,
            'height': self.height,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficLight:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        tl = cls(
            name=str(data.get('name', 'Traffic Light')),
            state=str(data.get('state', 'green')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            green_duration=float(data.get('green_duration', 12.0)),
            yellow_duration=float(data.get('yellow_duration', 3.0)),
            red_duration=float(data.get('red_duration', 10.0)),
            width=float(data.get('width', 0.6)),
            height=float(data.get('height', 3.2)),
            is_collidable=bool(data.get('is_collidable', False)),
            entity_id=data.get('entity_id')
        )
        tl.timer = float(data.get('timer', 0.0))
        return tl


class CheckpointEntity(WorldEntity):
    """
    Checkpoint waypoint gate placed in the environment.
    """
    def __init__(
        self,
        name: str = "Checkpoint Gate",
        index: int = 0,
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        gate_width: float = 12.0,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="checkpoint",
            semantic_label="waypoint",
            pos=pos or Vec3(0.0, 0.0, 0.0),
            yaw=yaw,
            is_collidable=False,
            entity_id=entity_id
        )
        self.index = int(index)
        self.gate_width = float(gate_width)

    def get_gate_endpoints(self) -> Tuple[Vec2, Vec2]:
        """Returns 2D endpoints of checkpoint gate line."""
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        norm_x = -sin_y
        norm_y = cos_y
        half_w = self.gate_width * 0.5
        left = Vec2(self.pos.x + norm_x * half_w, self.pos.y + norm_y * half_w)
        right = Vec2(self.pos.x - norm_x * half_w, self.pos.y - norm_y * half_w)
        return left, right

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'index': self.index,
            'gate_width': self.gate_width,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointEntity:
        pos_raw = data.get('pos', [0.0, 0.0, 0.0])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Checkpoint Gate')),
            index=int(data.get('index', 0)),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            gate_width=float(data.get('gate_width', 12.0)),
            entity_id=data.get('entity_id')
        )


class SpawnEntity(WorldEntity):
    """
    Vehicle spawn location entity.
    """
    def __init__(
        self,
        name: str = "Spawn Point",
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        initial_speed: float = 0.0,
        entity_id: Optional[str] = None
    ):
        super().__init__(
            name=name,
            entity_type="spawn",
            semantic_label="spawn",
            pos=pos or Vec3(0.0, 0.0, 0.2),
            yaw=yaw,
            is_collidable=False,
            entity_id=entity_id
        )
        self.initial_speed = float(initial_speed)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'initial_speed': self.initial_speed,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpawnEntity:
        pos_raw = data.get('pos', [0.0, 0.0, 0.2])
        pos = Vec3(float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        return cls(
            name=str(data.get('name', 'Spawn Point')),
            pos=pos,
            yaw=float(data.get('yaw', 0.0)),
            initial_speed=float(data.get('initial_speed', 0.0)),
            entity_id=data.get('entity_id')
        )


# Factory registry for dynamic entity instantiation
ENTITY_CLASS_MAP = {
    'obstacle': StaticObstacle,
    'static_obstacle': StaticObstacle,
    'barrier': Barrier,
    'cone': TrafficCone,
    'traffic_cone': TrafficCone,
    'traffic_sign': TrafficSign,
    'sign': TrafficSign,
    'traffic_light': TrafficLight,
    'checkpoint': CheckpointEntity,
    'spawn': SpawnEntity,
}


def create_entity(entity_type: str, name: Optional[str] = None, pos: Optional[Vec3] = None, yaw: float = 0.0, **kwargs) -> WorldEntity:
    """Factory helper to construct any entity by type name."""
    cls = ENTITY_CLASS_MAP.get(entity_type.lower())
    if cls is None:
        raise ValueError(f"Unknown entity type: '{entity_type}'. Supported: {list(ENTITY_CLASS_MAP.keys())}")
    inst_name = name or entity_type.replace('_', ' ').title()
    return cls(name=inst_name, pos=pos, yaw=yaw, **kwargs)


def entity_from_dict(data: Dict[str, Any]) -> WorldEntity:
    """Deserializes dictionary representation into appropriate WorldEntity subclass."""
    ent_type = str(data.get('entity_type', data.get('type', 'obstacle'))).lower()
    cls = ENTITY_CLASS_MAP.get(ent_type, StaticObstacle)
    return cls.from_dict(data)
