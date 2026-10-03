"""
Semantic world entities base class.
Provides identity, type classification, and semantic segmentation labeling.
"""

from __future__ import annotations
import uuid
from typing import Dict, Any, Optional
from sim_core.math_utils import Vec2, Vec3


class WorldEntity:
    """
    Base entity with semantic identity for AI perception, segmentation, and tracking.
    """
    def __init__(
        self,
        name: str,
        entity_type: str,
        semantic_label: str,
        pos: Optional[Vec3] = None,
        yaw: float = 0.0,
        entity_id: Optional[str] = None
    ):
        self.entity_id = entity_id or str(uuid.uuid4())[:8]
        self.name = name
        self.entity_type = entity_type        # e.g., "obstacle", "vehicle", "traffic_light"
        self.semantic_label = semantic_label  # e.g., "hazard", "vehicle", "traffic_control"
        self.pos = pos or Vec3(0.0, 0.0, 0.0)
        self.yaw = float(yaw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'type': self.entity_type,
            'semantic_label': self.semantic_label,
            'pos': self.pos.to_tuple(),
            'yaw': self.yaw,
        }
