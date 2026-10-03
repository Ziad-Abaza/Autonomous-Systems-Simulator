"""
2D Spatial Hash Grid broadphase acceleration structure for collision detection
and spatial queries against track boundaries, obstacles, and world entities.
Guarantees zero false negatives while reducing collision check complexity from O(N) to O(1) expected.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Set, Any, Optional
from sim_core.math_utils import Vec2, OBB2D


class SpatialHashGrid2D:
    """
    Uniform 2D spatial hash grid partitioning continuous 2D space into discrete buckets.
    Accelerates segment collision queries, entity intersections, and raycasting.
    """
    def __init__(self, cell_size: float = 12.0):
        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size
        # Map: (cell_x, cell_y) -> List[int] (indices into self.segments)
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        # Indexed segments: [(Vec2(start), Vec2(end)), ...]
        self.segments: List[Tuple[Vec2, Vec2]] = []
        # Entities grid: (cell_x, cell_y) -> List[Any]
        self.entity_grid: Dict[Tuple[int, int], List[Any]] = {}
        self.entities: List[Any] = []

    def clear(self) -> None:
        """Clears all indexed geometry and entities."""
        self.grid.clear()
        self.segments.clear()
        self.entity_grid.clear()
        self.entities.clear()

    def build_from_segments(self, segments: List[Tuple[Vec2, Vec2]]) -> None:
        """
        Populates grid with static boundary line segments.
        Rebuilds the entire broadphase in milliseconds.
        """
        self.grid.clear()
        self.segments = list(segments)
        for idx, (p1, p2) in enumerate(self.segments):
            self._insert_segment(idx, p1, p2)

    def _coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor(x * self.inv_cell_size)), int(math.floor(y * self.inv_cell_size))

    def _insert_segment(self, idx: int, p1: Vec2, p2: Vec2) -> None:
        min_x = p1.x if p1.x < p2.x else p2.x
        max_x = p1.x if p1.x > p2.x else p2.x
        min_y = p1.y if p1.y < p2.y else p2.y
        max_y = p1.y if p1.y > p2.y else p2.y

        cx0 = int(math.floor(min_x * self.inv_cell_size))
        cx1 = int(math.floor(max_x * self.inv_cell_size))
        cy0 = int(math.floor(min_y * self.inv_cell_size))
        cy1 = int(math.floor(max_y * self.inv_cell_size))

        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = (cx, cy)
                cell = self.grid.get(key)
                if cell is None:
                    self.grid[key] = [idx]
                else:
                    cell.append(idx)

    def insert_entity(self, entity: Any) -> None:
        """Registers a dynamic or static entity into the entity broadphase."""
        self.entities.append(entity)
        obb = entity.get_obb() if hasattr(entity, 'get_obb') else None
        if obb is not None:
            c = obb.center
            r = max(obb.half_length, obb.half_width) + 0.5
            cx0 = int(math.floor((c.x - r) * self.inv_cell_size))
            cx1 = int(math.floor((c.x + r) * self.inv_cell_size))
            cy0 = int(math.floor((c.y - r) * self.inv_cell_size))
            cy1 = int(math.floor((c.y + r) * self.inv_cell_size))
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    key = (cx, cy)
                    cell = self.entity_grid.get(key)
                    if cell is None:
                        self.entity_grid[key] = [entity]
                    else:
                        cell.append(entity)
        else:
            # Point-based entity
            pos = getattr(entity, 'pos', None)
            if pos is not None:
                cell_key = self._coord_to_cell(pos.x, pos.y)
                cell = self.entity_grid.get(cell_key)
                if cell is None:
                    self.entity_grid[cell_key] = [entity]
                else:
                    cell.append(entity)

    def query_candidate_segment_indices(self, center: Vec2, radius: float) -> List[int]:
        """
        Returns list of segment indices in cells overlapping circle (center, radius).
        Guarantees no false negatives for any geometry within the circle.
        """
        cx0 = int(math.floor((center.x - radius) * self.inv_cell_size))
        cx1 = int(math.floor((center.x + radius) * self.inv_cell_size))
        cy0 = int(math.floor((center.y - radius) * self.inv_cell_size))
        cy1 = int(math.floor((center.y + radius) * self.inv_cell_size))

        candidates: Set[int] = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                cell = self.grid.get((cx, cy))
                if cell is not None:
                    candidates.update(cell)
        return list(candidates)

    def query_candidate_segments(self, center: Vec2, radius: float) -> List[Tuple[Vec2, Vec2]]:
        """Returns candidate segments directly for narrowphase testing."""
        indices = self.query_candidate_segment_indices(center, radius)
        return [self.segments[i] for i in indices]

    def query_candidate_entities(self, center: Vec2, radius: float) -> List[Any]:
        """Returns candidate entities in cells overlapping circle (center, radius)."""
        cx0 = int(math.floor((center.x - radius) * self.inv_cell_size))
        cx1 = int(math.floor((center.x + radius) * self.inv_cell_size))
        cy0 = int(math.floor((center.y - radius) * self.inv_cell_size))
        cy1 = int(math.floor((center.y + radius) * self.inv_cell_size))

        seen = set()
        candidates = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                cell = self.entity_grid.get((cx, cy))
                if cell is not None:
                    for ent in cell:
                        eid = getattr(ent, 'entity_id', id(ent))
                        if eid not in seen:
                            seen.add(eid)
                            candidates.append(ent)
        return candidates

    def get_memory_footprint(self) -> Dict[str, Any]:
        """Returns diagnostic memory statistics of the spatial grid."""
        import sys
        cell_count = len(self.grid)
        total_refs = sum(len(v) for v in self.grid.values())
        approx_bytes = sys.getsizeof(self.grid) + sys.getsizeof(self.segments)
        return {
            'cell_count': cell_count,
            'total_segment_refs': total_refs,
            'indexed_segments': len(self.segments),
            'approx_memory_bytes': approx_bytes,
            'cell_size': self.cell_size
        }
