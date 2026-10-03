"""
Snapshot-based undo/redo history for the track editor.

Editor documents are small (a handful of control points + entities), so
full-state snapshots are cheap and far more robust than inverse-command
math — every mutation path (drags, inserts, deletes, placement,
inspector property changes) is covered uniformly.

State = (road_definition_dict, entities_list_of_dicts) as produced by
RoadDefinition.to_dict / WorldEntity.to_dict — pure data, no pygame.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

Snapshot = Tuple[Dict[str, Any], List[Dict[str, Any]]]


class EditHistory:
    """Linear undo stack with redo truncation and a size cap."""

    def __init__(self, limit: int = 64):
        self.limit = limit
        self._stack: List[Snapshot] = []
        self._idx: int = -1

    # ------------------------------------------------------------ stack

    def push(self, snapshot: Snapshot) -> None:
        """Record a state AFTER a mutation (or the initial state)."""
        snap = (copy.deepcopy(snapshot[0]), copy.deepcopy(snapshot[1]))
        if self._idx >= 0 and self._stack[self._idx] == snap:
            return  # no-op mutation, don't pollute the stack
        del self._stack[self._idx + 1:]  # truncate redo branch
        self._stack.append(snap)
        if len(self._stack) > self.limit:
            self._stack.pop(0)
        self._idx = len(self._stack) - 1

    def clear(self, initial: Optional[Snapshot] = None) -> None:
        self._stack = []
        self._idx = -1
        if initial is not None:
            self.push(initial)

    # ----------------------------------------------------------- access

    @property
    def can_undo(self) -> bool:
        return self._idx > 0

    @property
    def can_redo(self) -> bool:
        return self._idx < len(self._stack) - 1

    @property
    def size(self) -> int:
        return len(self._stack)

    def undo(self) -> Optional[Snapshot]:
        if not self.can_undo:
            return None
        self._idx -= 1
        return self._stack[self._idx]

    def redo(self) -> Optional[Snapshot]:
        if not self.can_redo:
            return None
        self._idx += 1
        return self._stack[self._idx]
