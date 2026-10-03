"""
Trajectory / dataset storage.

Per-step transition records for training data and trajectory inspection.
JSONL format, buffered writes. CRITICAL INVARIANT: agent-facing data and
diagnostic/oracle data are stored under separate keys — diagnostic fields
must never be mistaken for agent observations.

File layout (one file per episode):

    trajectories/ep_<episode_id>.jsonl
        {"type": "header", "env_fingerprint": ..., "scenario_id": ...,
         "seed": ..., "episode_id": ..., "observation_schema": ...,
         "action_schema": ...}
        {"type": "step", "episode_id": ..., "step": n,
         "agent_data": {"obs": [...], "action": [...], "reward": r,
                        "terminated": b, "truncated": b,
                        "termination_reason": "..."},
         "diagnostic_data": {"speed": ..., "lateral_offset": ...,
                             "heading_error": ..., "pos": [...], "yaw": ...}}
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, Any, List, Optional


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [_clean(x) for x in v.tolist()]
        if isinstance(v, (np.floating, np.integer)):
            return _clean(v.item())
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except ImportError:
        pass
    return v


class TrajectoryWriter:
    """Writes per-episode trajectory files under a trajectories directory."""

    def __init__(self, trajectories_dir: str, buffer_size: int = 128):
        self.dir = trajectories_dir
        self.buffer_size = max(1, buffer_size)
        self._buf: List[Dict[str, Any]] = []
        self._fh = None
        self._episode_id: Optional[str] = None
        os.makedirs(self.dir, exist_ok=True)

    def start_episode(
        self,
        episode_id: str,
        env_fingerprint: str,
        scenario_id: str,
        seed: int,
        observation_schema: Optional[Dict[str, Any]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        episode_seed: Optional[int] = None,
        curriculum_stage_index: Optional[int] = None,
    ) -> None:
        self.close_episode()
        if os.path.basename(episode_id) != episode_id:
            raise ValueError(f"Invalid episode_id: {episode_id!r}")
        self._episode_id = episode_id
        self._fh = open(os.path.join(self.dir, f"{episode_id}.jsonl"), "w", encoding="utf-8")
        self._emit({
            "type": "header",
            "episode_id": episode_id,
            "env_fingerprint": env_fingerprint,
            "scenario_id": scenario_id,
            "seed": int(seed),
            # Exact seed this episode was reset with (for reproducible
            # re-simulation); distinct from the run-level `seed`.
            "episode_seed": int(episode_seed) if episode_seed is not None else None,
            "curriculum_stage_index": (
                int(curriculum_stage_index)
                if curriculum_stage_index is not None else None),
            "observation_schema": observation_schema or {},
            "action_schema": action_schema or {},
        })

    def record_step(
        self,
        step: int,
        agent_data: Dict[str, Any],
        diagnostic_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """agent_data must contain obs/action/reward/terminated/truncated."""
        if self._fh is None:
            raise RuntimeError("start_episode() must be called before record_step()")
        for key in ("obs", "action", "reward"):
            if key not in agent_data:
                raise ValueError(f"agent_data missing required key: {key}")
        self._emit({
            "type": "step",
            "episode_id": self._episode_id,
            "step": int(step),
            "agent_data": _clean(dict(agent_data)),
            "diagnostic_data": _clean(dict(diagnostic_data or {})),
        })

    def close_episode(self) -> None:
        self._flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._episode_id = None

    def _emit(self, record: Dict[str, Any]) -> None:
        self._buf.append(record)
        if len(self._buf) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf or self._fh is None:
            return
        for rec in self._buf:
            self._fh.write(json.dumps(rec) + "\n")
        self._buf.clear()

    def close(self) -> None:
        self.close_episode()


class TrajectoryReader:
    """Reads a trajectory file back into header + step records."""

    def __init__(self, path: str):
        self.path = path

    def read(self) -> Dict[str, Any]:
        header: Dict[str, Any] = {}
        steps: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "header":
                    header = rec
                elif rec.get("type") == "step":
                    steps.append(rec)
        return {"header": header, "steps": steps}
