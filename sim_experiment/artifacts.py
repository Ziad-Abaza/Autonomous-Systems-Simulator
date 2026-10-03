"""
Artifact Registry.

Tracks external artifacts (checkpoints, evaluation results, replays,
trajectories) produced inside a run directory. Artifact files are opaque
— the registry stores references and metadata, never model internals.

Index format (artifacts/registry.jsonl, append-only):

    {"kind": "checkpoint"|"evaluation"|"replay"|"trajectory"|"other",
     "path": "<relative path inside run dir>",
     "created_at": <ts>, "step": <int>, "metadata": {...}}
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, List, Optional

KINDS = ("checkpoint", "evaluation", "replay", "trajectory", "other")


class ArtifactRegistry:
    """Append-only artifact index rooted at a run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.index_path = os.path.join(self.run_dir, "artifacts", "registry.jsonl")

    def register(
        self,
        kind: str,
        rel_path: str,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"Unknown artifact kind: {kind}")
        abs_path = self._resolve(rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Artifact does not exist: {abs_path}")
        entry = {
            "kind": kind,
            "path": rel_path.replace("\\", "/"),
            "created_at": round(time.time(), 4),
            "step": int(step),
            "metadata": dict(metadata or {}),
        }
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def entries(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.exists(self.index_path):
            return out
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind is None or e.get("kind") == kind:
                    out.append(e)
        return out

    def latest_of_kind(self, kind: str) -> Optional[Dict[str, Any]]:
        entries = self.entries(kind)
        return entries[-1] if entries else None

    def find(self, kind: str, **metadata_match) -> List[Dict[str, Any]]:
        return [
            e for e in self.entries(kind)
            if all(e.get("metadata", {}).get(k) == v for k, v in metadata_match.items())
        ]

    def _resolve(self, rel_path: str) -> str:
        """Resolves a run-relative path, rejecting traversal outside the run dir."""
        abs_path = os.path.abspath(os.path.join(self.run_dir, rel_path))
        if os.path.commonpath([self.run_dir, abs_path]) != self.run_dir:
            raise ValueError(f"Artifact path escapes run directory: {rel_path!r}")
        return abs_path
