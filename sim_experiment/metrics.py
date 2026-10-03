"""
Structured metrics pipeline.

Append-only JSONL storage, buffered so nothing is written per simulation
step. Every record is step-indexed and scoped:

    {"scope": "step"|"episode"|"evaluation"|"run",
     "seq": <int>, "ts": <wall clock>, "timestep": <int>,
     "metrics": {...}}

Non-finite values (NaN/Inf) are sanitized to null on write so the stream
never emits invalid JSON and corrupt values can't silently poison
aggregations.
"""

from __future__ import annotations
import json
import math
import os
import time
from typing import Dict, Any, List, Optional, Iterator

SCOPES = ("step", "episode", "evaluation", "run")


def _sanitize(value: Any) -> Any:
    """Replaces NaN/Inf floats with None recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


class MetricsWriter:
    """Buffered append-only writer for a metrics.jsonl stream."""

    def __init__(self, path: str, buffer_size: int = 64):
        self.path = path
        self.buffer_size = max(1, buffer_size)
        self._buf: List[Dict[str, Any]] = []
        self._seq = self._initial_seq()

    def _initial_seq(self) -> int:
        """Continues sequence numbering across restarts."""
        if not os.path.exists(self.path):
            return 0
        try:
            last = None
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = line
            if last:
                return int(json.loads(last).get("seq", -1)) + 1
        except (json.JSONDecodeError, OSError):
            pass
        return 0

    def write(self, scope: str, timestep: int, metrics: Dict[str, Any]) -> None:
        if scope not in SCOPES:
            raise ValueError(f"Unknown metric scope: {scope}")
        self._buf.append({
            "scope": scope,
            "seq": self._seq,
            "ts": round(time.time(), 4),
            "timestep": int(timestep),
            "metrics": _sanitize(metrics),
        })
        self._seq += 1
        if len(self._buf) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in self._buf:
                f.write(json.dumps(rec) + "\n")
        self._buf.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MetricsReader:
    """Reads a metrics.jsonl stream; tolerant of a torn final line."""

    def __init__(self, path: str):
        self.path = path

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip torn tail line

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self.iter_rows())

    def by_scope(self, scope: str) -> List[Dict[str, Any]]:
        return [r for r in self.iter_rows() if r.get("scope") == scope]

    def latest(self) -> Optional[Dict[str, Any]]:
        last = None
        for row in self.iter_rows():
            last = row
        return last

    def latest_metrics(self) -> Dict[str, Any]:
        row = self.latest()
        return dict(row.get("metrics", {})) if row else {}

    def tail(self, n: int = 50) -> List[Dict[str, Any]]:
        rows = self.read_all()
        return rows[-n:]

    def aggregate(self, scope: str = "episode") -> Dict[str, float]:
        """Mean/min/max/last per metric key for a scope."""
        rows = self.by_scope(scope)
        acc: Dict[str, List[float]] = {}
        for r in rows:
            for k, v in r.get("metrics", {}).items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    acc.setdefault(k, []).append(float(v))
        out: Dict[str, float] = {}
        for k, vals in acc.items():
            if not vals:
                continue
            out[f"{k}/mean"] = sum(vals) / len(vals)
            out[f"{k}/min"] = min(vals)
            out[f"{k}/max"] = max(vals)
            out[f"{k}/last"] = vals[-1]
        return out
