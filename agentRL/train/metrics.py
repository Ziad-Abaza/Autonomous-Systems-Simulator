"""MetricsLogger — append-only metrics.jsonl per run directory.

Every row: {seq, ts, timestep, scope, metrics}. NaN/Inf are serialized
as null (JSON has no NaN; emitting the literal would corrupt parsers).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any


def _sanitize(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return f if math.isfinite(f) else None
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:  # pragma: no cover
        pass
    return v


class MetricsLogger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._seq = 0

    def _write(self, scope: str, timestep: int,
               metrics: dict[str, Any]) -> None:
        row = {"seq": self._seq, "ts": round(time.time(), 3),
               "timestep": int(timestep), "scope": scope,
               "metrics": _sanitize(metrics)}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self._seq += 1

    def episode(self, timestep: int, metrics: dict[str, Any]) -> None:
        self._write("episode", timestep, metrics)

    def update(self, timestep: int, metrics: dict[str, Any]) -> None:
        self._write("update", timestep, metrics)

    def eval(self, timestep: int, metrics: dict[str, Any]) -> None:
        self._write("eval", timestep, metrics)

    def heartbeat(self, timestep: int, metrics: dict[str, Any]) -> None:
        self._write("heartbeat", timestep, metrics)

    def failure(self, timestep: int, metrics: dict[str, Any]) -> None:
        self._write("failure", timestep, metrics)

    def close(self) -> None:
        self._fh.close()
