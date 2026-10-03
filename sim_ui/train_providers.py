"""
UI data providers for the TRAIN inspector panel — pure functions over
service-layer objects, no pygame dependency, unit-testable headless.

    worker_rows(scheduler, registry) -> worker status rows
    dataset_preview(dataset_dir)     -> validation + stats + episode list
    comparison_to_multichart(cmp)    -> chart-ready multi-series rows
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


def worker_rows(scheduler=None, registry=None) -> List[Dict[str, Any]]:
    """
    Worker status rows for the WORKERS inspector section.
    Live scheduler workers take precedence (they have job assignments);
    registry records supply persistence/offline history when no scheduler
    view is available.
    """
    rows: List[Dict[str, Any]] = []
    if scheduler is not None:
        now = time.time()
        for w in scheduler.workers:
            d = w.to_dict() if hasattr(w, "to_dict") else dict(w.__dict__)
            hb = getattr(w, "last_heartbeat", None) or d.get("last_heartbeat")
            rows.append({
                "worker_id": getattr(w, "remote_worker_id", None)
                             or d.get("remote_worker_id")
                             or d.get("worker_id", "?"),
                "status": d.get("status", "?"),
                "job": d.get("current_job_id"),
                "heartbeat_age_s": (round(now - hb, 1)
                                    if hb else None),
                "missed_heartbeats": getattr(w, "missed_heartbeats",
                                           d.get("missed_heartbeats", 0)),
            })
        return rows
    if registry is not None:
        for rec in registry.list_workers():
            rows.append({
                "worker_id": rec["worker_id"],
                "status": rec["status"],
                "job": None,
                "heartbeat_age_s": (round(time.time() - rec["last_heartbeat"], 1)
                                    if rec.get("last_heartbeat") else None),
                "reason": rec.get("offline_reason", ""),
            })
    return rows


def dataset_preview(dataset_dir: str, max_episodes: int = 8) -> Dict[str, Any]:
    """
    Dataset preview for the DATASET inspector section:
    validation state, aggregate stats, and the first N episode summaries.
    """
    from sim_experiment.dataset_inspect import inspect_dataset
    rep = inspect_dataset(dataset_dir)
    episodes = []
    path = os.path.join(dataset_dir, "episodes.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episodes.append({
                    "episode_id": ep.get("episode_id"),
                    "total_return": ep.get("total_return"),
                    "length": ep.get("length"),
                    "termination_reason": ep.get("termination_reason", ""),
                })
                if len(episodes) >= max_episodes:
                    break
    return {
        "valid": rep.get("valid", False),
        "errors": rep.get("errors", []),
        "warnings": rep.get("warnings", []),
        "stats": rep.get("statistics", {}),
        "episodes": episodes,
        "total_episodes": rep.get("statistics", {}).get("episode_count", 0),
    }


def comparison_to_multichart(comparison: Optional[Dict[str, Any]]
                             ) -> List[Dict[str, Any]]:
    """
    Converts compare_runs output into chart-ready multi-series rows:
        [{"label": run_label, "data": [v0, v1, ...]}]
    Prefers the smoothed series; falls back to raw.
    """
    if not comparison:
        return []
    rows = []
    for s in comparison.get("series") or []:
        series = s.get("smoothed") or s.get("raw") or []
        data = [float(v) for _ts, v in series]
        rows.append({"label": s.get("label") or s.get("run_id", "?"),
                     "data": data})
    return rows
