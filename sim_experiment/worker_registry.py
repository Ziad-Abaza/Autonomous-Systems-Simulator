"""
Persistent worker registry — experiments_root/workers/registry.json.

Records every worker that ever registered (remote or local), its
capabilities, last heartbeat, and lifecycle state. Survives scheduler
restarts: a worker reconnecting with the same worker_id reuses its record.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


class WorkerRegistry:
    def __init__(self, experiments_root: str):
        self.dir = os.path.join(os.path.abspath(experiments_root), "workers")
        self.path = os.path.join(self.dir, "registry.json")
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------ persistence

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"workers": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"workers": {}}

    def _save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ operations

    def register(self, worker_id: str, capabilities: Optional[Dict[str, Any]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Idempotent registration — same worker_id reuses its record."""
        rec = self._data["workers"].get(worker_id)
        if rec is None:
            rec = {
                "worker_id": worker_id,
                "capabilities": dict(capabilities or {}),
                "meta": dict(meta or {}),
                "status": "ONLINE",
                "registered_at": round(time.time(), 4),
                "last_heartbeat": None,
            }
            self._data["workers"][worker_id] = rec
        else:
            if capabilities:
                rec["capabilities"] = dict(capabilities)
            rec["status"] = "ONLINE"
        self._save()
        return dict(rec)

    def heartbeat(self, worker_id: str) -> None:
        rec = self._data["workers"].get(worker_id)
        if rec is None:
            rec = self.register(worker_id)
        rec["last_heartbeat"] = round(time.time(), 4)
        if rec["status"] == "OFFLINE":
            rec["status"] = "ONLINE"
        self._save()

    def mark_offline(self, worker_id: str, reason: str = "") -> None:
        rec = self._data["workers"].get(worker_id)
        if rec is None:
            return
        rec["status"] = "OFFLINE"
        rec["offline_reason"] = reason
        rec["offline_at"] = round(time.time(), 4)
        self._save()

    def get(self, worker_id: str) -> Optional[Dict[str, Any]]:
        rec = self._data["workers"].get(worker_id)
        return dict(rec) if rec else None

    def list_workers(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in sorted(
            self._data["workers"].values(), key=lambda r: r["worker_id"])]
