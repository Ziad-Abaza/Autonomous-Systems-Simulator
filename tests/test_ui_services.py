"""Phase 6 — UI service providers: workers view, dataset preview, multichart."""
import json
import os

import pytest


# ------------------------------------------------------------------ workers

class _FakeScheduler:
    def __init__(self, workers):
        self.workers = workers


class _FakeWorker:
    def __init__(self, wid, status="IDLE", job=None, remote_id=None):
        self.worker_id = wid
        self.status = status
        self.current_job_id = job
        self.remote_worker_id = remote_id
        self.last_heartbeat = 0.0
        self.missed_heartbeats = 0

    def to_dict(self):
        return {
            "worker_id": self.worker_id, "status": self.status,
            "current_job_id": self.current_job_id,
            "remote_worker_id": self.remote_worker_id,
            "last_heartbeat": self.last_heartbeat,
            "missed_heartbeats": self.missed_heartbeats,
        }


def test_worker_rows_from_scheduler():
    from sim_ui.train_providers import worker_rows
    sched = _FakeScheduler([
        _FakeWorker("w0", "RUNNING", "job3"),
        _FakeWorker("w1", "OFFLINE"),
    ])
    rows = worker_rows(scheduler=sched)
    assert rows[0]["worker_id"] == "w0"
    assert rows[0]["status"] == "RUNNING"
    assert rows[0]["job"] == "job3"
    assert rows[1]["status"] == "OFFLINE"


def test_worker_rows_from_registry(tmp_path):
    from sim_ui.train_providers import worker_rows
    from sim_experiment.worker_registry import WorkerRegistry
    reg = WorkerRegistry(str(tmp_path))
    reg.register("wA", capabilities={"type": "remote"})
    reg.register("wB")
    reg.mark_offline("wB", "gone")
    rows = worker_rows(registry=reg)
    ids = [r["worker_id"] for r in rows]
    assert ids == ["wA", "wB"]
    assert rows[1]["status"] == "OFFLINE"
    assert rows[1]["reason"] == "gone"


# -------------------------------------------------------------- dataset preview

def _write_ds(ds_dir, n=4):
    os.makedirs(ds_dir, exist_ok=True)
    with open(os.path.join(ds_dir, "episodes.jsonl"), "w") as f:
        for i in range(n):
            ep = {
                "episode_id": f"e{i}", "seed": i, "env_fingerprint": "fp",
                "total_return": float(i), "length": 3,
                "termination_reason": "collision" if i % 2 == 0 else "lap_completed",
                "steps": [{"obs": [0.0], "action": [0.0], "reward": float(i),
                           "terminated": t == 2, "truncated": False}
                          for t in range(3)],
            }
            f.write(json.dumps(ep) + "\n")
    with open(os.path.join(ds_dir, "manifest.json"), "w") as f:
        json.dump({"dataset_format": "transitions_v1", "episodes": n,
                   "steps": 3 * n, "env_fingerprint": "fp"}, f)


def test_dataset_preview_stats_and_episodes(tmp_path):
    from sim_ui.train_providers import dataset_preview
    ds = str(tmp_path / "ds")
    _write_ds(ds)
    prev = dataset_preview(ds)
    assert prev["valid"] is True
    assert prev["stats"]["episode_count"] == 4
    assert prev["stats"]["termination_reasons"]["collision"] == 2
    eps = prev["episodes"]
    assert len(eps) == 4
    assert eps[0]["episode_id"] == "e0"
    assert eps[0]["total_return"] == 0.0
    assert eps[0]["length"] == 3


def test_dataset_preview_invalid(tmp_path):
    from sim_ui.train_providers import dataset_preview
    prev = dataset_preview(str(tmp_path / "nope"))
    assert prev["valid"] is False
    assert prev["errors"]


# ----------------------------------------------------------------- multichart

def test_comparison_to_multichart():
    from sim_ui.train_providers import comparison_to_multichart
    cmp_payload = {
        "series": [
            {"label": "run_a", "smoothed": [(0, 1.0), (10, 2.0), (20, 3.0)]},
            {"label": "run_b", "smoothed": [(0, 2.0), (10, 4.0)]},
        ]
    }
    rows = comparison_to_multichart(cmp_payload)
    assert len(rows) == 2
    assert rows[0]["label"] == "run_a"
    assert rows[0]["data"] == [1.0, 2.0, 3.0]
    assert rows[1]["data"] == [2.0, 4.0]


def test_comparison_to_multichart_empty():
    from sim_ui.train_providers import comparison_to_multichart
    assert comparison_to_multichart(None) == []
    assert comparison_to_multichart({"series": []}) == []
