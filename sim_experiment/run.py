"""
Run entity and Run Manager.

An Experiment describes what should be executed; a Run is one actual
execution. Runs are append-mostly records inside the experiment directory:

    runs/<run_id>/
        run.json          # lifecycle record (status, progress, error, refs)
        metrics.jsonl     # structured metrics stream (MetricsWriter)
        contract.json     # external trainer contract (orchestrator)
        logs/             # trainer stdout/stderr
        checkpoints/      # model artifacts (opaque files + registry)
        evaluation/       # evaluation result artifacts
        replays/          # episode replay files
        trajectories/     # trajectory datasets
        artifacts/        # misc artifacts

Historical run records are never rewritten except for status/progress
bookkeeping; resuming creates a NEW run linked to its parent.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


class RunStatus:
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"

    TERMINAL = (COMPLETED, FAILED, CANCELLED, INTERRUPTED)


# Allowed status transitions. Terminal states accept no further transitions.
_TRANSITIONS: Dict[str, tuple] = {
    RunStatus.CREATED: (RunStatus.QUEUED, RunStatus.STARTING, RunStatus.CANCELLED, RunStatus.FAILED),
    RunStatus.QUEUED: (RunStatus.STARTING, RunStatus.CANCELLED, RunStatus.FAILED),
    RunStatus.STARTING: (RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.INTERRUPTED),
    RunStatus.RUNNING: (RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED,
                        RunStatus.CANCELLED, RunStatus.INTERRUPTED),
    RunStatus.PAUSED: (RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED,
                       RunStatus.INTERRUPTED),
    RunStatus.COMPLETED: (),
    RunStatus.FAILED: (),
    RunStatus.CANCELLED: (),
    RunStatus.INTERRUPTED: (),
}


@dataclass
class Run:
    """One execution of an experiment."""
    run_id: str
    experiment_id: str
    status: str = RunStatus.CREATED
    seed: int = 42
    created_at: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    pid: Optional[int] = None
    current_timestep: int = 0
    episode_count: int = 0
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)   # relative paths
    evaluation_results: List[str] = field(default_factory=list)
    replays: List[str] = field(default_factory=list)
    resume_from: Optional[Dict[str, Any]] = None           # {parent_run_id, checkpoint}
    error: Optional[Dict[str, Any]] = None
    exit_code: Optional[int] = None
    trainer: str = ""
    env_mode: str = "inprocess"
    num_envs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Run":
        r = cls(run_id=str(data.get("run_id", "")), experiment_id=str(data.get("experiment_id", "")))
        for f_name in (
            "status", "seed", "created_at", "started_at", "ended_at", "pid",
            "current_timestep", "episode_count", "latest_metrics", "checkpoints",
            "evaluation_results", "replays", "resume_from", "error", "exit_code",
            "trainer", "env_mode", "num_envs",
        ):
            if f_name in data:
                setattr(r, f_name, data[f_name])
        return r


class RunManager:
    """Owns run records and the run directory layout."""

    RUNS_SUBDIR = "runs"

    # ------------------------------------------------------------ paths

    def runs_root(self, experiment_dir: str) -> str:
        return os.path.join(experiment_dir, self.RUNS_SUBDIR)

    def run_dir(self, experiment_dir: str, run_id: str) -> str:
        if not run_id or os.path.basename(run_id) != run_id:
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return os.path.join(self.runs_root(experiment_dir), run_id)

    # ------------------------------------------------------------ lifecycle

    def create_run(
        self,
        experiment_dir: str,
        seed: int,
        experiment_id: str = "",
        trainer: str = "",
        env_mode: str = "inprocess",
        num_envs: int = 1,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Creates a new run record + directory layout."""
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run = Run(
            run_id=run_id,
            experiment_id=experiment_id,
            seed=int(seed),
            created_at=time.time(),
            trainer=trainer,
            env_mode=env_mode,
            num_envs=int(num_envs),
            resume_from=dict(resume_from) if resume_from else None,
        )
        rd = self.run_dir(experiment_dir, run_id)
        for sub in ("", "logs", "checkpoints", "evaluation", "replays",
                    "trajectories", "artifacts"):
            os.makedirs(os.path.join(rd, sub), exist_ok=True)
        if not run.experiment_id:
            run.experiment_id = os.path.basename(os.path.abspath(experiment_dir))
        self._write(exp_dir=experiment_dir, run=run)
        return run

    def load_run(self, experiment_dir: str, run_id: str) -> Run:
        path = os.path.join(self.run_dir(experiment_dir, run_id), "run.json")
        with open(path, "r", encoding="utf-8") as f:
            return Run.from_dict(json.load(f))

    def list_runs(self, experiment_dir: str) -> List[Dict[str, Any]]:
        # ~1 s TTL cache — UI panels call this per frame and every run dir
        # costs a run.json read. Status writes invalidate immediately.
        bucket = int(time.time() / 1.0)
        cache = getattr(self, "_runs_cache", {}).get(experiment_dir)
        if cache and cache[0] == bucket:
            return [dict(r) for r in cache[1]]
        root = self.runs_root(experiment_dir)
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(root):
            return out
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry, "run.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            out.append({
                "run_id": d.get("run_id", entry),
                "status": d.get("status", ""),
                "seed": d.get("seed", 0),
                "created_at": d.get("created_at", 0.0),
                "started_at": d.get("started_at", 0.0),
                "ended_at": d.get("ended_at", 0.0),
                "current_timestep": d.get("current_timestep", 0),
                "episode_count": d.get("episode_count", 0),
                "trainer": d.get("trainer", ""),
            })
        if not hasattr(self, "_runs_cache"):
            self._runs_cache = {}
        self._runs_cache[experiment_dir] = (bucket, out)
        return out

    def _invalidate_runs_cache(self, experiment_dir: str) -> None:
        if hasattr(self, "_runs_cache"):
            self._runs_cache.pop(experiment_dir, None)

    def set_status(
        self,
        experiment_dir: str,
        run_id: str,
        status: str,
        error: Optional[Dict[str, Any]] = None,
        pid: Optional[int] = None,
        exit_code: Optional[int] = None,
    ) -> Run:
        """Validated status transition; persists run.json."""
        run = self.load_run(experiment_dir, run_id)
        allowed = _TRANSITIONS.get(run.status, ())
        if status not in allowed:
            raise RuntimeError(
                f"Invalid run status transition {run.status} -> {status} for {run_id}"
            )
        run.status = status
        if status == RunStatus.RUNNING and not run.started_at:
            run.started_at = time.time()
        if status in RunStatus.TERMINAL:
            run.ended_at = time.time()
        if error is not None:
            run.error = error
        if pid is not None:
            run.pid = pid
        if exit_code is not None:
            run.exit_code = exit_code
        self._write(exp_dir=experiment_dir, run=run)
        return run

    def update_progress(
        self,
        experiment_dir: str,
        run_id: str,
        timestep: Optional[int] = None,
        episode: Optional[int] = None,
        latest_metrics: Optional[Dict[str, Any]] = None,
    ) -> Run:
        run = self.load_run(experiment_dir, run_id)
        if timestep is not None:
            run.current_timestep = int(timestep)
        if episode is not None:
            run.episode_count = int(episode)
        if latest_metrics is not None:
            run.latest_metrics = dict(latest_metrics)
        self._write(exp_dir=experiment_dir, run=run)
        return run

    def attach_pid(self, experiment_dir: str, run_id: str, pid: int) -> None:
        run = self.load_run(experiment_dir, run_id)
        run.pid = int(pid)
        self._write(exp_dir=experiment_dir, run=run)

    def add_artifact_ref(self, experiment_dir: str, run_id: str, kind: str, rel_path: str) -> None:
        """Appends a checkpoint/evaluation/replay reference to run.json."""
        run = self.load_run(experiment_dir, run_id)
        target = {
            "checkpoint": run.checkpoints,
            "evaluation": run.evaluation_results,
            "replay": run.replays,
        }.get(kind)
        if target is None:
            raise ValueError(f"Unknown artifact kind: {kind}")
        if rel_path not in target:
            target.append(rel_path)
        self._write(exp_dir=experiment_dir, run=run)

    # ------------------------------------------------------------ io

    def _write(self, exp_dir: str, run: Run) -> None:
        path = os.path.join(self.run_dir(exp_dir, run.run_id), "run.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2)
        os.replace(tmp, path)
        self._invalidate_runs_cache(exp_dir)
