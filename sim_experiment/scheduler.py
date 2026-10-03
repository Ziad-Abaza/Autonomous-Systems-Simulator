"""
Batch Scheduler + Local Worker Pool.

Phase 4 could expand a batch into run specs but had no execution engine.
The BatchScheduler owns a queue of jobs and a pool of local workers; each
worker executes at most one run at a time through the existing
LocalTrainingOrchestrator (trainer subprocess = process isolation,
run-dir isolation, deterministic seeds — unchanged).

    BatchScheduler
        └── WorkerPool (N local workers, capacity-limited)
                └── LocalTrainingOrchestrator
                        └── external trainer subprocess

Persistence (per experiment):
    <experiment_dir>/batches/<batch_id>/batch.json       # live state, updated on transitions
    <experiment_dir>/batches/<batch_id>/batch_result.json # final immutable result

One failed run never terminates the batch — failures are isolated per job;
retry policy decides whether a new attempt is queued.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sim_experiment.manifest import ExperimentManifest
from sim_experiment.run import RunManager, RunStatus
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.batch import expand_run_specs
from sim_env.scenario_designer import ScenarioDefinition


class WorkerState:
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    STOPPING = "STOPPING"
    OFFLINE = "OFFLINE"


class JobStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    TERMINAL = (COMPLETED, FAILED, CANCELLED)


# Infrastructure-level failures that may succeed on retry.
RETRYABLE_ERROR_TYPES = frozenset({
    "trainer_crash", "trainer_exception", "launch_failure",
    "timeout", "worker_crash",
})
# Configuration/determinism errors — retrying can never help.
NON_RETRYABLE_ERROR_TYPES = frozenset({
    "invalid_contract", "invalid_curriculum", "invalid_experiment",
    "unsupported_algorithm", "curriculum_mismatch",
    "curriculum_state_missing", "protocol_version_mismatch",
    "launch_rejected", "incompatible_environment",
})


def is_retryable_error(
    error: Optional[Dict[str, Any]],
    retryable_types: Optional[frozenset] = None,
) -> bool:
    """Classifies a run error dict as retryable (transient) or not.

    `retryable_types` lets a RetryPolicy extend the retryable set; the
    NON_RETRYABLE set always wins (a policy cannot make config errors
    retryable).
    """
    if not error:
        return False
    etype = error.get("type", "")
    if etype in NON_RETRYABLE_ERROR_TYPES:
        return False
    allowed = set(RETRYABLE_ERROR_TYPES)
    if retryable_types:
        allowed |= set(retryable_types)
    if etype in allowed:
        return True
    return False  # unknown types are not retried (conservative)


@dataclass
class RetryPolicy:
    max_retries: int = 0
    retryable_error_types: frozenset = RETRYABLE_ERROR_TYPES

    def to_dict(self) -> Dict[str, Any]:
        return {"max_retries": self.max_retries,
                "retryable_error_types": sorted(self.retryable_error_types)}


class LocalAdapter:
    """Worker adapter executing via the in-process LocalTrainingOrchestrator."""

    def __init__(self, orchestrator: LocalTrainingOrchestrator):
        self.orch = orchestrator

    def capabilities(self) -> Dict[str, Any]:
        return {
            "type": "local",
            "trainers": sorted(self.orch.trainer_modules()),
            "max_envs_per_run": None,
        }

    def launch(self, manifest, experiment_dir, trainer, env_mode, run_overrides) -> str:
        return self.orch.launch(
            manifest, experiment_dir, trainer=trainer, env_mode=env_mode,
            run_overrides=run_overrides)

    def poll(self, experiment_dir: str, run_id: str) -> Dict[str, Any]:
        return self.orch.poll(experiment_dir, run_id)

    def cancel(self, experiment_dir: str, run_id: str) -> None:
        self.orch.cancel(experiment_dir, run_id)


@dataclass
class Worker:
    """One execution slot; executes at most one run at a time.

    `adapter` is the execution backend — LocalAdapter (in-process
    orchestrator) or RemoteWorkerAdapter (TCP worker service).
    """
    worker_id: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = WorkerState.IDLE
    current_run_id: Optional[str] = None
    current_job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    adapter: Any = None
    # Lease/heartbeat bookkeeping (remote workers). A worker whose adapter
    # exposes heartbeat() is probed every heartbeat_interval_s; missing the
    # lease_ttl_s window marks it OFFLINE and reclaims its jobs.
    last_heartbeat: float = 0.0
    missed_heartbeats: int = 0
    remote_worker_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "capabilities": self.capabilities,
            "status": self.status,
            "current_run_id": self.current_run_id,
            "current_job_id": self.current_job_id,
            "metadata": self.metadata,
            "last_heartbeat": self.last_heartbeat,
            "missed_heartbeats": self.missed_heartbeats,
        }


class BatchScheduler:
    """Queues run jobs across a fixed pool of local workers."""

    def __init__(
        self,
        experiments_root: str = "experiments",
        max_workers: int = 2,
        orchestrator: Optional[LocalTrainingOrchestrator] = None,
        workers: Optional[List[Worker]] = None,
        heartbeat_interval_s: float = 10.0,
        lease_ttl_s: float = 30.0,
    ):
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.lease_ttl_s = float(lease_ttl_s)
        self.experiments_root = os.path.abspath(experiments_root)
        # Persistent worker registry (experiments_root/workers/registry.json)
        # — survives scheduler restarts; reconnecting worker_ids reuse records.
        from sim_experiment.worker_registry import WorkerRegistry
        self.registry = WorkerRegistry(self.experiments_root)
        self.orch = orchestrator or LocalTrainingOrchestrator(
            experiments_root=self.experiments_root)
        self.run_manager = RunManager()
        if workers is not None:
            if not workers:
                raise ValueError("workers must not be empty")
            self.workers = workers
            self._local_adapter = LocalAdapter(self.orch)
            for w in self.workers:
                if w.adapter is None:
                    w.adapter = self._local_adapter
        else:
            if max_workers < 1:
                raise ValueError("max_workers must be >= 1")
            self._local_adapter = LocalAdapter(self.orch)
            self.workers = [
                Worker(
                    worker_id=f"worker_{i}",
                    capabilities=self._local_adapter.capabilities(),
                    adapter=self._local_adapter,
                )
                for i in range(max_workers)
            ]
        self._batches: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------ batch CRUD

    def create_batch(
        self,
        manifest: ExperimentManifest,
        experiment_dir: str,
        specs: Optional[List[Dict[str, Any]]] = None,
        seeds: Optional[List[int]] = None,
        scenario_ids: Optional[List[str]] = None,
        trainer: str = "ppo",
        env_mode: str = "inprocess",
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Dict[str, Any]:
        """
        Creates a persistent batch of queued jobs. `specs` may carry per-job
        overrides: {"seed", "scenario_id", "algorithm_config"}.
        """
        if specs is None:
            specs = expand_run_specs(manifest, seeds=seeds, scenario_ids=scenario_ids)
        if not specs:
            raise ValueError("Batch requires at least one run spec")

        experiment_dir = os.path.abspath(experiment_dir)
        batch_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch_dir = os.path.join(experiment_dir, "batches", batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        jobs = [
            {
                "job_id": f"{batch_id}_job{i}",
                "index": i,
                "spec": dict(spec),
                "status": JobStatus.QUEUED,
                "attempts": [],
            }
            for i, spec in enumerate(specs)
        ]
        batch = {
            "batch_id": batch_id,
            "experiment_id": manifest.experiment_id,
            "trainer": trainer,
            "env_mode": env_mode,
            "retry_policy": (retry_policy or RetryPolicy()).to_dict(),
            "created_at": round(time.time(), 4),
            "jobs": jobs,
        }
        rec = {
            "batch_id": batch_id,
            "experiment_id": manifest.experiment_id,
            "experiment_dir": experiment_dir,
            "batch_dir": batch_dir,
            "manifest": manifest,
            "retry_policy": retry_policy or RetryPolicy(),
            "data": batch,
        }
        self._batches[batch_id] = rec
        self._persist(rec)
        return {"batch_id": batch_id, "batch_path": batch_dir, "jobs": len(jobs)}

    def status(self, batch_id: str) -> Dict[str, Any]:
        data = self._batch(batch_id)["data"]
        counts = {s: 0 for s in (JobStatus.QUEUED, JobStatus.RUNNING,
                                 JobStatus.COMPLETED, JobStatus.FAILED,
                                 JobStatus.CANCELLED)}
        for j in data["jobs"]:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        return {
            "batch_id": batch_id,
            "experiment_id": data["experiment_id"],
            "total": len(data["jobs"]),
            "queued": counts.get(JobStatus.QUEUED, 0),
            "running": counts.get(JobStatus.RUNNING, 0),
            "completed": counts.get(JobStatus.COMPLETED, 0),
            "failed": counts.get(JobStatus.FAILED, 0),
            "cancelled": counts.get(JobStatus.CANCELLED, 0),
            "finished": sum(counts[s] for s in JobStatus.TERMINAL),
        }

    # ------------------------------------------------------------ execution

    def tick(self, batch_id: Optional[str] = None) -> None:
        """One scheduling round: dispatch queued jobs, poll running ones."""
        self._check_heartbeats()
        for rec in self._batches.values():
            if batch_id and rec["batch_id"] != batch_id:
                continue
            self._dispatch(rec)
            self._poll_jobs(rec)
            self._persist(rec)

    # ------------------------------------------------------------ leases

    def _check_heartbeats(self) -> None:
        """
        Probes remote workers (adapters exposing heartbeat()). Missed
        heartbeats past lease_ttl_s mark the worker OFFLINE and reclaim its
        in-flight jobs for retry on healthy workers.
        """
        now = time.time()
        for w in self.workers:
            hb = getattr(w.adapter, "heartbeat", None)
            if hb is None or w.status == WorkerState.OFFLINE:
                continue
            if w.last_heartbeat and now - w.last_heartbeat < self.heartbeat_interval_s:
                continue
            try:
                resp = hb()
                if resp.get("worker_id"):
                    w.remote_worker_id = resp["worker_id"]
                self.registry.heartbeat(w.remote_worker_id or w.worker_id)
                w.last_heartbeat = now
                w.missed_heartbeats = 0
                continue
            except Exception:
                w.missed_heartbeats += 1
            last = w.last_heartbeat or 0.0
            if now - last > self.lease_ttl_s:
                self._mark_offline(w, reason="heartbeat_lease_expired")

    def _mark_offline(self, w: Worker, reason: str) -> None:
        """Marks a worker OFFLINE and requeues/rejects its in-flight jobs."""
        w.status = WorkerState.OFFLINE
        w.metadata["offline_reason"] = reason
        self.registry.mark_offline(w.remote_worker_id or w.worker_id, reason)
        job_id = w.current_job_id
        w.current_run_id = None
        w.current_job_id = None
        for rec in self._batches.values():
            policy = rec["retry_policy"]
            for job in rec["data"]["jobs"]:
                if job["status"] != JobStatus.RUNNING:
                    continue
                attempt = job["attempts"][-1] if job["attempts"] else None
                if attempt is None or attempt.get("worker_id") != w.worker_id:
                    continue
                if job_id is not None and job["job_id"] != job_id:
                    continue
                attempt["status"] = "FAILED"
                attempt["ended_at"] = round(time.time(), 4)
                attempt["error"] = {
                    "type": "worker_crash",
                    "message": f"Worker {w.worker_id} unreachable ({reason})",
                }
                # A lost worker is infrastructure failure, not a job attempt
                # failure — the job re-queues without consuming its retry
                # budget so a healthy worker can pick it up.
                job["status"] = JobStatus.QUEUED
                self._persist(rec)

    def run_until_complete(
        self, batch_id: str, timeout_s: float = 600.0, poll_s: float = 0.5
    ) -> Dict[str, Any]:
        """Drives the batch until all jobs are terminal; writes batch_result.json."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.tick(batch_id)
            st = self.status(batch_id)
            if st["finished"] >= st["total"]:
                return self._finalize_batch(self._batches[batch_id], "completed")
            time.sleep(poll_s)
        raise TimeoutError(f"Batch {batch_id} did not finish in {timeout_s}s")

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancels running jobs via the orchestrator; marks queued jobs cancelled."""
        rec = self._batch(batch_id)
        exp_dir = rec["experiment_dir"]
        for job in rec["data"]["jobs"]:
            if job["status"] == JobStatus.RUNNING:
                for attempt in job["attempts"]:
                    if attempt["status"] == "RUNNING" and attempt.get("run_id"):
                        worker = self._worker_by_id(attempt.get("worker_id"))
                        adapter = (worker.adapter if worker is not None
                                   else self._local_adapter)
                        adapter.cancel(exp_dir, attempt["run_id"])
                        attempt["status"] = "CANCELLED"
                job["status"] = JobStatus.CANCELLED
            elif job["status"] == JobStatus.QUEUED:
                job["status"] = JobStatus.CANCELLED
        for w in self.workers:
            if w.current_job_id and any(
                j["job_id"] == w.current_job_id and j["status"] == JobStatus.CANCELLED
                for j in rec["data"]["jobs"]
            ):
                w.status = WorkerState.IDLE
                w.current_run_id = None
                w.current_job_id = None
        self._persist(rec)
        return self._finalize_batch(rec, "cancelled")

    # ------------------------------------------------------------ internals

    def _batch(self, batch_id: str) -> Dict[str, Any]:
        if batch_id not in self._batches:
            raise KeyError(f"Unknown batch: {batch_id}")
        return self._batches[batch_id]

    def _idle_worker(self, trainer: str) -> Optional[Worker]:
        for w in self.workers:
            if w.status == WorkerState.IDLE and trainer in w.capabilities.get("trainers", []):
                return w
        return None

    def _resolve_scenario(self, rec: Dict[str, Any], spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sid = spec.get("scenario_id")
        manifest = rec["manifest"]
        if not sid or sid == manifest.scenario_id:
            return None
        scen = ScenarioDefinition.get_standard_scenarios().get(sid)
        if scen is None:
            raise ValueError(f"Batch spec references unknown scenario_id '{sid}'")
        return scen.to_dict()

    def _dispatch(self, rec: Dict[str, Any]) -> None:
        data = rec["data"]
        for job in data["jobs"]:
            if job["status"] != JobStatus.QUEUED:
                continue
            # Duplicate-dispatch guard: never start a second attempt while a
            # previous attempt is still marked RUNNING (active lease).
            if job["attempts"] and job["attempts"][-1]["status"] == "RUNNING":
                continue
            worker = self._idle_worker(data["trainer"])
            if worker is None:
                return
            worker.status = WorkerState.STARTING
            attempt = {
                "attempt": len(job["attempts"]) + 1,
                "started_at": round(time.time(), 4),
                "status": "STARTING",
                "run_id": None,
                "worker_id": worker.worker_id,
            }
            try:
                run_id = worker.adapter.launch(
                    rec["manifest"], rec["experiment_dir"],
                    trainer=data["trainer"], env_mode=data["env_mode"],
                    run_overrides={
                        "seed": job["spec"].get("seed"),
                        "scenario_dict": self._resolve_scenario(rec, job["spec"]),
                        "algorithm_config": job["spec"].get("algorithm_config"),
                    },
                )
            except Exception as e:
                worker.status = WorkerState.IDLE
                attempt["status"] = "FAILED"
                attempt["error"] = {"type": "launch_rejected", "message": str(e)}
                job["attempts"].append(attempt)
                job["status"] = JobStatus.FAILED  # config errors: no retry
                continue
            attempt["run_id"] = run_id
            attempt["status"] = "RUNNING"
            job["attempts"].append(attempt)
            job["status"] = JobStatus.RUNNING
            worker.status = WorkerState.RUNNING
            worker.current_run_id = run_id
            worker.current_job_id = job["job_id"]

    def _poll_jobs(self, rec: Dict[str, Any]) -> None:
        data = rec["data"]
        policy = rec["retry_policy"]
        for job in data["jobs"]:
            if job["status"] != JobStatus.RUNNING:
                continue
            attempt = job["attempts"][-1]
            worker = self._worker_by_id(attempt.get("worker_id"))
            adapter = worker.adapter if worker is not None else self._local_adapter
            try:
                summary = adapter.poll(rec["experiment_dir"], attempt["run_id"])
            except Exception as e:
                # Poll failure = unreachable worker. Count it toward the
                # heartbeat lease; on expiry the worker goes OFFLINE and the
                # job is reclaimed via _mark_offline.
                if worker is not None:
                    worker.missed_heartbeats += 1
                    if time.time() - (worker.last_heartbeat or 0.0) > self.lease_ttl_s:
                        self._mark_offline(worker, reason="poll_unreachable")
                continue
            if summary["status"] not in RunStatus.TERMINAL:
                continue
            attempt["status"] = summary["status"]
            attempt["ended_at"] = round(time.time(), 4)
            attempt["error"] = summary.get("error")
            self._release_worker(job["job_id"])

            failed = summary["status"] in (RunStatus.FAILED, RunStatus.INTERRUPTED)
            retryable = failed and is_retryable_error(
                summary.get("error"),
                retryable_types=policy.retryable_error_types)
            if retryable and len(job["attempts"]) <= policy.max_retries:
                job["status"] = JobStatus.QUEUED   # re-queue a fresh attempt
            else:
                job["status"] = summary["status"]

    def _release_worker(self, job_id: str) -> None:
        for w in self.workers:
            if w.current_job_id == job_id:
                w.status = WorkerState.IDLE
                w.current_run_id = None
                w.current_job_id = None

    def _worker_by_id(self, worker_id: Optional[str]) -> Optional[Worker]:
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        return None

    def _persist(self, rec: Dict[str, Any]) -> None:
        path = os.path.join(rec["batch_dir"], "batch.json")
        tmp = path + ".tmp"
        data = dict(rec["data"])
        data["workers"] = [w.to_dict() for w in self.workers]
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _finalize_batch(self, rec: Dict[str, Any], batch_status: str) -> Dict[str, Any]:
        """Writes the immutable batch_result.json for a finished batch."""
        data = rec["data"]
        exp_dir = rec["experiment_dir"]
        runs: List[Dict[str, Any]] = []
        for job in data["jobs"]:
            run_ids = [a["run_id"] for a in job["attempts"] if a.get("run_id")]
            entry = {
                "job_id": job["job_id"],
                "spec": job["spec"],
                "status": job["status"],
                "run_ids": run_ids,
                "run_id": run_ids[-1] if run_ids else None,
                "attempts": job["attempts"],
                "retries": max(0, len(job["attempts"]) - 1),
            }
            # metrics/eval summary + artifact refs from the final run
            if entry["run_id"]:
                try:
                    run = self.run_manager.load_run(exp_dir, entry["run_id"])
                    entry["metrics"] = dict(run.latest_metrics)
                    entry["timesteps"] = run.current_timestep
                    entry["episode_count"] = run.episode_count
                    entry["artifacts"] = {
                        "checkpoints": list(run.checkpoints),
                        "evaluations": list(run.evaluation_results),
                        "replays": list(run.replays),
                    }
                except OSError:
                    pass
            runs.append(entry)

        result = {
            "batch_id": data["batch_id"],
            "experiment_id": data["experiment_id"],
            "trainer": data["trainer"],
            "env_mode": data["env_mode"],
            "status": batch_status,
            "created_at": data["created_at"],
            "finished_at": round(time.time(), 4),
            "duration_s": round(time.time() - data["created_at"], 3),
            "runs": runs,
        }
        out_path = os.path.join(rec["batch_dir"], "batch_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result
