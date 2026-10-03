"""Phase 6 — worker hardening: registration, heartbeats, leases, reclaim."""
import os
import time

import pytest

from sim_experiment.scheduler import (
    BatchScheduler, Worker, WorkerState, JobStatus, RetryPolicy,
    is_retryable_error,
)
from sim_experiment.remote_worker import WorkerService, RemoteWorkerAdapter
from sim_env.templates import EnvironmentTemplateManager


TOKEN = "testtoken123"


@pytest.fixture
def worker_service(tmp_path):
    svc = WorkerService("127.0.0.1", 0, str(tmp_path / "exps"), TOKEN)
    svc.start()
    yield svc
    svc.stop()


def _adapter(svc):
    return RemoteWorkerAdapter("127.0.0.1", svc.port, TOKEN, timeout=5.0)


# ------------------------------------------------------------- service protocol

def test_hello_advertises_worker_id_and_heartbeat(worker_service):
    ack = _adapter(worker_service).handshake()
    assert ack["capabilities"].get("heartbeat") is True
    assert ack["capabilities"].get("worker_id")


def test_heartbeat_roundtrip(worker_service):
    adapter = _adapter(worker_service)
    hb = adapter.heartbeat()
    assert hb["alive"] is True
    assert hb["worker_id"] == adapter.worker_id


def test_register_assigns_stable_worker_id(worker_service):
    adapter = _adapter(worker_service)
    resp = adapter.register()
    assert resp["worker_id"]
    adapter2 = RemoteWorkerAdapter("127.0.0.1", worker_service.port, TOKEN)
    # Same service instance -> same worker_id (identity survives reconnect)
    assert adapter2.register()["worker_id"] == resp["worker_id"]


def test_launch_rejects_path_outside_root(worker_service):
    adapter = _adapter(worker_service)
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=4),
        evaluation=EvaluationConfig(), name="x", random_seed=1)
    with pytest.raises(RuntimeError, match="experiment_dir|path|escape"):
        adapter.launch(m, "C:\\\\Windows\\\\Temp\\\\evil", "dummy",
                       "inprocess", {})


def test_auth_still_required_for_heartbeat(worker_service):
    bad = RemoteWorkerAdapter("127.0.0.1", worker_service.port, "wrongtoken")
    with pytest.raises(RuntimeError, match="auth_failed"):
        bad.heartbeat()


# --------------------------------------------------------------- scheduler lease

class _DyingAdapter:
    """Remote adapter that launches fine then goes dark mid-run."""

    def __init__(self):
        self.dead = False
        self._run = 0

    def capabilities(self):
        return {"type": "remote", "trainers": ["dummy"], "worker_id": "dead0"}

    def launch(self, manifest, experiment_dir, trainer, env_mode, run_overrides):
        self._run += 1
        return f"run_{self._run}"

    def poll(self, experiment_dir, run_id):
        if self.dead:
            raise ConnectionError("worker unreachable")
        return {"status": "RUNNING"}

    def heartbeat(self):
        if self.dead:
            raise ConnectionError("worker unreachable")
        return {"alive": True, "worker_id": "dead0"}

    def cancel(self, experiment_dir, run_id):
        raise ConnectionError("worker unreachable")


class _LocalDoneAdapter:
    """Local adapter that finishes runs immediately."""

    def __init__(self):
        self._run = 0

    def capabilities(self):
        return {"type": "local", "trainers": ["dummy"]}

    def launch(self, manifest, experiment_dir, trainer, env_mode, run_overrides):
        self._run += 1
        return f"local_{self._run}"

    def poll(self, experiment_dir, run_id):
        return {"status": "COMPLETED"}

    def cancel(self, experiment_dir, run_id):
        pass


def _manifest(tmp_path):
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    mgr = ExperimentManager(root_dir=str(tmp_path / "exps"))
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=4),
        evaluation=EvaluationConfig(), name="w", random_seed=1)
    return m, mgr.create(m)


def test_dead_worker_job_reclaimed(tmp_path):
    m, exp_dir = _manifest(tmp_path)
    dead = _DyingAdapter()
    live = _LocalDoneAdapter()
    sched = BatchScheduler(
        experiments_root=str(tmp_path / "exps"),
        workers=[
            Worker("w_dead", capabilities=dead.capabilities(), adapter=dead),
            Worker("w_live", capabilities=live.capabilities(), adapter=live),
        ],
        heartbeat_interval_s=0.01,
        lease_ttl_s=0.05,
    )
    batch = sched.create_batch(m, exp_dir, specs=[{"seed": 1}], trainer="dummy")
    sched.tick(batch["batch_id"])
    # Job dispatched to w_dead (first idle capable worker)
    job = sched._batches[batch["batch_id"]]["data"]["jobs"][0]
    assert job["status"] == JobStatus.RUNNING
    assert job["attempts"][0]["worker_id"] == "w_dead"

    dead.dead = True
    for _ in range(30):
        sched.tick(batch["batch_id"])
        if sched.status(batch["batch_id"])["completed"]:
            break
        time.sleep(0.02)

    st = sched.status(batch["batch_id"])
    assert st["completed"] == 1
    w_dead = sched.workers[0]
    assert w_dead.status == WorkerState.OFFLINE
    job = sched._batches[batch["batch_id"]]["data"]["jobs"][0]
    assert len(job["attempts"]) >= 2  # re-queued and retried on live worker


def test_healthy_remote_worker_stays_online(tmp_path):
    m, exp_dir = _manifest(tmp_path)
    live = _LocalDoneAdapter()

    class _HbAdapter(_LocalDoneAdapter):
        def heartbeat(self):
            return {"alive": True, "worker_id": "hb"}

    hb = _HbAdapter()
    sched = BatchScheduler(
        experiments_root=str(tmp_path / "exps"),
        workers=[Worker("w1", capabilities=hb.capabilities(), adapter=hb)],
        heartbeat_interval_s=0.01, lease_ttl_s=60.0,
    )
    batch = sched.create_batch(m, exp_dir, specs=[{"seed": 1}], trainer="dummy")
    for _ in range(5):
        sched.tick(batch["batch_id"])
        time.sleep(0.015)
    assert sched.workers[0].status != WorkerState.OFFLINE


def test_offline_worker_not_dispatched(tmp_path):
    m, exp_dir = _manifest(tmp_path)
    live = _LocalDoneAdapter()

    class _Hb(_LocalDoneAdapter):
        def heartbeat(self):
            return {"alive": True}

    w = Worker("w1", capabilities=_Hb().capabilities(), adapter=_Hb())
    w.status = WorkerState.OFFLINE
    sched = BatchScheduler(
        experiments_root=str(tmp_path / "exps"), workers=[w])
    batch = sched.create_batch(m, exp_dir, specs=[{"seed": 1}], trainer="dummy")
    sched.tick(batch["batch_id"])
    job = sched._batches[batch["batch_id"]]["data"]["jobs"][0]
    assert job["status"] == JobStatus.QUEUED


def test_poll_cancel_root_checked(worker_service):
    """POLL/CANCEL cannot reference experiment dirs outside the worker root."""
    adapter = _adapter(worker_service)
    for rpc in (adapter.poll, adapter.cancel):
        with pytest.raises(RuntimeError, match="invalid_experiment_dir|escapes"):
            rpc("C:\\\\Windows\\\\Temp", "run_x")


def test_registry_persists_and_reuses(tmp_path):
    from sim_experiment.worker_registry import WorkerRegistry
    reg = WorkerRegistry(str(tmp_path))
    rec = reg.register("w1", capabilities={"trainers": ["ppo"]})
    assert rec["status"] == "ONLINE"
    reg.heartbeat("w1")
    assert reg.get("w1")["last_heartbeat"] is not None
    reg.mark_offline("w1", "test")
    # New instance reads the same file (persistence) — offline survives.
    reg2 = WorkerRegistry(str(tmp_path))
    assert reg2.get("w1")["status"] == "OFFLINE"
    # Same worker_id re-registering reuses (not duplicates) the record.
    reg2.register("w1")
    assert len(reg2.list_workers()) == 1
    assert reg2.get("w1")["status"] == "ONLINE"


def test_status_message(worker_service):
    adapter = _adapter(worker_service)
    resp = adapter._rpc({"type": "STATUS"})
    assert resp["worker_id"] == adapter.worker_id or resp["alive"] is True


def test_duplicate_dispatch_guard(tmp_path):
    m, exp_dir = _manifest(tmp_path)
    live = _LocalDoneAdapter()
    sched = BatchScheduler(
        experiments_root=str(tmp_path / "exps"),
        workers=[Worker("w1", capabilities=live.capabilities(), adapter=live)])
    batch = sched.create_batch(m, exp_dir, specs=[{"seed": 1}], trainer="dummy")
    # Force an inconsistent state: job QUEUED but last attempt RUNNING.
    job = sched._batches[batch["batch_id"]]["data"]["jobs"][0]
    job["attempts"].append({"attempt": 1, "status": "RUNNING",
                            "worker_id": "w1", "run_id": "ghost"})
    sched.tick(batch["batch_id"])
    assert len(job["attempts"]) == 1  # no second attempt launched


def test_custom_retryable_types_honored(tmp_path):
    """RetryPolicy.retryable_error_types is consulted, not just the module set."""
    pol = RetryPolicy(max_retries=1, retryable_error_types=frozenset({"weird_custom"}))
    # "weird_custom" is retryable under this policy but not under the default.
    assert not is_retryable_error({"type": "weird_custom"})
    assert is_retryable_error({"type": "weird_custom"},
                              retryable_types=pol.retryable_error_types)
