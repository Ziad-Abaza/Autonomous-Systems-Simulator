"""Phase 5 — remote/TCP worker service + RemoteWorkerAdapter E2E tests."""
import os

import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.remote_worker import (
    WorkerService, RemoteWorkerAdapter, WORKER_PROTOCOL_VERSION,
)
from sim_experiment.scheduler import BatchScheduler, Worker, WorkerState
from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def service(tmp_path):
    svc = WorkerService(host="127.0.0.1", port=0,
                        experiments_root=str(tmp_path / "experiments"),
                        token="secret-token")
    svc.start()
    yield svc
    svc.stop()


class TestWorkerService:
    def test_hello_returns_capabilities(self, service):
        adapter = RemoteWorkerAdapter("127.0.0.1", service.port, "secret-token")
        caps = adapter.handshake()
        assert caps["protocol_version"] == WORKER_PROTOCOL_VERSION
        assert "trainers" in caps["capabilities"]

    def test_wrong_token_rejected(self, service):
        adapter = RemoteWorkerAdapter("127.0.0.1", service.port, "bad-token")
        with pytest.raises(RuntimeError, match="auth"):
            adapter.handshake()

    def test_empty_token_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="token"):
            WorkerService(host="127.0.0.1", port=0,
                          experiments_root=str(tmp_path), token="")


class TestRemoteDispatch:
    def test_remote_worker_runs_dummy(self, tmp_path, project, service):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(algorithm="dummy", total_timesteps=30),
            evaluation=EvaluationConfig(), name="remote_test", random_seed=1,
        )
        exp_dir = mgr.create(m)

        adapter = RemoteWorkerAdapter("127.0.0.1", service.port, "secret-token")
        worker = Worker(worker_id="remote_1",
                        capabilities=adapter.capabilities())
        worker.adapter = adapter

        s = BatchScheduler(mgr.root_dir, workers=[worker])
        batch = s.create_batch(m, exp_dir, specs=[{"seed": 1}, {"seed": 2}],
                               trainer="dummy")
        result = s.run_until_complete(batch["batch_id"], timeout_s=120)
        assert result["status"] == "completed"
        assert all(r["status"] == "COMPLETED" for r in result["runs"])
        assert len(result["runs"]) == 2

    def test_remote_cancel(self, tmp_path, project, service):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(
                algorithm="dummy", total_timesteps=10**6,
                algorithm_config={"tick_seconds": 0.5}),
            evaluation=EvaluationConfig(), name="remote_cancel", random_seed=1,
        )
        exp_dir = mgr.create(m)

        adapter = RemoteWorkerAdapter("127.0.0.1", service.port, "secret-token")
        worker = Worker(worker_id="remote_1",
                        capabilities=adapter.capabilities())
        worker.adapter = adapter

        s = BatchScheduler(mgr.root_dir, workers=[worker])
        batch = s.create_batch(m, exp_dir, specs=[{"seed": 1}], trainer="dummy")
        s.tick()
        status = s.status(batch["batch_id"])
        assert status["running"] == 1
        s.cancel_batch(batch["batch_id"])
        assert s.status(batch["batch_id"])["cancelled"] == 1
