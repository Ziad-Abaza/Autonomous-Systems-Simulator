"""Phase 5 — BatchScheduler + local WorkerPool tests."""
import json
import os
import time

import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.scheduler import (
    BatchScheduler, RetryPolicy, WorkerState, is_retryable_error,
)
from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


def _manifest(project, alg_cfg=None, total=30):
    return ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=total,
                                algorithm_config=alg_cfg or {}),
        evaluation=EvaluationConfig(), name="batch_test", random_seed=7,
    )


@pytest.fixture
def setup(tmp_path, project):
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    manifest = _manifest(project)
    exp_dir = mgr.create(manifest)
    return {"root": mgr.root_dir, "mgr": mgr, "manifest": manifest, "exp_dir": exp_dir}


class TestErrorClassification:
    def test_retryable(self):
        assert is_retryable_error({"type": "trainer_crash"})
        assert is_retryable_error({"type": "trainer_exception"})
        assert is_retryable_error({"type": "timeout"})

    def test_non_retryable(self):
        for t in ("invalid_contract", "invalid_curriculum",
                  "unsupported_algorithm", "curriculum_mismatch",
                  "protocol_version_mismatch"):
            assert not is_retryable_error({"type": t})

    def test_none_error_not_retryable(self):
        assert not is_retryable_error(None)


class TestWorkers:
    def test_pool_capacity(self, setup):
        s = BatchScheduler(setup["root"], max_workers=3)
        assert len(s.workers) == 3
        assert all(w.status == WorkerState.IDLE for w in s.workers)

    def test_worker_capabilities(self, setup):
        s = BatchScheduler(setup["root"], max_workers=2)
        w = s.workers[0]
        assert w.worker_id
        assert "trainers" in w.capabilities


class TestBatchLifecycle:
    def test_create_batch_persists(self, setup):
        s = BatchScheduler(setup["root"], max_workers=2)
        batch = s.create_batch(
            setup["manifest"], setup["exp_dir"],
            specs=[{"seed": 1}, {"seed": 2}, {"seed": 3}],
            trainer="dummy")
        status = s.status(batch["batch_id"])
        assert status["total"] == 3
        assert status["queued"] == 3
        assert os.path.exists(batch["batch_path"])

    def test_batch_completes_all_runs(self, setup):
        s = BatchScheduler(setup["root"], max_workers=2)
        batch = s.create_batch(
            setup["manifest"], setup["exp_dir"],
            specs=[{"seed": 1}, {"seed": 2}, {"seed": 3}],
            trainer="dummy")
        result = s.run_until_complete(batch["batch_id"], timeout_s=120)
        assert result["status"] == "completed"
        assert len(result["runs"]) == 3
        assert all(r["status"] == "COMPLETED" for r in result["runs"])
        assert all(r["run_id"] for r in result["runs"])
        # batch_result.json persisted and not overwritten
        assert os.path.exists(os.path.join(batch["batch_path"], "batch_result.json"))

    def test_worker_parallelism_respected(self, setup):
        s = BatchScheduler(setup["root"], max_workers=1)
        batch = s.create_batch(
            setup["manifest"], setup["exp_dir"],
            specs=[{"seed": 1}, {"seed": 2}], trainer="dummy")
        s.tick()
        running = [w for w in s.workers if w.status == WorkerState.RUNNING]
        assert len(running) == 1
        status = s.status(batch["batch_id"])
        assert status["running"] == 1
        s.cancel_batch(batch["batch_id"])

    def test_failed_run_does_not_kill_batch(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        good = _manifest(project)
        exp_dir = mgr.create(good)
        s = BatchScheduler(mgr.root_dir, max_workers=2)
        # job 0 fails (fail_at_step), jobs 1-2 succeed — via per-job alg
        # config embedded in the spec.
        batch = s.create_batch(
            good, exp_dir,
            specs=[{"seed": 1, "algorithm_config": {"fail_at_step": 5}},
                   {"seed": 2}, {"seed": 3}],
            trainer="dummy")
        result = s.run_until_complete(batch["batch_id"], timeout_s=120)
        statuses = sorted(r["status"] for r in result["runs"])
        assert statuses == ["COMPLETED", "COMPLETED", "FAILED"]
        assert result["status"] == "completed"

    def test_retry_creates_new_attempt(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = _manifest(project)
        exp_dir = mgr.create(m)
        s = BatchScheduler(mgr.root_dir, max_workers=1)
        batch = s.create_batch(
            m, exp_dir,
            specs=[{"seed": 1, "algorithm_config": {"fail_at_step": 5}}],
            trainer="dummy",
            retry_policy=RetryPolicy(max_retries=2))
        result = s.run_until_complete(batch["batch_id"], timeout_s=180)
        job = result["runs"][0]
        assert job["status"] == "FAILED"
        assert len(job["attempts"]) == 3          # initial + 2 retries
        assert len({a["run_id"] for a in job["attempts"]}) == 3

    def test_no_retry_by_default(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = _manifest(project)
        exp_dir = mgr.create(m)
        s = BatchScheduler(mgr.root_dir, max_workers=1)
        batch = s.create_batch(
            m, exp_dir,
            specs=[{"seed": 1, "algorithm_config": {"fail_at_step": 5}}],
            trainer="dummy")
        result = s.run_until_complete(batch["batch_id"], timeout_s=120)
        assert len(result["runs"][0]["attempts"]) == 1

    def test_cancel_batch(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = _manifest(project, alg_cfg={"tick_seconds": 0.5}, total=10**6)
        exp_dir = mgr.create(m)
        s = BatchScheduler(mgr.root_dir, max_workers=2)
        batch = s.create_batch(m, exp_dir, specs=[{"seed": 1}, {"seed": 2}, {"seed": 3}],
                               trainer="dummy")
        s.tick()
        s.cancel_batch(batch["batch_id"])
        status = s.status(batch["batch_id"])
        assert status["cancelled"] >= 1
        assert status["running"] == 0

    def test_seed_override_propagates(self, setup):
        s = BatchScheduler(setup["root"], max_workers=1)
        batch = s.create_batch(
            setup["manifest"], setup["exp_dir"],
            specs=[{"seed": 99}], trainer="dummy")
        result = s.run_until_complete(batch["batch_id"], timeout_s=120)
        run_id = result["runs"][0]["attempts"][0]["run_id"]
        from sim_experiment.run import RunManager
        run = RunManager().load_run(setup["exp_dir"], run_id)
        assert run.seed == 99
