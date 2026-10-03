"""
Phase 4 experiment-domain tests: manifest, identity, ExperimentManager,
Run lifecycle, metrics pipeline, artifact registry, trainer contract.
"""

import json
import os
import time
import math
import pytest

from sim_project.serializer import EnvironmentProject
from sim_env.templates import EnvironmentTemplateManager
from sim_experiment.manifest import (
    ExperimentManifest, TrainingConfig, EvaluationConfig, MANIFEST_VERSION
)
from sim_experiment.manager import ExperimentManager
from sim_experiment.run import Run, RunStatus, RunManager
from sim_experiment.metrics import MetricsWriter, MetricsReader
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trainer_contract import (
    build_contract, validate_contract, TRAINER_CONTRACT_VERSION
)


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def manifest(project, tmp_path):
    training = TrainingConfig(algorithm="ppo", total_timesteps=512)
    evaluation = EvaluationConfig(eval_seeds=[1, 2], num_episodes=2)
    return ExperimentManifest.from_project(
        project=project,
        scenario=project.scenario_def,
        training=training,
        evaluation=evaluation,
        name="test experiment",
        random_seed=42,
    )


# ---------------------------------------------------------------- Manifest

class TestManifest:
    def test_snapshot_contains_environment_identity(self, manifest, project):
        d = manifest.to_dict()
        assert d["environment_fingerprint"] == project.compute_fingerprint()
        assert d["environment_version"] == project.environment_version
        assert d["environment"]["agent"]["agent_id"] == project.agent.agent_id
        assert d["scenario_id"] == project.scenario_def.scenario_id
        assert d["manifest_version"] == MANIFEST_VERSION

    def test_schemas_captured(self, manifest):
        d = manifest.to_dict()
        assert d["observation_schema"]["channels"]
        assert d["action_schema"]["channels"]
        assert d["reward_configuration"]["components"]
        assert d["termination_configuration"]["rules"]
        assert d["episode_configuration"]["max_steps"] > 0

    def test_identity_is_deterministic(self, manifest, project):
        m2 = ExperimentManifest.from_project(
            project=project,
            scenario=project.scenario_def,
            training=TrainingConfig(algorithm="ppo", total_timesteps=512),
            evaluation=EvaluationConfig(eval_seeds=[1, 2], num_episodes=2),
            name="different name",      # name is metadata, not identity
            random_seed=42,
        )
        assert m2.experiment_fingerprint == manifest.experiment_fingerprint
        assert m2.experiment_id == manifest.experiment_id

    def test_identity_changes_with_config(self, manifest, project):
        m2 = ExperimentManifest.from_project(
            project=project,
            scenario=project.scenario_def,
            training=TrainingConfig(algorithm="sac", total_timesteps=512),
            evaluation=EvaluationConfig(eval_seeds=[1, 2], num_episodes=2),
            name="x",
            random_seed=42,
        )
        assert m2.experiment_fingerprint != manifest.experiment_fingerprint

    def test_identity_changes_with_seed(self, manifest, project):
        m2 = ExperimentManifest.from_project(
            project=project,
            scenario=project.scenario_def,
            training=TrainingConfig(algorithm="ppo", total_timesteps=512),
            evaluation=EvaluationConfig(eval_seeds=[1, 2], num_episodes=2),
            name="x",
            random_seed=7,
        )
        assert m2.experiment_id != manifest.experiment_id

    def test_roundtrip_serialization(self, manifest):
        d = manifest.to_dict()
        m2 = ExperimentManifest.from_dict(d)
        assert m2.to_dict() == d
        assert m2.experiment_id == manifest.experiment_id

    def test_algorithm_config_namespaced(self, project):
        t = TrainingConfig(
            algorithm="ppo",
            algorithm_config={"clip_coef": 0.15, "ent_coef": 0.001}
        )
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=t, evaluation=EvaluationConfig(), name="x", random_seed=1
        )
        assert m.to_dict()["training"]["algorithm_config"]["clip_coef"] == 0.15


# ---------------------------------------------------------------- Manager

class TestExperimentManager:
    def test_create_writes_manifest_and_snapshot(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        exp_dir = mgr.create(manifest)
        assert os.path.isdir(exp_dir)
        assert os.path.exists(os.path.join(exp_dir, "experiment.json"))
        assert os.path.exists(os.path.join(exp_dir, "environment.json"))
        assert os.path.exists(os.path.join(exp_dir, "scenario.json"))

    def test_load_and_list(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        mgr.create(manifest)
        loaded = mgr.load(manifest.experiment_id)
        assert loaded.experiment_id == manifest.experiment_id
        listing = mgr.list_experiments()
        assert len(listing) == 1
        assert listing[0]["experiment_id"] == manifest.experiment_id
        assert listing[0]["environment_fingerprint"] == manifest.environment_fingerprint

    def test_validation_rejects_bad_manifest(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        bad = ExperimentManifest.from_dict(manifest.to_dict())
        bad.environment_fingerprint = "deadbeef"
        with pytest.raises(ValueError):
            mgr.create(bad)

    def test_immutable_after_launch(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        mgr.create(manifest)
        mgr.mark_launched(manifest.experiment_id)
        with pytest.raises(RuntimeError):
            mgr.create(manifest)  # same experiment id, already launched
        loaded = mgr.load(manifest.experiment_id)
        assert loaded.launched is True

    def test_clone_returns_unlaunched_copy(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        mgr.create(manifest)
        clone = mgr.clone(manifest.experiment_id)
        assert clone.launched is False
        assert clone.experiment_fingerprint == manifest.experiment_fingerprint

    def test_export_copies_full_directory(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        mgr.create(manifest)
        dest = str(tmp_path / "exported" / manifest.experiment_id)
        mgr.export(manifest.experiment_id, dest)
        assert os.path.exists(os.path.join(dest, "experiment.json"))
        reloaded = ExperimentManifest.from_dict(
            json.load(open(os.path.join(dest, "experiment.json")))
        )
        assert reloaded.experiment_id == manifest.experiment_id

    def test_archive_marks_experiment(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        mgr.create(manifest)
        mgr.archive(manifest.experiment_id)
        assert mgr.list_experiments()[0]["archived"] is True
        assert mgr.list_experiments(include_archived=False) == []


# ---------------------------------------------------------------- Runs

class TestRunLifecycle:
    def _exp_dir(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        return mgr.create(manifest)

    def test_create_run_layout(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=42)
        assert run.status == RunStatus.CREATED
        rd = rmg.run_dir(exp_dir, run.run_id)
        assert os.path.isdir(rd)
        for sub in ("logs", "checkpoints", "evaluation", "replays", "trajectories", "artifacts"):
            assert os.path.isdir(os.path.join(rd, sub))
        assert os.path.exists(os.path.join(rd, "run.json"))

    def test_status_transitions(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=1)
        rmg.set_status(exp_dir, run.run_id, RunStatus.QUEUED)
        rmg.set_status(exp_dir, run.run_id, RunStatus.STARTING)
        rmg.set_status(exp_dir, run.run_id, RunStatus.RUNNING)
        rmg.set_status(exp_dir, run.run_id, RunStatus.COMPLETED)
        loaded = rmg.load_run(exp_dir, run.run_id)
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.ended_at > 0

    def test_invalid_transition_rejected(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=1)
        with pytest.raises(RuntimeError):
            rmg.set_status(exp_dir, run.run_id, RunStatus.RUNNING)  # must go through QUEUED/STARTING
        rmg.set_status(exp_dir, run.run_id, RunStatus.QUEUED)
        rmg.set_status(exp_dir, run.run_id, RunStatus.STARTING)
        rmg.set_status(exp_dir, run.run_id, RunStatus.RUNNING)
        rmg.set_status(exp_dir, run.run_id, RunStatus.COMPLETED)
        with pytest.raises(RuntimeError):
            rmg.set_status(exp_dir, run.run_id, RunStatus.RUNNING)  # terminal state

    def test_failure_records_error(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=1)
        for s in (RunStatus.QUEUED, RunStatus.STARTING, RunStatus.RUNNING):
            rmg.set_status(exp_dir, run.run_id, s)
        rmg.set_status(exp_dir, run.run_id, RunStatus.FAILED,
                       error={"type": "trainer_crash", "message": "exit code 1"})
        loaded = rmg.load_run(exp_dir, run.run_id)
        assert loaded.error["type"] == "trainer_crash"

    def test_progress_update(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=1)
        rmg.update_progress(exp_dir, run.run_id,
                            timestep=500, episode=3,
                            latest_metrics={"mean_reward": 1.5})
        loaded = rmg.load_run(exp_dir, run.run_id)
        assert loaded.current_timestep == 500
        assert loaded.episode_count == 3
        assert loaded.latest_metrics["mean_reward"] == 1.5

    def test_resume_creates_child_run(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=1)
        child = rmg.create_run(exp_dir, seed=1, resume_from={
            "parent_run_id": run.run_id,
            "checkpoint": "checkpoints/policy.pt"
        })
        loaded = rmg.load_run(exp_dir, child.run_id)
        assert loaded.resume_from["parent_run_id"] == run.run_id

    def test_list_runs(self, tmp_path, manifest):
        exp_dir = self._exp_dir(tmp_path, manifest)
        rmg = RunManager()
        rmg.create_run(exp_dir, seed=1)
        rmg.create_run(exp_dir, seed=2)
        assert len(rmg.list_runs(exp_dir)) == 2


# ---------------------------------------------------------------- Metrics

class TestMetrics:
    def test_write_read_scoped(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        w = MetricsWriter(path)
        w.write("episode", timestep=100, metrics={"reward": 5.0, "length": 42})
        w.write("run", timestep=100, metrics={"sps": 300.0})
        w.flush()
        rows = MetricsReader(path).read_all()
        assert len(rows) == 2
        assert rows[0]["scope"] == "episode"
        assert rows[0]["metrics"]["reward"] == 5.0

    def test_latest_and_scope_filter(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        w = MetricsWriter(path)
        for i in range(5):
            w.write("episode", timestep=i * 10, metrics={"reward": float(i)})
        w.write("evaluation", timestep=40, metrics={"mean_reward": 3.3})
        w.flush()
        r = MetricsReader(path)
        assert r.latest()["scope"] == "evaluation"
        assert len(r.by_scope("episode")) == 5

    def test_nan_inf_sanitized(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        w = MetricsWriter(path)
        w.write("step", timestep=1, metrics={"bad": float("nan"), "inf": float("inf"), "ok": 1.0})
        w.flush()
        rows = MetricsReader(path).read_all()
        assert rows[0]["metrics"]["ok"] == 1.0
        assert rows[0]["metrics"]["bad"] is None
        assert rows[0]["metrics"]["inf"] is None

    def test_buffered_not_written_until_flush(self, tmp_path):
        path = str(tmp_path / "metrics.jsonl")
        w = MetricsWriter(path)
        w.write("step", timestep=1, metrics={"a": 1})
        assert not os.path.exists(path) or os.path.getsize(path) == 0
        w.flush()
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------- Artifacts

class TestArtifacts:
    def test_register_and_lookup(self, tmp_path):
        rd = str(tmp_path)
        ckpt_dir = os.path.join(rd, "checkpoints")
        os.makedirs(ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "policy_100.pt")
        with open(ckpt, "wb") as f:
            f.write(b"weights")
        reg = ArtifactRegistry(rd)
        reg.register("checkpoint", "checkpoints/policy_100.pt",
                     step=100, metadata={"algorithm": "ppo"})
        latest = reg.latest_of_kind("checkpoint")
        assert latest["path"] == "checkpoints/policy_100.pt"
        assert latest["step"] == 100

    def test_missing_artifact_rejected(self, tmp_path):
        reg = ArtifactRegistry(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            reg.register("checkpoint", "checkpoints/nope.pt")

    def test_path_traversal_rejected(self, tmp_path):
        reg = ArtifactRegistry(str(tmp_path))
        with pytest.raises(ValueError):
            reg.register("checkpoint", "../outside.pt")


# ---------------------------------------------------------------- Contract

class TestTrainerContract:
    def test_contract_shape(self, tmp_path, manifest):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        exp_dir = mgr.create(manifest)
        rmg = RunManager()
        run = rmg.create_run(exp_dir, seed=42)
        rd = rmg.run_dir(exp_dir, run.run_id)
        contract = build_contract(manifest, exp_dir, rd, run.run_id)
        errors = validate_contract(contract)
        assert errors == []
        assert contract["contract_version"] == TRAINER_CONTRACT_VERSION
        assert contract["experiment_id"] == manifest.experiment_id
        assert contract["paths"]["run_dir"] == "."
        from sim_experiment.trainer_contract import resolve_contract_paths
        resolved = resolve_contract_paths(contract, rd)
        assert resolved["paths"]["run_dir"] == os.path.abspath(rd)
        assert resolved["paths"]["metrics_file"] == os.path.join(
            os.path.abspath(rd), "metrics.jsonl")
        assert contract["training"]["algorithm"] == "ppo"
        assert contract["seed"] == 42

    def test_validate_catches_missing(self):
        errors = validate_contract({"contract_version": TRAINER_CONTRACT_VERSION})
        assert errors
