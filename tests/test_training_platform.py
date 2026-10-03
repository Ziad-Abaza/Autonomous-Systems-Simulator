"""
Phase 4 integration tests: orchestration lifecycle, trainer contract,
PPO smoke run, evaluation, trajectories, batch expansion,
reproducibility checks, and headless environments.
"""

import json
import os
import time
import pytest

from sim_env.templates import EnvironmentTemplateManager
from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.run import RunManager, RunStatus
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.metrics import MetricsReader
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.evaluation import evaluate_policy, EvaluationResult
from sim_experiment.trajectory import TrajectoryWriter, TrajectoryReader
from sim_experiment.batch import expand_run_specs
from sim_experiment.reproduce import check_reproducibility
from sim_experiment.headless import build_env_from_dicts, HeadlessEnvPool


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def setup(tmp_path, project):
    """Creates a manager + manifest + experiment dir under a temp root."""
    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    manifest = ExperimentManifest.from_project(
        project=project,
        scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=30),
        evaluation=EvaluationConfig(eval_seeds=[5], num_episodes=1),
        name="itest",
        random_seed=42,
    )
    exp_dir = mgr.create(manifest)
    return {"root": root, "mgr": mgr, "manifest": manifest, "exp_dir": exp_dir, "project": project}


# ------------------------------------------------------------ Orchestrator

class TestOrchestrator:
    def test_launch_completes_and_writes_artifacts(self, setup):
        orch = LocalTrainingOrchestrator(experiments_root=setup["root"])
        run_id = orch.launch(setup["manifest"], setup["exp_dir"], trainer="dummy")
        summary = orch.wait(setup["exp_dir"], run_id, timeout_s=60)

        assert summary["status"] == RunStatus.COMPLETED
        rd = os.path.join(setup["exp_dir"], "runs", run_id)
        assert os.path.exists(os.path.join(rd, "contract.json"))
        assert os.path.exists(os.path.join(rd, "metrics.jsonl"))
        assert os.path.exists(os.path.join(rd, "run_result.json"))
        assert os.path.getsize(os.path.join(rd, "logs", "stdout.log")) >= 0

        rows = MetricsReader(os.path.join(rd, "metrics.jsonl")).read_all()
        assert any(r["scope"] == "episode" for r in rows)
        assert any(r["scope"] == "run" for r in rows)

        # checkpoint registered + synced into run.json
        assert any("dummy_" in c for c in summary["checkpoints"])

    def test_trainer_failure_is_first_class(self, tmp_path, setup):
        manifest = ExperimentManifest.from_project(
            project=setup["project"], scenario=setup["project"].scenario_def,
            training=TrainingConfig(
                algorithm="dummy", total_timesteps=50,
                algorithm_config={"fail_at_step": 5},
            ),
            evaluation=EvaluationConfig(), name="failtest", random_seed=1,
        )
        exp_dir = setup["mgr"].create(manifest)
        orch = LocalTrainingOrchestrator(experiments_root=setup["root"])
        run_id = orch.launch(manifest, exp_dir, trainer="dummy")
        summary = orch.wait(exp_dir, run_id, timeout_s=60)
        assert summary["status"] == RunStatus.FAILED
        assert summary["error"]["type"] in ("trainer_exception", "trainer_crash")

    def test_cancel_running_trainer(self, tmp_path, setup):
        manifest = ExperimentManifest.from_project(
            project=setup["project"], scenario=setup["project"].scenario_def,
            training=TrainingConfig(
                algorithm="dummy", total_timesteps=10_000_000,
                algorithm_config={"tick_seconds": 0.05},
            ),
            evaluation=EvaluationConfig(), name="canceltest", random_seed=1,
        )
        exp_dir = setup["mgr"].create(manifest)
        orch = LocalTrainingOrchestrator(experiments_root=setup["root"])
        run_id = orch.launch(manifest, exp_dir, trainer="dummy")
        time.sleep(0.5)
        summary = orch.cancel(exp_dir, run_id)
        assert summary["status"] == RunStatus.CANCELLED
        # artifacts from partial run remain on disk
        rd = os.path.join(exp_dir, "runs", run_id)
        assert os.path.exists(os.path.join(rd, "metrics.jsonl"))

    def test_trainer_whitelist_enforced(self, setup):
        orch = LocalTrainingOrchestrator(experiments_root=setup["root"])
        with pytest.raises(ValueError):
            orch.launch(setup["manifest"], setup["exp_dir"], trainer="os.system")


# ------------------------------------------------------------ Evaluation

class TestEvaluation:
    def test_evaluate_policy_returns_aggregates(self, setup):
        env = build_env_from_dicts(
            setup["project"].to_dict(), setup["project"].scenario_def.to_dict(), seed=0
        )
        result = evaluate_policy(
            env, lambda obs: [0.0, 0.4, 0.0],
            seeds=[3], num_episodes=1, deterministic=True,
            result_kwargs={"env_fingerprint": "abc", "scenario_id": "s"},
        )
        assert result.aggregate["episode_count"] == 1
        assert "mean_reward" in result.aggregate
        assert "collision_rate" in result.aggregate
        assert result.episodes[0]["length"] > 0

    def test_evaluation_result_roundtrip(self, tmp_path):
        r = EvaluationResult(eval_id="e1", checkpoint_path="c.pt",
                             aggregate={"mean_reward": 1.0}, episodes=[{"seed": 1}])
        p = str(tmp_path / "eval.json")
        r.save(p)
        loaded = EvaluationResult.from_dict(json.load(open(p)))
        assert loaded.eval_id == "e1"
        assert loaded.aggregate["mean_reward"] == 1.0


# ------------------------------------------------------------ Trajectory

class TestTrajectory:
    def test_agent_and_diagnostic_data_separated(self, tmp_path):
        tdir = str(tmp_path / "traj")
        w = TrajectoryWriter(tdir)
        w.start_episode("ep0", env_fingerprint="fp", scenario_id="s", seed=1)
        w.record_step(0, agent_data={
            "obs": [0.1, 0.2], "action": [0.0, 0.5, 0.0],
            "reward": 0.5, "terminated": False, "truncated": False,
            "termination_reason": "running",
        }, diagnostic_data={"speed": 3.0, "is_colliding": False})
        w.close_episode()
        data = TrajectoryReader(os.path.join(tdir, "ep0.jsonl")).read()
        assert data["header"]["env_fingerprint"] == "fp"
        step = data["steps"][0]
        assert step["agent_data"]["obs"] == [0.1, 0.2]
        assert step["diagnostic_data"]["speed"] == 3.0
        assert "speed" not in step["agent_data"]

    def test_missing_agent_key_rejected(self, tmp_path):
        w = TrajectoryWriter(str(tmp_path / "t"))
        w.start_episode("e", "fp", "s", 0)
        with pytest.raises(ValueError):
            w.record_step(0, agent_data={"obs": [1]})  # no action/reward
        w.close()


# ------------------------------------------------------------ Batch

class TestBatch:
    def test_deterministic_expansion(self, setup):
        specs = expand_run_specs(setup["manifest"], seeds=[1, 2, 3])
        assert [s["seed"] for s in specs] == [1, 2, 3]
        assert all(s["scenario_id"] == setup["manifest"].scenario_id for s in specs)

    def test_scenario_expansion(self, setup):
        specs = expand_run_specs(setup["manifest"], seeds=[7],
                                 scenario_ids=["basic_lane_following", "high_speed_racing"])
        assert len(specs) == 2
        assert specs[0]["scenario_id"] != specs[1]["scenario_id"]


# ------------------------------------------------------------ Reproducibility

class TestReproducibility:
    def test_clean_experiment_is_reproducible(self, setup):
        report = check_reproducibility(setup["exp_dir"])
        assert report["reproducible"] is True
        assert not report["fatal_failures"]

    def test_tampered_environment_detected(self, setup):
        env_path = os.path.join(setup["exp_dir"], "environment.json")
        with open(env_path) as f:
            d = json.load(f)
        d["agent"]["reward_function"]["components"][0]["weight"] = 999.0
        with open(env_path, "w") as f:
            json.dump(d, f)
        report = check_reproducibility(setup["exp_dir"])
        assert report["reproducible"] is False
        assert "environment fingerprint matches manifest" in report["fatal_failures"]

    def test_missing_files_detected(self, tmp_path):
        report = check_reproducibility(str(tmp_path / "nonexistent"))
        assert report["reproducible"] is False


# ------------------------------------------------------------ Headless

class TestHeadless:
    def test_independent_envs_and_seeds(self, setup):
        env_dict = setup["project"].to_dict()
        scen = setup["project"].scenario_def.to_dict()
        pool = HeadlessEnvPool(env_dict, scen, num_envs=2, base_seed=10)
        assert len(pool) == 2
        obs_a, _ = pool.envs[0].reset(seed=10)
        obs_b, _ = pool.envs[1].reset(seed=11)
        # different seeds -> different randomization state, independent episodes
        pool.envs[0].step([0.0, 0.5, 0.0])
        assert pool.envs[0].current_step == 1
        assert pool.envs[1].current_step == 0

    def test_headless_env_needs_no_ui(self, setup):
        env = build_env_from_dicts(setup["project"].to_dict(), seed=1)
        obs, _ = env.reset(seed=1)
        obs, r, term, trunc, info = env.step([0.0, 0.5, 0.0])
        assert isinstance(r, float)


# ------------------------------------------------------------ PPO smoke

class TestPPOEndToEnd:
    def test_ppo_run_through_orchestrator(self, tmp_path, project):
        root = str(tmp_path / "experiments")
        mgr = ExperimentManager(root_dir=root)
        manifest = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(
                algorithm="ppo", total_timesteps=256,
                rollout_length=128, batch_size=64, epochs=2,
                checkpoint_frequency=128,
            ),
            evaluation=EvaluationConfig(eval_seeds=[0], num_episodes=1),
            name="ppo smoke", random_seed=42,
        )
        exp_dir = mgr.create(manifest)
        mgr.mark_launched(manifest.experiment_id)

        orch = LocalTrainingOrchestrator(experiments_root=root)
        run_id = orch.launch(manifest, exp_dir, trainer="ppo")
        summary = orch.wait(exp_dir, run_id, timeout_s=300)

        assert summary["status"] == RunStatus.COMPLETED
        rd = os.path.join(exp_dir, "runs", run_id)
        assert summary["checkpoints"], "final checkpoint must be registered"
        assert os.path.exists(os.path.join(rd, "checkpoints", "policy_final.pt"))

        # metrics stream consistent with run_result
        metrics = MetricsReader(os.path.join(rd, "metrics.jsonl")).read_all()
        assert any(r["scope"] == "run" for r in metrics)
        result = json.load(open(os.path.join(rd, "run_result.json")))
        n_ep_metric_rows = len([r for r in metrics if r["scope"] == "episode"])
        assert n_ep_metric_rows == result["episodes_completed"]

        # evaluate the produced checkpoint headlessly
        from sim_experiment.evaluation import make_policy_from_checkpoint
        env = build_env_from_dicts(manifest.environment, manifest.scenario_configuration, seed=0)
        policy = make_policy_from_checkpoint(
            os.path.join(rd, "checkpoints", "policy_final.pt"), algorithm="ppo"
        )
        result = evaluate_policy(env, policy, seeds=[0], num_episodes=1)
        assert result.aggregate["episode_count"] == 1
