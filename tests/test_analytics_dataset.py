"""Phase 5 — experiment analytics/comparison + trajectory dataset export."""
import json
import os

import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.analytics import (
    load_metrics_series, smooth, compare_runs, metric_summary,
)
from sim_experiment.dataset import export_dataset, list_episodes
from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def run_dirs(tmp_path, project):
    """Two completed dummy runs in the same experiment."""
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=60,
                                algorithm_config={"trajectory_episodes": 0}),
        evaluation=EvaluationConfig(), name="analytics", random_seed=1,
    )
    exp_dir = mgr.create(m)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
    ids = []
    for _ in range(2):
        rid = orch.launch(m, exp_dir, trainer="dummy")
        assert orch.wait(exp_dir, rid, timeout_s=120)["status"] == "COMPLETED"
        ids.append(rid)
    return exp_dir, [os.path.join(exp_dir, "runs", r) for r in ids]


class TestMetricsSeries:
    def test_load_episode_scope(self, run_dirs):
        _, dirs = run_dirs
        series = load_metrics_series(dirs[0], "episode", "reward")
        assert len(series) >= 3
        assert all(isinstance(t, int) and isinstance(v, float) for t, v in series)
        assert [t for t, _ in series] == sorted(t for t, _ in series)

    def test_missing_metric_returns_empty(self, run_dirs):
        _, dirs = run_dirs
        assert load_metrics_series(dirs[0], "episode", "nonexistent") == []

    def test_smooth_moving_average(self):
        s = smooth([(0, 0.0), (1, 10.0), (2, 20.0), (3, 30.0)], window=3)
        assert s[-1][1] == pytest.approx(20.0)
        assert s[0][1] == pytest.approx(0.0)


class TestComparison:
    def test_compare_runs_aligned(self, run_dirs):
        _, dirs = run_dirs
        cmp_result = compare_runs(
            dirs, metric="reward", scope="episode", smooth_window=2)
        assert cmp_result["metric"] == "reward"
        assert len(cmp_result["series"]) == 2
        for entry in cmp_result["series"]:
            assert entry["raw"] and entry["smoothed"]
            assert "max" in entry and "final" in entry and "mean" in entry

    def test_metric_summary(self, run_dirs):
        _, dirs = run_dirs
        summary = metric_summary(dirs[0])
        assert summary["total_timesteps"] == 60
        assert summary["episodes"] >= 1

    def test_compare_empty_graceful(self, tmp_path):
        cmp_result = compare_runs([str(tmp_path / "nope")], metric="reward")
        assert cmp_result["series"] == []


class TestDatasetExport:
    def _trajectory_run(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(
                algorithm="ppo", total_timesteps=96, rollout_length=32,
                batch_size=32, epochs=1, eval_frequency=0,
                checkpoint_frequency=0, num_envs=1,
                algorithm_config={"trajectory_episodes": 5}),
            evaluation=EvaluationConfig(), name="traj", random_seed=3,
        )
        exp_dir = mgr.create(m)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        rid = orch.launch(m, exp_dir, trainer="ppo")
        assert orch.wait(exp_dir, rid, timeout_s=240)["status"] == "COMPLETED"
        return os.path.join(exp_dir, "runs", rid)

    def test_export_produces_jsonl_and_manifest(self, tmp_path, project):
        run_dir = self._trajectory_run(tmp_path, project)
        out = str(tmp_path / "dataset")
        report = export_dataset(run_dir, out)
        assert os.path.exists(os.path.join(out, "manifest.json"))
        assert os.path.exists(os.path.join(out, "episodes.jsonl"))
        assert report["episodes"] >= 1
        assert report["steps"] > 0
        with open(os.path.join(out, "manifest.json")) as f:
            mf = json.load(f)
        assert mf["dataset_format"] == "transitions_v1"
        assert "env_fingerprint" in mf

    def test_export_steps_agent_only(self, tmp_path, project):
        """Exported steps must contain ONLY agent-visible data — no
        diagnostic keys (speed/lateral_offset/etc.) at step level."""
        run_dir = self._trajectory_run(tmp_path, project)
        out = str(tmp_path / "dataset2")
        export_dataset(run_dir, out)
        diag_keys = {"speed", "lateral_offset", "heading_error",
                     "is_colliding", "is_on_road"}
        with open(os.path.join(out, "episodes.jsonl")) as f:
            for line in f:
                ep = json.loads(line)
                for step in ep["steps"]:
                    assert not diag_keys.intersection(step.keys()), \
                        f"diagnostic field leaked into dataset: {step.keys()}"
                break  # first episode suffices

    def test_fingerprint_filter(self, tmp_path, project):
        run_dir = self._trajectory_run(tmp_path, project)
        out = str(tmp_path / "dataset3")
        report = export_dataset(run_dir, out, env_fingerprint="bogus_fp")
        assert report["episodes"] == 0

    def test_min_return_filter(self, tmp_path, project):
        run_dir = self._trajectory_run(tmp_path, project)
        out = str(tmp_path / "dataset4")
        report = export_dataset(run_dir, out, min_return=1e9)
        assert report["episodes"] == 0

    def test_list_episodes(self, tmp_path, project):
        run_dir = self._trajectory_run(tmp_path, project)
        eps = list_episodes(run_dir)
        assert eps
        assert all("episode_id" in e and "steps" in e for e in eps)
