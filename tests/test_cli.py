"""Phase 5 — CLI coverage for new commands."""
import json
import os

import pytest

from sim_experiment.cli import main as cli_main
from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project(tmp_path):
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def exp(tmp_path, project):
    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=30),
        evaluation=EvaluationConfig(), name="cli_exp", random_seed=1,
    )
    exp_dir = mgr.create(m)
    return {"root": root, "manifest": m, "exp_dir": exp_dir,
            "exp_id": m.experiment_id}


def test_trainers_command_lists_capabilities(exp, capsys):
    assert cli_main(["--root", exp["root"], "trainers"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "ppo" in data and "sac" in data and "dqn" in data
    assert "continuous" in data["ppo"]["action_types"]


def test_batch_run_command(exp, capsys):
    rc = cli_main(["--root", exp["root"], "batch-run", exp["exp_id"],
                   "--seeds", "1", "2", "--workers", "2", "--trainer", "dummy",
                   "--timeout", "120"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "batch_" in out
    # batch_result.json exists under the experiment
    batches_dir = os.path.join(exp["exp_dir"], "batches")
    assert os.path.isdir(batches_dir)
    batch_ids = os.listdir(batches_dir)
    assert batch_ids
    assert os.path.exists(
        os.path.join(batches_dir, batch_ids[0], "batch_result.json"))


def test_compare_command(exp, capsys):
    # run two dummy runs first
    cli_main(["--root", exp["root"], "batch-run", exp["exp_id"],
              "--seeds", "1", "2", "--workers", "1", "--trainer", "dummy",
              "--timeout", "120"])
    capsys.readouterr()
    rc = cli_main(["--root", exp["root"], "compare", exp["exp_id"],
                   "--metric", "reward", "--scope", "episode"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["metric"] == "reward"
    assert len(data["series"]) == 2


def test_curriculum_status_command(tmp_path, project, capsys):
    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=10),
        evaluation=EvaluationConfig(), name="nocurr", random_seed=1,
    )
    mgr.create(m)
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    orch = LocalTrainingOrchestrator(experiments_root=root)
    rid = orch.launch(m, mgr.experiment_dir(m.experiment_id), trainer="dummy")
    orch.wait(mgr.experiment_dir(m.experiment_id), rid, timeout_s=60)
    capsys.readouterr()
    rc = cli_main(["--root", root, "curriculum", m.experiment_id, rid])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no curriculum" in out.lower()


def test_dataset_export_no_trajectories(tmp_path, exp, capsys):
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    orch = LocalTrainingOrchestrator(experiments_root=exp["root"])
    rid = orch.launch(exp["manifest"], exp["exp_dir"], trainer="dummy")
    orch.wait(exp["exp_dir"], rid, timeout_s=60)
    capsys.readouterr()
    rc = cli_main(["--root", exp["root"], "dataset-export",
                   exp["exp_id"], rid, "--dest", str(tmp_path / "ds_out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "episodes" in out


def test_worker_serve_and_status(tmp_path):
    from sim_experiment.remote_worker import WorkerService, RemoteWorkerAdapter
    svc = WorkerService("127.0.0.1", 0, str(tmp_path / "exps"), token="tok")
    svc.start()
    try:
        rc = cli_main(["worker-status", "--host", "127.0.0.1",
                       "--port", str(svc.port), "--token", "tok"])
        assert rc == 0
    finally:
        svc.stop()
