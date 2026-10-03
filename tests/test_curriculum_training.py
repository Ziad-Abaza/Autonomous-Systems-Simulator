"""Phase 5 — curriculum integration: contract, checkpoints, resume, smoke run."""
import json
import os

import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.trainer_contract import build_contract, validate_contract
from sim_experiment.curriculum_runtime import curriculum_fingerprint
from sim_env.curriculum import CurriculumDefinition, CurriculumStage
from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def curriculum():
    return CurriculumDefinition(name="test_curr", stages=[
        CurriculumStage(stage_id=1, name="s1", description="",
                        scenario_id="basic_lane_following",
                        target_metric="mean_return",
                        advancement_threshold=-1e9, min_episodes=1,
                        environment_overrides={"time_limit": 0.4}),
        CurriculumStage(stage_id=2, name="s2", description="",
                        scenario_id="basic_lane_following",
                        target_metric="mean_return",
                        advancement_threshold=1e18, min_episodes=1),
    ])


@pytest.fixture
def manifest(project, curriculum):
    project.curriculum = curriculum
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=64,
                                rollout_length=32, batch_size=32, epochs=1,
                                eval_frequency=32, checkpoint_frequency=32,
                                num_envs=1),
        evaluation=EvaluationConfig(eval_seeds=[7], num_episodes=1),
        name="curr_test", random_seed=42,
    )
    return m


def test_contract_carries_curriculum(manifest):
    contract = build_contract(manifest, "exp", "run", "run_1")
    assert contract["curriculum"] is not None
    assert contract["curriculum_fingerprint"] == curriculum_fingerprint(
        manifest.curriculum_configuration
    )
    assert validate_contract(contract) == []


def test_contract_without_curriculum(project):
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=64),
        evaluation=EvaluationConfig(), name="nocurr", random_seed=1,
    )
    contract = build_contract(m, "exp", "run", "run_1")
    assert contract["curriculum"] is None
    assert validate_contract(contract) == []


def test_curriculum_rejected_over_tcp(manifest):
    contract = build_contract(manifest, "exp", "run", "run_1",
                              env_mode="tcp", tcp_ports=[9000])
    errors = validate_contract(contract)
    assert any("curriculum" in e and "tcp" in e.lower() for e in errors)


def test_invalid_curriculum_rejected_at_launch(tmp_path, project):
    project.curriculum = CurriculumDefinition(name="bad", stages=[
        CurriculumStage(stage_id=1, name="bad_stage", description="",
                        scenario_id="nope_scenario",
                        target_metric="mean_return",
                        advancement_threshold=0.0, min_episodes=1),
    ])
    manifest = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="dummy", total_timesteps=10),
        evaluation=EvaluationConfig(), name="badcurr", random_seed=1,
    )
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
    with pytest.raises(ValueError, match="[Cc]urriculum"):
        orch.launch(manifest, exp_dir, trainer="dummy")


class TestPPORunnerSetEnvs:
    def test_set_envs_forces_rereset(self, project):
        from sim_client.agents.ppo_baseline import PPORunner
        from sim_experiment.headless import build_env_from_dicts
        env1 = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        runner = PPORunner(env=env1, num_steps=8, num_epochs=1, batch_size=8, seed=3)
        env2 = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=99)
        runner.set_envs([env2])
        assert runner.envs[0] is env2
        assert runner._envs_dirty is True

    def test_set_envs_count_mismatch_rejected(self, project):
        from sim_client.agents.ppo_baseline import PPORunner
        from sim_experiment.headless import build_env_from_dicts
        env1 = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        runner = PPORunner(env=env1, num_steps=8, num_epochs=1, batch_size=8, seed=3)
        env2 = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=2)
        env3 = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=3)
        with pytest.raises(ValueError, match="env"):
            runner.set_envs([env2, env3])

    def test_checkpoint_extra_roundtrip(self, project, tmp_path):
        from sim_client.agents.ppo_baseline import PPORunner
        from sim_experiment.headless import build_env_from_dicts
        env = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        runner = PPORunner(env=env, num_steps=8, num_epochs=1, batch_size=8, seed=3)
        ck = str(tmp_path / "ck.pt")
        runner.save_checkpoint(ck, step=10, extra={"curriculum_state": {"stage_index": 2}})
        import torch
        data = torch.load(ck, map_location="cpu", weights_only=False)
        assert data["curriculum_state"]["stage_index"] == 2


def test_curriculum_smoke_run_advances(tmp_path, manifest):
    """Full PPO run with a trivially-advancing curriculum (threshold -1e9)."""
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo")
    summary = orch.wait(exp_dir, run_id, timeout_s=240)
    assert summary["status"] == "COMPLETED", summary.get("error")

    rd = os.path.join(exp_dir, "runs", run_id)
    state_path = os.path.join(rd, "curriculum_state.json")
    assert os.path.exists(state_path)
    with open(state_path) as f:
        state = json.load(f)
    assert state["stage_index"] == 1
    assert state["history"]

    # checkpoint carries curriculum state
    import torch
    ck = os.path.join(rd, "checkpoints", "policy_final.pt")
    data = torch.load(ck, map_location="cpu", weights_only=False)
    assert data["curriculum_state"]["stage_index"] == 1
    assert data["curriculum_state"]["curriculum_fingerprint"] == state["curriculum_fingerprint"]


def test_curriculum_resume_restores_stage(tmp_path, manifest):
    """Resume from a stage-1 checkpoint restores stage_index, doesn't restart."""
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo")
    assert orch.wait(exp_dir, run_id, timeout_s=240)["status"] == "COMPLETED"

    ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")
    run_id2 = orch.launch(manifest, exp_dir, trainer="ppo",
                          resume_from={"parent_run_id": run_id, "checkpoint": ck})
    summary = orch.wait(exp_dir, run_id2, timeout_s=240)
    assert summary["status"] == "COMPLETED", summary.get("error")
    with open(os.path.join(exp_dir, "runs", run_id2, "curriculum_state.json")) as f:
        state = json.load(f)
    # resumed run starts at stage 1 (final stage), not stage 0
    assert state["stage_index"] == 1
    assert state["stage_name"] == "s2"


def test_curriculum_resume_mismatch_fails(tmp_path, manifest, curriculum):
    """A checkpoint from a different curriculum must not silently resume."""
    mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo")
    assert orch.wait(exp_dir, run_id, timeout_s=240)["status"] == "COMPLETED"
    ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")

    # Build a different experiment with a different curriculum but try to
    # resume from the first run's checkpoint.
    project2 = EnvironmentTemplateManager.create_project_from_template("lane_following")
    diff = CurriculumDefinition(name="different_curr", stages=[
        CurriculumStage(stage_id=1, name="x", description="",
                        scenario_id="basic_lane_following",
                        target_metric="mean_return",
                        advancement_threshold=0.0, min_episodes=1),
    ])
    project2.curriculum = diff
    m2 = ExperimentManifest.from_project(
        project=project2, scenario=project2.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=32,
                                rollout_length=32, batch_size=32, epochs=1),
        evaluation=EvaluationConfig(eval_seeds=[7], num_episodes=1),
        name="curr_other", random_seed=42,
    )
    exp_dir2 = mgr.create(m2)
    run_id2 = orch.launch(m2, exp_dir2, trainer="ppo",
                          resume_from={"parent_run_id": run_id, "checkpoint": ck})
    summary = orch.wait(exp_dir2, run_id2, timeout_s=240)
    assert summary["status"] == "FAILED"
    assert "curriculum" in str(summary.get("error", "")).lower()
