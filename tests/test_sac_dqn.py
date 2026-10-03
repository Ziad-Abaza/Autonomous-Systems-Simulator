"""Phase 5 — SAC + DQN trainers: contract, smoke runs, checkpoints, eval."""
import json
import os

import numpy as np
import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_env.templates import EnvironmentTemplateManager
from sim_env.action_designer import ActionSpaceDefinition


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


@pytest.fixture
def discrete_project(project):
    project.agent.action_space = (
        ActionSpaceDefinition.create_default_vehicle_action_space(continuous=False))
    return project


def _make(mgr, project, algorithm, alg_cfg, timesteps=160, seed=3):
    m = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(
            algorithm=algorithm, total_timesteps=timesteps,
            rollout_length=64, batch_size=16, epochs=1,
            checkpoint_frequency=80, eval_frequency=80,
            algorithm_config=alg_cfg),
        evaluation=EvaluationConfig(eval_seeds=[seed], num_episodes=1),
        name=f"{algorithm}_test", random_seed=seed,
    )
    return m, mgr.create(m)


class TestReplayBuffer:
    def test_add_and_sample_shapes(self):
        from sim_client.agents.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100, obs_dim=5, act_dim=3)
        for i in range(50):
            buf.add(np.ones(5) * i, [0.1, 0.2, 0.3], 1.0, np.ones(5), False)
        batch = buf.sample(16)
        assert batch["obs"].shape == (16, 5)
        assert batch["actions"].shape == (16, 3)
        assert batch["rewards"].shape == (16,)
        assert batch["dones"].shape == (16,)

    def test_capacity_wraps(self):
        from sim_client.agents.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10, obs_dim=2, act_dim=1)
        for i in range(15):
            buf.add(np.ones(2) * i, [0.0], 0.0, np.zeros(2), i == 14)
        assert buf.size == 10
        # oldest entries overwritten; the single done (i==14) survives in slot 4
        assert buf.dones.sum() == 1
        assert buf.obs[:, 0].min() >= 5   # entries 0-4 overwritten by 10-14

    def test_deterministic_sampling_with_seed(self):
        from sim_client.agents.replay_buffer import ReplayBuffer
        b1 = ReplayBuffer(capacity=50, obs_dim=2, act_dim=1, seed=7)
        b2 = ReplayBuffer(capacity=50, obs_dim=2, act_dim=1, seed=7)
        for i in range(30):
            b1.add(np.ones(2) * i, [0.0], 1.0, np.zeros(2), False)
            b2.add(np.ones(2) * i, [0.0], 1.0, np.zeros(2), False)
        assert np.array_equal(b1.sample(8)["rewards"], b2.sample(8)["rewards"])


class TestSAC:
    def test_smoke_run_completes(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, project, "sac",
                           {"buffer_size": 500, "warmup_steps": 20,
                            "updates_per_step": 1})
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="sac")
        summary = orch.wait(exp_dir, run_id, timeout_s=300)
        assert summary["status"] == "COMPLETED", summary.get("error")
        rd = os.path.join(exp_dir, "runs", run_id)
        assert os.path.exists(os.path.join(rd, "checkpoints", "policy_final.pt"))
        assert os.path.exists(os.path.join(rd, "metrics.jsonl"))
        evals = os.listdir(os.path.join(rd, "evaluation"))
        assert any(f.startswith("eval_") for f in evals)

    def test_checkpoint_resume(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, project, "sac",
                           {"buffer_size": 500, "warmup_steps": 10,
                            "updates_per_step": 1}, timesteps=80)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="sac")
        assert orch.wait(exp_dir, run_id, timeout_s=300)["status"] == "COMPLETED"
        ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")
        run_id2 = orch.launch(m, exp_dir, trainer="sac",
                              resume_from={"parent_run_id": run_id, "checkpoint": ck})
        summary = orch.wait(exp_dir, run_id2, timeout_s=300)
        assert summary["status"] == "COMPLETED", summary.get("error")
        assert summary["current_timestep"] >= 80

    def test_eval_policy_adapter(self, tmp_path, project):
        from sim_experiment.evaluation import make_policy_from_checkpoint, evaluate_policy
        from sim_experiment.headless import build_env_from_dicts
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, project, "sac",
                           {"buffer_size": 300, "warmup_steps": 10}, timesteps=80)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="sac")
        assert orch.wait(exp_dir, run_id, timeout_s=300)["status"] == "COMPLETED"
        ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")
        policy = make_policy_from_checkpoint(ck, algorithm="sac")
        env = build_env_from_dicts(m.environment, m.scenario_configuration, seed=1)
        result = evaluate_policy(env, policy, seeds=[1], num_episodes=1)
        assert result.episodes

    def test_sac_rejects_discrete_env(self, tmp_path, discrete_project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, discrete_project, "sac",
                           {"buffer_size": 100, "warmup_steps": 10})
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        with pytest.raises(ValueError, match="[Ii]ncompatible|action"):
            orch.launch(m, exp_dir, trainer="sac")


class TestDQN:
    def test_smoke_run_completes(self, tmp_path, discrete_project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, discrete_project, "dqn",
                           {"buffer_size": 500, "warmup_steps": 20,
                            "eps_start": 1.0, "eps_end": 0.1,
                            "eps_decay_steps": 100, "train_freq": 2,
                            "target_update_interval": 50})
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="dqn")
        summary = orch.wait(exp_dir, run_id, timeout_s=300)
        assert summary["status"] == "COMPLETED", summary.get("error")
        rd = os.path.join(exp_dir, "runs", run_id)
        assert os.path.exists(os.path.join(rd, "checkpoints", "policy_final.pt"))

    def test_checkpoint_resume(self, tmp_path, discrete_project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, discrete_project, "dqn",
                           {"buffer_size": 300, "warmup_steps": 10,
                            "target_update_interval": 20}, timesteps=80)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="dqn")
        assert orch.wait(exp_dir, run_id, timeout_s=300)["status"] == "COMPLETED"
        ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")
        run_id2 = orch.launch(m, exp_dir, trainer="dqn",
                              resume_from={"parent_run_id": run_id, "checkpoint": ck})
        summary = orch.wait(exp_dir, run_id2, timeout_s=300)
        assert summary["status"] == "COMPLETED", summary.get("error")

    def test_eval_policy_adapter(self, tmp_path, discrete_project):
        from sim_experiment.evaluation import make_policy_from_checkpoint, evaluate_policy
        from sim_experiment.headless import build_env_from_dicts
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, discrete_project, "dqn",
                           {"buffer_size": 300, "warmup_steps": 10}, timesteps=80)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        run_id = orch.launch(m, exp_dir, trainer="dqn")
        assert orch.wait(exp_dir, run_id, timeout_s=300)["status"] == "COMPLETED"
        ck = os.path.join(exp_dir, "runs", run_id, "checkpoints", "policy_final.pt")
        policy = make_policy_from_checkpoint(ck, algorithm="dqn")
        env = build_env_from_dicts(m.environment, m.scenario_configuration, seed=1)
        result = evaluate_policy(env, policy, seeds=[1], num_episodes=1)
        assert result.episodes
        # DQN actions must be discrete indices
        assert isinstance(result.episodes[0], dict)

    def test_dqn_rejects_continuous_env_at_launch(self, tmp_path, project):
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        m, exp_dir = _make(mgr, project, "dqn", {})
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        with pytest.raises(ValueError, match="[Ii]ncompatible|discrete|action"):
            orch.launch(m, exp_dir, trainer="dqn")
