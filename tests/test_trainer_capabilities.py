"""Phase 5 — trainer capability declaration + pre-launch compatibility."""
import pytest

from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.capabilities import (
    TRAINER_CAPABILITIES, check_compatibility, describe_trainer,
)
from sim_env.templates import EnvironmentTemplateManager
from sim_env.action_designer import ActionSpaceDefinition


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


def _manifest(project, algorithm="ppo", **kw):
    return ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm=algorithm, total_timesteps=32, **kw),
        evaluation=EvaluationConfig(), name="cap_test", random_seed=1,
    )


class TestCapabilityDeclarations:
    def test_all_registered_trainers_declared(self):
        modules = LocalTrainingOrchestrator.trainer_modules()
        for name in modules:
            assert name in TRAINER_CAPABILITIES, f"{name} missing capabilities"

    def test_ppo_declares_continuous_only(self):
        cap = TRAINER_CAPABILITIES["ppo"]
        assert "ppo" in cap["algorithms"]
        assert "continuous" in cap["action_types"]
        assert "discrete" not in cap["action_types"]

    def test_sac_declares_continuous_only(self):
        assert "continuous" in TRAINER_CAPABILITIES["sac"]["action_types"]
        assert "discrete" not in TRAINER_CAPABILITIES["sac"]["action_types"]

    def test_dqn_declares_discrete_only(self):
        assert TRAINER_CAPABILITIES["dqn"]["action_types"] == ["discrete"]

    def test_describe_trainer_shape(self):
        d = describe_trainer("ppo")
        for k in ("algorithms", "action_types", "observation_types",
                  "multi_env", "checkpoint_format", "evaluation"):
            assert k in d


class TestCompatibility:
    def test_ppo_continuous_ok(self, project):
        assert check_compatibility(_manifest(project, "ppo"), "ppo") == []

    def test_dqn_continuous_env_rejected(self, project):
        errors = check_compatibility(_manifest(project, "dqn"), "dqn")
        assert any("discrete" in e.lower() or "action" in e.lower() for e in errors)

    def test_dqn_discrete_env_ok(self, project):
        project.agent.action_space = (
            ActionSpaceDefinition.create_default_vehicle_action_space(continuous=False))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(algorithm="dqn", total_timesteps=32),
            evaluation=EvaluationConfig(), name="dqn_env", random_seed=1,
        )
        assert check_compatibility(m, "dqn") == []

    def test_ppo_on_discrete_env_rejected(self, project):
        project.agent.action_space = (
            ActionSpaceDefinition.create_default_vehicle_action_space(continuous=False))
        m = ExperimentManifest.from_project(
            project=project, scenario=project.scenario_def,
            training=TrainingConfig(algorithm="ppo", total_timesteps=32),
            evaluation=EvaluationConfig(), name="ppo_disc", random_seed=1,
        )
        errors = check_compatibility(m, "ppo")
        assert errors

    def test_algorithm_mismatch_rejected(self, project):
        errors = check_compatibility(_manifest(project, "sac"), "ppo")
        assert any("algorithm" in e.lower() for e in errors)

    def test_unknown_trainer_rejected(self, project):
        errors = check_compatibility(_manifest(project, "ppo"), "nonexistent_trainer")
        assert errors

    def test_launch_blocks_incompatible_before_process(self, tmp_path, project):
        """DQN + continuous env must fail before any subprocess launches."""
        m = _manifest(project, "dqn")
        mgr = ExperimentManager(root_dir=str(tmp_path / "experiments"))
        exp_dir = mgr.create(m)
        orch = LocalTrainingOrchestrator(experiments_root=mgr.root_dir)
        with pytest.raises(ValueError, match="[Ii]ncompatible|discrete|action"):
            orch.launch(m, exp_dir, trainer="dqn")
