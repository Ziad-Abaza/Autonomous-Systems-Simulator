"""Phase 5 — Curriculum runtime controller tests."""
import pytest

from sim_env.curriculum import CurriculumDefinition, CurriculumStage
from sim_env.scenario_designer import ScenarioDefinition


def _two_stage_curriculum():
    return CurriculumDefinition(
        name="test_curr",
        stages=[
            CurriculumStage(
                stage_id=1, name="easy", description="",
                scenario_id="basic_lane_following",
                target_metric="mean_return", advancement_threshold=100.0,
                min_episodes=3,
                environment_overrides={"target_speed": 8.0},
            ),
            CurriculumStage(
                stage_id=2, name="hard", description="",
                scenario_id="wet_adverse_weather",
                target_metric="lap_completion_rate", advancement_threshold=0.5,
                min_episodes=2,
            ),
        ],
    )


class TestValidation:
    def test_valid_definition(self):
        from sim_experiment.curriculum_runtime import validate_curriculum
        assert validate_curriculum(_two_stage_curriculum().to_dict()) == []

    def test_empty_stages_invalid(self):
        from sim_experiment.curriculum_runtime import validate_curriculum
        d = CurriculumDefinition(name="x", stages=[]).to_dict()
        errors = validate_curriculum(d)
        assert any("stage" in e.lower() for e in errors)

    def test_unknown_scenario_id(self):
        from sim_experiment.curriculum_runtime import validate_curriculum
        d = _two_stage_curriculum().to_dict()
        d["stages"][0]["scenario_id"] = "does_not_exist"
        errors = validate_curriculum(d)
        assert any("does_not_exist" in e for e in errors)

    def test_unknown_metric(self):
        from sim_experiment.curriculum_runtime import validate_curriculum
        d = _two_stage_curriculum().to_dict()
        d["stages"][0]["target_metric"] = "nonsense_metric"
        errors = validate_curriculum(d)
        assert any("metric" in e.lower() for e in errors)

    def test_min_episodes_positive(self):
        from sim_experiment.curriculum_runtime import validate_curriculum
        d = _two_stage_curriculum().to_dict()
        d["stages"][0]["min_episodes"] = 0
        errors = validate_curriculum(d)
        assert any("min_episodes" in e for e in errors)


class TestController:
    def test_initial_stage(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        assert c.stage_index == 0
        assert c.current_stage.name == "easy"
        assert c.episodes_in_stage == 0
        assert not c.is_complete

    def test_stage_scenario_resolution_and_overrides(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        scen = c.stage_scenario_dict()
        assert scen["scenario_id"] == "basic_lane_following"
        # environment_overrides {"target_speed": 8.0} -> target_speed_override
        assert scen["target_speed_override"] == 8.0

    def test_episode_counting(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(2)
        assert c.episodes_in_stage == 2

    def test_no_advance_below_min_episodes(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(2)  # min_episodes=3
        rec = c.evaluate_advancement({"mean_reward": 500.0}, timestep=100)
        assert rec is not None and rec["advanced"] is False
        assert c.stage_index == 0

    def test_no_advance_below_threshold(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(5)
        rec = c.evaluate_advancement({"mean_reward": 50.0}, timestep=100)
        assert rec["advanced"] is False
        assert c.stage_index == 0

    def test_advance_when_threshold_and_episodes_met(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(4)
        rec = c.evaluate_advancement({"mean_reward": 250.0}, timestep=100)
        assert rec["advanced"] is True
        assert rec["metric"] == "mean_return"
        assert rec["value"] == 250.0
        assert rec["threshold"] == 100.0
        assert rec["episodes"] == 4
        assert c.stage_index == 1
        assert c.episodes_in_stage == 0
        assert c.current_stage.name == "hard"
        # new stage scenario resolves
        assert c.stage_scenario_dict()["scenario_id"] == "wet_adverse_weather"

    def test_metric_alias_mean_return(self):
        """'mean_return' maps to evaluation aggregate 'mean_reward'."""
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(4)
        rec = c.evaluate_advancement({"mean_reward": 101.0}, timestep=1)
        assert rec["advanced"] is True

    def test_metric_alias_completion_rate(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(4)
        c.evaluate_advancement({"mean_reward": 200.0}, timestep=1)
        c.record_training_episodes(3)
        rec = c.evaluate_advancement({"completion_rate": 0.9}, timestep=2)
        # stage 2 is the final stage: threshold met but nowhere to advance
        assert rec["advanced"] is False
        assert rec["value"] == 0.9
        assert c.is_complete

    def test_missing_metric_no_advance(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(10)
        rec = c.evaluate_advancement({"something_else": 1.0}, timestep=1)
        assert rec["advanced"] is False
        assert rec["metric_value_available"] is False

    def test_lower_is_better_direction(self):
        """collision_rate should advance when value <= threshold."""
        from sim_experiment.curriculum_runtime import CurriculumController
        d = CurriculumDefinition(name="c", stages=[
            CurriculumStage(stage_id=1, name="a", description="",
                            scenario_id="basic_lane_following",
                            target_metric="collision_rate",
                            advancement_threshold=0.1, min_episodes=1),
            CurriculumStage(stage_id=2, name="b", description="",
                            scenario_id="basic_lane_following",
                            target_metric="mean_return",
                            advancement_threshold=1.0, min_episodes=1),
        ])
        c = CurriculumController(d, base_seed=1)
        c.record_training_episodes(1)
        rec = c.evaluate_advancement({"collision_rate": 0.05}, timestep=1)
        assert rec["advanced"] is True

    def test_final_stage_never_advances(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        d = CurriculumDefinition(name="c", stages=[
            CurriculumStage(stage_id=1, name="only", description="",
                            scenario_id="basic_lane_following",
                            target_metric="mean_return",
                            advancement_threshold=1.0, min_episodes=1),
        ])
        c = CurriculumController(d, base_seed=1)
        c.record_training_episodes(5)
        rec = c.evaluate_advancement({"mean_reward": 999.0}, timestep=1)
        assert rec["advanced"] is False
        assert c.is_complete

    def test_history_recorded(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(4)
        c.evaluate_advancement({"mean_reward": 200.0}, timestep=7)
        assert len(c.history) == 1
        assert c.history[0]["advanced"] is True
        assert c.history[0]["timestep"] == 7

    def test_state_roundtrip(self):
        from sim_experiment.curriculum_runtime import CurriculumController
        c = CurriculumController(_two_stage_curriculum(), base_seed=42)
        c.record_training_episodes(4)
        c.evaluate_advancement({"mean_reward": 200.0}, timestep=7)
        c.record_training_episodes(2)
        state = c.to_state()
        c2 = CurriculumController.from_state(_two_stage_curriculum(), state)
        assert c2.stage_index == 1
        assert c2.episodes_in_stage == 2
        assert len(c2.history) == 1

    def test_fingerprint_deterministic(self):
        from sim_experiment.curriculum_runtime import curriculum_fingerprint
        d = _two_stage_curriculum().to_dict()
        assert curriculum_fingerprint(d) == curriculum_fingerprint(d)
        d2 = _two_stage_curriculum().to_dict()
        d2["stages"][0]["advancement_threshold"] = 999.0
        assert curriculum_fingerprint(d) != curriculum_fingerprint(d2)
