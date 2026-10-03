"""Phase 5 — observation contract hardening + ep_len regression tests."""
import numpy as np
import pytest

from sim_env.templates import EnvironmentTemplateManager


@pytest.fixture
def project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


class TestDiagnosticStateContract:
    def test_diagnostic_fields_declared(self):
        from sim_env.observation_contract import diagnostic_state_contract
        contract = diagnostic_state_contract()
        assert "speed" in contract
        assert contract["speed"]["classification"] == "diagnostic"
        assert "reward_breakdown" in contract

    def test_no_agent_fields_in_diagnostic_contract(self):
        """Diagnostic contract must never silently claim agent fields."""
        from sim_env.observation_contract import diagnostic_state_contract
        for meta in diagnostic_state_contract().values():
            assert meta["classification"] == "diagnostic"

    def test_env_builds_diagnostic_state(self, project):
        from sim_experiment.headless import build_env_from_dicts
        from sim_env.observation_contract import build_diagnostic_state
        env = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        env.reset()
        env.step([0.0, 0.5, 0.0])
        st = build_diagnostic_state(env)
        assert "speed" in st and "pos" in st and "sim_time" in st
        assert "total_reward" in st and "reward_breakdown" in st


class TestChannelClassification:
    def test_classification_map(self, project):
        from sim_env.observation_contract import channel_classification_map
        m = channel_classification_map(project.agent)
        assert m
        for name, cat in m.items():
            assert cat in ("agent_observation", "debug_telemetry",
                           "oracle_ground_truth")

    def test_leakage_is_structural_not_naming(self, project):
        """A channel named 'speed' with oracle category must be flagged;
        a channel named 'oracle_x' with agent category must pass."""
        from sim_env.observation_designer import ObservationChannelConfig
        obs = project.agent.observation_space
        # safe name, privileged category -> violation
        obs.channels.append(ObservationChannelConfig(
            name="speed", channel_type="scalar", category="oracle_ground_truth",
            source_sensor="vehicle_state", source_key="speed"))
        ok, issues = obs.validate_no_leakage()
        assert not ok and issues
        obs.channels.pop()
        # scary name, agent category -> fine
        obs.channels.append(ObservationChannelConfig(
            name="oracle_ground_truth_speed", channel_type="scalar",
            category="agent_observation",
            source_sensor="vehicle_state", source_key="speed"))
        ok, _ = obs.validate_no_leakage()
        assert ok
        obs.channels.pop()


class TestSensorNames:
    def test_validator_uses_agent_sensor_names(self, project):
        """Validator must check channels against agent-declared sensors,
        not a hardcoded fallback list."""
        from sim_env.validator import EnvironmentValidator
        project.agent.sensor_names = ["vehicle_state"]  # no lidar/imu declared
        obs = project.agent.observation_space
        # add a channel referencing a sensor the agent does not declare
        from sim_env.observation_designer import ObservationChannelConfig
        obs.channels.append(ObservationChannelConfig(
            name="custom_ray", channel_type="vector", shape=(4,),
            source_sensor="custom_lidar", source_key="ranges_norm"))
        report = EnvironmentValidator.validate(
            road_def=project.road_def, agent=project.agent,
            entities=project.entities)
        joined = " ".join(i.message for e in report.errors for i in [e])
        assert "custom_lidar" in joined
        obs.channels.pop()


class TestEpisodeLengthConsistency:
    def test_env_step_count_matches_info(self, project):
        """info['step'] is the authoritative episode length."""
        from sim_experiment.headless import build_env_from_dicts
        env = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        env.reset()
        for i in range(5):
            _, _, _, _, info = env.step([0.0, 0.5, 0.0])
        assert info["step"] == 5
        assert env.current_step == 5

    def test_eval_episode_length_uses_authoritative_count(self, project):
        """eval length == env's own step counter, not a parallel counter."""
        from sim_experiment.headless import build_env_from_dicts
        from sim_experiment.evaluation import evaluate_policy
        env = build_env_from_dicts(project.to_dict(), project.scenario_def.to_dict(), seed=1)
        # scenario with tiny time limit forces truncation quickly
        scen = project.scenario_def.to_dict()
        scen["time_limit_override"] = 0.2   # 12 steps at 60 Hz
        env = build_env_from_dicts(project.to_dict(), scen, seed=1)
        result = evaluate_policy(env, lambda o: [0.0, 0.5, 0.0],
                                 seeds=[1], num_episodes=1)
        assert result.episodes[0]["timed_out"] is True
        # reported length equals the env's own authoritative step counter
        assert result.episodes[0]["length"] == env.current_step
        assert result.episodes[0]["length"] > 0
