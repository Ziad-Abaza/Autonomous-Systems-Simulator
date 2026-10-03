"""
Unit tests for Environment Validator, RL Readiness Gate, and diagnostic issue reporting.
"""

import pytest

from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint
from sim_core.world.entity import create_entity
from sim_core.math_utils import Vec3
from sim_env.agent import AgentDefinition
from sim_env.observation_designer import ObservationSpaceDefinition, ObservationChannelConfig, ChannelCategory
from sim_env.action_designer import ActionSpaceDefinition, ActionChannelConfig
from sim_env.reward_designer import RewardFunctionDefinition
from sim_env.termination_designer import TerminationDefinition
from sim_env.validator import EnvironmentValidator, IssueSeverity


def test_validator_on_healthy_environment():
    road = RoadDefinition.create_default_oval()
    agent = AgentDefinition.create_default_vehicle_agent()
    report = EnvironmentValidator.validate(road_def=road, agent=agent)

    assert report.is_valid_for_rl is True
    assert len(report.errors) == 0
    assert len(report.infos) > 0


def test_validator_detects_missing_agent():
    road = RoadDefinition.create_default_oval()
    report = EnvironmentValidator.validate(road_def=road, agent=None)

    assert report.is_valid_for_rl is False
    assert any("No agent" in e.message for e in report.errors)


def test_validator_detects_degenerate_track():
    # Only 2 control points
    road = RoadDefinition(name="Degenerate Track")
    road.add_control_point(0.0, 0.0, 0.0, 10.0)
    road.add_control_point(10.0, 0.0, 0.0, 10.0)
    agent = AgentDefinition.create_default_vehicle_agent()

    report = EnvironmentValidator.validate(road_def=road, agent=agent)
    assert report.is_valid_for_rl is False
    assert any("control points" in e.message for e in report.errors)


def test_validator_detects_spawn_obstacle_overlap():
    road = RoadDefinition.create_default_oval()
    road.spawn_point = SpawnPoint(x=10.0, y=20.0, z=0.0)
    agent = AgentDefinition.create_default_vehicle_agent()

    # Place a collidable barrier directly on top of spawn point
    barrier = create_entity("barrier", pos=Vec3(10.5, 20.2, 0.0))
    report = EnvironmentValidator.validate(road_def=road, agent=agent, entities=[barrier])

    assert report.is_valid_for_rl is False
    assert any("inside or too close" in e.message for e in report.errors)


def test_validator_detects_observation_leakage():
    road = RoadDefinition.create_default_oval()
    agent = AgentDefinition.create_default_vehicle_agent()

    # Inject debug/oracle channel into agent observation
    agent.observation_space.add_channel(ObservationChannelConfig(
        name="oracle_leak",
        category=ChannelCategory.ORACLE_GROUND_TRUTH,
        source_sensor="oracle",
        source_key="secret"
    ))

    report = EnvironmentValidator.validate(road_def=road, agent=agent)
    assert report.is_valid_for_rl is False
    assert any("SECURITY LEAKAGE" in e.message for e in report.errors)


def test_validator_detects_missing_termination_conditions():
    road = RoadDefinition.create_default_oval()
    agent = AgentDefinition.create_default_vehicle_agent()
    # Disable all termination rules
    for r in agent.termination_rules.rules:
        r.enabled = False

    report = EnvironmentValidator.validate(road_def=road, agent=agent)
    assert report.is_valid_for_rl is False
    assert any("No active termination" in e.message for e in report.errors)
