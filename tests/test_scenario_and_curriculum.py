"""
Unit tests for Scenario Designer, environmental overrides, serialization,
and Curriculum sequential progression.
"""

import pytest

from sim_env.scenario_designer import ScenarioDefinition
from sim_env.curriculum import CurriculumDefinition, CurriculumStage


def test_standard_scenarios_library():
    scenarios = ScenarioDefinition.get_standard_scenarios()
    assert "basic_lane_following" in scenarios
    assert "high_speed_racing" in scenarios
    assert "wet_adverse_weather" in scenarios
    assert "obstacle_evasion" in scenarios

    wet = scenarios["wet_adverse_weather"]
    assert wet.weather == "rain"
    assert wet.surface_friction_mult < 1.0

    obs_scen = scenarios["obstacle_evasion"]
    assert len(obs_scen.obstacle_overrides) >= 2


def test_scenario_serialization_roundtrip():
    scen = ScenarioDefinition(
        scenario_id="custom_night_run",
        name="Custom Night Run",
        description="Testing headlights and low visibility",
        weather="clear",
        time_of_day="night",
        ambient_light=0.2,
        surface_friction_mult=0.9,
        target_speed_override=25.0
    )
    d = scen.to_dict()
    restored = ScenarioDefinition.from_dict(d)

    assert restored.scenario_id == "custom_night_run"
    assert restored.ambient_light == 0.2
    assert restored.target_speed_override == 25.0


def test_curriculum_stages_and_advancement():
    curriculum = CurriculumDefinition.create_default()
    assert len(curriculum.stages) >= 4
    assert curriculum.current_stage_idx == 0

    s1 = curriculum.get_current_stage()
    assert s1.stage_id == 1
    assert "Lane Keeping" in s1.name

    # Advance stage
    adv = curriculum.advance_stage()
    assert adv is True
    assert curriculum.current_stage_idx == 1
    s2 = curriculum.get_current_stage()
    assert s2.stage_id == 2
    assert "High Speed" in s2.name


def test_curriculum_serialization():
    curriculum = CurriculumDefinition.create_default()
    curriculum.advance_stage()
    d = curriculum.to_dict()
    restored = CurriculumDefinition.from_dict(d)

    assert restored.current_stage_idx == 1
    assert len(restored.stages) == len(curriculum.stages)
    assert restored.stages[0].name == curriculum.stages[0].name


def test_scenario_runtime_overrides():
    from sim_env.environment import SimulationEnvironment
    from sim_env.agent import AgentDefinition

    agent = AgentDefinition.create_default_vehicle_agent()
    scen = ScenarioDefinition(
        scenario_id="speed_override_run",
        name="Speed Override Run",
        target_speed_override=32.0,
        spawn_override={"pos": [15.0, 25.0], "yaw_deg": 45.0, "initial_speed": 10.0}
    )

    env = SimulationEnvironment(agent=agent, scenario_def=scen)
    obs, info = env.reset()

    # Verify spawn override took effect
    assert env.vehicle.state.pos.x == 15.0
    assert env.vehicle.state.pos.y == 25.0
    assert abs(env.vehicle.state.speed - 10.0) < 1e-3

    # Verify target speed override applied to active speed reward component
    speed_comp = [c for c in env.compiled_reward_engine.active_components if c.component_type == "speed"]
    assert len(speed_comp) == 1
    assert speed_comp[0].params["target_speed_ms"] == 32.0

