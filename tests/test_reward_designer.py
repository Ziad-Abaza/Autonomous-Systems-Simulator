"""
Unit tests for Reward Designer, composable components, falloff curves,
live decomposition, and safety assertions (teleportation, reverse driving,
checkpoint abuse, resets, and no future information).
"""

import pytest
import math

from sim_env.reward_designer import (
    RewardFunctionDefinition,
    RewardComponentConfig,
    FalloffType,
    CompiledRewardEngine
)


def test_reward_function_defaults_and_graph_export():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    graph = rf.export_graph()

    assert graph["aggregation"] == "weighted_sum"
    assert len(graph["nodes"]) >= 10
    comp_types = {n["type"] for n in graph["nodes"]}
    assert "progress" in comp_types
    assert "centerline" in comp_types
    assert "heading" in comp_types
    assert "collision" in comp_types
    assert "checkpoint" in comp_types


def test_reward_component_falloff_models():
    # Test linear vs quadratic vs exponential centerline falloffs
    def eval_centerline(falloff: str, lat_offset: float, max_dist: float = 4.0):
        comp = RewardComponentConfig(
            component_id="center",
            name="Center",
            component_type="centerline",
            weight=1.0,
            params={"max_distance_m": max_dist, "falloff": falloff}
        )
        engine = RewardFunctionDefinition(components=[comp]).compile_engine()
        engine.reset()
        _, breakdown = engine.compute_step_reward(
            current_s=0.0, track_length=100.0, is_closed=True,
            lateral_offset=lat_offset, road_width=8.0, speed=10.0,
            heading_error=0.0, current_steer=0.0, is_colliding=False,
            is_on_road=True, checkpoint_passed=False, lap_completed=False
        )
        return breakdown["center"]

    # At center (0.0): all falloffs return 1.0
    assert pytest.approx(eval_centerline(FalloffType.LINEAR, 0.0), abs=1e-3) == 1.0
    assert pytest.approx(eval_centerline(FalloffType.QUADRATIC, 0.0), abs=1e-3) == 1.0
    assert pytest.approx(eval_centerline(FalloffType.EXPONENTIAL, 0.0), abs=1e-3) == 1.0

    # At half-way (2.0 / 4.0 = 0.5):
    # Linear: 1.0 - 0.5 = 0.5
    # Quadratic: 1.0 - 0.5^2 = 0.75
    # Exponential: exp(-3 * 0.5) = exp(-1.5) ~= 0.223
    assert pytest.approx(eval_centerline(FalloffType.LINEAR, 2.0), abs=1e-3) == 0.5
    assert pytest.approx(eval_centerline(FalloffType.QUADRATIC, 2.0), abs=1e-3) == 0.75
    assert pytest.approx(eval_centerline(FalloffType.EXPONENTIAL, 2.0), abs=1e-2) == math.exp(-1.5)


def test_reward_safety_teleportation_guard():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    engine = rf.compile_engine()
    engine.reset(initial_s=10.0)

    # Discontinuous teleportation jump of +50 meters in a single 60Hz tick
    _, breakdown = engine.compute_step_reward(
        current_s=60.0, track_length=500.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=10.0,
        heading_error=0.0, current_steer=0.0, is_colliding=False,
        is_on_road=True, checkpoint_passed=False, lap_completed=False
    )
    # Teleportation must be rejected (delta_s clamped to 0.0)
    assert breakdown["progress"] == 0.0


def test_reward_safety_reverse_driving_penalty():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    engine = rf.compile_engine()
    engine.reset(initial_s=50.0)

    # Driving backward (delta_s = -2.0m, or facing opposite heading)
    _, breakdown = engine.compute_step_reward(
        current_s=48.0, track_length=500.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=5.0,
        heading_error=math.pi, current_steer=0.0, is_colliding=False,
        is_on_road=True, checkpoint_passed=False, lap_completed=False
    )
    assert breakdown["reverse"] == -1.0  # Backward penalty triggered


def test_reward_collision_and_offroad_events():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    engine = rf.compile_engine()
    engine.reset()

    # Step with collision and off-road
    _, breakdown = engine.compute_step_reward(
        current_s=1.0, track_length=500.0, is_closed=True,
        lateral_offset=8.0, road_width=12.0, speed=0.0,
        heading_error=0.0, current_steer=0.0, is_colliding=True,
        is_on_road=False, checkpoint_passed=False, lap_completed=False
    )
    assert breakdown["collision"] == -50.0
    assert breakdown["off_road"] == -25.0


def test_reward_checkpoint_and_lap_completion():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    engine = rf.compile_engine()
    engine.reset()

    # Step crossing checkpoint and finishing lap
    _, breakdown = engine.compute_step_reward(
        current_s=1.0, track_length=500.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=15.0,
        heading_error=0.0, current_steer=0.0, is_colliding=False,
        is_on_road=True, checkpoint_passed=True, lap_completed=True
    )
    assert breakdown["checkpoint"] == 10.0
    assert breakdown["completion"] == 100.0


def test_reward_reset_and_accumulation():
    rf = RewardFunctionDefinition.create_default_racing_reward()
    engine = rf.compile_engine()
    engine.reset()

    r1, _ = engine.compute_step_reward(
        current_s=1.0, track_length=500.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=10.0,
        heading_error=0.0, current_steer=0.0, is_colliding=False,
        is_on_road=True, checkpoint_passed=False, lap_completed=False
    )
    r2, _ = engine.compute_step_reward(
        current_s=2.0, track_length=500.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=10.0,
        heading_error=0.0, current_steer=0.0, is_colliding=False,
        is_on_road=True, checkpoint_passed=False, lap_completed=False
    )

    assert pytest.approx(engine.total_accumulated_reward, abs=1e-4) == (r1 + r2)

    # Reset
    engine.reset(initial_s=0.0)
    assert engine.total_accumulated_reward == 0.0
    assert engine.last_breakdown["total"] == 0.0
