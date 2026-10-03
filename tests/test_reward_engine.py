"""
Unit tests for reward calculation and component decomposition.
"""

import pytest
from sim_env.reward_engine import RewardEngine, RewardConfig


def test_reward_decomposition():
    engine = RewardEngine(RewardConfig(
        weight_progress=1.0,
        weight_centering=1.0,
        collision_penalty=50.0,
        checkpoint_bonus=10.0
    ))
    engine.reset(initial_s=0.0)

    # Step forward 2.0 meters, centered, no collision
    r, breakdown = engine.compute_step_reward(
        current_s=2.0,
        track_length=200.0,
        is_closed=True,
        lateral_offset=0.0,
        road_width=12.0,
        speed=10.0,
        heading_error=0.0,
        current_steer=0.0,
        is_colliding=False,
        is_on_road=True,
        checkpoint_passed=False,
        lap_completed=False
    )

    assert breakdown['progress'] == 2.0
    assert breakdown['centering'] == 1.0
    assert breakdown['collision'] == 0.0
    assert breakdown['checkpoint'] == 0.0
    assert r > 3.0

    # Step with collision
    r_col, breakdown_col = engine.compute_step_reward(
        current_s=2.1,
        track_length=200.0,
        is_closed=True,
        lateral_offset=6.0,
        road_width=12.0,
        speed=0.0,
        heading_error=0.0,
        current_steer=0.0,
        is_colliding=True,
        is_on_road=True,
        checkpoint_passed=False,
        lap_completed=False
    )
    assert breakdown_col['collision'] == -50.0
    assert r_col < -40.0
