"""
Unit tests for Reward Configuration, Authoring Validation, and Decomposition.
Validates that reward components are configurable, validated against negative weights,
and cleanly decomposed at runtime.
"""

import pytest
from sim_env.reward_engine import RewardConfig, RewardEngine


def test_reward_config_validation():
    # Valid config
    cfg = RewardConfig()
    assert cfg.validate() is True

    # Invalid negative progress weight
    invalid_cfg = RewardConfig(weight_progress=-1.0)
    with pytest.raises(ValueError, match="weight_progress"):
        invalid_cfg.validate()

    # Invalid target speed (must be > 0)
    invalid_speed = RewardConfig(target_speed=0.0)
    with pytest.raises(ValueError, match="target_speed"):
        invalid_speed.validate()

    # Invalid collision penalty (must be >= 0)
    invalid_penalty = RewardConfig(collision_penalty=-10.0)
    with pytest.raises(ValueError, match="collision_penalty"):
        invalid_penalty.validate()


def test_reward_config_serialization_roundtrip():
    cfg = RewardConfig(
        weight_progress=2.5,
        weight_centering=1.2,
        weight_speed=0.8,
        target_speed=25.0,
        checkpoint_bonus=15.0,
        collision_penalty=100.0,
    )
    d = cfg.to_dict()
    restored = RewardConfig.from_dict(d)
    assert restored.weight_progress == 2.5
    assert restored.weight_centering == 1.2
    assert restored.weight_speed == 0.8
    assert restored.target_speed == 25.0
    assert restored.checkpoint_bonus == 15.0
    assert restored.collision_penalty == 100.0
    assert restored.validate() is True


def test_reward_decomposition_integrity():
    cfg = RewardConfig(
        weight_progress=1.0,
        weight_centering=0.5,
        weight_speed=0.2,
        target_speed=20.0,
        weight_heading=0.3,
        weight_action_smoothness=0.1,
        checkpoint_bonus=10.0,
        collision_penalty=50.0,
    )
    engine = RewardEngine(config=cfg)
    engine.reset(initial_s=0.0)

    # Step forward with some lateral offset and speed
    total, breakdown = engine.compute_step_reward(
        current_s=0.5,
        track_length=200.0,
        is_closed=True,
        lateral_offset=1.0,
        road_width=10.0,
        speed=15.0,
        heading_error=0.05,
        current_steer=0.1,
        is_colliding=False,
        is_on_road=True,
        checkpoint_passed=True,
        lap_completed=False,
    )

    assert "total" in breakdown
    assert breakdown["total"] == total
    assert breakdown["checkpoint"] == 10.0
    assert breakdown["progress"] > 0.0

    # Verify sum of components equals total
    sum_components = sum(v for k, v in breakdown.items() if k != "total")
    assert pytest.approx(total, abs=1e-5) == sum_components
