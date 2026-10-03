"""Reward/termination preset tests — pins the stationary-policy exploit fix."""
from __future__ import annotations

import pytest


KNOWN_COMPONENT_TYPES = {
    "progress", "centerline", "speed", "heading", "smooth_steer",
    "checkpoint", "completion", "collision", "off_road", "reverse",
    "time_penalty",
}


def _idle_step(engine, s=0.0):
    return engine.compute_step_reward(
        current_s=s, track_length=400.0, is_closed=True,
        lateral_offset=0.0, road_width=12.0, speed=0.0,
        heading_error=0.0, current_steer=0.0,
        is_colliding=False, is_on_road=True,
        checkpoint_passed=False, lap_completed=False,
    )[0]


def test_stationary_return_nonpositive():
    """The core exploit fix: idling at spawn must not be a winning strategy."""
    from agentRL.rewards.presets import reward_preset
    from sim_env.reward_designer import CompiledRewardEngine

    engine = CompiledRewardEngine(reward_preset("drive_v1"))
    engine.reset(initial_s=0.0)
    total = sum(_idle_step(engine) for _ in range(100))
    assert total <= 0.0, (
        f"stationary policy still earns positive reward ({total}); "
        "preset reintroduces the degenerate local optimum"
    )


def test_driving_beats_idle():
    from agentRL.rewards.presets import reward_preset
    from sim_env.reward_designer import CompiledRewardEngine

    engine = CompiledRewardEngine(reward_preset("drive_v1"))
    engine.reset(initial_s=0.0)
    idle = _idle_step(engine)
    engine.reset(initial_s=0.0)
    s = 0.0
    driving = 0.0
    for _ in range(10):
        s += 15.0 / 60.0  # 15 m/s at 60 Hz
        driving += engine.compute_step_reward(
            current_s=s, track_length=400.0, is_closed=True,
            lateral_offset=0.0, road_width=12.0, speed=15.0,
            heading_error=0.0, current_steer=0.0,
            is_colliding=False, is_on_road=True,
            checkpoint_passed=False, lap_completed=False,
        )[0]
    assert driving > 10.0 * idle, "forward driving must dominate idling"


def test_preset_components_valid():
    from agentRL.rewards.presets import reward_preset

    for comp in reward_preset("drive_v1").components:
        assert comp.component_type in KNOWN_COMPONENT_TYPES, (
            f"unknown component_type {comp.component_type}"
        )


def test_termination_preset_semantics():
    from agentRL.rewards.presets import termination_preset

    rules = {r.condition_type: r for r in termination_preset("term_v1").rules}
    assert rules["collision"].is_truncation is False
    assert rules["off_road"].is_truncation is False
    assert rules["wrong_direction"].is_truncation is False
    assert rules["max_steps"].is_truncation is True
    assert rules["checkpoint_timeout"].is_truncation is True


def test_unknown_preset_raises():
    from agentRL.rewards.presets import reward_preset, termination_preset

    with pytest.raises(KeyError):
        reward_preset("nonexistent")
    with pytest.raises(KeyError):
        termination_preset("nonexistent")
