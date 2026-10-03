"""Reward + termination presets built on the declarative sim_env schemas.

DRIVE_V1 is the anti-exploit preset: the default sim reward pays a
stationary, centered, aligned vehicle ~+0.8/step (centering 0.5 +
heading 0.3) — every recorded baseline converged to parking at spawn.
Here the dense shaping terms are down-weighted to 0.05 each and a
-0.12/step time penalty makes idling strictly losing:

    idle   = 0.05 + 0.05 - 0.12 = -0.02/step
    15 m/s = 0.50 + 0.50 + 0.05 + 0.05 - 0.12 = +0.98/step

TERM_V1 enables lap completion (success signal) and a 20 s
checkpoint-timeout truncation (the built-in stuck detector) plus a
1500-step horizon tuned for ~one oval lap at moderate speed.
"""
from __future__ import annotations

from sim_env.reward_designer import (
    FalloffType,
    RewardComponentConfig,
    RewardFunctionDefinition,
)
from sim_env.termination_designer import (
    TerminationDefinition,
    TerminationRuleConfig,
)

_REWARDS: dict[str, RewardFunctionDefinition] = {}
_TERMINATIONS: dict[str, TerminationDefinition] = {}


def _c(cid: str, ctype: str, weight: float,
       params: dict | None = None, name: str | None = None,
       enabled: bool = True) -> RewardComponentConfig:
    return RewardComponentConfig(
        component_id=cid, name=name or cid, component_type=ctype,
        weight=weight, params=params or {}, enabled=enabled,
    )


def _t(rid: str, ctype: str, trunc: bool,
       params: dict | None = None, name: str | None = None,
       enabled: bool = True) -> TerminationRuleConfig:
    return TerminationRuleConfig(
        rule_id=rid, name=name or rid, condition_type=ctype,
        is_truncation=trunc, params=params or {}, enabled=enabled,
    )


_REWARDS["drive_v1"] = RewardFunctionDefinition(components=[
    _c("progress", "progress", 2.0, {"max_step_delta_m": 5.0},
       "Centerline Progress"),
    _c("speed", "speed", 0.5, {"target_speed_ms": 15.0},
       "Target Speed"),
    _c("centering", "centerline", 0.05,
       {"max_distance_m": 6.0, "falloff": FalloffType.LINEAR},
       "Centerline Deviation"),
    _c("heading", "heading", 0.05, {}, "Heading Alignment"),
    _c("smooth_steer", "smooth_steer", -0.02, {}, "Steering Smoothness"),
    _c("checkpoint", "checkpoint", 5.0, {}, "Checkpoint Crossing"),
    _c("completion", "completion", 100.0, {}, "Lap Completion"),
    _c("collision", "collision", -50.0, {}, "Collision Penalty"),
    _c("off_road", "off_road", -25.0, {}, "Off-Road Penalty"),
    _c("reverse", "reverse", -1.0, {"heading_threshold_deg": 100.0},
       "Reverse Driving Penalty"),
    _c("time_penalty", "time_penalty", -0.12, {}, "Time Step Penalty"),
])

_TERMINATIONS["term_v1"] = TerminationDefinition(rules=[
    _t("term_collision", "collision", False, {}, "Barrier Collision"),
    _t("term_off_road", "off_road", False, {}, "Off-Road Excursion"),
    _t("term_wrong_direction", "wrong_direction", False,
       {"max_angle_deg": 120.0}, "Wrong Direction"),
    _t("term_completion", "course_completion", False,
       {"target_laps": 1}, "Lap Completion"),
    _t("trunc_max_steps", "max_steps", True,
       {"max_steps": 1500}, "Step Limit"),
    _t("trunc_stuck", "checkpoint_timeout", True,
       {"max_seconds": 20.0}, "Stuck Timeout"),
])


def reward_preset(name: str) -> RewardFunctionDefinition:
    if name not in _REWARDS:
        raise KeyError(f"unknown reward preset {name!r}; "
                       f"known: {sorted(_REWARDS)}")
    return _REWARDS[name]


def termination_preset(name: str) -> TerminationDefinition:
    if name not in _TERMINATIONS:
        raise KeyError(f"unknown termination preset {name!r}; "
                       f"known: {sorted(_TERMINATIONS)}")
    return _TERMINATIONS[name]


def list_reward_presets() -> list[str]:
    return sorted(_REWARDS)


def list_termination_presets() -> list[str]:
    return sorted(_TERMINATIONS)
