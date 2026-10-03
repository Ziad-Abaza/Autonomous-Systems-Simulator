"""Failure classification — deterministic, auditable failure classes.

Every evaluated episode lands in exactly one class so regression reports
can diff failure *mix*, not just scalar metrics. Classes:

    completed          lap/goal reached
    collision          barrier or entity contact (terminated)
    off_track          left the road surface (terminated)
    wrong_direction    heading > threshold (terminated)
    stall_timeout      ended by timeout while effectively stationary
    no_progress        ended while moving but not advancing checkpoints
    timeout_progress   ran out of time while still making progress
    oscillation        high steer sign-flip rate with low progress
    protocol_error     env reported an error / invalid call
"""
from __future__ import annotations

from typing import Any

STALL_SPEED_MS = 0.5
OSCILLATION_FLIP_RATE = 0.3
OSCILLATION_MAX_PROGRESS = 0.3


def classify_failure(episode: dict[str, Any]) -> str:
    reason = str(episode.get("termination_reason", ""))
    speed = float(episode.get("mean_speed", 0.0))
    progress = float(episode.get("lap_progress", 0.0))
    flips = float(episode.get("steer_sign_flips", 0.0))

    if reason in ("completion", "course_completed", "course_completion"):
        return "completed"
    if reason == "collision":
        return "collision"
    if reason == "off_road":
        return "off_track"
    if reason == "wrong_direction":
        return "wrong_direction"
    if "error" in reason or reason == "invalid_call":
        return "protocol_error"

    # timeout family — subdivide by whether the car was moving/progressing
    if flips > OSCILLATION_FLIP_RATE and progress < OSCILLATION_MAX_PROGRESS:
        return "oscillation"
    if reason in ("stuck", "checkpoint_timeout"):
        return "stall_timeout" if speed < STALL_SPEED_MS else "no_progress"
    if reason in ("max_steps", "max_duration_exceeded",
                  "scenario_time_limit_exceeded"):
        if speed < STALL_SPEED_MS:
            return "stall_timeout"
        return "timeout_progress" if progress > 0.05 else "no_progress"
    if reason in ("", "running", "reset"):
        return "incomplete"
    return f"unclassified:{reason}"
