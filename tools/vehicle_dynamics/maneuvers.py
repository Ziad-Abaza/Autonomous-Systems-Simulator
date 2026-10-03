"""
Standardized vehicle-dynamics validation maneuvers.

Each maneuver is a ControlScript: t -> (steer_cmd, throttle_cmd, brake_cmd)
with steer in [-1, 1] (negative = left), throttle/brake in [0, 1].

Reproduction tests A-F map onto the general maneuver library.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from tools.vehicle_dynamics.harness import ControlScript


@dataclass
class Maneuver:
    id: str
    description: str
    script: ControlScript
    duration_s: float
    initial_speed: float = 0.0
    pre_roll_s: float = 0.0


def _throttle_for_speed(target: float, top: float = 45.0) -> float:
    """Rough feedforward throttle to hold a speed (drag-limited cruise)."""
    return min(1.0, max(0.15, target / top))


def straight_line(throttle: float = 0.5, duration: float = 6.0) -> Maneuver:
    """Test A: constant throttle, zero steering."""
    return Maneuver(
        id="straight_line",
        description="Constant throttle, zero steering — expect zero lateral motion",
        script=lambda t: (0.0, throttle, 0.0),
        duration_s=duration)


def constant_steer(steer: float, speed_ms: float, duration: float = 8.0) -> Maneuver:
    """Test B: fixed steering at controlled speed (pre-roll to speed)."""
    thr = _throttle_for_speed(speed_ms)
    return Maneuver(
        id=f"constant_steer_{steer:+.2f}_{speed_ms:.0f}ms",
        description=f"Constant steer cmd {steer:+.2f} at ~{speed_ms} m/s",
        script=lambda t: (steer, thr, 0.0),
        duration_s=duration,
        pre_roll_s=5.0)


def steer_step(steer: float, speed_ms: float, step_t: float = 0.0, duration: float = 6.0) -> Maneuver:
    """Test C: steering step input at cruise speed."""
    thr = _throttle_for_speed(speed_ms)
    return Maneuver(
        id=f"steer_step_{steer:+.2f}_{speed_ms:.0f}ms",
        description=f"Steering step to {steer:+.2f} at ~{speed_ms} m/s",
        script=lambda t: (steer if t >= step_t else 0.0, thr, 0.0),
        duration_s=duration,
        pre_roll_s=5.0)


def steer_ramp(steer_max: float, speed_ms: float, ramp_s: float = 2.0,
               hold_s: float = 3.0) -> Maneuver:
    """Steering ramp: linear increase then hold."""
    thr = _throttle_for_speed(speed_ms)
    def script(t):
        if t < ramp_s:
            s = steer_max * (t / ramp_s)
        else:
            s = steer_max
        return (s, thr, 0.0)
    return Maneuver(
        id=f"steer_ramp_{steer_max:+.2f}_{speed_ms:.0f}ms",
        description=f"Steering ramp to {steer_max:+.2f} over {ramp_s}s at ~{speed_ms} m/s",
        script=script, duration_s=ramp_s + hold_s, pre_roll_s=5.0)


def sine_steer(amplitude: float, freq_hz: float, speed_ms: float,
               duration: float = 6.0) -> Maneuver:
    thr = _throttle_for_speed(speed_ms)
    return Maneuver(
        id=f"sine_steer_{amplitude:.2f}_{freq_hz:.1f}hz_{speed_ms:.0f}ms",
        description=f"Sine steering ±{amplitude:.2f} at {freq_hz} Hz, ~{speed_ms} m/s",
        script=lambda t: (amplitude * math.sin(2 * math.pi * freq_hz * t), thr, 0.0),
        duration_s=duration, pre_roll_s=5.0)


def steer_reversal(steer: float, speed_ms: float, t1: float = 1.0,
                   t2: float = 2.0, duration: float = 5.0) -> Maneuver:
    """Test D: +steer -> 0 -> -steer."""
    thr = _throttle_for_speed(speed_ms)
    def script(t):
        s = steer if t < t1 else (0.0 if t < t2 else -steer)
        return (s, thr, 0.0)
    return Maneuver(
        id=f"steer_reversal_{steer:.2f}_{speed_ms:.0f}ms",
        description=f"Steer {steer:+.2f} -> 0 -> {-steer:+.2f} at ~{speed_ms} m/s",
        script=script, duration_s=duration, pre_roll_s=5.0)


def steer_escalation(speed_ms: float, step_s: float = 1.5,
                     levels=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)) -> Maneuver:
    """Test E: progressively increasing steering demand."""
    thr = _throttle_for_speed(speed_ms)
    def script(t):
        i = min(int(t / step_s), len(levels) - 1)
        return (levels[i], thr, 0.0)
    return Maneuver(
        id=f"steer_escalation_{speed_ms:.0f}ms",
        description=f"Progressive steering escalation at ~{speed_ms} m/s",
        script=script, duration_s=step_s * len(levels), pre_roll_s=5.0)


def braking(speed_ms: float, brake: float = 1.0, duration: float = 8.0) -> Maneuver:
    """Straight-line braking from speed."""
    return Maneuver(
        id=f"braking_{brake:.1f}_{speed_ms:.0f}ms",
        description=f"Brake {brake:.0%} from {speed_ms} m/s, zero steering",
        script=lambda t: (0.0, 0.0, brake),
        duration_s=duration, initial_speed=speed_ms)


def brake_and_steer(steer: float, brake: float, speed_ms: float,
                    duration: float = 6.0) -> Maneuver:
    """Test F: combined braking + steering — friction envelope check."""
    return Maneuver(
        id=f"brake_steer_{steer:+.2f}_{brake:.1f}_{speed_ms:.0f}ms",
        description=f"Brake {brake:.0%} + steer {steer:+.2f} from {speed_ms} m/s",
        script=lambda t: (steer, 0.0, brake),
        duration_s=duration, initial_speed=speed_ms)


def lane_change(speed_ms: float, duration: float = 5.0) -> Maneuver:
    """ISO 3888-style single lane change: +steer ramp, then -steer, then straighten."""
    thr = _throttle_for_speed(speed_ms)
    def script(t):
        if t < 0.5:
            return (0.0, thr, 0.0)
        if t < 1.5:
            return (0.25 * (t - 0.5), thr, 0.0)
        if t < 2.5:
            return (0.25 - 0.5 * (t - 1.5), thr, 0.0)
        if t < 3.5:
            return (-0.25 + 0.25 * (t - 2.5), thr, 0.0)
        return (0.0, thr, 0.0)
    return Maneuver(
        id=f"lane_change_{speed_ms:.0f}ms",
        description=f"Single lane change at ~{speed_ms} m/s",
        script=script, duration_s=duration, pre_roll_s=5.0)


def lateral_disturbance(speed_ms: float, kick: float = 1.0,
                        kick_s: float = 0.15, duration: float = 6.0) -> Maneuver:
    """Recovery from a brief full-lock steering kick, then release."""
    thr = _throttle_for_speed(speed_ms)
    def script(t):
        if 1.0 <= t < 1.0 + kick_s:
            return (kick, thr, 0.0)
        return (0.0, thr, 0.0)
    return Maneuver(
        id=f"lateral_disturbance_{speed_ms:.0f}ms",
        description=f"{kick_s*1000:.0f} ms steering kick at ~{speed_ms} m/s, hands off",
        script=script, duration_s=duration, pre_roll_s=5.0)


def standard_suite(speed_ms: float = 20.0) -> Dict[str, Maneuver]:
    """The complete validation maneuver suite (spec section 13)."""
    return {m.id: m for m in [
        straight_line(),
        constant_steer(0.10, speed_ms),          # ~3.3 deg road wheel
        constant_steer(0.25, speed_ms),          # ~8.3 deg
        steer_step(0.15, speed_ms),
        steer_ramp(0.3, speed_ms),
        sine_steer(0.15, 0.5, speed_ms),
        steer_reversal(0.2, speed_ms),
        steer_escalation(speed_ms),
        braking(speed_ms),
        brake_and_steer(0.2, 0.8, speed_ms),
        lane_change(speed_ms),
        lateral_disturbance(speed_ms),
    ]}
