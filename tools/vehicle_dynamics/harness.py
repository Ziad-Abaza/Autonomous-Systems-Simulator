"""
Deterministic vehicle-dynamics test harness.

Runs scripted control maneuvers against VehicleModel directly (no track,
no rendering, no sensors) and records a complete time series of vehicle
state plus derived tire metrics for quantitative validation.

All runs are deterministic: same config + same script + same dt produce
identical output because VehicleModel contains no randomness.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel

# ControlScript: t (seconds) -> (steer_cmd, throttle_cmd, brake_cmd)
ControlScript = Callable[[float], Tuple[float, float, float]]

SERIES_FIELDS = (
    "t", "vx", "vy", "speed", "yaw", "yaw_rate", "steering_angle",
    "ax", "ay", "pos_x", "pos_y", "sideslip", "alpha_f", "alpha_r",
    "fy_f", "fy_r", "f_long", "throttle", "brake", "steer_cmd",
)


@dataclass
class RunResult:
    """Recorded time series plus scalar summary metrics for one maneuver."""
    name: str
    dt: float
    series: Dict[str, List[float]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for f in SERIES_FIELDS:
            self.series.setdefault(f, [])


def _sideslip(st) -> float:
    vx, vy = st.vel_body.x, st.vel_body.y
    if vx * vx + vy * vy < 1.0:
        return 0.0
    return math.atan2(vy, max(0.1, vx))


def _tire_metrics(vehicle: VehicleModel, surface_friction: float) -> Tuple[float, float, float, float]:
    """Read per-axle slip angles and tire forces recorded by the model."""
    st = vehicle.state
    return st.alpha_f, st.alpha_r, st.fy_f, st.fy_r


def run_maneuver(
    name: str,
    script: ControlScript,
    duration_s: float,
    config: Optional[VehicleConfig] = None,
    dt: float = 1.0 / 60.0,
    initial_speed: float = 0.0,
    surface_friction: float = 1.0,
    pre_roll_s: float = 0.0,
) -> RunResult:
    """
    Runs one deterministic maneuver.

    pre_roll_s: seconds of zero-steering full-speed stabilization driven
    before t=0 of the recorded window, so tests can start at a target
    speed without measuring the spin-up transient.
    """
    vehicle = VehicleModel(config or VehicleConfig())
    vehicle.reset(pos=Vec3(0.0, 0.0, 0.0), yaw=0.0, initial_speed=initial_speed)

    if pre_roll_s > 0.0:
        n_pre = int(pre_roll_s / dt)
        for _ in range(n_pre):
            vehicle.step(0.0, 1.0, 0.0, dt, surface_friction)

    result = RunResult(name=name, dt=dt)
    n = int(round(duration_s / dt))
    for i in range(n):
        t = i * dt
        steer, throttle, brake = script(t)
        vehicle.step(steer, throttle, brake, dt, surface_friction)
        st = vehicle.state

        alpha_f, alpha_r, fy_f, fy_r = _tire_metrics(vehicle, surface_friction)
        s = result.series
        s["t"].append(t)
        s["vx"].append(st.vel_body.x)
        s["vy"].append(st.vel_body.y)
        s["speed"].append(st.speed_total)
        s["yaw"].append(st.yaw)
        s["yaw_rate"].append(st.yaw_rate)
        s["steering_angle"].append(st.steering_angle)
        s["ax"].append(st.accel_body.x)
        s["ay"].append(st.accel_body.y)
        s["pos_x"].append(st.pos.x)
        s["pos_y"].append(st.pos.y)
        s["sideslip"].append(_sideslip(st))
        s["alpha_f"].append(alpha_f)
        s["alpha_r"].append(alpha_r)
        s["fy_f"].append(fy_f)
        s["fy_r"].append(fy_r)
        s["f_long"].append(st.fx_f + st.fx_r)
        s["throttle"].append(throttle)
        s["brake"].append(brake)
        s["steer_cmd"].append(steer)

    result.metrics = compute_metrics(result)
    return result


def compute_metrics(r: RunResult) -> Dict[str, float]:
    """Scalar summary metrics over the recorded window."""
    s = r.series
    n = len(s["t"])
    if n == 0:
        return {}

    def peak(key): return max(abs(v) for v in s[key])
    def final(key): return s[key][-1]
    def rms(key): return math.sqrt(sum(v * v for v in s[key]) / n)
    def steady(key, frac=0.25):
        k = max(1, int(n * frac))
        return sum(s[key][-k:]) / k

    # Path curvature proxy: total yaw change / path length
    path_len = 0.0
    for i in range(1, n):
        dx = s["pos_x"][i] - s["pos_x"][i - 1]
        dy = s["pos_y"][i] - s["pos_y"][i - 1]
        path_len += math.hypot(dx, dy)
    yaw_change = s["yaw"][-1] - s["yaw"][0]

    return {
        "duration_s": s["t"][-1] + r.dt,
        "path_length_m": path_len,
        "final_speed_ms": final("speed"),
        "final_vy_ms": final("vy"),
        "peak_vy_ms": peak("vy"),
        "rms_vy_ms": rms("vy"),
        "peak_sideslip_deg": math.degrees(peak("sideslip")),
        "final_sideslip_deg": math.degrees(final("sideslip")),
        "steady_sideslip_deg": math.degrees(steady("sideslip")),
        "peak_yaw_rate_dps": math.degrees(peak("yaw_rate")),
        "steady_yaw_rate_dps": math.degrees(steady("yaw_rate")),
        "final_yaw_deg": math.degrees(final("yaw")),
        "yaw_change_deg": math.degrees(yaw_change),
        "peak_ay_ms2": peak("ay"),
        "steady_ay_ms2": steady("ay"),
        "peak_ax_ms2": peak("ax"),
        "peak_alpha_f_deg": math.degrees(peak("alpha_f")),
        "peak_alpha_r_deg": math.degrees(peak("alpha_r")),
        "peak_fy_f_n": peak("fy_f"),
        "peak_fy_r_n": peak("fy_r"),
        "mean_curvature_1pm": (yaw_change / path_len) if path_len > 1e-6 else 0.0,
        "lateral_drift_m": abs(final("pos_y")),
    }
