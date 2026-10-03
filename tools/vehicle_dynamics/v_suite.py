"""
V01-V22 formal physics validation suite.

Each check produces machine-readable metrics into
benchmarks/vehicle_dynamics/v_suite/results.json plus per-maneuver CSVs.

Standard alignment:
  - V04/V06 compute SAE J266-style steady-state / transient metrics
    (understeer gradient, yaw-velocity gain, sideslip gain, response delay).
  - V08 uses an ISO 3888-1-inspired double-lane-change steer profile.
  - V13 sweeps combined slip and verifies the friction envelope.
  - V18/V19/V20 exercise environment physics (friction transition,
    banking gravity bias, boundary contact response).
  - V21/V22 verify timestep independence and determinism.

Usage:  python tools/vehicle_dynamics/v_suite.py [--out DIR]
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim_core.math_utils import Vec2, Vec3  # noqa: E402
from sim_core.vehicle.vehicle_config import VehicleConfig  # noqa: E402
from sim_core.vehicle.vehicle_model import VehicleModel  # noqa: E402
from sim_core.vehicle.collision import VehicleCollisionChecker  # noqa: E402
from tools.vehicle_dynamics.harness import run_maneuver, RunResult  # noqa: E402
from tools.vehicle_dynamics.maneuvers import (  # noqa: E402
    straight_line, constant_steer, steer_step, steer_ramp, sine_steer,
    steer_reversal, steer_escalation, braking, brake_and_steer,
    lane_change, lateral_disturbance, s_curve, throttle_in_corner,
    drift_initiation, drift_recovery, full_throttle_steer_correction,
)

G = 9.81


def _write_csv(path: str, r: RunResult) -> None:
    keys = [k for k in r.series if r.series[k]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(r.series["t"])
        for i in range(n):
            w.writerow([r.series[k][i] for k in keys])


def _beta(st) -> float:
    vx, vy = st.vel_body.x, st.vel_body.y
    if vx * vx + vy * vy < 1.0:
        return 0.0
    return math.atan2(vy, max(0.1, vx))


def _steady(series, frac=0.3):
    n = len(series)
    k = max(1, int(n * frac))
    return sum(series[-k:]) / k


# ---------------------------------------------------------------------
# Standard-referenced analysis
# ---------------------------------------------------------------------

def steady_state_gradient(speed_ms: float = 20.0,
                          levels=(0.02, 0.04, 0.06, 0.08, 0.10)) -> Dict[str, Any]:
    """SAE J266 / ISO 4138-style: steady-state steer wheel angle vs
    lateral acceleration -> understeer gradient K (rad per m/s^2) and
    yaw-velocity/sideslip gains at fixed speed."""
    rows = []
    for s in levels:
        m = constant_steer(s, speed_ms, duration=10.0)
        r = run_maneuver(m.id, m.script, m.duration_s, initial_speed=m.initial_speed,
                         pre_roll_s=m.pre_roll_s)
        wz = _steady(r.series["yaw_rate"])
        ay = _steady(r.series["ay"])
        vx = _steady(r.series["vx"])
        beta = _steady(r.series["sideslip"])
        delta = abs(_steady(r.series["steering_angle"]))
        kappa = abs(wz) / max(1.0, vx)
        rows.append({
            "steer_cmd": s, "delta_rad": delta, "ay_ms2": ay,
            "wz_rads": wz, "kappa_1pm": kappa, "beta_deg": math.degrees(beta),
            "vx_ms": vx,
        })
    # understeer gradient: linear fit delta = L*kappa + K*|ay|
    # -> K = slope(delta - L*kappa, |ay|). Positive = understeer.
    L = VehicleConfig().wheelbase
    xs = [abs(row["ay_ms2"]) for row in rows]
    ys = [row["delta_rad"] - L * row["kappa_1pm"] for row in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    K = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom > 1e-9 else 0.0
    # yaw velocity gain: steady wz per rad of wheel angle (1/s per rad)
    yaw_gain = abs(_steady([r_['wz_rads'] for r_ in rows])) / max(1e-6, _steady([r_['delta_rad'] for r_ in rows]))
    return {
        "points": rows,
        "understeer_gradient_rad_per_ms2": K,
        "understeer_gradient_deg_per_g": math.degrees(K) * G,
        "yaw_velocity_gain_per_s_per_rad": yaw_gain,
        "speed_ms": speed_ms,
    }


def step_steer_transient(steer: float = 0.08, speed_ms: float = 27.8) -> Dict[str, float]:
    """ISO 7401-style step steer: yaw-rate response delay, overshoot,
    settling time, steady-state gains."""
    m = steer_step(steer, speed_ms, step_t=0.5, duration=5.0)
    r = run_maneuver(m.id, m.script, m.duration_s, initial_speed=m.initial_speed,
                     pre_roll_s=m.pre_roll_s)
    s = r.series
    wz = s["yaw_rate"]
    t = s["t"]
    wz_ss = _steady(wz)
    peak = max(wz) if wz_ss >= 0 else min(wz)
    t50 = next((tt for tt, w in zip(t, wz)
                if abs(w) >= 0.5 * abs(wz_ss)), None)
    t90 = next((tt for tt, w in zip(t, wz)
                if abs(w) >= 0.9 * abs(wz_ss)), None)
    overshoot = (abs(peak) - abs(wz_ss)) / abs(wz_ss) if abs(wz_ss) > 1e-9 else 0.0
    band = 0.05 * abs(wz_ss)
    settle_t = None
    for i in range(len(t) - 1, -1, -1):
        if abs(wz[i] - wz_ss) > band:
            settle_t = t[i] - 0.5
            break
    ay_ss = _steady(s["ay"])
    delta_ss = _steady(s["steering_angle"])
    return {
        "wz_ss_dps": math.degrees(wz_ss),
        "ay_ss_ms2": ay_ss,
        "delay_50_s": (t50 - 0.5) if t50 is not None else float("nan"),
        "delay_90_s": (t90 - 0.5) if t90 is not None else float("nan"),
        "overshoot_frac": overshoot,
        "settle_s": settle_t if settle_t is not None else float("nan"),
        "ay_gain_per_rad": abs(ay_ss) / max(1e-6, abs(delta_ss)),
        "yaw_gain_per_s_per_rad": abs(wz_ss) / max(1e-6, abs(delta_ss)),
    }


# ---------------------------------------------------------------------
# V-maneuver implementations
# ---------------------------------------------------------------------

def run_model_suite(out_dir: str, tag: str, dt: float = 1 / 60) -> Dict[str, Any]:
    """V01-V17 vehicle-level checks."""
    res: Dict[str, Any] = {}
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    def rec(man):
        r = run_maneuver(man.id, man.script, man.duration_s,
                         initial_speed=man.initial_speed, pre_roll_s=man.pre_roll_s,
                         dt=dt)
        _write_csv(os.path.join(csv_dir, f"{man.id}.csv"), r)
        return r.metrics

    res["V01_straight_acceleration"] = rec(straight_line(throttle=1.0, duration=8.0))
    res["V02_constant_speed_straight"] = rec(straight_line(throttle=0.4, duration=6.0))
    res["V03_braking"] = rec(braking(25.0))
    res["V04_constant_radius_sweep"] = steady_state_gradient(20.0)
    # V05 increasing-speed corner: constant steer + throttle ramp
    def spiral(t):
        thr = min(1.0, 0.3 + 0.5 * t)
        return (0.12, thr, 0.0)
    res["V05_increasing_speed_corner"] = run_maneuver(
        "v05", spiral, 8.0, dt=dt, initial_speed=12.0).metrics
    res["V06_step_steer_ISO7401"] = step_steer_transient()
    res["V07_sine_steer"] = rec(sine_steer(0.15, 0.5, 20.0))
    # V08 double lane change: ISO 3888-1-inspired profile at 20 m/s
    def dlc(t):
        if t < 0.5:
            s = 0.0
        elif t < 1.4:
            s = -0.22 * math.sin(math.pi * (t - 0.5) / 0.9)
        elif t < 2.4:
            s = 0.22 * math.sin(math.pi * (t - 1.4) / 1.0)
        else:
            s = 0.0
        return (s, 0.45, 0.0)
    res["V08_double_lane_change"] = run_maneuver(
        "v08_dlc", dlc, 4.5, dt=dt, initial_speed=20.0).metrics
    res["V09_s_curve"] = rec(s_curve(20.0))
    # V10 lateral correction: brief small correction then centre
    def corr(t):
        if t < 0.5:
            s = 0.0
        elif t < 0.9:
            s = -0.04
        elif t < 1.6:
            s = 0.05
        else:
            s = 0.0
        return (s, 0.45, 0.0)
    res["V10_lateral_correction"] = run_maneuver(
        "v10_corr", corr, 5.0, dt=dt, initial_speed=20.0).metrics
    res["V11_accel_cornering"] = rec(throttle_in_corner(0.15, 15.0, throttle=1.0))
    res["V12_brake_cornering"] = rec(brake_and_steer(0.2, 0.8, 20.0))
    # V13 combined-slip sweep: steady corner then brake ramp
    def cs_sweep(t):
        if t < 2.0:
            return (0.10, 0.45, 0.0)
        return (0.10, 0.0, min(1.0, (t - 2.0) * 0.8))
    res["V13_combined_slip_sweep"] = run_maneuver(
        "v13_cs", cs_sweep, 6.0, dt=dt, initial_speed=22.0).metrics
    res["V14_drift_initiation"] = rec(drift_initiation())
    # V15 drift recovery WITH proper countersteer protocol
    def v15(dt):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=30.0)
        peak_b = 0.0
        recovered = None
        for i in range(int(8 / dt)):
            tt = i * dt
            if tt < 1.5:
                cmd, thr = (0.15, 1.0)
            else:
                b = _beta(car.state)
                cmd = max(-1.0, min(1.0, -b / 0.58))
                thr = 0.4
            car.step(cmd, thr, 0.0, dt)
            b = _beta(car.state)
            peak_b = max(peak_b, abs(b))
            if tt > 1.5 and recovered is None and abs(b) < math.radians(3):
                recovered = tt - 1.5
        return {
            "peak_sideslip_deg": math.degrees(peak_b),
            "recovery_time_s": recovered if recovered is not None else float("nan"),
            "final_sideslip_deg": math.degrees(abs(_beta(car.state))),
            "final_yaw_rate_dps": math.degrees(abs(car.state.yaw_rate)),
            "recovered": recovered is not None,
        }
    res["V15_drift_recovery_countersteer"] = v15(dt)
    res["V16_disturbance_recovery"] = rec(lateral_disturbance(20.0))
    # V17 high-speed stability sweep
    v17 = {}
    for v in (20.0, 25.0, 30.0, 35.0, 40.0, 44.0):
        for st_cmd in (0.02, 0.05):
            rr = run_maneuver(f"hs_{v}_{st_cmd}",
                              lambda t, s=st_cmd: (s, 0.9, 0.0), 6.0,
                              dt=dt, initial_speed=v)
            v17[f"v{v:.0f}_s{st_cmd:.2f}"] = {
                "peak_sideslip_deg": rr.metrics["peak_sideslip_deg"],
                "steady_sideslip_deg": rr.metrics["steady_sideslip_deg"],
                "peak_ay_g": rr.metrics["peak_ay_g"],
                "peak_util_f": rr.metrics["peak_util_f"],
                "peak_util_r": rr.metrics["peak_util_r"],
                "peak_yaw_rate_dps": rr.metrics["peak_yaw_rate_dps"],
            }
    res["V17_high_speed_stability"] = v17

    # V18 surface friction transition: cornering then mu drops to 0.55
    def surf_fn(t):
        return 1.0 if t < 2.0 else 0.55
    r18 = run_maneuver("v18_fric", lambda t: (0.08, 0.45, 0.0), 6.0,
                       dt=dt, initial_speed=20.0, surface_fn=surf_fn)
    ay_before = _steady(r18.series["ay"][:int(1.5 / dt)])
    ay_after = _steady(r18.series["ay"][-int(1.0 / dt):])
    res["V18_friction_transition"] = dict(r18.metrics)
    res["V18_friction_transition"]["ay_before_ms2"] = ay_before
    res["V18_friction_transition"]["ay_after_ms2"] = ay_after
    res["V18_friction_transition"]["ay_drop_frac"] = (
        abs(ay_after) / max(1e-6, abs(ay_before)))
    _write_csv(os.path.join(csv_dir, "v18_fric.csv"), r18)

    # V19 banking bias: lateral gravity pull at constant speed, hands off
    def bias_fn(t):
        return Vec2(0.0, -G * math.sin(math.radians(10.0)))
    r19 = run_maneuver("v19_bank", lambda t: (0.0, 0.4, 0.0), 4.0,
                       dt=dt, initial_speed=20.0, bias_fn=bias_fn)
    res["V19_banking"] = {
        "lateral_drift_m": r19.series["pos_y"][-1],
        "final_vy_ms": r19.series["vy"][-1],
        "peak_ay_ms2": r19.metrics["peak_ay_ms2"],
        "expected_bias_ms2": G * math.sin(math.radians(10.0)),
    }

    # V20 boundary collision response: head-on + oblique into a wall segment
    def v20(head_on: bool):
        car = VehicleModel()
        wall_a, wall_b = Vec2(30.0, -10.0), Vec2(30.0, 10.0)
        # head-on: drive +X into the wall; oblique: 20 deg incidence that
        # still lands within the wall segment span (|y| < 10 m)
        yaw = 0.0 if head_on else -math.radians(20.0)
        car.reset(Vec3(5.0, 0.0, 0.0), yaw, initial_speed=20.0)
        ke0 = 0.5 * car.config.mass * car.state.speed_total ** 2
        contacts = 0
        max_pen_pos_x = -1e9
        for i in range(int(5 / dt)):
            car.step(0.0, 0.0, 0.0, dt)
            obb = car.get_obb()
            if obb.intersects_segment(wall_a, wall_b):
                contacts += 1
                VehicleCollisionChecker.apply_boundary_contact(
                    car, wall_a, wall_b, interior_hint=Vec2(-1.0, 0.0))
            max_pen_pos_x = max(max_pen_pos_x, max(
                c.x for c in obb.get_corners()))
        ke1 = 0.5 * car.config.mass * car.state.speed_total ** 2
        return {
            "contacts": contacts,
            "ke_start_J": ke0, "ke_end_J": ke1,
            "ke_ratio": ke1 / ke0,
            "final_speed_ms": car.state.speed_total,
            "max_corner_x_m": max_pen_pos_x,
            "wall_x_m": 30.0,
            "crossed_wall": max_pen_pos_x > 30.0 + 0.6,
        }
    res["V20_collision_head_on"] = v20(True)
    res["V20_collision_oblique"] = v20(False)

    # V21 timestep independence: same script at 30/60/120/240 Hz
    finals = []
    for hz in (30, 60, 120, 240):
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=25.0)
        d = 1.0 / hz
        tt = 0.0
        while tt < 6.0:
            s_cmd = 0.1 * math.sin(2 * math.pi * 0.4 * tt)
            car.step(s_cmd, 0.5, 0.0, d)
            tt += d
        finals.append((car.state.pos.x, car.state.pos.y, car.state.yaw))
    ref = finals[-1]
    deltas = [math.hypot(f[0] - ref[0], f[1] - ref[1]) for f in finals]
    res["V21_timestep_independence"] = {
        "pos_delta_m_vs_240hz": deltas,
        "yaw_delta_deg_vs_240hz": [math.degrees(abs(f[2] - ref[2])) for f in finals],
        "max_pos_delta_m": max(deltas),
    }

    # V22 determinism
    def det_run():
        car = VehicleModel()
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=20.0)
        for i in range(300):
            tt = i / 60
            car.step(0.15 * math.sin(2 * tt), 0.4, 0.0, 1 / 60)
        s = car.state
        return (s.pos.x, s.pos.y, s.yaw, s.vel_body.x, s.vel_body.y, s.yaw_rate)
    res["V22_determinism"] = {"bitwise_equal": det_run() == det_run()}

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "benchmarks", "vehicle_dynamics", "v_suite"))
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = run_model_suite(args.out, "v_suite", dt=1.0 / args.hz)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps({k: (v if not isinstance(v, dict) or len(str(v)) < 400
                          else "…") for k, v in results.items()},
                     indent=2, default=str)[:6000])
    print(f"\n[v_suite] results -> {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
