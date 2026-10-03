"""
Vehicle-dynamics validation runner.

Usage:
    python tools/vehicle_dynamics/run_validation.py --tag baseline
    python tools/vehicle_dynamics/run_validation.py --tag fixed --speed 25
    python tools/vehicle_dynamics/run_validation.py --compare baseline fixed

Writes benchmarks/vehicle_dynamics/<tag>/ containing:
    results.json      scalar metrics per maneuver
    manifest.json     full reproducibility manifest
    <maneuver>.csv    time series per maneuver
    plots/<maneuver>.png  (if matplotlib available)
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import platform
from dataclasses import asdict
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim_core.vehicle.vehicle_config import VehicleConfig  # noqa: E402
from tools.vehicle_dynamics.harness import run_maneuver, RunResult  # noqa: E402
from tools.vehicle_dynamics.maneuvers import standard_suite, Maneuver  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sim_version import SIM_VERSION
except ImportError:
    SIM_VERSION = "unknown"


def build_manifest(config: VehicleConfig, dt: float, speed: float) -> Dict:
    return {
        "sim_version": SIM_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "physics_dt_s": dt,
        "physics_hz": 1.0 / dt,
        "surface_friction": 1.0,
        "target_speed_ms": speed,
        "tire_model": config.tire_model,
        "vehicle_config": config.to_dict(),
        "parameters_note": "Generic passenger-vehicle representative parameters; "
                           "not measured from a real vehicle.",
    }


def write_csv(path: str, r: RunResult) -> None:
    keys = [k for k in r.series if r.series[k]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(r.series["t"])
        for i in range(n):
            w.writerow([r.series[k][i] for k in keys])


def plot_result(path: str, r: RunResult) -> None:
    if not HAS_MPL:
        return
    s = r.series
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.suptitle(r.name)

    def ax(i, j): return axes[i][j]

    ax(0, 0).plot(s["pos_x"], s["pos_y"])
    ax(0, 0).set_title("Trajectory (m)"); ax(0, 0).axis("equal")

    ax(0, 1).plot(s["t"], s["speed"], label="speed")
    ax(0, 1).plot(s["t"], s["vy"], label="vy")
    ax(0, 1).set_title("Velocity (m/s)"); ax(0, 1).legend()

    ax(1, 0).plot(s["t"], s["steer_cmd"], label="cmd")
    ax(1, 0).plot(s["t"], s["steering_angle"], label="wheel rad")
    ax(1, 0).set_title("Steering"); ax(1, 0).legend()

    ax(1, 1).plot(s["t"], [v * 57.2958 for v in s["yaw_rate"]])
    ax(1, 1).set_title("Yaw rate (deg/s)")

    ax(2, 0).plot(s["t"], s["ay"], label="ay")
    ax(2, 0).plot(s["t"], s["ax"], label="ax")
    ax(2, 0).set_title("Body accel (m/s^2)"); ax(2, 0).legend()

    ax(2, 1).plot(s["t"], [v * 57.2958 for v in s["sideslip"]])
    ax(2, 1).set_title("Sideslip beta (deg)")

    ax(3, 0).plot(s["t"], [v * 57.2958 for v in s["alpha_f"]], label="front")
    ax(3, 0).plot(s["t"], [v * 57.2958 for v in s["alpha_r"]], label="rear")
    ax(3, 0).set_title("Slip angles (deg)"); ax(3, 0).legend()

    ax(3, 1).plot(s["t"], s["fy_f"], label="Fy front")
    ax(3, 1).plot(s["t"], s["fy_r"], label="Fy rear")
    ax(3, 1).set_title("Tire lateral force (N)"); ax(3, 1).legend()

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run_suite(tag: str, speed: float, dt: float, out_root: str,
              config: VehicleConfig | None = None) -> Dict[str, Dict[str, float]]:
    cfg = config or VehicleConfig()
    out_dir = os.path.join(out_root, tag)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    suite = standard_suite(speed_ms=speed)
    results: Dict[str, Dict[str, float]] = {}
    for m in suite.values():
        r = run_maneuver(m.id, m.script, m.duration_s, config=cfg,
                         dt=dt, initial_speed=m.initial_speed,
                         pre_roll_s=m.pre_roll_s)
        results[m.id] = r.metrics
        write_csv(os.path.join(out_dir, f"{m.id}.csv"), r)
        plot_result(os.path.join(out_dir, "plots", f"{m.id}.png"), r)
        print(f"  {m.id:42s} peak|beta|={r.metrics['peak_sideslip_deg']:6.2f} deg  "
              f"peak|vy|={r.metrics['peak_vy_ms']:5.2f} m/s  "
              f"peak|wz|={r.metrics['peak_yaw_rate_dps']:6.1f} dps")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(build_manifest(cfg, dt, speed), f, indent=2)
    return results


def compare(tag_a: str, tag_b: str, out_root: str) -> None:
    with open(os.path.join(out_root, tag_a, "results.json")) as f:
        a = json.load(f)
    with open(os.path.join(out_root, tag_b, "results.json")) as f:
        b = json.load(f)
    keys = ["peak_sideslip_deg", "steady_sideslip_deg", "peak_vy_ms",
            "rms_vy_ms", "peak_yaw_rate_dps", "peak_ay_ms2",
            "peak_alpha_f_deg", "peak_alpha_r_deg", "lateral_drift_m"]
    print(f"{'maneuver':40s} {'metric':22s} {tag_a:>10s} {tag_b:>10s}")
    for mid in a:
        if mid not in b:
            continue
        for k in keys:
            if k in a[mid] and k in b[mid]:
                print(f"{mid:40s} {k:22s} {a[mid][k]:10.3f} {b[mid][k]:10.3f}")
        print("-" * 86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tire-model", default="bicycle",
                    choices=["bicycle", "pacejka4"])
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "benchmarks", "vehicle_dynamics"))
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1], args.out)
        return

    print(f"[vehicle-dynamics] Running validation suite -> tag '{args.tag}'")
    run_suite(args.tag, args.speed, 1.0 / args.hz, args.out,
              config=VehicleConfig(tire_model=args.tire_model))
    print(f"[vehicle-dynamics] Results written to "
          f"{os.path.join(args.out, args.tag)}")


if __name__ == "__main__":
    main()
