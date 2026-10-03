"""
Vehicle parameter calibration sweep.

Sweeps one VehicleConfig parameter over a range and reports how the
standard validation metrics respond. Produces a CSV + console table so
parameter sensitivity is measurable instead of tuned by feel.

Usage:
    python tools/vehicle_dynamics/calibrate.py --param tire_friction --values 0.6 0.8 1.0 1.2
    python tools/vehicle_dynamics/calibrate.py --param cornering_stiffness_front --sweep 0.5 2.0 7
    python tools/vehicle_dynamics/calibrate.py --param mass --sweep 0.8 1.3 6
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim_core.vehicle.vehicle_config import VehicleConfig  # noqa: E402
from tools.vehicle_dynamics.harness import run_maneuver  # noqa: E402
from tools.vehicle_dynamics.maneuvers import standard_suite  # noqa: E402

SWEEPABLE = [f for f in VehicleConfig().__dict__.keys()]

# Metrics that best characterize handling quality
REPORT_KEYS = [
    "steady_sideslip_deg", "peak_sideslip_deg", "steady_yaw_rate_dps",
    "mean_curvature_1pm", "peak_ay_ms2", "final_speed_ms",
]

# Maneuvers that best expose a parameter's effect
PROBE_IDS = [
    "constant_steer_+0.10_20ms",
    "steer_step_+0.15_20ms",
    "steer_reversal_0.20_20ms",
    "brake_steer_+0.20_0.8_20ms",
]


def sweep(param: str, values: List[float], speed: float,
          out_csv: str | None = None) -> List[dict]:
    base = VehicleConfig()
    if not hasattr(base, param):
        raise SystemExit(f"Unknown parameter '{param}'. Sweepable: {SWEEPABLE}")

    probes = [m for m in standard_suite(speed_ms=speed).values()
              if m.id in PROBE_IDS]

    rows: List[dict] = []
    for v in values:
        cfg = replace(base, **{param: type(getattr(base, param))(v)})
        row = {param: v}
        for m in probes:
            r = run_maneuver(m.id, m.script, m.duration_s, config=cfg,
                             initial_speed=m.initial_speed,
                             pre_roll_s=m.pre_roll_s)
            for k in REPORT_KEYS:
                row[f"{m.id}:{k}"] = r.metrics.get(k, float("nan"))
        rows.append(row)
        print(f"  {param}={v:g}  "
              f"beta_ss(cs0.10)={row['constant_steer_+0.10_20ms:steady_sideslip_deg']:+5.2f}deg  "
              f"beta_pk(step)={row['steer_step_+0.15_20ms:peak_sideslip_deg']:5.2f}deg  "
              f"beta_pk(rev)={row['steer_reversal_0.20_20ms:peak_sideslip_deg']:5.2f}deg")

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[calibrate] wrote {out_csv}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", type=float)
    ap.add_argument("--sweep", nargs=3, type=float,
                    metavar=("LO", "HI", "N"), help="linear sweep")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "benchmarks", "vehicle_dynamics", "calibration"))
    args = ap.parse_args()

    if args.values:
        values = args.values
    elif args.sweep:
        lo, hi, n = args.sweep
        values = [lo + (hi - lo) * i / max(1, int(n) - 1) for i in range(int(n))]
    else:
        raise SystemExit("Provide --values or --sweep LO HI N")

    print(f"[calibrate] sweeping {args.param} over {values}")
    out_csv = os.path.join(args.out, f"sweep_{args.param}.csv")
    sweep(args.param, values, args.speed, out_csv)


if __name__ == "__main__":
    main()
