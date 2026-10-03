"""Compare two run directories' metrics.jsonl side by side.

  python tests/agent/compare_runs.py <run_dir_a> <run_dir_b>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def summarize(run_dir: str) -> dict:
    path = Path(run_dir) / "metrics.jsonl"
    eps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scope"] == "episode":
            eps.append(row["metrics"])
    if not eps:
        return {"episodes": 0}
    last = eps[-10:]
    return {
        "episodes": len(eps),
        "mean_return_last10": sum(e["return"] for e in last) / len(last),
        "max_return": max(e["return"] for e in eps),
        "mean_len_last10": sum(e["length"] for e in last) / len(last),
        "mean_speed_last10": sum(e.get("mean_speed", 0) for e in last)
                            / len(last),
        "best_progress": max(e.get("lap_progress", 0) for e in eps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()
    a, b = summarize(args.run_a), summarize(args.run_b)
    keys = sorted(set(a) | set(b))
    print(f"{'metric':<20}{'A':>14}{'B':>14}")
    for k in keys:
        va = a.get(k, 0)
        vb = b.get(k, 0)
        print(f"{k:<20}{va:>14.3f}{vb:>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
