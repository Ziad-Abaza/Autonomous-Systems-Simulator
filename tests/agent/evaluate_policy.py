"""Evaluate a checkpoint on a track; write eval JSON.

  python tests/agent/evaluate_policy.py --ckpt <path.pt> --track oval --out eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentRL.checkpoints.io import load_agent          # noqa: E402
from agentRL.envs.factory import EnvFactory            # noqa: E402
from agentRL.envs.track_registry import TrackRegistry  # noqa: E402
from agentRL.eval.evaluate import evaluate_policy      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--seeds", default="42-46",
                    help="e.g. 42-46 or 1,2,3")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-", 1)
        seeds = tuple(range(int(lo), int(hi) + 1))
    else:
        seeds = tuple(int(s) for s in args.seeds.split(","))

    agent = load_agent(args.ckpt)
    res = evaluate_policy(EnvFactory(),
                          TrackRegistry.default().load(args.track),
                          agent, seeds=seeds, max_steps=args.max_steps)
    text = json.dumps(res, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"mean_return={res['aggregate']['mean_return']:.2f} "
          f"completion={res['aggregate']['completion_rate']:.2f} "
          f"collision={res['aggregate']['collision_rate']:.2f} "
          f"speed={res['aggregate']['mean_speed']:.2f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
