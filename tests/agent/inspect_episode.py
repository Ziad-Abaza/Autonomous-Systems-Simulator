"""Dump one episode step-by-step: obs, action, reward breakdown, state.

  python tests/agent/inspect_episode.py --ckpt <path.pt> --track oval --out ep.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentRL.checkpoints.io import load_agent          # noqa: E402
from agentRL.envs.factory import EnvFactory            # noqa: E402
from agentRL.envs.track_registry import TrackRegistry  # noqa: E402
from agentRL.obs.encoder import ObsEncoder             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", default="episode.jsonl")
    args = ap.parse_args()

    agent = load_agent(args.ckpt)
    factory = EnvFactory(obs_spec=agent.obs_spec)
    env = factory.build(TrackRegistry.default().load(args.track), seed=0)
    encoder = ObsEncoder(agent.obs_spec)
    obs, _ = env.reset(seed=args.seed)
    encoder.reset()
    enc = encoder.encode(np.asarray(obs, np.float32), None)

    with open(args.out, "w", encoding="utf-8") as fh:
        for step in range(args.max_steps):
            a, aux = agent.act(enc, deterministic=True)
            nobs, r, term, trunc, info = env.step(a)
            fh.write(json.dumps({
                "step": step,
                "obs": np.asarray(obs).tolist(),
                "action": np.asarray(a).tolist(),
                "reward": r,
                "reward_breakdown": info.get("reward_breakdown", {}),
                "speed": info.get("speed"),
                "lateral_offset": info.get("lateral_offset"),
                "heading_error": info.get("heading_error"),
                "lap_progress": info.get("lap_progress"),
                "terminated": term, "truncated": trunc,
                "termination_reason": info.get("termination_reason"),
            }, default=str) + "\n")
            obs = nobs
            enc = encoder.encode(np.asarray(nobs, np.float32),
                                 info.get("last_action", a))
            if term or trunc:
                break
    print(f"wrote {step + 1} steps -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
