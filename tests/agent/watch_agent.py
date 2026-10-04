"""Watch a trained checkpoint drive in the 3D studio (TCP).

  1) python main.py --track oval --port 8765     # studio renders
  2) python tests/agent/watch_agent.py --ckpt <best.pt> --episodes 5

The script connects to the studio's external-AI TCP server, optionally
applies a spawn_override scenario (initial_speed matching training),
and drives episodes while the user watches the rendered window.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentRL.checkpoints.io import load_agent  # noqa: E402
from agentRL.obs.encoder import ObsEncoder     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of its mean")
    ap.add_argument("--initial-speed", type=float, default=8.0,
                    help="spawn speed override (0 = authored spawn)")
    args = ap.parse_args()

    from sim_client import SimGymEnv
    agent = load_agent(args.ckpt)
    encoder = ObsEncoder(agent.obs_spec)
    env = SimGymEnv(port=args.port)

    try:
        if args.initial_speed > 0:
            # apply a scenario carrying the training spawn speed
            env.set_scenario(
                {"spawn_override": {"initial_speed": args.initial_speed}},
                reset=True)
        for ep in range(args.episodes):
            obs, _ = env.reset()
            encoder.reset()
            enc = encoder.encode(np.asarray(obs, np.float32).ravel(), None)
            ret, speed_sum, last_info = 0.0, 0.0, {}
            for _ in range(args.max_steps):
                a, _ = agent.act(enc, deterministic=not args.stochastic)
                obs, r, term, trunc, info = env.step(a)
                ret += r
                speed_sum += info.get("speed", 0.0)
                enc = encoder.encode(np.asarray(obs, np.float32).ravel(),
                                     info.get("last_action", a))
                last_info = info
                if term or trunc:
                    break
                time.sleep(0.0)  # keep TCP roundtrip pace
            n = max(1, int(last_info.get("step", 1)))
            print(f"ep{ep}: return={ret:.1f} len={last_info.get('step')} "
                  f"speed={speed_sum / n:.2f} "
                  f"progress={last_info.get('lap_progress', 0):.3f} "
                  f"ckpts={last_info.get('checkpoints_passed', 0)} "
                  f"reason={last_info.get('termination_reason', '')}",
                  flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
