"""Run a checkpoint policy on a track — the smoke entry point.

  python tests/agent/run_policy.py --ckpt <path.pt> --track oval --episodes 3
  python tests/agent/run_policy.py --ckpt <path.pt> --tcp 8765 --episodes 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentRL.checkpoints.io import load_agent          # noqa: E402
from agentRL.envs.factory import EnvFactory            # noqa: E402
from agentRL.envs.track_registry import TrackRegistry  # noqa: E402
from agentRL.obs.encoder import ObsEncoder             # noqa: E402


def run_in_process(agent, track_id: str, episodes: int, max_steps: int):
    factory = EnvFactory(obs_spec=agent.obs_spec)
    env = factory.build(TrackRegistry.default().load(track_id), seed=0)
    encoder = ObsEncoder(agent.obs_spec)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        encoder.reset()
        enc = encoder.encode(np.asarray(obs, np.float32), None)
        ret, speed = 0.0, 0.0
        for _ in range(max_steps):
            a, _ = agent.act(enc, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            speed += info.get("speed", 0.0)
            enc = encoder.encode(np.asarray(obs, np.float32),
                                 info.get("last_action", a))
            if term or trunc:
                break
        n = max(1, info.get("step", 1))
        print(f"ep{ep}: return={ret:.1f} len={info.get('step')} "
              f"speed={speed / n:.2f} progress={info.get('lap_progress', 0):.2f} "
              f"reason={info.get('termination_reason', '')}")


def run_tcp(agent, port: int, episodes: int, max_steps: int):
    from sim_client import SimGymEnv
    env = SimGymEnv(port=port)
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            ret = 0.0
            enc = np.asarray(obs, np.float32).ravel()
            for _ in range(max_steps):
                a, _ = agent.act(enc, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                ret += r
                enc = np.asarray(obs, np.float32).ravel()
                if term or trunc:
                    break
            print(f"ep{ep} (tcp): return={ret:.1f} "
                  f"reason={info.get('termination_reason', '')}")
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--tcp", type=int, default=None,
                    help="port of a running headless sim (skip in-process)")
    args = ap.parse_args()
    agent = load_agent(args.ckpt)
    if args.tcp:
        run_tcp(agent, args.tcp, args.episodes, args.max_steps)
    else:
        run_in_process(agent, args.track, args.episodes, args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
