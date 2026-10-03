"""Experiment matrix runner — config-driven, reproducible runs.

  python -m agentRL.experiments.matrix --exp E001
  python -m agentRL.experiments.matrix --exp E001 --steps-override 400

Each config is a small JSON contract in agentRL/experiments/configs/.
Run artifacts land in agentRL/runs/<EID>/ and a results summary JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from agentRL.checkpoints.io import load_agent
from agentRL.core.config import AgentConfig, TrainConfig
from agentRL.core.seeding import seed_tree, set_global_seeds
from agentRL.envs.factory import EnvFactory
from agentRL.envs.scenario_gen import ScenarioMutator
from agentRL.envs.track_registry import TrackRegistry
from agentRL.eval.matrix import EvalMatrix
from agentRL.obs.spec import ObservationSpec, PRESETS
from agentRL.train.continual import ContinualTrainer, Phase
from agentRL.train.trainer import MixedTrackTrainer, make_trainer

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
EXPERIMENT_IDS = [f"E{i:03d}" for i in range(1, 11)]

ACT_SPACE = {"low": np.array([-1.0, 0.0, 0.0]),
             "high": np.array([1.0, 1.0, 1.0]),
             "pos_only": (2,)}  # brake: raw<0 = released (brake>=0.1 parks car)


def load_config(path: os.PathLike | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _config_for(exp_id: str) -> dict[str, Any]:
    path = CONFIGS_DIR / f"{exp_id.lower()}.json"
    if not path.exists():
        raise KeyError(f"no config for {exp_id} in {CONFIGS_DIR}")
    cfg = load_config(path)
    cfg.setdefault("id", exp_id)
    return cfg


def _build_agent(algo: str, cfg: dict, seed: int):
    obs = ObservationSpec(
        channel_names=PRESETS[cfg.get("obs_preset", "full23")],
        frame_stack=int(cfg.get("frame_stack", 1)),
        prev_action=bool(cfg.get("prev_action", False)))
    common = dict(hidden_sizes=tuple(cfg.get("hidden", (256, 256))),
                  lr=float(cfg.get("lr", 3e-4)),
                  gamma=float(cfg.get("gamma", 0.99)),
                  extra={"seed": seed, **cfg.get("agent_overrides", {})})
    if algo == "sac":
        from agentRL.algos.sac import SACAgent
        return SACAgent(obs, ACT_SPACE, AgentConfig(algo_id="sac", **common))
    if algo == "ppo":
        from agentRL.algos.ppo import PPOAgent
        return PPOAgent(obs, ACT_SPACE, AgentConfig(algo_id="ppo", **common))
    raise KeyError(f"unknown algo {algo!r}")


def _mutator(cfg: dict, seed: int) -> ScenarioMutator | None:
    m = cfg.get("mutator")
    if not m:
        return None
    kwargs = dict(m)
    for k in ("n_obstacles", "lateral_frac", "s_range", "spawn_jitter",
              "friction", "noise", "entity_types"):
        if k in kwargs:
            kwargs[k] = tuple(kwargs[k])
    return ScenarioMutator(seed=seed, **kwargs)


def run_experiment(exp_id: str, run_dir: str | None = None,
                   steps_override: int | None = None,
                   eval_episodes: int | None = None,
                   algos: list[str] | None = None) -> dict[str, Any]:
    cfg = _config_for(exp_id)
    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)
    seeds = seed_tree(seed)
    reg = TrackRegistry.default()
    factory = EnvFactory(
        reward=cfg.get("reward", "drive_v1"),
        termination=cfg.get("termination", "term_v1"),
        obs_spec=ObservationSpec(
            channel_names=PRESETS[cfg.get("obs_preset", "full23")],
            frame_stack=int(cfg.get("frame_stack", 1)),
            prev_action=bool(cfg.get("prev_action", False))))
    results: dict[str, Any] = {}
    base_dir = Path(run_dir or RUNS_ROOT / exp_id)

    for algo in algos or cfg.get("algos", [cfg["algo"]]):
        adir = base_dir / algo
        agent = _build_agent(algo, cfg, seeds["torch"])
        tc = TrainConfig(
            total_steps=int(steps_override or cfg.get("steps", 60_000)),
            eval_interval=int(cfg.get("eval_interval", 10_000)),
            ckpt_interval=int(cfg.get("ckpt_interval", 10_000)),
            num_envs=int(cfg.get("num_envs", 1)),
            seed=seeds["env"], run_dir=str(adir),
            eval_episodes=int(eval_episodes or cfg.get("eval_episodes", 3)))
        kind = cfg.get("kind", "single")
        mutator = _mutator(cfg, seeds["mutator"])

        if kind == "single":
            trainer = make_trainer(factory, reg.load(cfg["track"]),
                                   agent, tc, mutator=mutator)
            results[algo] = trainer.train()
        elif kind == "mixed":
            tracks = [reg.load(t) for t in cfg["tracks"]]
            trainer = MixedTrackTrainer(factory, tracks, agent, tc,
                                        mutator=mutator)
            results[algo] = trainer.train()
        elif kind == "continual":
            ct = ContinualTrainer(
                factory, agent, tc,
                holdout_tracks=cfg.get("holdouts", []),
                eval_seeds=tuple(cfg.get("eval_seeds", (42, 43, 44))),
                eval_max_steps=int(cfg.get("eval_max_steps", 1500)))
            phases = [Phase(p["track_id"],
                            int(steps_override or p["steps"]),
                            mutator=mutator, tag=p.get("tag", ""))
                      for p in cfg["phases"]]
            results[algo] = {"report": str(
                adir / "continual_report.json")}
            ct.run(phases)
        elif kind == "eval":
            agent = load_agent(cfg["ckpt"])
            tracks = [reg.load(t) for t in cfg["tracks"]]
            grid = EvalMatrix(factory).run(
                {agent.algo_id: agent}, tracks, out_dir=str(adir),
                seeds=tuple(cfg.get("eval_seeds", (42, 43, 44, 45, 46))),
                mutator=mutator,
                max_steps=int(cfg.get("eval_max_steps", 2000)))
            results[algo] = grid
        else:
            raise KeyError(f"unknown experiment kind {kind!r}")

    out = dict(cfg)
    out["results"] = {
        a: (r if isinstance(r, dict) else str(r))
        for a, r in results.items()}
    os.makedirs(base_dir, exist_ok=True)
    with open(base_dir / "experiment_result.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="e.g. E001")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--steps-override", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--algos", default=None, help="comma list, e.g. sac,ppo")
    args = ap.parse_args()
    res = run_experiment(args.exp, run_dir=args.run_dir,
                         steps_override=args.steps_override,
                         eval_episodes=args.eval_episodes,
                         algos=(args.algos.split(",")
                                if args.algos else None))
    print(json.dumps({"exp": args.exp,
                      "results": list(res)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
