"""evaluate_policy — frozen-policy rollout evaluation on a track.

Deterministic policy, N seeded episodes. Per-episode record carries the
failure class; aggregate carries rates + means. All metrics derive from
info[] diagnostics — never oracle ground truth.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from agentRL.core.agent import BaseRLAgent
from agentRL.envs.factory import EnvFactory
from agentRL.envs.scenario_gen import ScenarioMutator
from agentRL.envs.track_registry import TrackSpec
from agentRL.eval.failures import classify_failure
from agentRL.obs.encoder import ObsEncoder


def evaluate_policy(
    factory: EnvFactory,
    track: TrackSpec,
    agent: BaseRLAgent,
    seeds: Iterable[int] = (42, 43, 44, 45, 46),
    mutator: ScenarioMutator | None = None,
    max_steps: int = 2000,
    out_rows: list[dict] | None = None,
) -> dict[str, Any]:
    """Run deterministic eval episodes; return per_episode + aggregate."""
    env = factory.build(track, seed=0)
    encoder = ObsEncoder(factory.obs_spec)
    episodes: list[dict[str, Any]] = []

    for seed in seeds:
        if mutator is not None:
            mutator.draw_and_apply(env)
        obs, _ = env.reset(seed=int(seed))
        encoder.reset()
        encoded = encoder.encode(np.asarray(obs, np.float32), None)
        ret, speed, flips, prev_steer = 0.0, 0.0, 0, 0.0
        lat_max, lat_sum = 0.0, 0.0
        head_sum = 0.0
        info: dict[str, Any] = {}
        for _ in range(max_steps):
            a_env, _ = agent.act(encoded, deterministic=True)
            nobs, reward, terminated, truncated, info = env.step(a_env)
            ret += float(reward)
            speed += float(info.get("speed", 0.0))
            lat = abs(float(info.get("lateral_offset", 0.0)))
            lat_max, lat_sum = max(lat_max, lat), lat_sum + lat
            head_sum += abs(float(info.get("heading_error", 0.0)))
            if prev_steer * float(a_env[0]) < 0 and abs(prev_steer) > 0.05:
                flips += 1
            prev_steer = float(a_env[0])
            done = terminated or truncated
            encoded = encoder.encode(np.asarray(nobs, np.float32),
                                     info.get("last_action", a_env))
            if done:
                break
        if not (terminated or truncated):
            # eval horizon cut — semantically a max_steps truncation
            terminated, truncated = False, True
            info["termination_reason"] = "max_steps"
        steps = max(1, int(info.get("step", 0)) or 1)
        ep = {
            "seed": int(seed), "return": ret, "length": steps,
            "mean_speed": speed / steps,
            "mean_lateral": lat_sum / steps, "max_lateral": lat_max,
            "mean_heading_err": head_sum / steps,
            "steer_sign_flips": flips / steps,
            "lap_progress": float(info.get("lap_progress", 0.0)),
            "laps_completed": int(info.get("laps_completed", 0)),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "termination_reason": info.get("termination_reason", ""),
            "is_colliding": bool(info.get("is_colliding", False)),
        }
        ep["failure_class"] = classify_failure(ep)
        episodes.append(ep)
        if out_rows is not None:
            out_rows.append({"track": track.track_id,
                             "algo": getattr(agent, "algo_id", "?"),
                             **ep})

    def rate(cls: str) -> float:
        return sum(1 for e in episodes if e["failure_class"] == cls) \
            / len(episodes)

    timeouts = {"stall_timeout", "no_progress", "timeout_progress",
                "oscillation"}
    agg = {
        "n_episodes": len(episodes),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "std_return": float(np.std([e["return"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "completion_rate": rate("completed"),
        "collision_rate": rate("collision"),
        "off_road_rate": rate("off_track"),
        "wrong_direction_rate": rate("wrong_direction"),
        "timeout_rate": sum(rate(c) for c in timeouts),
        "mean_progress": float(np.mean([e["lap_progress"]
                                        for e in episodes])),
        "mean_speed": float(np.mean([e["mean_speed"]
                                     for e in episodes])),
        "mean_lateral": float(np.mean([e["mean_lateral"]
                                       for e in episodes])),
        "max_lateral": float(np.max([e["max_lateral"]
                                     for e in episodes])),
        "mean_heading_err": float(np.mean([e["mean_heading_err"]
                                           for e in episodes])),
        "steer_smoothness": float(1.0 - np.mean(
            [e["steer_sign_flips"] for e in episodes])),
        "failure_classes": {
            c: sum(1 for e in episodes if e["failure_class"] == c)
            for c in sorted({e["failure_class"] for e in episodes})},
    }
    return {"per_episode": episodes, "aggregate": agg}
