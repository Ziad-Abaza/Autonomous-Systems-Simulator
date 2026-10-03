"""
Deterministic trajectory sampling.

Replaces prefix-based capture limits with seed-keyed episode selection:
the same (run_dir, count, seed) always yields the same episode subset,
independent of episode completion order or file timestamps.
"""

from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from sim_experiment.dataset import _episode_files, _read_episode


def _sample_key(episode_id: str, seed: int) -> str:
    """Stable hash key — identical across processes and platforms."""
    return hashlib.sha256(f"{int(seed)}:{episode_id}".encode("utf-8")).hexdigest()


def sample_trajectories(
    run_dir: str,
    count: int,
    seed: int,
    strategy: str = "uniform",
) -> List[Dict[str, Any]]:
    """
    Deterministically selects up to `count` episodes from a run's
    trajectories/ directory.

    strategy="uniform"    — hash-keyed uniform subset (default)
    strategy="best"       — highest total_return episodes (seed tiebreak)
    strategy="mixed"      — half best, half uniform from the rest

    Returns episode dicts {episode_id, seed, env_fingerprint,
    total_return, steps (raw records), path}.
    """
    if strategy not in ("uniform", "best", "mixed"):
        raise ValueError(f"unknown sampling strategy: {strategy!r}")

    episodes = []
    for path in _episode_files(run_dir):
        ep = _read_episode(path)
        if ep is None:
            continue
        md = ep["metadata"]
        agent_steps = [s.get("agent_data", {}) for s in ep["steps"]]
        episodes.append({
            "episode_id": md.get("episode_id"),
            "seed": md.get("seed"),
            "episode_seed": md.get("episode_seed"),
            "curriculum_stage_index": md.get("curriculum_stage_index"),
            "env_fingerprint": md.get("env_fingerprint"),
            "scenario_id": md.get("scenario_id"),
            "total_return": sum(float(a.get("reward", 0.0)) for a in agent_steps),
            "steps": ep["steps"],
            "path": path,
        })

    if len(episodes) <= count:
        return episodes
    if count <= 0:
        return []

    if strategy == "best":
        ranked = sorted(episodes,
                        key=lambda e: (-e["total_return"],
                                       _sample_key(e["episode_id"], seed)))
        return ranked[:count]

    if strategy == "mixed":
        n_best = count // 2
        ranked = sorted(episodes,
                        key=lambda e: (-e["total_return"],
                                       _sample_key(e["episode_id"], seed)))
        best = ranked[:n_best]
        remaining = [e for e in episodes if e not in best]
        remaining.sort(key=lambda e: _sample_key(e["episode_id"], seed))
        return best + remaining[:count - n_best]

    episodes.sort(key=lambda e: _sample_key(e["episode_id"], seed))
    return episodes[:count]
