"""
Trajectory dataset export — imitation-learning-ready transitions.

Reads run trajectories/ JSONL (which already separates agent_data from
diagnostic_data) and exports a flat transition dataset. Agent-facing fields
ONLY at step level — diagnostic data (speed, offsets, collision flags) is
structurally excluded from steps; it can appear only as episode-level
summary metadata.

Schema-compat enforcement: episodes are filtered by env fingerprint and
observation/action schema hashes; mismatched episodes are skipped and
counted in the report — never silently mixed.
"""

from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

DATASET_FORMAT = "transitions_v1"


def _episode_files(run_dir: str) -> List[str]:
    traj_dir = os.path.join(run_dir, "trajectories")
    if not os.path.isdir(traj_dir):
        return []
    return sorted(
        os.path.join(traj_dir, f) for f in os.listdir(traj_dir)
        if f.endswith(".jsonl")
    )


def _read_episode(path: str) -> Optional[Dict[str, Any]]:
    metadata = None
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metadata is None:
                metadata = rec
                continue
            steps.append(rec)
    if metadata is None:
        return None
    return {"metadata": metadata, "steps": steps, "path": path}


def list_episodes(run_dir: str) -> List[Dict[str, Any]]:
    """Episode-level index for the trajectory explorer."""
    out = []
    for path in _episode_files(run_dir):
        ep = _read_episode(path)
        if ep is None:
            continue
        md = ep["metadata"]
        last = ep["steps"][-1] if ep["steps"] else {}
        agent = last.get("agent_data", {})
        total_return = sum(
            float(s.get("agent_data", {}).get("reward", 0.0)) for s in ep["steps"])
        out.append({
            "episode_id": md.get("episode_id"),
            "env_fingerprint": md.get("env_fingerprint"),
            "scenario_id": md.get("scenario_id"),
            "seed": md.get("seed"),
            "steps": len(ep["steps"]),
            "total_return": round(total_return, 4),
            "termination_reason": agent.get("termination_reason", ""),
            "file": os.path.basename(path),
        })
    return out


def _schema_hash(metadata: Dict[str, Any]) -> str:
    payload = {
        "observation_schema": metadata.get("observation_schema"),
        "action_schema": metadata.get("action_schema"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def export_dataset(
    run_dir: str,
    output_dir: str,
    env_fingerprint: Optional[str] = None,
    min_return: Optional[float] = None,
    termination_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Exports agent-visible transitions to <output_dir>:
        manifest.json    — dataset metadata + filters applied
        episodes.jsonl   — one episode per line:
            {episode_id, seed, total_return, length, termination_reason,
             steps: [{obs, action, reward, terminated, truncated}]}

    Steps contain ONLY agent_data — diagnostics never cross the boundary.
    """
    os.makedirs(output_dir, exist_ok=True)
    wanted_reasons = set(termination_reasons or [])

    episodes_out = []
    skipped_fp = skipped_return = skipped_reason = 0
    schema_hash = None
    total_steps = 0

    for path in _episode_files(run_dir):
        ep = _read_episode(path)
        if ep is None:
            continue
        md = ep["metadata"]
        if env_fingerprint and md.get("env_fingerprint") != env_fingerprint:
            skipped_fp += 1
            continue
        if schema_hash is None:
            schema_hash = _schema_hash(md)
        elif _schema_hash(md) != schema_hash:
            skipped_fp += 1  # schema mismatch is a compat failure
            continue

        steps = ep["steps"]
        agent_steps = [s.get("agent_data", {}) for s in steps]
        total_return = sum(float(a.get("reward", 0.0)) for a in agent_steps)
        last_reason = agent_steps[-1].get("termination_reason", "") if agent_steps else ""

        if min_return is not None and total_return < min_return:
            skipped_return += 1
            continue
        if wanted_reasons and last_reason not in wanted_reasons:
            skipped_reason += 1
            continue

        # Strict agent-data isolation: whitelist only the legal fields.
        clean_steps = [{
            "obs": a.get("obs"),
            "action": a.get("action"),
            "reward": a.get("reward"),
            "terminated": bool(a.get("terminated")),
            "truncated": bool(a.get("truncated")),
            "termination_reason": a.get("termination_reason", ""),
        } for a in agent_steps]

        episodes_out.append({
            "episode_id": md.get("episode_id"),
            "seed": md.get("seed"),
            "env_fingerprint": md.get("env_fingerprint"),
            "scenario_id": md.get("scenario_id"),
            "total_return": round(total_return, 4),
            "length": len(clean_steps),
            "termination_reason": last_reason,
            "steps": clean_steps,
        })
        total_steps += len(clean_steps)

    out_path = os.path.join(output_dir, "episodes.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + "\n")

    manifest = {
        "dataset_format": DATASET_FORMAT,
        "source_run_dir": os.path.abspath(run_dir),
        "schema_hash": schema_hash,
        "env_fingerprint": env_fingerprint,
        "episodes": len(episodes_out),
        "steps": total_steps,
        "filters": {
            "env_fingerprint": env_fingerprint,
            "min_return": min_return,
            "termination_reasons": sorted(wanted_reasons),
        },
        "skipped": {
            "fingerprint_or_schema_mismatch": skipped_fp,
            "below_min_return": skipped_return,
            "termination_reason": skipped_reason,
        },
        "isolation": "steps contain agent_data only; diagnostic_data excluded",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "episodes": len(episodes_out),
        "steps": total_steps,
        "skipped": manifest["skipped"],
        "output_dir": os.path.abspath(output_dir),
    }
