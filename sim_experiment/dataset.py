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
    """
    Reads one trajectory JSONL file. The first record MUST carry
    type == "header" and subsequent records type == "step" — positional
    assumptions are not a contract.
    """
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
                if rec.get("type") != "header":
                    # Malformed file: first record is not a header.
                    return None
                metadata = rec
                continue
            if rec.get("type") != "step":
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
    skipped_fp = skipped_return = skipped_reason = skipped_schema = 0
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
            skipped_schema += 1  # schema mismatch is a compat failure
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
            # Combined key kept for older consumers; the split keys are the
            # authoritative accounting.
            "fingerprint_or_schema_mismatch": skipped_fp + skipped_schema,
            "fingerprint_mismatch": skipped_fp,
            "schema_mismatch": skipped_schema,
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


# ------------------------------------------------------------------- validation

_AGENT_STEP_FIELDS = {"obs", "action", "reward", "terminated", "truncated",
                      "termination_reason"}


def _read_episode_jsonl(path: str) -> List[Dict[str, Any]]:
    """Reads a transitions_v1 episodes.jsonl line-by-line."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Independently validates a transitions_v1 dataset directory.

    Returns a report dict:
        valid, errors[], warnings[], episodes, steps,
        fingerprints (distinct), schema_hash, step_field_violations
    """
    errors: List[str] = []
    warnings: List[str] = []
    episodes = 0
    steps = 0
    fingerprints: List[str] = []
    step_violations = 0

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    episodes_path = os.path.join(dataset_dir, "episodes.jsonl")
    manifest: Dict[str, Any] = {}

    if not os.path.exists(manifest_path):
        errors.append("missing manifest.json")
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"manifest.json is not valid JSON: {e}")
        else:
            if manifest.get("dataset_format") != DATASET_FORMAT:
                errors.append(
                    f"dataset_format {manifest.get('dataset_format')!r} "
                    f"!= expected '{DATASET_FORMAT}'")

    if not os.path.exists(episodes_path):
        errors.append("missing episodes.jsonl")
    else:
        try:
            eps = _read_episode_jsonl(episodes_path)
        except json.JSONDecodeError as e:
            errors.append(f"episodes.jsonl contains invalid JSON: {e}")
            eps = []
        episodes = len(eps)
        seen_ids = set()
        for ei, ep in enumerate(eps):
            eid = ep.get("episode_id")
            if eid is None:
                errors.append(f"episode[{ei}]: missing episode_id")
            elif eid in seen_ids:
                errors.append(f"episode[{ei}]: duplicate episode_id '{eid}'")
            else:
                seen_ids.add(eid)
            fp = ep.get("env_fingerprint")
            if fp and fp not in fingerprints:
                fingerprints.append(fp)
            if manifest.get("env_fingerprint") and fp != manifest["env_fingerprint"]:
                warnings.append(
                    f"episode[{ei}]: env_fingerprint '{fp}' differs from "
                    f"manifest '{manifest['env_fingerprint']}'")
            ep_steps = ep.get("steps")
            if not isinstance(ep_steps, list) or not ep_steps:
                errors.append(f"episode[{ei}]: missing or empty steps list")
                continue
            steps += len(ep_steps)
            declared_len = ep.get("length")
            if declared_len is not None and int(declared_len) != len(ep_steps):
                warnings.append(
                    f"episode[{ei}]: length field {declared_len} != "
                    f"actual {len(ep_steps)} steps")
            for si, st in enumerate(ep_steps):
                if not isinstance(st, dict):
                    errors.append(f"episode[{ei}].steps[{si}]: not an object")
                    step_violations += 1
                    continue
                missing = {"obs", "action", "reward"} - set(st.keys())
                if missing:
                    errors.append(
                        f"episode[{ei}].steps[{si}]: missing required "
                        f"fields {sorted(missing)}")
                    step_violations += 1
                extra = set(st.keys()) - _AGENT_STEP_FIELDS
                if extra:
                    # Diagnostic leakage is a security contract violation.
                    errors.append(
                        f"episode[{ei}].steps[{si}]: unexpected diagnostic "
                        f"fields {sorted(extra)}")
                    step_violations += 1
            last = ep_steps[-1] if ep_steps else {}
            if not (last.get("terminated") or last.get("truncated")):
                warnings.append(
                    f"episode[{ei}]: last step is neither terminated nor truncated")

        # Manifest consistency
        if manifest:
            if "episodes" in manifest and int(manifest["episodes"]) != episodes:
                errors.append(
                    f"manifest episodes count {manifest['episodes']} != "
                    f"actual {episodes}")
            if "steps" in manifest and int(manifest["steps"]) != steps:
                errors.append(
                    f"manifest steps count {manifest['steps']} != actual {steps}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "episodes": episodes,
        "steps": steps,
        "fingerprints": fingerprints,
        "schema_hash": manifest.get("schema_hash"),
        "step_field_violations": step_violations,
    }


# ------------------------------------------------------------------- splitting

def split_dataset(
    dataset_dir: str,
    seed: int,
    ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """
    Deterministically partitions dataset episodes into train/val/test by
    hashing (episode_id, seed) — reproducible across runs/machines for the
    same inputs. Writes <dataset_dir>/splits.json and returns
    {split_name: [episode_ids]}.
    """
    ratios = dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1})
    if not ratios:
        raise ValueError("ratios must be a non-empty dict")
    episodes_path = os.path.join(dataset_dir, "episodes.jsonl")
    eps = _read_episode_jsonl(episodes_path)
    ids = [str(e.get("episode_id")) for e in eps]

    def _key(eid: str) -> str:
        return hashlib.sha256(f"{seed}:{eid}".encode("utf-8")).hexdigest()

    ordered = sorted(ids, key=_key)
    n = len(ordered)
    names = list(ratios.keys())
    total_ratio = sum(float(ratios[k]) for k in names)
    if total_ratio <= 0:
        raise ValueError("ratios must sum to > 0")

    # Largest-remainder allocation so counts sum exactly to n.
    raw = [(k, n * float(ratios[k]) / total_ratio) for k in names]
    counts = {k: int(v) for k, v in raw}
    remainder = n - sum(counts.values())
    by_frac = sorted(names, key=lambda k: (raw[names.index(k)][1] - int(raw[names.index(k)][1])), reverse=True)
    for k in by_frac[:remainder]:
        counts[k] += 1

    splits: Dict[str, List[str]] = {}
    idx = 0
    for k in names:
        splits[k] = ordered[idx:idx + counts[k]]
        idx += counts[k]

    with open(os.path.join(dataset_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": int(seed), "ratios": ratios,
            "episodes": n, "splits": splits,
        }, f, indent=2)
    return splits
