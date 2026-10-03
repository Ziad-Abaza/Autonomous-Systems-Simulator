"""
Dataset statistics + inspection service for transitions_v1 datasets.

Pure read-only analysis over episodes.jsonl + manifest.json — used by the
UI dataset-inspection surface and by tooling that wants sanity checks
before imitation training.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from sim_experiment.dataset import (
    validate_dataset, _read_episode_jsonl, DATASET_FORMAT,
)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def dataset_statistics(dataset_dir: str) -> Dict[str, Any]:
    """
    Aggregate statistics for a transitions_v1 dataset:
    episode/step counts, return & length distributions, termination-reason
    histogram, obs/action dims, per-split counts if splits.json exists.
    """
    episodes_path = os.path.join(dataset_dir, "episodes.jsonl")
    eps = _read_episode_jsonl(episodes_path) if os.path.exists(episodes_path) else []

    returns: List[float] = []
    lengths: List[float] = []
    reasons: Dict[str, int] = {}
    obs_dims: set = set()
    act_dims: set = set()
    fingerprints: set = set()
    scenario_ids: set = set()

    for ep in eps:
        steps = ep.get("steps") or []
        returns.append(float(ep.get("total_return",
                          sum(float(s.get("reward", 0.0)) for s in steps))))
        lengths.append(len(steps))
        reasons[str(ep.get("termination_reason", ""))] = \
            reasons.get(str(ep.get("termination_reason", "")), 0) + 1
        if ep.get("env_fingerprint"):
            fingerprints.add(ep["env_fingerprint"])
        if ep.get("scenario_id"):
            scenario_ids.add(ep["scenario_id"])
        if steps:
            o = steps[0].get("obs")
            a = steps[0].get("action")
            if isinstance(o, (list, tuple)):
                obs_dims.add(len(o))
            if isinstance(a, (list, tuple)):
                act_dims.add(len(a))
            elif isinstance(a, (int, float)):
                act_dims.add(1)

    splits = {}
    splits_path = os.path.join(dataset_dir, "splits.json")
    if os.path.exists(splits_path):
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                sp = json.load(f)
            splits = {k: len(v) for k, v in (sp.get("splits") or {}).items()}
        except json.JSONDecodeError:
            splits = {"error": "splits.json is not valid JSON"}

    return {
        "episode_count": len(eps),
        "step_count": int(sum(lengths)),
        "return": _stats(returns),
        "length": _stats(lengths),
        "termination_reasons": reasons,
        "obs_dims": sorted(obs_dims),
        "action_dims": sorted(act_dims),
        "fingerprints": sorted(fingerprints),
        "scenario_ids": sorted(scenario_ids),
        "splits": splits,
    }


def inspect_dataset(dataset_dir: str, validate: bool = True) -> Dict[str, Any]:
    """
    Full inspection report for the UI: validation + statistics + manifest
    fields + the recorded split (if any).
    """
    manifest = {}
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = {"_error": "manifest.json is not valid JSON"}

    report: Dict[str, Any] = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "dataset_format": manifest.get("dataset_format"),
        "manifest": manifest,
        "statistics": dataset_statistics(dataset_dir),
    }
    if validate:
        rep = validate_dataset(dataset_dir)
        report.update({
            "valid": rep["valid"],
            "errors": rep["errors"],
            "warnings": rep["warnings"],
        })
    return report
