"""
Behavior Cloning data loading over transitions_v1 datasets.

Compatibility gates (checked BEFORE training, hard-fail with BCDataError):
  - dataset passes transitions_v1 validation (format + structure)
  - manifest env_fingerprint matches the caller's expected fingerprint
  - observation vectors match the expected obs_dim

Splitting is delegated to dataset.split_dataset so BC consumes exactly
the same deterministic partitions other tooling sees.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import numpy as np

from sim_experiment.dataset import (
    validate_dataset, split_dataset, _read_episode_jsonl,
)


class BCDataError(ValueError):
    """Dataset is structurally valid JSON but incompatible for BC."""


def load_transitions(
    dataset_dir: str,
    split: str = "train",
    seed: int = 42,
    obs_dim: Optional[int] = None,
    expected_env_fingerprint: Optional[str] = None,
    ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads (obs, action) arrays for one dataset split.

    split: "train" | "val" | "test" | "all"
    Returns {"obs": float32 [N, obs_dim], "actions": [N, ...], "episode_ids": [...]}
    """
    if not os.path.isdir(dataset_dir):
        raise BCDataError(f"dataset directory does not exist: {dataset_dir}")

    rep = validate_dataset(dataset_dir)
    if not rep["valid"]:
        raise BCDataError(
            "dataset failed transitions_v1 validation: "
            + "; ".join(rep["errors"][:5]))

    manifest_fp = rep.get("fingerprints") or [None]
    if expected_env_fingerprint is not None:
        found = [f for f in manifest_fp if f == expected_env_fingerprint]
        if not found:
            raise BCDataError(
                f"env fingerprint mismatch: dataset contains "
                f"{manifest_fp}, expected '{expected_env_fingerprint}'")

    episodes = _read_episode_jsonl(os.path.join(dataset_dir, "episodes.jsonl"))

    if split != "all":
        splits = split_dataset(dataset_dir, seed=seed, ratios=ratios)
        if split not in splits:
            raise BCDataError(f"split '{split}' not in {sorted(splits.keys())}")
        wanted = set(splits[split])
        episodes = [e for e in episodes if str(e.get("episode_id")) in wanted]

    obs_list: List[Any] = []
    act_list: List[Any] = []
    ep_ids: List[str] = []
    for ep in episodes:
        eid = str(ep.get("episode_id"))
        for st in ep.get("steps") or []:
            o = st.get("obs")
            if isinstance(o, dict):
                o = o.get("vector")
            o = np.asarray(o, dtype=np.float32).reshape(-1)
            if obs_dim is not None and o.shape[0] != obs_dim:
                raise BCDataError(
                    f"obs dim mismatch: episode '{eid}' has obs_dim "
                    f"{o.shape[0]}, expected {obs_dim}")
            obs_list.append(o)
            act_list.append(st.get("action"))
            ep_ids.append(eid)

    if not obs_list:
        raise BCDataError(f"split '{split}' contains no transitions")

    obs = np.stack(obs_list).astype(np.float32)
    actions = np.asarray(act_list)
    if actions.dtype == object:
        actions = np.asarray(act_list, dtype=np.float32)
    return {"obs": obs, "actions": actions, "episode_ids": ep_ids}
