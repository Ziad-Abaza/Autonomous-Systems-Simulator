"""Checkpoint io — save/load dispatch + run-state sidecars."""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from agentRL.core.agent import BaseRLAgent
from agentRL.core.versioning import VersionError

_ALGOS = {}


def _algo_map() -> dict[str, type[BaseRLAgent]]:
    if not _ALGOS:
        from agentRL.algos.ppo import PPOAgent
        from agentRL.algos.sac import SACAgent
        _ALGOS.update({"sac": SACAgent, "ppo": PPOAgent})
    return _ALGOS


def save_checkpoint(agent: BaseRLAgent, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    agent.save(path)


def load_agent(path: str) -> BaseRLAgent:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    algo = payload.get("algo_id")
    cls = _algo_map().get(algo)
    if cls is None:
        raise VersionError(f"unknown algo_id {algo!r} in {path}")
    return cls.load(path)


def write_run_state(run_dir: str, state: dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_state.json"), "w",
              encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def read_run_state(run_dir: str) -> dict[str, Any]:
    p = os.path.join(run_dir, "run_state.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)
