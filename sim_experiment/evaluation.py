"""
Evaluation System.

Evaluation is strictly separate from training state: it runs a frozen
policy (loaded from an opaque checkpoint artifact) in a fresh environment
over a fixed seed list, and persists an immutable result artifact.

The policy adapter is algorithm-agnostic: `evaluate_policy` takes any
callable(obs) -> action. `make_policy_from_checkpoint` currently supports
the PPO ActorCritic format; other algorithms can register their own
adapter without touching the evaluator.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable

import numpy as np


@dataclass
class EvaluationResult:
    """Immutable evaluation artifact."""
    eval_id: str = ""
    checkpoint_path: str = ""
    algorithm: str = ""
    env_fingerprint: str = ""
    scenario_id: str = ""
    deterministic_policy: bool = True
    seeds: List[int] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        r = cls()
        for k in ("eval_id", "checkpoint_path", "algorithm", "env_fingerprint",
                  "scenario_id", "deterministic_policy", "seeds", "episodes",
                  "aggregate", "created_at"):
            if k in data:
                setattr(r, k, data[k])
        return r

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_policy(
    env: Any,
    act_fn: Callable[[Any], Any],
    seeds: List[int],
    num_episodes: int,
    deterministic: bool = True,
    result_kwargs: Optional[Dict[str, Any]] = None,
    step_observer: Optional[Callable[[int, int, Any, float, Dict[str, Any]], None]] = None,
) -> EvaluationResult:
    """
    Runs `num_episodes` evaluation episodes cycling through `seeds`.
    Does not touch any training state — the caller provides a fresh env
    and a frozen act_fn. `step_observer(episode_idx, step_idx, action,
    reward, info)` allows callers to capture replays/trajectories.
    """
    episodes: List[Dict[str, Any]] = []
    for i in range(num_episodes):
        seed = seeds[i % len(seeds)] if seeds else 0
        obs, _ = env.reset(seed=seed)

        ep_return = 0.0
        ep_len = 0
        lat_errs: List[float] = []
        head_errs: List[float] = []
        speeds: List[float] = []
        reason = "max_steps"
        collided = False
        off_road = False
        completed = False

        while True:
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if step_observer is not None:
                step_observer(i, ep_len, action, float(reward), info)
            lat_errs.append(abs(float(info.get("lateral_offset", 0.0))))
            head_errs.append(abs(float(info.get("heading_error", 0.0))))
            speeds.append(float(info.get("speed", 0.0)))
            if terminated or truncated:
                # env-reported step count is authoritative for episode length
                ep_len = int(info.get("step", ep_len))
                reason = str(info.get("termination_reason", "unknown"))
                collided = "collision" in reason or bool(info.get("is_colliding", False))
                off_road = "off_road" in reason
                completed = "lap_completed" in reason or "course" in reason
                break

        episodes.append({
            "seed": seed,
            "reward": round(ep_return, 4),
            "length": ep_len,
            "termination_reason": reason,
            "completed": completed,
            "collided": collided,
            "off_road": off_road,
            "timed_out": truncated and not terminated,
            "mean_lateral_error": round(float(np.mean(lat_errs)), 4) if lat_errs else 0.0,
            "mean_heading_error": round(float(np.mean(head_errs)), 4) if head_errs else 0.0,
            "mean_speed": round(float(np.mean(speeds)), 4) if speeds else 0.0,
        })

    rewards = [e["reward"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    n = max(1, len(episodes))
    aggregate = {
        "episode_count": len(episodes),
        "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
        "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
        "min_reward": round(float(min(rewards)), 4) if rewards else 0.0,
        "max_reward": round(float(max(rewards)), 4) if rewards else 0.0,
        "completion_rate": round(sum(1 for e in episodes if e["completed"]) / n, 4),
        "collision_rate": round(sum(1 for e in episodes if e["collided"]) / n, 4),
        "off_road_rate": round(sum(1 for e in episodes if e["off_road"]) / n, 4),
        "timeout_rate": round(sum(1 for e in episodes if e["timed_out"]) / n, 4),
        "mean_episode_length": round(float(np.mean(lengths)), 2) if lengths else 0.0,
        "mean_lateral_error": round(float(np.mean([e["mean_lateral_error"] for e in episodes])), 4),
        "mean_heading_error": round(float(np.mean([e["mean_heading_error"] for e in episodes])), 4),
        "mean_speed": round(float(np.mean([e["mean_speed"] for e in episodes])), 4),
    }

    result = EvaluationResult(
        eval_id=f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        deterministic_policy=deterministic,
        seeds=list(seeds),
        episodes=episodes,
        aggregate=aggregate,
        created_at=time.time(),
    )
    for k, v in (result_kwargs or {}).items():
        setattr(result, k, v)
    return result


def export_transitions(
    env: Any,
    act_fn: Callable[[Any], Any],
    seeds: List[int],
    num_episodes: int,
    output_dir: str,
    env_fingerprint: str = "",
    scenario_id: str = "",
) -> Dict[str, Any]:
    """
    Runs eval episodes with a frozen policy and exports every transition as
    a complete transitions_v1 dataset (episodes.jsonl + manifest.json) —
    the format consumed by imitation training and dataset inspection.

    Agent-facing fields only: obs/action/reward/terminated/truncated/
    termination_reason — diagnostic fields never cross the boundary.
    """
    from sim_experiment.dataset import DATASET_FORMAT

    os.makedirs(output_dir, exist_ok=True)
    episodes_out = []
    total_steps = 0

    for i in range(num_episodes):
        seed = seeds[i % len(seeds)] if seeds else 0
        obs, _ = env.reset(seed=seed)
        steps = []
        ep_return = 0.0
        last_reason = ""
        while True:
            action = act_fn(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            o = obs
            if isinstance(o, dict):
                o = o.get("vector")
            steps.append({
                "obs": np.asarray(o, dtype=np.float32).tolist(),
                "action": np.asarray(action).tolist()
                        if not isinstance(action, (int, float)) else action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": str(info.get("termination_reason", "")),
            })
            obs = next_obs
            last_reason = str(info.get("termination_reason", ""))
            if terminated or truncated:
                break

        episodes_out.append({
            "episode_id": f"eval_ep{i}",
            "seed": int(seed),
            "episode_seed": int(seed),
            "env_fingerprint": env_fingerprint,
            "scenario_id": scenario_id,
            "total_return": round(ep_return, 4),
            "length": len(steps),
            "termination_reason": last_reason,
            "steps": steps,
        })
        total_steps += len(steps)

    with open(os.path.join(output_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + "\n")
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_format": DATASET_FORMAT,
            "source": "evaluation_export",
            "env_fingerprint": env_fingerprint,
            "episodes": len(episodes_out),
            "steps": total_steps,
            "filters": {},
            "skipped": {},
            "isolation": "steps contain agent-visible fields only",
        }, f, indent=2)

    return {
        "episodes": len(episodes_out),
        "steps": total_steps,
        "output_dir": os.path.abspath(output_dir),
    }


def _obs_vec(obs, obs_dim: int):
    if isinstance(obs, dict):
        obs = obs.get("vector", np.zeros(obs_dim, dtype=np.float32))
    return np.asarray(obs, dtype=np.float32)


def make_policy_from_checkpoint(
    checkpoint_path: str,
    algorithm: str = "ppo",
    deterministic: bool = True,
) -> Callable[[Any], Any]:
    """
    Builds a policy callable from an opaque checkpoint artifact.
    Supports the PPO, SAC, and DQN checkpoint formats.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])

    if algorithm == "ppo":
        from sim_client.agents.ppo_baseline import ActorCritic
        act_dim = int(ckpt["act_dim"])
        net = ActorCritic(obs_dim, act_dim)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

        def policy(obs):
            x = torch.tensor(_obs_vec(obs, obs_dim), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    feat = net.actor_backbone(x)
                    mean = net.actor_mean(feat)
                    a = mean.squeeze(0).cpu().numpy()
                else:
                    a, _, _, _ = net.get_action_and_value(x)
                    a = a.squeeze(0).cpu().numpy()
            return [
                float(np.clip(a[0], -1.0, 1.0)),
                float(np.clip(a[1], 0.0, 1.0)),
                float(np.clip(a[2], 0.0, 1.0)),
            ]
        return policy

    if algorithm == "sac":
        from sim_client.agents.sac_baseline import SACActor
        act_dim = int(ckpt["act_dim"])
        actor = SACActor(obs_dim, act_dim)
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()
        low = np.asarray(ckpt.get("action_low", [-1.0, 0.0, 0.0]), dtype=np.float32)
        high = np.asarray(ckpt.get("action_high", [1.0, 1.0, 1.0]), dtype=np.float32)

        def policy(obs):
            x = torch.tensor(_obs_vec(obs, obs_dim), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    a = actor.deterministic(x).squeeze(0).cpu().numpy()
                else:
                    a, _ = actor.sample(x)
                    a = a.squeeze(0).cpu().numpy()
            return low + (a + 1.0) * 0.5 * (high - low)
        return policy

    if algorithm == "dqn":
        from sim_client.agents.dqn_baseline import QNetwork
        num_actions = int(ckpt["num_actions"])
        net = QNetwork(obs_dim, num_actions)
        net.load_state_dict(ckpt["q_state_dict"])
        net.eval()

        def policy(obs):
            x = torch.tensor(_obs_vec(obs, obs_dim), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = net(x)
            return int(q_vals.argmax(dim=1).item())
        return policy

    if algorithm == "bc":
        from sim_experiment.bc.bc_model import BCPolicy
        action_mode = ckpt.get("action_mode", "continuous")
        act_dim = int(ckpt["act_dim"])
        hidden = tuple(ckpt.get("hidden") or (64, 64))
        net = BCPolicy(obs_dim, act_dim, hidden, action_mode)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

        def policy(obs):
            x = torch.tensor(_obs_vec(obs, obs_dim), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = net(x).squeeze(0)
            if action_mode == "discrete":
                return int(out.argmax().item())
            return out.cpu().numpy()
        return policy

    raise ValueError(f"No checkpoint adapter for algorithm: {algorithm!r}")
