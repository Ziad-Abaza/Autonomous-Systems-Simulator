"""
Deep Q-Network (DQN) implementation for discrete action spaces.

Double-DQN-style target computation, hard target-network updates, and a
linear epsilon-greedy schedule. Requires a discrete environment action
space; the discrete action index is decoded by the environment.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_client.agents.replay_buffer import ReplayBuffer
from sim_client.agents.ppo_baseline import layer_init


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.ReLU(),
            layer_init(nn.Linear(128, 128)), nn.ReLU(),
            layer_init(nn.Linear(128, num_actions), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNRunner:
    """
    Off-policy DQN training loop. Discrete action spaces ONLY — the runner
    validates num_actions >= 2 at construction; environment compatibility
    is enforced earlier by the trainer capability check.
    """

    def __init__(
        self,
        env,
        obs_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        warmup_steps: int = 1_000,
        batch_size: int = 128,
        train_freq: int = 4,
        target_update_interval: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        update_interval: int = 1_000,
        seed: int = 42,
        device: str = "cpu",
        on_update=None,
        on_step=None,
        resume_checkpoint: Optional[str] = None,
    ):
        if num_actions < 2:
            raise ValueError(
                f"DQN requires a discrete action space with >= 2 actions "
                f"(got {num_actions})")
        from sim_experiment.vec_env import is_vec_env
        self._venv = env if is_vec_env(env) else None
        if self._venv is not None:
            self.envs = None
            self.env = None
            self.num_envs = env.num_envs
        else:
            self.envs = list(env) if isinstance(env, (list, tuple)) else [env]
            self.env = self.envs[0]
            self.num_envs = len(self.envs)
        self.obs_dim = obs_dim
        self.num_actions = int(num_actions)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = int(warmup_steps)
        self.train_freq = max(1, int(train_freq))
        self.target_update_interval = max(1, int(target_update_interval))
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = max(1, int(eps_decay_steps))
        self.update_interval = max(1, int(update_interval))
        self.seed = seed
        self.device = torch.device(device)
        self.on_update = on_update
        self.on_step = on_step
        self.global_step_offset = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._explore_rng = np.random.default_rng(seed + 7)

        self.q = QNetwork(obs_dim, num_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, num_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, 1,
                                   discrete_actions=True, seed=seed)

        self._envs_dirty = True
        self._env_epoch = 0
        self._reset_counts = [1] * self.num_envs
        self._ep_seeds: List[Optional[int]] = [None] * self.num_envs

        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        self.metrics: Dict[str, Any] = {
            "total_timesteps": 0, "episodes_completed": 0,
            "episode_returns": [], "episode_lengths": [],
            "mean_lateral_errors": [], "mean_speeds": [], "completion_rates": [],
            "collision_count": 0, "off_road_count": 0,
            "td_losses": [], "epsilons": [],
            "sps": 0.0, "wall_clock_time": 0.0,
        }

    # ------------------------------------------------------------- env mgmt

    def set_envs(self, envs) -> None:
        from sim_experiment.vec_env import is_vec_env
        if is_vec_env(envs):
            if self._venv is None:
                raise ValueError("cannot swap a VectorEnv into a list-mode runner")
            if envs.num_envs != self.num_envs:
                raise ValueError(
                    f"set_envs requires exactly {self.num_envs} envs, got {envs.num_envs}")
            self._venv = envs
        else:
            if self._venv is not None:
                raise ValueError("cannot swap an env list into a vec-mode runner")
            new_envs = list(envs) if isinstance(envs, (list, tuple)) else [envs]
            if len(new_envs) != self.num_envs:
                raise ValueError(
                    f"set_envs requires exactly {self.num_envs} envs, got {len(new_envs)}")
            self.envs = new_envs
            self.env = new_envs[0]
        self._env_epoch += 1
        self._reset_counts = [1] * self.num_envs
        self._envs_dirty = True

    def _env_reset(self, env_idx: int):
        seed = self._next_reset_seed(env_idx)
        self._ep_seeds[env_idx] = seed
        if self._venv is not None:
            return self._venv.reset_at(env_idx, seed=seed)
        return self.envs[env_idx].reset(seed=seed)[0]

    def _next_reset_seed(self, env_idx: int) -> int:
        seed = (self.seed + self._env_epoch * 1_000_003
                + env_idx * 79_919 + self._reset_counts[env_idx])
        self._reset_counts[env_idx] += 1
        return int(seed)

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _vec(self, o: Any) -> np.ndarray:
        if isinstance(o, dict):
            o = o.get("vector", np.zeros(self.obs_dim, dtype=np.float32))
        return np.asarray(o, dtype=np.float32)

    # ------------------------------------------------------------- training

    def _consume_step(self, e, action_idx, next_obs, reward, terminated,
                      truncated, info, S):
        """Shared post-step bookkeeping for list and vector rollout modes."""
        done = terminated or truncated
        self.buffer.add(S["obs"][e], [action_idx], reward,
                        self._vec(next_obs), bool(terminated))
        S["obs"][e] = self._vec(next_obs)
        S["gstep"] += 1

        if self.on_step is not None:
            self.on_step({
                "env_idx": e, "episode_seed": self._ep_seeds[e],
                "obs": np.asarray(S["obs"][e]).tolist(),
                "action": int(action_idx),
                "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "info": info,
            })

        S["ep_return"][e] += float(reward)
        S["ep_len"][e] += 1
        S["ep_lat"][e].append(abs(float(info.get("lateral_offset", 0.0))))
        S["ep_spd"][e].append(float(info.get("speed", 0.0)))

        if done:
            ep_stats = {
                "env_idx": e,
                "return": float(S["ep_return"][e]),
                "length": int(info.get("step", S["ep_len"][e])),
                "termination_reason": str(info.get("termination_reason", "")),
                "checkpoints_passed": int(info.get("checkpoints_passed", 0)),
                "mean_lateral_error": float(np.mean(S["ep_lat"][e])) if S["ep_lat"][e] else 0.0,
                "mean_speed": float(np.mean(S["ep_spd"][e])) if S["ep_spd"][e] else 0.0,
                "collided": bool(info.get("is_colliding", False)),
                "off_road": not bool(info.get("is_on_road", True)),
                "completion": float(info.get("checkpoints_passed", 0)),
            }
            S["new_episodes"].append(ep_stats)
            self.metrics["episodes_completed"] += 1
            self.metrics["episode_returns"].append(ep_stats["return"])
            self.metrics["episode_lengths"].append(ep_stats["length"])
            self.metrics["mean_lateral_errors"].append(ep_stats["mean_lateral_error"])
            self.metrics["mean_speeds"].append(ep_stats["mean_speed"])
            self.metrics["completion_rates"].append(ep_stats["completion"])
            if ep_stats["collided"]:
                self.metrics["collision_count"] += 1
            if ep_stats["off_road"]:
                self.metrics["off_road_count"] += 1
            S["ep_return"][e] = 0.0
            S["ep_len"][e] = 0
            S["ep_lat"][e] = []
            S["ep_spd"][e] = []
            S["obs"][e] = self._vec(self._env_reset(e))

        # Gradient updates
        g = S["gstep"]
        if (g >= self.warmup_steps
                and g % self.train_freq == 0
                and self.buffer.size >= self.batch_size):
            S["last_td_loss"] = self._update()
            if g % self.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())

    def train(self, total_timesteps: int = 50_000) -> Dict[str, Any]:
        start_time = time.perf_counter()
        num_envs = self.num_envs

        obs = [None] * num_envs
        ep_return = [0.0] * num_envs
        ep_len = [0] * num_envs
        ep_lat: List[List[float]] = [[] for _ in range(num_envs)]
        ep_spd: List[List[float]] = [[] for _ in range(num_envs)]
        new_episodes: List[Dict[str, Any]] = []

        global_step = self.global_step_offset
        last_td_loss = 0.0

        while global_step - self.global_step_offset < total_timesteps:
            if self._envs_dirty:
                self._envs_dirty = False
                if self._venv is not None:
                    seeds = [self._next_reset_seed(e) for e in range(num_envs)]
                    self._ep_seeds = list(seeds)
                    for e, o in enumerate(self._venv.reset_all(seeds)):
                        obs[e] = self._vec(o)
                else:
                    for e, env in enumerate(self.envs):
                        s_e = self._next_reset_seed(e)
                        self._ep_seeds[e] = s_e
                        o, _ = env.reset(seed=s_e)
                        obs[e] = self._vec(o)
                ep_return = [0.0] * num_envs
                ep_len = [0] * num_envs
                ep_lat = [[] for _ in range(num_envs)]
                ep_spd = [[] for _ in range(num_envs)]

            S = {
                "obs": obs, "gstep": global_step,
                "ep_return": ep_return, "ep_len": ep_len,
                "ep_lat": ep_lat, "ep_spd": ep_spd,
                "new_episodes": new_episodes, "last_td_loss": last_td_loss,
            }

            if self._venv is not None:
                # Batched: all envs step together (process-parallel backends
                # execute the steps concurrently). Explore-RNG consumption
                # order stays env-major, identical to the sequential path.
                action_idxs = []
                for e in range(num_envs):
                    eps = self.epsilon(global_step + e)
                    if self._explore_rng.random() < eps or global_step + e < self.warmup_steps:
                        action_idx = int(self._explore_rng.integers(0, self.num_actions))
                    else:
                        with torch.no_grad():
                            q_vals = self.q(torch.tensor(
                                obs[e], dtype=torch.float32, device=self.device).unsqueeze(0))
                        action_idx = int(q_vals.argmax(dim=1).item())
                    action_idxs.append(action_idx)
                results = self._venv.step_all(action_idxs)
                for e, (n_obs, rew, term, trunc, info) in enumerate(results):
                    self._consume_step(e, action_idxs[e], n_obs, rew, term,
                                       trunc, info, S)
                global_step = S["gstep"]
                last_td_loss = S["last_td_loss"]
            else:
                for e, env in enumerate(self.envs):
                    eps = self.epsilon(global_step)
                    if self._explore_rng.random() < eps or global_step < self.warmup_steps:
                        action_idx = int(self._explore_rng.integers(0, self.num_actions))
                    else:
                        with torch.no_grad():
                            q_vals = self.q(torch.tensor(
                                obs[e], dtype=torch.float32, device=self.device).unsqueeze(0))
                        action_idx = int(q_vals.argmax(dim=1).item())

                    next_obs, reward, terminated, truncated, info = env.step(action_idx)
                    self._consume_step(e, action_idx, next_obs, reward,
                                       terminated, truncated, info, S)
                    global_step = S["gstep"]
                    last_td_loss = S["last_td_loss"]

            if (global_step - self.global_step_offset) % self.update_interval == 0 or \
               global_step - self.global_step_offset >= total_timesteps:
                elapsed = time.perf_counter() - start_time
                local = global_step - self.global_step_offset
                if self.on_update is not None:
                    self.on_update({
                        "global_step": int(global_step),
                        "sps": local / max(1e-4, elapsed),
                        "td_loss": last_td_loss,
                        "epsilon": self.epsilon(global_step),
                        "buffer_size": self.buffer.size,
                        "episodes": new_episodes,
                        "episodes_completed": self.metrics["episodes_completed"],
                    })
                    new_episodes = []

        total_wall = time.perf_counter() - start_time
        self.metrics["total_timesteps"] = global_step
        self.metrics["wall_clock_time"] = round(total_wall, 2)
        self.metrics["sps"] = round(global_step / max(1e-4, total_wall), 1)
        total_eps = max(1, self.metrics["episodes_completed"])
        self.metrics["collision_rate"] = round(self.metrics["collision_count"] / total_eps, 3)
        self.metrics["off_road_rate"] = round(self.metrics["off_road_count"] / total_eps, 3)
        self.metrics["overall_mean_return"] = (
            round(float(np.mean(self.metrics["episode_returns"])), 2)
            if self.metrics["episode_returns"] else 0.0)
        self.metrics["overall_mean_length"] = (
            round(float(np.mean(self.metrics["episode_lengths"])), 1)
            if self.metrics["episode_lengths"] else 0.0)
        return self.metrics

    def _update(self) -> float:
        batch = self.buffer.sample(self.batch_size)
        obs = torch.tensor(batch["obs"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device).squeeze(-1)
        rewards = torch.tensor(batch["rewards"], device=self.device)
        next_obs = torch.tensor(batch["next_obs"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device)

        q_vals = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(dim=1).values
            td_target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.mse_loss(q_vals, td_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str, step: Optional[int] = None,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "optimizer": self.opt.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "seed": self.seed,
            "metrics": self.metrics,
            "timestep": int(step if step is not None else self.metrics.get("total_timesteps", 0)),
        }
        payload.update(dict(extra or {}))
        torch.save(payload, path)
        print(f"[DQN Runner] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q_state_dict"])
        if "q_target_state_dict" in ckpt:
            self.q_target.load_state_dict(ckpt["q_target_state_dict"])
        else:
            self.q_target.load_state_dict(self.q.state_dict())
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        self.global_step_offset = int(
            ckpt.get("timestep") or ckpt.get("metrics", {}).get("total_timesteps", 0))
        print(f"[DQN Runner] Resumed from checkpoint {path} "
              f"(timestep offset {self.global_step_offset})")
        return ckpt
