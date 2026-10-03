"""
Soft Actor-Critic (SAC) implementation for continuous action spaces.

Gaussian policy with tanh squashing, twin Q networks, soft target updates,
and automatic entropy temperature tuning. Actions are sampled in normalized
[-1,1] space and mapped to the environment's declared continuous bounds.

Multi-environment: envs step synchronously each iteration; gradient updates
run once per environment step (updates_per_step x num_envs).
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from sim_client.agents.replay_buffer import ReplayBuffer
from sim_client.agents.ppo_baseline import layer_init


class SACActor(nn.Module):
    """Gaussian policy -> tanh-squashed action in [-1,1]^act_dim."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 256)), nn.ReLU(),
        )
        self.mean = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(256, act_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        log_std = torch.clamp(self.log_std(feat), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return self.mean(feat), log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        a = torch.tanh(x)
        # tanh correction: log|det(1 - tanh^2(x))|
        logp = (dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class SACQNet(nn.Module):
    """Q(obs, action) network."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SACRunner:
    """
    Off-policy SAC training loop against the hardened environment.
    Same hook contract as PPORunner: on_step(rec) per env step,
    on_update(stats) per logging interval.
    """

    def __init__(
        self,
        env,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        warmup_steps: int = 1_000,
        batch_size: int = 256,
        alpha_init: float = 0.2,
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        updates_per_step: int = 1,
        target_update_interval: int = 1,
        update_interval: int = 1_000,
        seed: int = 42,
        device: str = "cpu",
        on_update=None,
        on_step=None,
        resume_checkpoint: Optional[str] = None,
    ):
        self.envs = list(env) if isinstance(env, (list, tuple)) else [env]
        self.env = self.envs[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = int(warmup_steps)
        self.updates_per_step = int(updates_per_step)
        self.target_update_interval = int(target_update_interval)
        self.update_interval = max(1, int(update_interval))
        self.seed = seed
        self.device = torch.device(device)
        self.on_update = on_update
        self.on_step = on_step
        self.global_step_offset = 0

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = SACActor(obs_dim, act_dim).to(self.device)
        self.q1 = SACQNet(obs_dim, act_dim).to(self.device)
        self.q2 = SACQNet(obs_dim, act_dim).to(self.device)
        self.q1_target = SACQNet(obs_dim, act_dim).to(self.device)
        self.q2_target = SACQNet(obs_dim, act_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.auto_entropy = bool(auto_entropy)
        self.target_entropy = (float(target_entropy) if target_entropy is not None
                               else -float(act_dim))
        self.log_alpha = torch.tensor(
            float(np.log(alpha_init)), device=self.device, requires_grad=self.auto_entropy)
        self.alpha_opt = (torch.optim.Adam([self.log_alpha], lr=lr)
                          if self.auto_entropy else None)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim, seed=seed)

        self._envs_dirty = True
        self._env_epoch = 0
        self._reset_counts = [1] * len(self.envs)

        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        self.metrics: Dict[str, Any] = {
            "total_timesteps": 0, "episodes_completed": 0,
            "episode_returns": [], "episode_lengths": [],
            "mean_lateral_errors": [], "mean_speeds": [], "completion_rates": [],
            "collision_count": 0, "off_road_count": 0,
            "actor_losses": [], "critic_losses": [], "alphas": [],
            "sps": 0.0, "wall_clock_time": 0.0,
        }

    # ------------------------------------------------------------- env mgmt

    def set_envs(self, envs) -> None:
        new_envs = list(envs) if isinstance(envs, (list, tuple)) else [envs]
        if len(new_envs) != len(self.envs):
            raise ValueError(
                f"set_envs requires exactly {len(self.envs)} envs, got {len(new_envs)}")
        self.envs = new_envs
        self.env = new_envs[0]
        self._env_epoch += 1
        self._reset_counts = [1] * len(new_envs)
        self._envs_dirty = True

    def _next_reset_seed(self, env_idx: int) -> int:
        seed = (self.seed + self._env_epoch * 1_000_003
                + env_idx * 79_919 + self._reset_counts[env_idx])
        self._reset_counts[env_idx] += 1
        return int(seed)

    def _to_env_action(self, a_norm: np.ndarray) -> np.ndarray:
        """[-1,1]^d action -> environment [low,high] bounds."""
        return self.action_low + (np.asarray(a_norm, dtype=np.float32) + 1.0) * 0.5 * (
            self.action_high - self.action_low)

    # ------------------------------------------------------------- training

    def train(self, total_timesteps: int = 50_000) -> Dict[str, Any]:
        start_time = time.perf_counter()
        num_envs = len(self.envs)

        obs = [None] * num_envs
        ep_return = [0.0] * num_envs
        ep_len = [0] * num_envs
        ep_lat: List[List[float]] = [[] for _ in range(num_envs)]
        ep_spd: List[List[float]] = [[] for _ in range(num_envs)]
        new_episodes: List[Dict[str, Any]] = []

        global_step = self.global_step_offset
        last_losses = {"actor": 0.0, "critic": 0.0, "alpha": float(self.log_alpha.exp().item())}

        while global_step - self.global_step_offset < total_timesteps:
            if self._envs_dirty:
                self._envs_dirty = False
                for e, env in enumerate(self.envs):
                    o, _ = env.reset(seed=self._next_reset_seed(e))
                    obs[e] = self._vec(o)
                ep_return = [0.0] * num_envs
                ep_len = [0] * num_envs
                ep_lat = [[] for _ in range(num_envs)]
                ep_spd = [[] for _ in range(num_envs)]

            for e, env in enumerate(self.envs):
                obs_t = torch.tensor(obs[e], dtype=torch.float32, device=self.device).unsqueeze(0)
                if global_step < self.warmup_steps:
                    a_norm = np.random.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
                else:
                    with torch.no_grad():
                        a_t, _ = self.actor.sample(obs_t)
                    a_norm = a_t.squeeze(0).cpu().numpy()

                env_action = self._to_env_action(a_norm)
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                # Buffer stores terminated-only as done (truncation bootstraps).
                self.buffer.add(obs[e], a_norm, reward, self._vec(next_obs), bool(terminated))
                obs[e] = self._vec(next_obs)
                global_step += 1

                if self.on_step is not None:
                    self.on_step({
                        "env_idx": e, "obs": np.asarray(obs[e]).tolist(),
                        "action": np.asarray(env_action).tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "info": info,
                    })

                ep_return[e] += float(reward)
                ep_len[e] += 1
                ep_lat[e].append(abs(float(info.get("lateral_offset", 0.0))))
                ep_spd[e].append(float(info.get("speed", 0.0)))

                if done:
                    ep_stats = {
                        "env_idx": e,
                        "return": float(ep_return[e]),
                        "length": int(info.get("step", ep_len[e])),
                        "termination_reason": str(info.get("termination_reason", "")),
                        "checkpoints_passed": int(info.get("checkpoints_passed", 0)),
                        "mean_lateral_error": float(np.mean(ep_lat[e])) if ep_lat[e] else 0.0,
                        "mean_speed": float(np.mean(ep_spd[e])) if ep_spd[e] else 0.0,
                        "collided": bool(info.get("is_colliding", False)),
                        "off_road": not bool(info.get("is_on_road", True)),
                        "completion": float(info.get("checkpoints_passed", 0)),
                    }
                    new_episodes.append(ep_stats)
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
                    ep_return[e] = 0.0
                    ep_len[e] = 0
                    ep_lat[e] = []
                    ep_spd[e] = []
                    o, _ = env.reset(seed=self._next_reset_seed(e))
                    obs[e] = self._vec(o)

                # Gradient updates after warmup
                if global_step >= self.warmup_steps and self.buffer.size >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        last_losses = self._update()
                    if global_step % self.target_update_interval == 0:
                        self._soft_update()

            if (global_step - self.global_step_offset) % self.update_interval == 0 or \
               global_step - self.global_step_offset >= total_timesteps:
                elapsed = time.perf_counter() - start_time
                local = global_step - self.global_step_offset
                if self.on_update is not None:
                    self.on_update({
                        "global_step": int(global_step),
                        "sps": local / max(1e-4, elapsed),
                        "actor_loss": last_losses["actor"],
                        "critic_loss": last_losses["critic"],
                        "alpha": last_losses["alpha"],
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

    def _vec(self, o: Any) -> np.ndarray:
        if isinstance(o, dict):
            o = o.get("vector", np.zeros(self.obs_dim, dtype=np.float32))
        return np.asarray(o, dtype=np.float32)

    def _update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size)
        obs = torch.tensor(batch["obs"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device)
        rewards = torch.tensor(batch["rewards"], device=self.device)
        next_obs = torch.tensor(batch["next_obs"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device)
        alpha = self.log_alpha.exp().detach()

        # --- critic update ---
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            q_next = torch.min(
                self.q1_target(next_obs, next_a),
                self.q2_target(next_obs, next_a),
            ).squeeze(-1) - alpha * next_logp
            q_target = rewards + self.gamma * (1.0 - dones) * q_next
        q1_pred = self.q1(obs, actions).squeeze(-1)
        q2_pred = self.q2(obs, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.q_opt.zero_grad()
        critic_loss.backward()
        self.q_opt.step()

        # --- actor update ---
        a_new, logp = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, a_new), self.q2(obs, a_new)).squeeze(-1)
        actor_loss = (alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- temperature update ---
        if self.auto_entropy and self.alpha_opt is not None:
            alpha_loss = (-self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        return {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str, step: Optional[int] = None,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "q_optimizer": self.q_opt.state_dict(),
            "log_alpha": float(self.log_alpha.item()),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "seed": self.seed,
            "metrics": self.metrics,
            "timestep": int(step if step is not None else self.metrics.get("total_timesteps", 0)),
        }
        payload.update(dict(extra or {}))
        torch.save(payload, path)
        print(f"[SAC Runner] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])
        if "q1_target_state_dict" in ckpt:
            self.q1_target.load_state_dict(ckpt["q1_target_state_dict"])
            self.q2_target.load_state_dict(ckpt["q2_target_state_dict"])
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
        if "actor_optimizer" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_optimizer"])
            self.q_opt.load_state_dict(ckpt["q_optimizer"])
        self.log_alpha.data = torch.tensor(
            float(ckpt.get("log_alpha", np.log(0.2))),
            device=self.device, requires_grad=self.auto_entropy)
        self.global_step_offset = int(
            ckpt.get("timestep") or ckpt.get("metrics", {}).get("total_timesteps", 0))
        print(f"[SAC Runner] Resumed from checkpoint {path} "
              f"(timestep offset {self.global_step_offset})")
        return ckpt
