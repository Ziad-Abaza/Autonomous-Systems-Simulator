"""SACAgent — soft actor-critic, the primary agentRL algorithm.

Off-policy + maximum-entropy: sample-efficient on CPU, robust to the
mixed-distribution buffers continual learning creates (rehearsal across
tracks). Twin critics, tanh-squashed Gaussian actor, auto temperature.

done = terminated AND NOT truncated — time-limit truncations bootstrap
through the target critic (governing contract per audit).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentRL.core.agent import BaseRLAgent
from agentRL.core.config import AgentConfig
from agentRL.core.versioning import (
    AGENT_VERSION, CKPT_SCHEMA_V, VersionError, check_schema)
from agentRL.memory.replay import TrackRehearsalBuffer
from agentRL.obs.spec import ObservationSpec

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
EPS = 1e-6


def _mlp(sizes: list[int], act: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1]) if len(sizes) > 1 else nn.Identity()


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.trunk = _mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.mu(h), torch.clamp(self.log_std(h),
                                       LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        a = torch.tanh(x)
        logp = (normal.log_prob(x)
                - torch.log(1.0 - a.pow(2) + EPS)).sum(-1)
        return a, logp, torch.tanh(mu)


class _Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.q = _mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], -1)).squeeze(-1)


class SACAgent(BaseRLAgent):
    algo_id = "sac"
    is_on_policy = False

    def __init__(self, obs_spec: ObservationSpec,
                 act_space: dict[str, np.ndarray], cfg: AgentConfig):
        super().__init__(obs_spec, act_space, cfg)
        e = cfg.extra
        seed = int(e.get("seed", 0))
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        obs_dim = obs_spec.input_dim
        act_dim = int(self.act_space["low"].size)
        hidden = tuple(cfg.hidden_sizes)
        self.actor = _Actor(obs_dim, act_dim, hidden)
        self.c1 = _Critic(obs_dim, act_dim, hidden)
        self.c2 = _Critic(obs_dim, act_dim, hidden)
        self.c1_t = _Critic(obs_dim, act_dim, hidden)
        self.c2_t = _Critic(obs_dim, act_dim, hidden)
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())
        for p in list(self.c1_t.parameters()) + list(self.c2_t.parameters()):
            p.requires_grad_(False)

        lr = cfg.lr
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c1 = torch.optim.Adam(self.c1.parameters(), lr=lr)
        self.opt_c2 = torch.optim.Adam(self.c2.parameters(), lr=lr)

        init_alpha = float(e.get("init_alpha", 0.2))
        self.log_alpha = torch.tensor(math.log(init_alpha),
                                      requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = float(e.get("target_entropy", -act_dim))

        self.tau = float(e.get("tau", 0.005))
        self.warmup = int(e.get("warmup", 2000))
        self.batch = int(e.get("batch", 256))
        self.updates_per_step = int(e.get("updates_per_step", 1))
        self.grad_clip = float(e.get("grad_clip", 10.0))
        self.random_steps = int(e.get("random_steps", self.warmup))

        self.memory = TrackRehearsalBuffer(
            per_track_capacity=int(e.get("buffer", 200_000)),
            rehearsal_fraction=float(e.get("rehearsal_fraction", 0.25)),
            seed=seed, obs_dim=obs_dim, act_dim=act_dim)
        self.gamma = cfg.gamma
        self._device = torch.device("cpu")

    # ---------------------------------------------------------------- acting

    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> tuple[np.ndarray, dict[str, Any]]:
        if not deterministic and \
                self.train_state["steps"] < self.random_steps:
            raw = self._rng.uniform(-1.0, 1.0,
                                    size=self.act_space["low"].size)
            return self.act_space_to_env(raw.astype(np.float32)), \
                {"phase": "warmup_random"}
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a, logp, mu_a = self.actor.sample(o)
            raw = (mu_a if deterministic else a).squeeze(0).numpy()
        return self.act_space_to_env(raw), {"logp": float(logp.item())}

    # ------------------------------------------------------------- observing

    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float,
                next_obs: np.ndarray, terminated: bool, truncated: bool,
                info: dict[str, Any]) -> None:
        if self.memory.current is None:
            self.memory.begin_track("default")
        self.memory.add(np.asarray(obs, np.float32),
                        self.env_to_act_space(action),
                        float(reward),
                        np.asarray(next_obs, np.float32),
                        terminated, truncated)
        self.train_state["steps"] += 1

    # --------------------------------------------------------------- update

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _finite_batch(self, batch: dict[str, torch.Tensor]) -> bool:
        return all(torch.isfinite(v).all().item() for v in batch.values())

    def update(self, step: int) -> dict[str, float]:
        buf_size = sum(len(b) for b in self.memory.buffers.values())
        if buf_size < self.batch:
            return {}
        metrics: dict[str, float] = {}
        for _ in range(self.updates_per_step):
            batch_np = self.memory.sample(self.batch)
            batch = {k: torch.as_tensor(v) for k, v in batch_np.items()}
            if not self._finite_batch(batch):
                self.train_state["nan_guards"] += 1
                metrics["nan_guard"] = float(self.train_state["nan_guards"])
                continue

            # --- critics ---
            with torch.no_grad():
                na, nlogp, _ = self.actor.sample(batch["next_obs"])
                q_next = torch.min(self.c1_t(batch["next_obs"], na),
                                   self.c2_t(batch["next_obs"], na)) \
                    - self.alpha.detach() * nlogp
                y = batch["reward"] + self.gamma * (1.0 - batch["done"]) * q_next
            l1 = F.mse_loss(self.c1(batch["obs"], batch["action"]), y)
            l2 = F.mse_loss(self.c2(batch["obs"], batch["action"]), y)
            critic_loss = l1 + l2
            if not torch.isfinite(critic_loss):
                self.train_state["nan_guards"] += 1
                metrics["nan_guard"] = float(self.train_state["nan_guards"])
                continue
            self.opt_c1.zero_grad()
            l1.backward()
            n1 = torch.nn.utils.clip_grad_norm_(self.c1.parameters(),
                                                self.grad_clip)
            if torch.isfinite(n1):
                self.opt_c1.step()
            self.opt_c2.zero_grad()
            l2.backward()
            n2 = torch.nn.utils.clip_grad_norm_(self.c2.parameters(),
                                                self.grad_clip)
            if torch.isfinite(n2):
                self.opt_c2.step()

            # --- actor + temperature ---
            a, logp, _ = self.actor.sample(batch["obs"])
            q_pi = torch.min(self.c1(batch["obs"], a),
                             self.c2(batch["obs"], a))
            actor_loss = (self.alpha.detach() * logp - q_pi).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            na_ = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                 self.grad_clip)
            if torch.isfinite(na_):
                self.opt_actor.step()

            alpha_loss = -(self.log_alpha
                           * (logp + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

            with torch.no_grad():
                for tp, p in zip(self.c1_t.parameters(), self.c1.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
                for tp, p in zip(self.c2_t.parameters(), self.c2.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)

            self.train_state["updates"] += 1
            metrics = {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()),
                "q_mean": float(q_pi.mean().item()),
                "buf": float(buf_size),
            }
        return metrics

    # ----------------------------------------------------------- checkpoint

    def _torch_state(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "c1": self.c1.state_dict(), "c2": self.c2.state_dict(),
            "c1_t": self.c1_t.state_dict(), "c2_t": self.c2_t.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_c1": self.opt_c1.state_dict(),
            "opt_c2": self.opt_c2.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
        }

    def _load_torch_state(self, st: dict[str, Any]) -> None:
        self.actor.load_state_dict(st["actor"])
        self.c1.load_state_dict(st["c1"])
        self.c2.load_state_dict(st["c2"])
        self.c1_t.load_state_dict(st["c1_t"])
        self.c2_t.load_state_dict(st["c2_t"])
        self.opt_actor.load_state_dict(st["opt_actor"])
        self.opt_c1.load_state_dict(st["opt_c1"])
        self.opt_c2.load_state_dict(st["opt_c2"])
        self.opt_alpha.load_state_dict(st["opt_alpha"])
        self.log_alpha.data.copy_(st["log_alpha"])

    def save(self, path: str) -> None:
        torch.save({
            "format": "agentRL.ckpt",
            "schema_version": CKPT_SCHEMA_V,
            "agent_version": AGENT_VERSION,
            "algo_id": self.algo_id,
            "agent_config": self.cfg.to_dict(),
            "obs_spec": self.obs_spec.to_dict(),
            "act_space": {"low": self.act_space["low"].tolist(),
                          "high": self.act_space["high"].tolist()},
            "torch": self._torch_state(),
            "train_state": dict(self.train_state),
            "memory_meta": self.memory.state_dict() if self.memory else {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "SACAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "agentRL.ckpt":
            raise VersionError(f"not an agentRL checkpoint: {path}")
        check_schema(int(payload.get("schema_version", -1)),
                     CKPT_SCHEMA_V, "checkpoint")
        if payload.get("algo_id") != cls.algo_id:
            raise VersionError(
                f"checkpoint algo {payload.get('algo_id')!r} != {cls.algo_id!r}")
        spec = ObservationSpec.from_dict(payload["obs_spec"])
        cfg = AgentConfig.from_dict(payload["agent_config"])
        act_space = {"low": np.asarray(payload["act_space"]["low"], np.float32),
                     "high": np.asarray(payload["act_space"]["high"], np.float32)}
        agent = cls(spec, act_space, cfg)
        agent._load_torch_state(payload["torch"])
        agent.train_state.update(payload.get("train_state", {}))
        if agent.memory and payload.get("memory_meta"):
            agent.memory.load_state_dict(payload["memory_meta"])
        return agent


class _EmptyBuf:
    def __len__(self) -> int:
        return 0
