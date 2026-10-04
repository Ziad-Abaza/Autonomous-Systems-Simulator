"""PPOAgent — clipped-objective on-policy agent (secondary algorithm).

Rollout design: observe() stores flat transitions tagged by env_id.
GAE is computed per env segment with the governing done semantics:
- delta_t = r + gamma*(1-absorbing)*V(next_obs_t) - V(obs_t); absorbing =
  terminated AND NOT truncated (truncation bootstraps through V(next_obs)).
- cont_t = 0 at any episode end (terminated OR truncated) so advantages
  never propagate across episode boundaries or horizon cuts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentRL.core.agent import BaseRLAgent
from agentRL.core.config import AgentConfig
from agentRL.core.versioning import (
    AGENT_VERSION, CKPT_SCHEMA_V, VersionError, check_schema)
from agentRL.obs.spec import ObservationSpec


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    return nn.Sequential(*layers[:-1])


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.mu = _mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(
            self.mu(obs), self.log_std.exp().expand_as(self.mu(obs)))


class _Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.v = _mlp([obs_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class PPOAgent(BaseRLAgent):
    algo_id = "ppo"
    is_on_policy = True

    def __init__(self, obs_spec: ObservationSpec,
                 act_space: dict[str, np.ndarray], cfg: AgentConfig):
        super().__init__(obs_spec, act_space, cfg)
        e = cfg.extra
        seed = int(e.get("seed", 0))
        torch.manual_seed(seed)

        obs_dim = obs_spec.input_dim
        act_dim = int(self.act_space["low"].size)
        hidden = tuple(cfg.hidden_sizes)
        self.actor = _Actor(obs_dim, act_dim, hidden)
        self.critic = _Critic(obs_dim, hidden)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr)

        self.gamma = cfg.gamma
        self.lam = float(e.get("gae_lambda", 0.95))
        self.clip = float(e.get("clip", 0.2))
        self.ent_coef = float(e.get("ent_coef", 0.01))
        self.vf_coef = float(e.get("vf_coef", 0.5))
        self.epochs = int(e.get("epochs", 4))
        self.minibatch = int(e.get("minibatch", 256))
        self.grad_clip = float(e.get("grad_clip", 0.5))
        self.steps_per_rollout = int(e.get("steps_per_rollout", 2048))

        self.rollout: dict[str, list] = {
            "obs": [], "act": [], "logp": [], "value": [], "reward": [],
            "term": [], "trunc": [], "env_id": [], "next_obs": [],
        }
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- acting

    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> tuple[np.ndarray, dict[str, Any]]:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = self.actor.dist(o)
            raw = dist.mean if deterministic else dist.sample()
            # logp evaluated at the SAME action the env sees (clipped);
            # otherwise the stored act != the logp_old and the PPO ratio
            # is biased whenever |raw| > 1.
            logp = dist.log_prob(torch.clamp(raw, -1.0, 1.0)).sum(-1)
            value = self.critic(o)
        raw_np = np.clip(raw.squeeze(0).numpy(), -1.0, 1.0).astype(np.float32)
        return self.act_space_to_env(raw_np), {
            "logp": float(logp.item()), "value": float(value.item()),
        }

    # ------------------------------------------------------------- observing

    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float,
                next_obs: np.ndarray, terminated: bool, truncated: bool,
                info: dict[str, Any]) -> None:
        r = self.rollout
        r["obs"].append(np.asarray(obs, np.float32))
        r["act"].append(self.env_to_act_space(action))
        r["logp"].append(float(info.get("logp", 0.0)))
        r["value"].append(float(info.get("value", 0.0)))
        r["reward"].append(float(reward))
        r["term"].append(bool(terminated))
        r["trunc"].append(bool(truncated))
        r["env_id"].append(int(info.get("env_id", 0)))
        r["next_obs"].append(np.asarray(next_obs, np.float32))
        self.train_state["steps"] += 1

    # --------------------------------------------------------------- update

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        """GAE per env segment. Returns (advantages, returns)."""
        r = self.rollout
        obs = torch.as_tensor(np.asarray(r["obs"]))
        nobs = torch.as_tensor(np.asarray(r["next_obs"]))
        with torch.no_grad():
            v = self.critic(obs).numpy()
            v_next = self.critic(nobs).numpy()
        rew = np.asarray(r["reward"], np.float32)
        term = np.asarray(r["term"], bool)
        trunc = np.asarray(r["trunc"], bool)
        env_id = np.asarray(r["env_id"])
        absorbing = term & ~trunc          # real terminal -> no bootstrap
        ended = term | trunc               # any episode end -> stop GAE

        T = len(rew)
        adv = np.zeros(T, np.float32)
        lastgae = np.zeros(int(env_id.max()) + 1 if T else 1, np.float32)
        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * (1 - absorbing[t]) * v_next[t] - v[t]
            cont = 0.0 if ended[t] else 1.0
            lastgae[env_id[t]] = delta + self.gamma * self.lam * cont \
                * lastgae[env_id[t]]
            adv[t] = lastgae[env_id[t]]
        ret = adv + v
        return adv, ret

    def update(self, step: int) -> dict[str, float]:
        if len(self.rollout["obs"]) < self.steps_per_rollout:
            return {}
        adv, ret = self._compute_gae()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(np.asarray(self.rollout["obs"]))
        act = torch.as_tensor(np.asarray(self.rollout["act"]))
        logp_old = torch.as_tensor(np.asarray(self.rollout["logp"]))
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        N = obs.size(0)

        metrics: dict[str, float] = {}
        for _ in range(self.epochs):
            for mb in self._minibatches(N):
                dist = self.actor.dist(obs[mb])
                logp = dist.log_prob(act[mb]).sum(-1)
                ratio = (logp - logp_old[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip,
                                    1 + self.clip) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(self.critic(obs[mb]), ret_t[mb])
                entropy = dist.entropy().sum(-1).mean()
                loss = actor_loss + self.vf_coef * v_loss \
                    - self.ent_coef * entropy
                if not torch.isfinite(loss):
                    self.train_state["nan_guards"] += 1
                    continue
                self.opt.zero_grad()
                loss.backward()
                n = torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()), self.grad_clip)
                if torch.isfinite(n):
                    self.opt.step()
                metrics = {
                    "actor_loss": float(actor_loss.item()),
                    "value_loss": float(v_loss.item()),
                    "entropy": float(entropy.item()),
                    "rollout": float(N),
                }
        for k in self.rollout:
            self.rollout[k].clear()
        self.train_state["updates"] += 1
        return metrics

    def _minibatches(self, n: int):
        perm = self._rng.permutation(n)
        for i in range(0, n, self.minibatch):
            yield perm[i:i + self.minibatch]

    # ----------------------------------------------------------- checkpoint

    def save(self, path: str) -> None:
        torch.save({
            "format": "agentRL.ckpt",
            "schema_version": CKPT_SCHEMA_V,
            "agent_version": AGENT_VERSION,
            "algo_id": self.algo_id,
            "agent_config": self.cfg.to_dict(),
            "obs_spec": self.obs_spec.to_dict(),
            "act_space": {"low": self.act_space["low"].tolist(),
                          "high": self.act_space["high"].tolist(),
                          "pos_only": list(self.act_space.get("pos_only", ()))},
            "torch": {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opt": self.opt.state_dict(),
            },
            "train_state": dict(self.train_state),
            "memory_meta": {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
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
                     "high": np.asarray(payload["act_space"]["high"], np.float32),
                     "pos_only": tuple(payload["act_space"].get("pos_only", ()))}
        agent = cls(spec, act_space, cfg)
        agent.actor.load_state_dict(payload["torch"]["actor"])
        agent.critic.load_state_dict(payload["torch"]["critic"])
        agent.opt.load_state_dict(payload["torch"]["opt"])
        agent.train_state.update(payload.get("train_state", {}))
        return agent
