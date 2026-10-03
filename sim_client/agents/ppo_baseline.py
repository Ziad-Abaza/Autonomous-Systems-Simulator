"""
Complete Proximal Policy Optimization (PPO) reinforcement learning implementation.
Includes ActorCritic neural network, RolloutBuffer with Generalized Advantage Estimation (GAE),
clipped surrogate loss, value function clipping, entropy bonus, and diagnostic metrics.
"""

from __future__ import annotations
import math
import time
import json
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from sim_env.environment import SimulationEnvironment
from sim_env.spaces import ObservationSchema, ActionSpaceConfig
from sim_project.presets import create_oval_circuit


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal parameter initialization for deep RL stability."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Continuous action Actor-Critic network.
    Outputs mean action and state value, with learnable diagonal Gaussian log-std.
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Shared feature representation
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        # Actor mean head (low gain for initial exploration)
        self.actor_mean = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

        # Critic value head
        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_backbone(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.actor_backbone(x)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(axis=-1)
        entropy = probs.entropy().sum(axis=-1)
        value = self.get_value(x)
        return action, log_prob, entropy, value.squeeze(-1)


class PPORunner:
    """
    Executes reproducible PPO training against the hardened AI environment.
    Supports multiple parallel environments (contiguous per-env rollout
    segments with per-segment GAE) and optional on_step/on_update hooks for
    training-platform metrics, checkpoints, and trajectory capture.
    """
    def __init__(
        self,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_steps: int = 1024,
        num_epochs: int = 4,
        batch_size: int = 256,
        seed: int = 42,
        device: str = "cpu",
        on_update=None,
        on_step=None,
        resume_checkpoint: Optional[str] = None
    ):
        # env may be a single environment, a list of independent envs, or a
        # VectorEnv (batched step_all/reset_all API — e.g. ProcessVectorEnv).
        from sim_experiment.vec_env import is_vec_env
        self._venv = env if is_vec_env(env) else None
        if self._venv is not None:
            self.envs = None
            self.env = None
            num_envs = self._venv.num_envs
        else:
            self.envs = list(env) if isinstance(env, (list, tuple)) else [env]
            self.env = self.envs[0]
            num_envs = len(self.envs)
        self.num_envs = num_envs
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.on_update = on_update
        self.on_step = on_step
        self.global_step_offset = 0

        # Fix RNG seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine dimensions
        if self._venv is not None:
            obs_sample = self._venv.reset_at(0, seed=seed)
        else:
            obs_sample = self.envs[0].reset(seed=seed)[0]
        if isinstance(obs_sample, dict):
            obs_sample = obs_sample.get('vector', np.zeros(23, dtype=np.float32))
        self.obs_dim = obs_sample.shape[0]
        self.act_dim = 3  # [steer, throttle, brake]

        # Policy & Optimizer
        self.agent = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        # Effective rollout length is a multiple of the env count so every env
        # contributes a contiguous segment (per-segment GAE stays correct).
        self.steps_per_env = max(1, num_steps // num_envs)
        self.num_steps = self.steps_per_env * num_envs

        # Rollout Storage Buffers
        self.obs_buf = torch.zeros((self.num_steps, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions_buf = torch.zeros((self.num_steps, self.act_dim), dtype=torch.float32, device=self.device)
        self.logprobs_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.rewards_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.dones_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.values_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        # terminated-only flags + V(final_obs) at truncation: time-limit
        # truncations bootstrap the value of the final observation instead of
        # being treated as terminal (standard episodic-MDP handling).
        self.terms_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.final_values_buf = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)

        # Env-swap / deterministic reset bookkeeping. _envs_dirty forces a
        # fresh re-initialization (reset with derived seeds) at the start of
        # the next update; _env_epoch bumps on every set_envs() swap so seed
        # streams never repeat across curriculum stages.
        self._envs_dirty = True
        self._env_epoch = 0
        self._reset_counts = [1] * num_envs
        # Exact per-episode reset seeds (for trajectory header provenance).
        self._ep_seeds: List[Optional[int]] = [None] * num_envs

        # Resume from a previous checkpoint if requested
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        # Performance and training metrics
        self.metrics: Dict[str, Any] = {
            'total_timesteps': 0,
            'episodes_completed': 0,
            'episode_returns': [],
            'episode_lengths': [],
            'mean_lateral_errors': [],
            'mean_speeds': [],
            'completion_rates': [],
            'collision_count': 0,
            'off_road_count': 0,
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'approx_kls': [],
            'explained_variances': [],
            'sps': 0.0,
            'wall_clock_time': 0.0
        }

    def set_envs(self, envs) -> None:
        """
        Atomically swaps the environment set (e.g. curriculum stage change).
        The rollout state is re-initialized at the start of the next update;
        env count must stay constant because rollout buffers are fixed-size.
        Accepts a plain env list (sequential mode) or a VectorEnv matching
        the current mode — modes cannot be mixed.
        """
        from sim_experiment.vec_env import is_vec_env
        if is_vec_env(envs):
            if self._venv is None:
                raise ValueError("cannot swap a VectorEnv into a list-mode runner")
            if envs.num_envs != self.num_envs:
                raise ValueError(
                    f"set_envs requires exactly {self.num_envs} envs, got {envs.num_envs}"
                )
            self._venv = envs
        else:
            if self._venv is not None:
                raise ValueError("cannot swap an env list into a vec-mode runner")
            new_envs = list(envs) if isinstance(envs, (list, tuple)) else [envs]
            if len(new_envs) != self.num_envs:
                raise ValueError(
                    f"set_envs requires exactly {self.num_envs} envs, got {len(new_envs)}"
                )
            self.envs = new_envs
            self.env = new_envs[0]
        self._env_epoch += 1
        self._reset_counts = [1] * self.num_envs
        self._envs_dirty = True

    def _env_reset(self, env_idx: int):
        """Deterministic reseed+reset for one env (either backend)."""
        seed = self._next_reset_seed(env_idx)
        self._ep_seeds[env_idx] = seed
        if self._venv is not None:
            return self._venv.reset_at(env_idx, seed=seed)
        return self.envs[env_idx].reset(seed=seed)[0]

    def _next_reset_seed(self, env_idx: int) -> int:
        """Deterministic per-(epoch, env, reset) seed — distinct every reset."""
        seed = (self.seed + self._env_epoch * 1_000_003
                + env_idx * 79_919 + self._reset_counts[env_idx])
        self._reset_counts[env_idx] += 1
        return int(seed)

    def _obs_to_vec(self, o):
        if isinstance(o, dict):
            o = o.get('vector', np.zeros(self.obs_dim, dtype=np.float32))
        return o

    def _consume_step(self, idx, env_idx, next_obs_raw, reward, terminated,
                      truncated, step_info, S, clamped_act):
        """
        Shared post-step bookkeeping for both rollout modes: buffer writes,
        truncation bootstrap, episode accounting, on_step hook, done reset.
        `S` carries the mutable per-update rollout state.
        """
        next_obs = self._obs_to_vec(next_obs_raw)

        self.rewards_buf[idx] = float(reward)
        done = terminated or truncated
        self.terms_buf[idx] = 1.0 if terminated else 0.0
        if truncated and not terminated:
            # Truncation is not terminal: bootstrap V(final_obs) so the
            # time limit does not fake a death. Computed BEFORE reset.
            with torch.no_grad():
                self.final_values_buf[idx] = self.agent.get_value(
                    torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).squeeze()
        else:
            self.final_values_buf[idx] = 0.0
        S['next_done'][env_idx] = torch.tensor(1.0 if done else 0.0, dtype=torch.float32, device=self.device)
        S['next_obs'][env_idx] = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        S['ep_return'][env_idx] += reward
        S['ep_len'][env_idx] += 1
        S['ep_lat'][env_idx].append(abs(step_info.get('lateral_offset', 0.0)))
        S['ep_spd'][env_idx].append(step_info.get('speed', 0.0))

        if self.on_step is not None:
            self.on_step({
                'env_idx': env_idx,
                'global_step': S['gsteps'][idx],
                'episode_seed': self._ep_seeds[env_idx],
                'obs': next_obs,
                'action': clamped_act,
                'reward': float(reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'info': step_info,
            })

        if done:
            ep_stats = {
                'env_idx': env_idx,
                'return': float(S['ep_return'][env_idx]),
                # env-reported step count is authoritative (single
                # source of truth; guards against counter drift)
                'length': int(step_info.get('step', S['ep_len'][env_idx])),
                'termination_reason': step_info.get('termination_reason', ''),
                'mean_lateral_error': float(np.mean(S['ep_lat'][env_idx])) if S['ep_lat'][env_idx] else 0.0,
                'mean_speed': float(np.mean(S['ep_spd'][env_idx])) if S['ep_spd'][env_idx] else 0.0,
                'checkpoints_passed': step_info.get('checkpoints_passed', 0),
                'is_colliding': bool(step_info.get('is_colliding', False)),
                'is_on_road': bool(step_info.get('is_on_road', True)),
            }
            S['new_episodes'].append(ep_stats)

            self.metrics['episodes_completed'] += 1
            self.metrics['episode_returns'].append(ep_stats['return'])
            self.metrics['episode_lengths'].append(ep_stats['length'])
            self.metrics['mean_lateral_errors'].append(ep_stats['mean_lateral_error'])
            self.metrics['mean_speeds'].append(ep_stats['mean_speed'])
            self.metrics['completion_rates'].append(ep_stats['checkpoints_passed'])

            reason = ep_stats['termination_reason']
            if 'collision' in reason:
                self.metrics['collision_count'] += 1
            elif 'off_road' in reason:
                self.metrics['off_road_count'] += 1

            S['ep_return'][env_idx] = 0.0
            S['ep_len'][env_idx] = 0
            S['ep_lat'][env_idx] = []
            S['ep_spd'][env_idx] = []
            reset_obs = self._env_reset(env_idx)
            S['next_obs'][env_idx] = torch.tensor(
                self._obs_to_vec(reset_obs), dtype=torch.float32, device=self.device)

    def _clamp_act(self, act_np):
        return [
            float(np.clip(act_np[0], -1.0, 1.0)),
            float(np.clip(act_np[1], 0.0, 1.0)),
            float(np.clip(act_np[2], 0.0, 1.0))
        ]

    def train(self, total_timesteps: int = 50000, log_interval: int = 2048) -> Dict[str, Any]:
        """Runs the PPO training loop for total_timesteps."""
        start_time = time.perf_counter()
        num_envs = self.num_envs
        spe = self.steps_per_env
        _to_vec = self._obs_to_vec

        # Per-env rollout state (initialized lazily via the _envs_dirty flag
        # so set_envs() mid-training triggers a clean re-reset).
        next_obs_tensors, next_done_tensors = [], []
        ep_return = [0.0] * num_envs
        ep_len = [0] * num_envs
        ep_lat_errors: List[List[float]] = [[] for _ in range(num_envs)]
        ep_speeds: List[List[float]] = [[] for _ in range(num_envs)]

        num_updates = max(1, total_timesteps // self.num_steps)
        global_step = self.global_step_offset

        print("=" * 75)
        print(f"  PPO TRAINING STARTING: {total_timesteps} STEPS ({num_updates} UPDATES, {num_envs} ENVS)")
        print(f"  Obs Dim: {self.obs_dim} | Act Dim: {self.act_dim} | Rollout Steps: {self.num_steps}")
        print("=" * 75)

        for update in range(1, num_updates + 1):
            if self._envs_dirty:
                self._envs_dirty = False
                next_obs_tensors, next_done_tensors = [], []
                ep_return = [0.0] * num_envs
                ep_len = [0] * num_envs
                ep_lat_errors = [[] for _ in range(num_envs)]
                ep_speeds = [[] for _ in range(num_envs)]
                if self._venv is not None:
                    seeds = [self._next_reset_seed(e) for e in range(num_envs)]
                    self._ep_seeds = list(seeds)
                    for o in self._venv.reset_all(seeds):
                        next_obs_tensors.append(torch.tensor(_to_vec(o), dtype=torch.float32, device=self.device))
                        next_done_tensors.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
                else:
                    for e, env in enumerate(self.envs):
                        s_e = self._next_reset_seed(e)
                        self._ep_seeds[e] = s_e
                        o, _ = env.reset(seed=s_e)
                        next_obs_tensors.append(torch.tensor(_to_vec(o), dtype=torch.float32, device=self.device))
                        next_done_tensors.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))

            new_episodes: List[Dict[str, Any]] = []
            S = {
                'next_obs': next_obs_tensors, 'next_done': next_done_tensors,
                'ep_return': ep_return, 'ep_len': ep_len,
                'ep_lat': ep_lat_errors, 'ep_spd': ep_speeds,
                'new_episodes': new_episodes,
                'gsteps': [0] * self.num_steps,
            }

            # 1. Rollout Collection
            if self._venv is not None:
                # Batched: every env steps once per tick (real parallelism
                # for process workers). Buffer layout stays per-env
                # contiguous (idx = e*spe + t) so GAE is unchanged.
                for t in range(spe):
                    acts = []
                    for e in range(num_envs):
                        idx = e * spe + t
                        global_step += 1
                        S['gsteps'][idx] = global_step
                        self.obs_buf[idx] = next_obs_tensors[e]
                        self.dones_buf[idx] = next_done_tensors[e]
                        with torch.no_grad():
                            action, logprob, _, value = self.agent.get_action_and_value(next_obs_tensors[e].unsqueeze(0))
                            self.values_buf[idx] = value.squeeze()
                        self.actions_buf[idx] = action.squeeze(0)
                        self.logprobs_buf[idx] = logprob.squeeze()
                        acts.append(self._clamp_act(action.squeeze(0).cpu().numpy()))
                    results = self._venv.step_all(acts)
                    for e, (n_obs, rew, term, trunc, sinfo) in enumerate(results):
                        self._consume_step(e * spe + t, e, n_obs, rew, term,
                                           trunc, sinfo, S, acts[e])
                    next_obs_tensors = S['next_obs']
                    next_done_tensors = S['next_done']
            else:
                # Sequential: contiguous segment per environment.
                for step in range(self.num_steps):
                    env_idx = step // spe
                    env = self.envs[env_idx]
                    global_step += 1
                    S['gsteps'][step] = global_step
                    self.obs_buf[step] = next_obs_tensors[env_idx]
                    self.dones_buf[step] = next_done_tensors[env_idx]

                    with torch.no_grad():
                        action, logprob, _, value = self.agent.get_action_and_value(next_obs_tensors[env_idx].unsqueeze(0))
                        self.values_buf[step] = value.squeeze()
                    self.actions_buf[step] = action.squeeze(0)
                    self.logprobs_buf[step] = logprob.squeeze()

                    # Step simulation
                    clamped_act = self._clamp_act(action.squeeze(0).cpu().numpy())
                    next_obs, reward, terminated, truncated, step_info = env.step(clamped_act)
                    self._consume_step(step, env_idx, next_obs, reward,
                                       terminated, truncated, step_info, S,
                                       clamped_act)
                    next_obs_tensors = S['next_obs']
                    next_done_tensors = S['next_done']

            # 2. Generalized Advantage Estimation (per contiguous env segment)
            with torch.no_grad():
                advantages = torch.zeros_like(self.rewards_buf)
                for e in range(num_envs):
                    lastgaelam = 0.0
                    for t in reversed(range(spe)):
                        idx = e * spe + t
                        if t == spe - 1:
                            # Segment boundary: carry stops if the step ended an
                            # episode; value bootstrap uses V(next_obs) only when
                            # the episode is still running.
                            cnon = 1.0 - next_done_tensors[e]
                            if next_done_tensors[e].item() == 0.0:
                                nextvalues = self.agent.get_value(
                                    next_obs_tensors[e].unsqueeze(0)).reshape(1, -1)
                            else:
                                nextvalues = self.final_values_buf[idx]
                        else:
                            # Carry stops at any episode end (terminated or
                            # truncated); value bootstrap uses V(s_{t+1}) for
                            # continuing steps, V(final_obs) for truncations,
                            # and 0 for true termination (via vnon below).
                            cnon = 1.0 - self.dones_buf[idx + 1]
                            if self.dones_buf[idx + 1].item() == 0.0:
                                nextvalues = self.values_buf[idx + 1]
                            else:
                                nextvalues = self.final_values_buf[idx]
                        vnon = 1.0 - self.terms_buf[idx]
                        delta = self.rewards_buf[idx] + self.gamma * nextvalues * vnon - self.values_buf[idx]
                        advantages[idx] = lastgaelam = delta + self.gamma * self.gae_lambda * cnon * lastgaelam
                returns = advantages + self.values_buf

            # 3. PPO Optimization Epochs
            b_obs = self.obs_buf.reshape((-1, self.obs_dim))
            b_logprobs = self.logprobs_buf.reshape(-1)
            b_actions = self.actions_buf.reshape((-1, self.act_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values_buf.reshape(-1)

            # Advantage normalization
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            b_inds = np.arange(self.num_steps)
            for epoch in range(self.num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.num_steps, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    # Clipped Surrogate Objective
                    mb_advantages = b_advantages[mb_inds]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value function loss with clipping
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Explained variance computation
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

            # Record iteration diagnostics
            elapsed = time.perf_counter() - start_time
            local_steps = global_step - self.global_step_offset
            current_sps = local_steps / max(1e-4, elapsed)
            self.metrics['policy_losses'].append(float(pg_loss.item()))
            self.metrics['value_losses'].append(float(v_loss.item()))
            self.metrics['entropies'].append(float(entropy_loss.item()))
            self.metrics['approx_kls'].append(float(approx_kl.item()))
            self.metrics['explained_variances'].append(float(explained_var))

            if self.on_update is not None:
                self.on_update({
                    'update': update,
                    'num_updates': num_updates,
                    'global_step': int(global_step),
                    'sps': float(current_sps),
                    'policy_loss': float(pg_loss.item()),
                    'value_loss': float(v_loss.item()),
                    'entropy': float(entropy_loss.item()),
                    'approx_kl': float(approx_kl.item()),
                    'explained_variance': float(explained_var),
                    'episodes': new_episodes,
                    'episodes_completed': self.metrics['episodes_completed'],
                })

            if update % max(1, num_updates // 10) == 0 or update == num_updates:
                mean_r = np.mean(self.metrics['episode_returns'][-10:]) if self.metrics['episode_returns'] else 0.0
                mean_l = np.mean(self.metrics['episode_lengths'][-10:]) if self.metrics['episode_lengths'] else 0.0
                mean_lat = np.mean(self.metrics['mean_lateral_errors'][-10:]) if self.metrics['mean_lateral_errors'] else 0.0
                print(
                    f"Update {update:3d}/{num_updates} | "
                    f"Steps: {global_step:6d} | "
                    f"SPS: {current_sps:5.0f} | "
                    f"Mean Return: {mean_r:7.2f} | "
                    f"Mean Len: {mean_l:4.0f} | "
                    f"Lat Err: {mean_lat:4.2f}m | "
                    f"KL: {approx_kl.item():.4f} | "
                    f"Loss(π): {pg_loss.item():+.3f}"
                )

        # Finalize metrics
        total_wall_clock = time.perf_counter() - start_time
        final_sps = global_step / max(1e-4, total_wall_clock)
        self.metrics['total_timesteps'] = global_step
        self.metrics['wall_clock_time'] = round(total_wall_clock, 2)
        self.metrics['sps'] = round(final_sps, 1)

        total_eps = max(1, self.metrics['episodes_completed'])
        self.metrics['collision_rate'] = round(self.metrics['collision_count'] / total_eps, 3)
        self.metrics['off_road_rate'] = round(self.metrics['off_road_count'] / total_eps, 3)
        self.metrics['overall_mean_return'] = round(float(np.mean(self.metrics['episode_returns'])), 2) if self.metrics['episode_returns'] else 0.0
        self.metrics['overall_mean_length'] = round(float(np.mean(self.metrics['episode_lengths'])), 1) if self.metrics['episode_lengths'] else 0.0

        print("=" * 75)
        print(f"  PPO TRAINING COMPLETE: {global_step} STEPS in {total_wall_clock:.2f}s ({final_sps:.1f} SPS)")
        print(f"  Total Episodes: {self.metrics['episodes_completed']} | Mean Return: {self.metrics['overall_mean_return']}")
        print(f"  Collision Rate: {self.metrics['collision_rate']} | Off-Road Rate: {self.metrics['off_road_rate']}")
        print("=" * 75)

        return self.metrics

    def save_checkpoint(self, path: str, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Saves model weights and optimizer state to disk.

        `extra` merges caller-owned metadata (e.g. curriculum_state) into the
        checkpoint payload at top level so it survives resume.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'seed': self.seed,
            'metrics': self.metrics,
            'timestep': int(step if step is not None else self.metrics.get('total_timesteps', 0)),
        }
        payload.update(dict(extra or {}))
        torch.save(payload, path)
        print(f"[PPO Runner] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Restores model/optimizer state; subsequent timesteps continue the count.

        Returns the raw checkpoint payload so callers can read extension
        fields (e.g. curriculum_state) without a second torch.load.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.global_step_offset = int(ckpt.get('timestep') or ckpt.get('metrics', {}).get('total_timesteps', 0))
        print(f"[PPO Runner] Resumed from checkpoint {path} (timestep offset {self.global_step_offset})")
        return ckpt
