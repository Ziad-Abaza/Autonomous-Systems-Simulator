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
    """
    def __init__(
        self,
        env: SimulationEnvironment,
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
        device: str = "cpu"
    ):
        self.env = env
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

        # Fix RNG seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine dimensions
        obs_sample = env.reset(seed=seed)[0]
        if isinstance(obs_sample, dict):
            obs_sample = obs_sample.get('vector', np.zeros(23, dtype=np.float32))
        self.obs_dim = obs_sample.shape[0]
        self.act_dim = 3  # [steer, throttle, brake]

        # Policy & Optimizer
        self.agent = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        # Rollout Storage Buffers
        self.obs_buf = torch.zeros((num_steps, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions_buf = torch.zeros((num_steps, self.act_dim), dtype=torch.float32, device=self.device)
        self.logprobs_buf = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        self.rewards_buf = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        self.dones_buf = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
        self.values_buf = torch.zeros(num_steps, dtype=torch.float32, device=self.device)

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

    def train(self, total_timesteps: int = 50000, log_interval: int = 2048) -> Dict[str, Any]:
        """Runs the PPO training loop for total_timesteps."""
        start_time = time.perf_counter()
        obs, info = self.env.reset(seed=self.seed)
        if isinstance(obs, dict):
            obs = obs.get('vector', np.zeros(self.obs_dim, dtype=np.float32))
        next_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_done_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        num_updates = max(1, total_timesteps // self.num_steps)
        global_step = 0

        # Episode stats accumulators
        ep_return = 0.0
        ep_len = 0
        ep_lat_errors = []
        ep_speeds = []

        print("=" * 75)
        print(f"  PPO TRAINING BENCHMARK STARTING: {total_timesteps} STEPS ({num_updates} UPDATES)")
        print(f"  Obs Dim: {self.obs_dim} | Act Dim: {self.act_dim} | Rollout Steps: {self.num_steps}")
        print("=" * 75)

        for update in range(1, num_updates + 1):
            # 1. Rollout Collection
            for step in range(self.num_steps):
                global_step += 1
                self.obs_buf[step] = next_obs_tensor
                self.dones_buf[step] = next_done_tensor

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs_tensor.unsqueeze(0))
                    self.values_buf[step] = value.squeeze()
                self.actions_buf[step] = action.squeeze(0)
                self.logprobs_buf[step] = logprob.squeeze()

                # Step simulation
                act_np = action.squeeze(0).cpu().numpy()
                clamped_act = [
                    float(np.clip(act_np[0], -1.0, 1.0)),
                    float(np.clip(act_np[1], 0.0, 1.0)),
                    float(np.clip(act_np[2], 0.0, 1.0))
                ]
                next_obs, reward, terminated, truncated, step_info = self.env.step(clamped_act)
                if isinstance(next_obs, dict):
                    next_obs = next_obs.get('vector', np.zeros(self.obs_dim, dtype=np.float32))

                self.rewards_buf[step] = float(reward)
                done = terminated or truncated
                next_done_tensor = torch.tensor(1.0 if done else 0.0, dtype=torch.float32, device=self.device)
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

                ep_return += reward
                ep_len += 1
                ep_lat_errors.append(abs(step_info.get('lateral_offset', 0.0)))
                ep_speeds.append(step_info.get('speed', 0.0))

                if done:
                    self.metrics['episodes_completed'] += 1
                    self.metrics['episode_returns'].append(float(ep_return))
                    self.metrics['episode_lengths'].append(ep_len)
                    self.metrics['mean_lateral_errors'].append(float(np.mean(ep_lat_errors)) if ep_lat_errors else 0.0)
                    self.metrics['mean_speeds'].append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
                    self.metrics['completion_rates'].append(step_info.get('checkpoints_passed', 0))

                    reason = step_info.get('termination_reason', '')
                    if 'collision' in reason:
                        self.metrics['collision_count'] += 1
                    elif 'off_road' in reason:
                        self.metrics['off_road_count'] += 1

                    ep_return = 0.0
                    ep_len = 0
                    ep_lat_errors = []
                    ep_speeds = []
                    reset_obs, _ = self.env.reset()
                    if isinstance(reset_obs, dict):
                        reset_obs = reset_obs.get('vector', np.zeros(self.obs_dim, dtype=np.float32))
                    next_obs_tensor = torch.tensor(reset_obs, dtype=torch.float32, device=self.device)

            # 2. Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs_tensor.unsqueeze(0)).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards_buf)
                lastgaelam = 0.0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_tensor
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones_buf[t + 1]
                        nextvalues = self.values_buf[t + 1]
                    delta = self.rewards_buf[t] + self.gamma * nextvalues * nextnonterminal - self.values_buf[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
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
            current_sps = global_step / max(1e-4, elapsed)
            self.metrics['policy_losses'].append(float(pg_loss.item()))
            self.metrics['value_losses'].append(float(v_loss.item()))
            self.metrics['entropies'].append(float(entropy_loss.item()))
            self.metrics['approx_kls'].append(float(approx_kl.item()))
            self.metrics['explained_variances'].append(float(explained_var))

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

    def save_checkpoint(self, path: str) -> None:
        """Saves model weights and optimizer state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'seed': self.seed,
            'metrics': self.metrics
        }, path)
        print(f"[PPO Runner] Checkpoint saved to {path}")
