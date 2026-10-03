"""
PyTorch Proximal Policy Optimization (PPO) reinforcement learning training script.
Demonstrates training a neural network policy to drive autonomously.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from sim_client.gym_env import SimGymEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        # Actor head (mean)
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        return mean, value

    def get_action(self, obs: np.ndarray):
        x = torch.as_tensor(obs, dtype=torch.float32)
        mean, value = self.forward(x)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().numpy(), log_prob.item(), value.item()


def train_ppo(host: str = "127.0.0.1", port: int = 8765, total_timesteps: int = 5000):
    print(f"[PPO Train] Initializing Gym environment on {host}:{port}...")
    env = SimGymEnv(host=host, port=port)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    obs, info = env.reset()
    episode_reward = 0.0
    episodes = 0

    print(f"[PPO Train] Starting training loop for {total_timesteps} timesteps...")
    for t in range(total_timesteps):
        action, log_prob, val = policy.get_action(obs)
        # Clamp action to [-1, 1] for steer, [0, 1] for throttle/brake
        clamped_action = [
            float(np.clip(action[0], -1.0, 1.0)),
            float(np.clip(action[1], 0.0, 1.0)),
            float(np.clip(action[2], 0.0, 1.0)),
        ]
        next_obs, reward, terminated, truncated, info = env.step(clamped_action)
        episode_reward += reward

        if terminated or truncated:
            episodes += 1
            print(f"Episode {episodes:3d} | Steps: {t:5d} | Return: {episode_reward:7.2f} | Reason: {info.get('termination_reason')}")
            episode_reward = 0.0
            obs, info = env.reset()
        else:
            obs = next_obs

    env.close()
    print("[PPO Train] Training run finished.")


if __name__ == "__main__":
    train_ppo()
