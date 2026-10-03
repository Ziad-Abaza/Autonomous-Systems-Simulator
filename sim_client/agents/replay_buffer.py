"""
Uniform-experience replay buffer for off-policy trainers (SAC, DQN).

Fixed-size circular storage of (obs, action, reward, next_obs, done)
transitions. Sampling is driven by a private Generator so batch draws are
deterministic under a given seed. `done` marks true termination only —
callers must not store time-limit truncations as terminal (they bootstrap).
"""

from __future__ import annotations
from typing import Any, Dict
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 discrete_actions: bool = False, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        act_dtype = np.int64 if discrete_actions else np.float32
        self.actions = np.zeros((capacity, act_dim), dtype=act_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        i = self.pos
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.actions[i] = np.asarray(action, dtype=self.actions.dtype)
        self.rewards[i] = float(reward)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.dones[i] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self._rng.integers(0, self.size, size=min(batch_size, self.size))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size
