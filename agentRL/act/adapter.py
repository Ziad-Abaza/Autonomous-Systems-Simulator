"""ActionAdapter — policy space [-1,1]^3 <-> env action bounds.

Policies always output normalized actions in [-1,1]. The adapter maps
them affinely into the env's per-channel bounds (steer [-1,1],
throttle [0,1], brake [0,1]) and tracks the last commanded env-space
action for prev_action observations. The env applies its own
dead-zone/rate-limit/validation downstream — we never double-clip
beyond the declared contract.
"""
from __future__ import annotations

import numpy as np


class ActionAdapter:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.dim = int(self.low.size)
        self._prev: np.ndarray | None = None

    def to_env(self, action: np.ndarray,
               validate: bool = False) -> np.ndarray | tuple[np.ndarray, bool]:
        a = np.asarray(action, dtype=np.float32).ravel()
        valid = bool(np.isfinite(a).all()) and a.size == self.dim
        if not valid:
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
            if a.size != self.dim:
                a = np.zeros(self.dim, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        env = self.low + (a + 1.0) * 0.5 * (self.high - self.low)
        env = env.astype(np.float32)
        self._prev = env.copy()
        return (env, valid) if validate else env

    def from_env(self, env_action: np.ndarray) -> np.ndarray:
        e = np.asarray(env_action, dtype=np.float32).ravel()
        span = np.maximum(self.high - self.low, 1e-8)
        return (2.0 * (e - self.low) / span - 1.0).astype(np.float32)

    @property
    def prev_action(self) -> np.ndarray | None:
        return self._prev

    def reset(self) -> None:
        self._prev = None
