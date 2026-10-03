"""ReplayBuffer and TrackRehearsalBuffer.

done semantics (governing contract, per audit):
- terminated=True, truncated=False -> absorbing, done=1
- terminated=False, truncated=True -> time-limit, done=0 (bootstrap)
- both True (env's step-after-done invalid reply) -> treated as
  truncation, done=0 — never reward an invalid call as absorbing.

TrackRehearsalBuffer is the continual-learning mechanism for off-policy
agents: each phase's transitions live in a per-track buffer; sampling
mixes `rehearsal_fraction` of the batch from previous tracks.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs,
            terminated: bool, truncated: bool) -> None:
        done = 1.0 if (terminated and not truncated) else 0.0
        i = self._ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=int(batch))
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }


class TrackRehearsalBuffer:
    """Per-track replay buffers with controlled cross-track rehearsal."""

    def __init__(self, per_track_capacity: int, rehearsal_fraction: float,
                 seed: int, obs_dim: int, act_dim: int):
        self.per_track_capacity = int(per_track_capacity)
        self.rehearsal_fraction = float(rehearsal_fraction)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.buffers: dict[str, ReplayBuffer] = {}
        self.track_order: list[str] = []
        self.current: str | None = None

    def begin_track(self, track_id: str) -> None:
        """Rotate the active buffer to track_id (previous buffers kept)."""
        if track_id not in self.buffers:
            self.buffers[track_id] = ReplayBuffer(
                self.per_track_capacity, self.obs_dim, self.act_dim,
                seed=self._seed + len(self.track_order))
            self.track_order.append(track_id)
        self.current = track_id

    def add(self, obs, action, reward, next_obs,
            terminated: bool, truncated: bool) -> None:
        if self.current is None:
            raise RuntimeError("begin_track() before add()")
        self.buffers[self.current].add(obs, action, reward, next_obs,
                                       terminated, truncated)

    def size_of(self, track_id: str) -> int:
        return len(self.buffers.get(track_id, ReplayBuffer(1, 1, 1)))

    def _previous_tracks(self) -> list[str]:
        if self.current is None:
            return []
        seen = self.track_order[: self.track_order.index(self.current)]
        return [t for t in seen if len(self.buffers[t]) > 0]

    def sample(self, batch: int) -> dict[str, np.ndarray]:
        cur = self.buffers[self.current]
        prev = self._previous_tracks()
        n_prev = int(batch * self.rehearsal_fraction) if prev else 0
        n_prev = min(n_prev, sum(len(self.buffers[t]) for t in prev))
        parts = [cur.sample(batch - n_prev)] if len(cur) else []
        if n_prev:
            per = max(1, n_prev // len(prev))
            for t in prev:
                parts.append(self.buffers[t].sample(per))
        merged = {
            k: np.concatenate([p[k] for p in parts]) for k in
            ("obs", "action", "reward", "next_obs", "done")
        }
        order = self._rng.permutation(len(merged["obs"]))
        return {k: v[order] for k, v in merged.items()}

    def state_dict(self) -> dict:
        return {
            "track_order": list(self.track_order),
            "current": self.current,
            "sizes": {t: len(b) for t, b in self.buffers.items()},
            "rehearsal_fraction": self.rehearsal_fraction,
        }

    def load_state_dict(self, state: dict) -> None:
        # Metadata only — buffer contents are not checkpointed (capacity-
        # bounded experience is re-gathered; optimizer/model state carries
        # the learned knowledge). Track order restored so rehearsal keeps
        # its semantics across resume.
        for tid in state.get("track_order", []):
            self.begin_track(tid)
        if state.get("current"):
            self.begin_track(state["current"])
