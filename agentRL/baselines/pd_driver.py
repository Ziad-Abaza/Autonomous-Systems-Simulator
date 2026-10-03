"""PD lane-follower + demonstration collection for warmstarting.

A hand-tuned controller on the standard obs channels. Calibrated on the
oval: survives 1500+ steps, passes ~5 checkpoints at moderate speed —
enough to seed a replay buffer with positive-progress transitions so a
critic sees the value landscape early (SAC+D style). The learned policy
is not constrained by the demos; they decay under FIFO eviction.

Channel layout (both state8 and full23 presets):
  [0] speed (norm x45 m/s), [3] yaw_rate,
  [5] distance_from_center (norm to half-width),
  [6] heading_error (norm by pi)
"""
from __future__ import annotations

import numpy as np

from agentRL.act.adapter import ActionAdapter

_SPD, _YAW, _LAT, _HEAD = 0, 3, 5, 6


class PDPolicy:
    """steer = k_lat*lat + k_head*head - k_yaw*yaw_rate; speed in corners."""

    algo_id = "pd_demo"
    obs_spec = None

    def __init__(self, vmax: float = 14.0, corner_slow: float = 40.0,
                 k_lat: float = 4.0, k_head: float = 2.0,
                 k_yaw: float = -0.8, k_spd: float = 0.15):
        self.vmax = vmax
        self.corner_slow = corner_slow
        self.k_lat, self.k_head, self.k_yaw = k_lat, k_head, k_yaw
        self.k_spd = k_spd

    def act(self, obs: np.ndarray, deterministic: bool = True
            ) -> tuple[np.ndarray, dict]:
        speed_ms = float(obs[_SPD]) * 45.0
        target = max(3.0, self.vmax - self.corner_slow * abs(float(obs[_HEAD])))
        steer = np.clip(self.k_lat * float(obs[_LAT])
                        + self.k_head * float(obs[_HEAD])
                        + self.k_yaw * float(obs[_YAW]), -1.0, 1.0)
        throttle = float(np.clip((target - speed_ms) * self.k_spd, 0.0, 1.0))
        brake = float(min(0.6, max(0.0, speed_ms - target - 2.0) * 0.1))
        return np.array([steer, throttle, brake], dtype=np.float32), {}


def collect_demos(env, adapter: ActionAdapter, memory, encoder,
                  n_steps: int, vmax: float = 14.0,
                  mutator=None, max_ep_steps: int = 2000) -> int:
    """Roll PD demos and write transitions into an off-policy buffer.

    Transitions use ENCODED obs and agent-space actions — identical to
    what agent.observe() records during normal training.
    """
    pd = PDPolicy(vmax=vmax)
    if memory.current is None:
        memory.begin_track("demos")
    total = 0
    while total < n_steps:
        if mutator is not None:
            mutator.draw_and_apply(env)
        obs, _ = env.reset()
        encoder.reset()
        enc = encoder.encode(np.asarray(obs, np.float32), None)
        for _ in range(max_ep_steps):
            a_env, _ = pd.act(np.asarray(obs, np.float32))
            nobs, r, term, trunc, info = env.step(a_env)
            nenc = encoder.encode(np.asarray(nobs, np.float32),
                                  info.get("last_action", a_env))
            memory.add(enc, adapter.from_env(a_env), float(r), nenc,
                       bool(term), bool(trunc))
            enc = nenc
            obs = nobs
            total += 1
            if term or trunc or total >= n_steps:
                break
    return total
