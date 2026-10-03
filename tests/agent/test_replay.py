"""ReplayBuffer + TrackRehearsalBuffer tests."""
from __future__ import annotations

import numpy as np

from agentRL.memory.replay import ReplayBuffer, TrackRehearsalBuffer


def _fill(buf, n, obs_dim=4, act_dim=3, start=0):
    for i in range(start, start + n):
        buf.add(np.full(obs_dim, i, dtype=np.float32),
                np.zeros(act_dim, dtype=np.float32),
                float(i), np.full(obs_dim, i + 1, dtype=np.float32),
                terminated=False, truncated=False)


def test_buffer_capacity_eviction():
    buf = ReplayBuffer(capacity=100, obs_dim=4, act_dim=3, seed=0)
    _fill(buf, 150)
    assert len(buf) == 100
    # oldest evicted: stored obs values should be in [50, 150)
    batch = buf.sample(1000)
    assert batch["obs"].min() >= 50.0


def test_sample_shapes():
    buf = ReplayBuffer(capacity=1000, obs_dim=4, act_dim=3, seed=0)
    _fill(buf, 500)
    b = buf.sample(64)
    assert b["obs"].shape == (64, 4)
    assert b["action"].shape == (64, 3)
    assert b["next_obs"].shape == (64, 4)
    for k in ("reward", "done"):
        assert b[k].shape == (64,)


def test_done_flag_semantics():
    """terminated=True is absorbing; truncated=True is NOT (bootstrap)."""
    buf = ReplayBuffer(capacity=10, obs_dim=2, act_dim=1, seed=0)
    buf.add(np.zeros(2), np.zeros(1), 1.0, np.ones(2),
            terminated=True, truncated=False)
    buf.add(np.zeros(2), np.zeros(1), 1.0, np.ones(2),
            terminated=False, truncated=True)
    buf.add(np.zeros(2), np.zeros(1), 1.0, np.ones(2),
            terminated=True, truncated=True)  # env's both-true case
    # term-only -> 1.0 ; trunc-only -> 0.0 ; both -> 0.0 (truncation wins)
    assert sorted(buf.done[:3].tolist()) == [0.0, 0.0, 1.0]


def test_rehearsal_fraction():
    rb = TrackRehearsalBuffer(per_track_capacity=1000,
                            rehearsal_fraction=0.25, seed=0,
                            obs_dim=4, act_dim=3)
    rb.begin_track("A")
    for i in range(500):
        rb.add(np.full(4, 1.0, dtype=np.float32), np.zeros(3), 1.0,
               np.zeros(4), terminated=False, truncated=False)
    rb.begin_track("B")
    for i in range(500):
        rb.add(np.full(4, 9.0, dtype=np.float32), np.zeros(3), 1.0,
               np.zeros(4), terminated=False, truncated=False)
    b = rb.sample(2000)
    frac_a = float((b["obs"][:, 0] == 1.0).mean())
    assert 0.15 < frac_a < 0.35, f"rehearsal fraction off: {frac_a}"


def test_state_roundtrip():
    rb = TrackRehearsalBuffer(per_track_capacity=100,
                            rehearsal_fraction=0.25, seed=0,
                            obs_dim=4, act_dim=3)
    rb.begin_track("A")
    for i in range(50):
        rb.add(np.zeros(4), np.zeros(3), 1.0, np.zeros(4),
               terminated=False, truncated=False)
    st = rb.state_dict()
    assert st["sizes"]["A"] == 50  # per-track counts recorded
    rb2 = TrackRehearsalBuffer(per_track_capacity=100,
                               rehearsal_fraction=0.25, seed=0,
                               obs_dim=4, act_dim=3)
    rb2.load_state_dict(st)
    # metadata restore: track structure + active buffer, contents regathered
    assert rb2.track_order == ["A"]
    assert rb2.current == "A"
